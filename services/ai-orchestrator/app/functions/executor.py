"""Function executor for running function calls."""

import asyncio
import structlog
import time
import traceback
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from app.functions.registry import FunctionRegistry, default_registry
from app.functions.schema import FunctionCall, FunctionResult


logger = structlog.get_logger()


class FunctionExecutionError(Exception):
    """Function execution error."""
    pass


class FunctionExecutor:
    """
    Executes function calls from LLM responses.

    Features:
    - Async and sync function support
    - Timeout handling
    - Error handling and reporting
    - Parallel execution
    - Confirmation handling
    """

    def __init__(
        self,
        registry: Optional[FunctionRegistry] = None,
        max_workers: int = 4,
        default_timeout: int = 30,
    ):
        self.registry = registry or default_registry
        self.max_workers = max_workers
        self.default_timeout = default_timeout

        # Thread pool for sync functions
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Execution statistics
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._total_execution_time = 0.0

    async def execute(
        self,
        call: FunctionCall,
        context: Optional[Dict[str, Any]] = None,
    ) -> FunctionResult:
        """
        Execute a single function call.

        Args:
            call: Function call to execute
            context: Execution context (session info, etc.)

        Returns:
            Function result
        """
        start_time = time.perf_counter()
        self._total_executions += 1

        logger.info(
            "function_execute_start",
            call_id=call.id,
            function=call.name,
            arguments=call.arguments,
        )

        try:
            # Get registered function
            registered = self.registry.get(call.name)
            if not registered:
                raise FunctionExecutionError(f"Unknown function: {call.name}")

            if not registered.enabled:
                raise FunctionExecutionError(f"Function disabled: {call.name}")

            # Validate arguments
            is_valid, error = self.registry.validate_call(call.name, call.arguments)
            if not is_valid:
                raise FunctionExecutionError(error)

            # Get timeout
            timeout = registered.schema.timeout_seconds or self.default_timeout

            # Execute function
            if registered.is_async:
                result = await asyncio.wait_for(
                    registered.handler(**call.arguments, _context=context)
                    if self._accepts_context(registered.handler)
                    else registered.handler(**call.arguments),
                    timeout=timeout,
                )
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._thread_pool,
                        lambda: registered.handler(**call.arguments),
                    ),
                    timeout=timeout,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._successful_executions += 1
            self._total_execution_time += duration_ms

            logger.info(
                "function_execute_success",
                call_id=call.id,
                function=call.name,
                duration_ms=round(duration_ms, 2),
            )

            return FunctionResult(
                call_id=call.id,
                name=call.name,
                success=True,
                result=result,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._failed_executions += 1

            logger.error(
                "function_execute_timeout",
                call_id=call.id,
                function=call.name,
                duration_ms=round(duration_ms, 2),
            )

            return FunctionResult(
                call_id=call.id,
                name=call.name,
                success=False,
                error=f"Function timed out after {self.default_timeout}s",
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._failed_executions += 1

            logger.error(
                "function_execute_error",
                call_id=call.id,
                function=call.name,
                error=str(e),
                traceback=traceback.format_exc(),
            )

            return FunctionResult(
                call_id=call.id,
                name=call.name,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    async def execute_batch(
        self,
        calls: List[FunctionCall],
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
    ) -> List[FunctionResult]:
        """
        Execute multiple function calls.

        Args:
            calls: List of function calls
            context: Execution context
            parallel: Execute in parallel if True

        Returns:
            List of function results
        """
        if not calls:
            return []

        if parallel:
            # Execute all in parallel
            tasks = [self.execute(call, context) for call in calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to results
            processed = []
            for call, result in zip(calls, results):
                if isinstance(result, Exception):
                    processed.append(FunctionResult(
                        call_id=call.id,
                        name=call.name,
                        success=False,
                        error=str(result),
                    ))
                else:
                    processed.append(result)
            return processed
        else:
            # Execute sequentially
            results = []
            for call in calls:
                result = await self.execute(call, context)
                results.append(result)
            return results

    def _accepts_context(self, handler) -> bool:
        """Check if handler accepts _context parameter."""
        import inspect
        sig = inspect.signature(handler)
        return "_context" in sig.parameters

    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        avg_time = (
            self._total_execution_time / self._total_executions
            if self._total_executions > 0
            else 0
        )

        return {
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "success_rate": round(
                self._successful_executions / max(1, self._total_executions), 3
            ),
            "average_execution_ms": round(avg_time, 2),
            "total_execution_time_ms": round(self._total_execution_time, 2),
        }

    def shutdown(self) -> None:
        """Shutdown executor."""
        self._thread_pool.shutdown(wait=True)


# Global executor instance
default_executor = FunctionExecutor()
