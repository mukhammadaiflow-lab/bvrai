"""
Flow Executor Engine.

Executes flows with support for dry-run, debug, and production modes.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import uuid

from ..config import ExecutionMode, NodeType, get_settings
from ..models import (
    Flow,
    NodeInstance,
    Connection,
    ExecutionContext,
    NodeExecutionResult,
    FlowExecutionResult,
)
from ..nodes import get_node_registry

logger = logging.getLogger(__name__)


class FlowExecutor:
    """
    Executes flows with variable substitution and node handling.

    Features:
    - Variable substitution with {{variable}} syntax
    - Dry-run mode for testing
    - Debug mode with step logging
    - Timeout handling
    - Error recovery
    """

    def __init__(self):
        """Initialize executor."""
        self.settings = get_settings()
        self.registry = get_node_registry()

        # Node handlers (maps node type to handler function)
        self._handlers: Dict[NodeType, Callable] = {}
        self._register_default_handlers()

    async def execute(
        self,
        flow: Flow,
        mode: ExecutionMode = ExecutionMode.DRY_RUN,
        input_vars: Optional[Dict[str, Any]] = None,
        mock_call: Optional[Dict[str, Any]] = None,
    ) -> FlowExecutionResult:
        """
        Execute a flow.

        Args:
            flow: Flow to execute
            mode: Execution mode
            input_vars: Input variables
            mock_call: Mock call data for testing

        Returns:
            FlowExecutionResult with outputs and node results
        """
        execution_id = str(uuid.uuid4())
        started_at = datetime.utcnow()

        # Create context
        context = ExecutionContext(
            execution_id=execution_id,
            flow_id=flow.id,
            mode=mode,
            started_at=started_at,
            global_vars=input_vars or {},
            flow_vars={v.name: v.default_value for v in flow.variables},
        )

        # Add mock call data
        if mock_call:
            context.call_sid = mock_call.get("call_sid", "mock-call-123")
            context.caller_number = mock_call.get("from", "+1234567890")
            context.called_number = mock_call.get("to", "+0987654321")
            context.session_vars["call"] = mock_call

        node_results: List[NodeExecutionResult] = []

        try:
            # Find trigger nodes
            trigger_nodes = [
                n for n in flow.nodes
                if n.type in [
                    NodeType.INCOMING_CALL,
                    NodeType.OUTBOUND_CALL,
                    NodeType.WEBHOOK,
                    NodeType.SCHEDULE,
                    NodeType.EVENT,
                ]
            ]

            if not trigger_nodes:
                return FlowExecutionResult(
                    execution_id=execution_id,
                    flow_id=flow.id,
                    success=False,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    error="No trigger nodes found",
                )

            # Start from first trigger
            current_node = trigger_nodes[0]
            current_port = "output"

            # Build connection lookup
            connections_by_source = self._build_connection_lookup(flow)

            # Execute nodes
            while current_node:
                context.current_node_id = current_node.id
                context.visited_nodes.append(current_node.id)

                # Execute node with timeout
                try:
                    result = await asyncio.wait_for(
                        self._execute_node(current_node, context, mode),
                        timeout=self.settings.execution.node_timeout_ms / 1000,
                    )
                except asyncio.TimeoutError:
                    result = NodeExecutionResult(
                        node_id=current_node.id,
                        success=False,
                        error="Node execution timed out",
                    )

                node_results.append(result)
                context.execution_path.append({
                    "node_id": current_node.id,
                    "node_type": current_node.type.value,
                    "success": result.success,
                    "duration_ms": result.duration_ms,
                })

                if not result.success:
                    return FlowExecutionResult(
                        execution_id=execution_id,
                        flow_id=flow.id,
                        success=False,
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                        node_results=node_results,
                        nodes_executed=len(node_results),
                        error=result.error,
                        error_node_id=current_node.id,
                    )

                # Find next node
                output_port = result.next_port_id or current_port
                next_conn = self._find_next_connection(
                    connections_by_source,
                    current_node.id,
                    output_port,
                )

                if next_conn:
                    current_node = next(
                        (n for n in flow.nodes if n.id == next_conn.target_node_id),
                        None,
                    )
                    current_port = "output"
                else:
                    current_node = None

                # Check for infinite loop
                if len(context.visited_nodes) > self.settings.execution.max_loop_iterations:
                    return FlowExecutionResult(
                        execution_id=execution_id,
                        flow_id=flow.id,
                        success=False,
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                        node_results=node_results,
                        nodes_executed=len(node_results),
                        error="Maximum loop iterations exceeded",
                    )

            # Success
            completed_at = datetime.utcnow()
            total_duration = (completed_at - started_at).total_seconds() * 1000

            return FlowExecutionResult(
                execution_id=execution_id,
                flow_id=flow.id,
                success=True,
                started_at=started_at,
                completed_at=completed_at,
                final_output={
                    "flow_vars": context.flow_vars,
                    "session_vars": context.session_vars,
                },
                node_results=node_results,
                nodes_executed=len(node_results),
                total_duration_ms=total_duration,
            )

        except Exception as e:
            logger.exception(f"Flow execution error: {e}")
            return FlowExecutionResult(
                execution_id=execution_id,
                flow_id=flow.id,
                success=False,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                node_results=node_results,
                nodes_executed=len(node_results),
                error=str(e),
            )

    async def _execute_node(
        self,
        node: NodeInstance,
        context: ExecutionContext,
        mode: ExecutionMode,
    ) -> NodeExecutionResult:
        """Execute a single node."""
        start_time = time.time()
        logs: List[str] = []

        if mode == ExecutionMode.DEBUG:
            logs.append(f"Executing node: {node.id} ({node.type.value})")

        # Skip disabled nodes
        if node.disabled:
            return NodeExecutionResult(
                node_id=node.id,
                success=True,
                output_data={"skipped": True},
                logs=["Node is disabled, skipping"],
            )

        # Substitute variables in node data
        processed_data = self._substitute_variables(node.data, context)

        if mode == ExecutionMode.DEBUG:
            logs.append(f"Processed data: {processed_data}")

        # Get handler
        handler = self._handlers.get(node.type)

        if handler:
            try:
                output_data, next_port = await handler(
                    node, processed_data, context, mode
                )
                success = True
                error = None
            except Exception as e:
                output_data = {}
                next_port = None
                success = False
                error = str(e)
                logs.append(f"Error: {error}")
        else:
            # Default handler for unimplemented nodes
            output_data = {"input": processed_data}
            next_port = "output"
            success = True
            error = None

            if mode == ExecutionMode.DEBUG:
                logs.append(f"No handler for {node.type.value}, using default")

        duration_ms = (time.time() - start_time) * 1000

        return NodeExecutionResult(
            node_id=node.id,
            success=success,
            output_data=output_data,
            next_port_id=next_port,
            error=error,
            duration_ms=duration_ms,
            logs=logs,
        )

    def _substitute_variables(
        self,
        data: Any,
        context: ExecutionContext,
    ) -> Any:
        """Substitute {{variable}} placeholders in data."""
        if isinstance(data, str):
            return self._substitute_string(data, context)
        elif isinstance(data, dict):
            return {k: self._substitute_variables(v, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_variables(v, context) for v in data]
        else:
            return data

    def _substitute_string(self, text: str, context: ExecutionContext) -> str:
        """Substitute variables in a string."""
        pattern = r"\{\{(\w+(?:\.\w+)*)\}\}"

        def replacer(match):
            var_path = match.group(1).split(".")

            # Look up in different scopes
            value = self._lookup_variable(var_path, context)

            if value is not None:
                return str(value)
            return match.group(0)  # Return original if not found

        return re.sub(pattern, replacer, text)

    def _lookup_variable(
        self,
        path: List[str],
        context: ExecutionContext,
    ) -> Any:
        """Look up a variable by path."""
        if not path:
            return None

        name = path[0]
        rest = path[1:]

        # Check scopes in order
        value = None
        if name in context.session_vars:
            value = context.session_vars[name]
        elif name in context.flow_vars:
            value = context.flow_vars[name]
        elif name in context.global_vars:
            value = context.global_vars[name]

        # Navigate nested path
        for key in rest:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _build_connection_lookup(
        self,
        flow: Flow,
    ) -> Dict[str, List[Connection]]:
        """Build lookup of connections by source node."""
        lookup: Dict[str, List[Connection]] = {}

        for conn in flow.connections:
            key = f"{conn.source_node_id}:{conn.source_port_id}"
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(conn)

        return lookup

    def _find_next_connection(
        self,
        lookup: Dict[str, List[Connection]],
        node_id: str,
        port_id: str,
    ) -> Optional[Connection]:
        """Find the next connection to follow."""
        key = f"{node_id}:{port_id}"
        connections = lookup.get(key, [])

        if connections:
            return connections[0]  # Return first match

        # Try default output
        key = f"{node_id}:output"
        connections = lookup.get(key, [])
        if connections:
            return connections[0]

        return None

    def _register_default_handlers(self):
        """Register default node handlers."""

        # Trigger handlers
        async def handle_incoming_call(node, data, context, mode):
            return {
                "call_sid": context.call_sid,
                "from": context.caller_number,
                "to": context.called_number,
                "greeting": data.get("greeting", ""),
            }, "output"

        self._handlers[NodeType.INCOMING_CALL] = handle_incoming_call

        # Speak handler
        async def handle_speak(node, data, context, mode):
            text = data.get("text", "")
            if mode == ExecutionMode.DRY_RUN:
                return {"spoken_text": text, "simulated": True}, "output"
            # In production, would call TTS service
            return {"spoken_text": text}, "output"

        self._handlers[NodeType.SPEAK] = handle_speak

        # Gather input handler
        async def handle_gather(node, data, context, mode):
            if mode == ExecutionMode.DRY_RUN:
                # Simulate input
                return {
                    "input": "simulated input",
                    "input_type": "speech",
                    "simulated": True,
                }, "speech"
            return {"waiting": True}, "timeout"

        self._handlers[NodeType.GATHER_INPUT] = handle_gather

        # Condition handler
        async def handle_condition(node, data, context, mode):
            condition = data.get("condition", "true")
            # Simple evaluation (in production, use safe eval)
            try:
                result = self._evaluate_condition(condition, context)
                return {"result": result}, "true" if result else "false"
            except Exception as e:
                return {"error": str(e)}, "false"

        self._handlers[NodeType.CONDITION] = handle_condition

        # Set variable handler
        async def handle_set_variable(node, data, context, mode):
            name = data.get("variable_name", "")
            value = data.get("value")
            scope = data.get("scope", "flow")

            if scope == "session":
                context.session_vars[name] = value
            elif scope == "global":
                context.global_vars[name] = value
            else:
                context.flow_vars[name] = value

            return {"variable": name, "value": value}, "output"

        self._handlers[NodeType.SET_VARIABLE] = handle_set_variable

        # Wait handler
        async def handle_wait(node, data, context, mode):
            duration_ms = data.get("duration_ms", 1000)
            if mode != ExecutionMode.DRY_RUN:
                await asyncio.sleep(duration_ms / 1000)
            return {"waited_ms": duration_ms}, "output"

        self._handlers[NodeType.WAIT] = handle_wait

        # Log handler
        async def handle_log(node, data, context, mode):
            message = data.get("message", "")
            level = data.get("level", "info")
            logger.log(
                getattr(logging, level.upper(), logging.INFO),
                f"[Flow {context.flow_id}] {message}",
            )
            return {"logged": message}, "output"

        self._handlers[NodeType.LOG] = handle_log

        # Hangup handler
        async def handle_hangup(node, data, context, mode):
            message = data.get("message", "")
            reason = data.get("reason", "completed")
            return {
                "goodbye_message": message,
                "reason": reason,
                "call_ended": True,
            }, None

        self._handlers[NodeType.HANGUP] = handle_hangup

    def _evaluate_condition(self, condition: str, context: ExecutionContext) -> bool:
        """Safely evaluate a condition expression."""
        # Substitute variables first
        condition = self._substitute_string(condition, context)

        # Very basic evaluation (in production, use a safe expression evaluator)
        condition = condition.strip().lower()

        if condition in ["true", "1", "yes"]:
            return True
        if condition in ["false", "0", "no", ""]:
            return False

        # Try numeric comparison
        try:
            parts = re.split(r'\s*(==|!=|>=|<=|>|<)\s*', condition)
            if len(parts) == 3:
                left, op, right = parts
                left_val = float(left)
                right_val = float(right)

                if op == "==":
                    return left_val == right_val
                elif op == "!=":
                    return left_val != right_val
                elif op == ">":
                    return left_val > right_val
                elif op == "<":
                    return left_val < right_val
                elif op == ">=":
                    return left_val >= right_val
                elif op == "<=":
                    return left_val <= right_val
        except (ValueError, TypeError):
            pass

        return bool(condition)

    def register_handler(
        self,
        node_type: NodeType,
        handler: Callable,
    ) -> None:
        """Register a custom node handler."""
        self._handlers[node_type] = handler
        logger.info(f"Registered handler for {node_type.value}")
