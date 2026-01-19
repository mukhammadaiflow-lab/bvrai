"""Function registry for managing available functions."""

import structlog
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field

from app.functions.schema import (
    FunctionSchema,
    FunctionParameter,
    FunctionDefinition,
    ParameterType,
)


logger = structlog.get_logger()


@dataclass
class RegisteredFunction:
    """Registered function with metadata."""
    schema: FunctionSchema
    handler: Callable
    is_async: bool
    agent_ids: List[str] = field(default_factory=list)  # Empty = all agents
    enabled: bool = True


class FunctionRegistry:
    """
    Registry for managing available functions.

    Features:
    - Register functions with schemas
    - Per-agent function availability
    - Function discovery
    - Validation
    """

    def __init__(self):
        self._functions: Dict[str, RegisteredFunction] = {}
        self._agent_functions: Dict[str, List[str]] = {}  # agent_id -> function names

    def register(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Optional[List[FunctionParameter]] = None,
        is_async: bool = True,
        requires_confirmation: bool = False,
        timeout_seconds: int = 30,
        agent_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> FunctionSchema:
        """
        Register a function.

        Args:
            name: Function name
            description: Function description
            handler: Function handler
            parameters: Function parameters
            is_async: Whether handler is async
            requires_confirmation: Requires user confirmation
            timeout_seconds: Execution timeout
            agent_ids: Limit to specific agents (None = all)
            tags: Function tags

        Returns:
            Registered function schema
        """
        schema = FunctionSchema(
            name=name,
            description=description,
            parameters=parameters or [],
            requires_confirmation=requires_confirmation,
            timeout_seconds=timeout_seconds,
            tags=tags or [],
        )

        registered = RegisteredFunction(
            schema=schema,
            handler=handler,
            is_async=is_async,
            agent_ids=agent_ids or [],
            enabled=True,
        )

        self._functions[name] = registered

        logger.info(
            "function_registered",
            name=name,
            params=len(parameters or []),
            agent_ids=agent_ids,
        )

        return schema

    def register_decorator(
        self,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[List[FunctionParameter]] = None,
        **kwargs,
    ):
        """
        Decorator for registering functions.

        Usage:
            @registry.register_decorator(
                description="Book an appointment",
                parameters=[...]
            )
            async def book_appointment(date: str, time: str):
                ...
        """
        def decorator(func: Callable):
            func_name = name or func.__name__

            # Try to get description from docstring
            desc = description or func.__doc__ or f"Function {func_name}"

            import asyncio
            is_async = asyncio.iscoroutinefunction(func)

            self.register(
                name=func_name,
                description=desc,
                handler=func,
                parameters=parameters,
                is_async=is_async,
                **kwargs,
            )

            return func

        return decorator

    def unregister(self, name: str) -> bool:
        """Unregister a function."""
        if name in self._functions:
            del self._functions[name]
            logger.info("function_unregistered", name=name)
            return True
        return False

    def get(self, name: str) -> Optional[RegisteredFunction]:
        """Get a registered function."""
        return self._functions.get(name)

    def get_schema(self, name: str) -> Optional[FunctionSchema]:
        """Get function schema."""
        func = self._functions.get(name)
        return func.schema if func else None

    def list_functions(
        self,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        enabled_only: bool = True,
    ) -> List[FunctionSchema]:
        """
        List available functions.

        Args:
            agent_id: Filter by agent
            tags: Filter by tags
            enabled_only: Only return enabled functions

        Returns:
            List of function schemas
        """
        results = []

        for name, func in self._functions.items():
            # Check enabled
            if enabled_only and not func.enabled:
                continue

            # Check agent access
            if agent_id and func.agent_ids:
                if agent_id not in func.agent_ids:
                    continue

            # Check tags
            if tags:
                if not any(t in func.schema.tags for t in tags):
                    continue

            results.append(func.schema)

        return results

    def get_tools_for_agent(
        self,
        agent_id: str,
        format: str = "openai",
    ) -> List[Dict[str, Any]]:
        """
        Get tools for a specific agent in LLM format.

        Args:
            agent_id: Agent ID
            format: Tool format (openai, anthropic)

        Returns:
            List of tool definitions
        """
        functions = self.list_functions(agent_id=agent_id)
        tools = []

        for func in functions:
            if format == "openai":
                tools.append(func.to_openai_tool())
            elif format == "anthropic":
                tools.append(func.to_anthropic_tool())
            else:
                tools.append(func.to_openai_tool())

        return tools

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a function."""
        if name in self._functions:
            self._functions[name].enabled = enabled
            return True
        return False

    def validate_call(
        self,
        name: str,
        arguments: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a function call.

        Args:
            name: Function name
            arguments: Call arguments

        Returns:
            (is_valid, error_message)
        """
        func = self._functions.get(name)
        if not func:
            return False, f"Unknown function: {name}"

        if not func.enabled:
            return False, f"Function disabled: {name}"

        # Validate required parameters
        for param in func.schema.parameters:
            if param.required and param.name not in arguments:
                return False, f"Missing required parameter: {param.name}"

            if param.name in arguments:
                value = arguments[param.name]

                # Type validation
                if not self._validate_type(value, param.type):
                    return False, f"Invalid type for {param.name}: expected {param.type}"

                # Enum validation
                if param.enum and value not in param.enum:
                    return False, f"Invalid value for {param.name}: must be one of {param.enum}"

        return True, None

    def _validate_type(self, value: Any, param_type: ParameterType) -> bool:
        """Validate value type."""
        type_map = {
            ParameterType.STRING: str,
            ParameterType.NUMBER: (int, float),
            ParameterType.INTEGER: int,
            ParameterType.BOOLEAN: bool,
            ParameterType.ARRAY: list,
            ParameterType.OBJECT: dict,
        }

        expected = type_map.get(param_type)
        if expected:
            return isinstance(value, expected)
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        enabled = sum(1 for f in self._functions.values() if f.enabled)
        by_tag: Dict[str, int] = {}

        for func in self._functions.values():
            for tag in func.schema.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        return {
            "total_functions": len(self._functions),
            "enabled_functions": enabled,
            "disabled_functions": len(self._functions) - enabled,
            "functions_by_tag": by_tag,
        }


# Global registry instance
default_registry = FunctionRegistry()
