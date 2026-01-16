"""
LLM Tool/Function Calling Framework

This module provides a robust framework for defining and executing
tools that can be called by LLMs.
"""

import asyncio
import functools
import inspect
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_type_hints,
)


logger = logging.getLogger(__name__)


class ParameterType(str, Enum):
    """JSON Schema parameter types."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: ParameterType
    description: str = ""
    required: bool = True
    default: Any = None

    # For string type
    enum: Optional[List[str]] = None
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    # For numeric types
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusive_minimum: Optional[float] = None
    exclusive_maximum: Optional[float] = None
    multiple_of: Optional[float] = None

    # For array type
    items: Optional[Dict[str, Any]] = None
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    unique_items: bool = False

    # For object type
    properties: Optional[Dict[str, "ToolParameter"]] = None
    additional_properties: bool = False

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {"type": self.type.value}

        if self.description:
            schema["description"] = self.description

        if self.default is not None:
            schema["default"] = self.default

        # String constraints
        if self.type == ParameterType.STRING:
            if self.enum:
                schema["enum"] = self.enum
            if self.pattern:
                schema["pattern"] = self.pattern
            if self.min_length is not None:
                schema["minLength"] = self.min_length
            if self.max_length is not None:
                schema["maxLength"] = self.max_length

        # Numeric constraints
        if self.type in (ParameterType.INTEGER, ParameterType.NUMBER):
            if self.minimum is not None:
                schema["minimum"] = self.minimum
            if self.maximum is not None:
                schema["maximum"] = self.maximum
            if self.exclusive_minimum is not None:
                schema["exclusiveMinimum"] = self.exclusive_minimum
            if self.exclusive_maximum is not None:
                schema["exclusiveMaximum"] = self.exclusive_maximum
            if self.multiple_of is not None:
                schema["multipleOf"] = self.multiple_of

        # Array constraints
        if self.type == ParameterType.ARRAY:
            if self.items:
                schema["items"] = self.items
            if self.min_items is not None:
                schema["minItems"] = self.min_items
            if self.max_items is not None:
                schema["maxItems"] = self.max_items
            if self.unique_items:
                schema["uniqueItems"] = True

        # Object constraints
        if self.type == ParameterType.OBJECT:
            if self.properties:
                schema["properties"] = {
                    name: param.to_json_schema()
                    for name, param in self.properties.items()
                }
                required = [
                    name for name, param in self.properties.items()
                    if param.required
                ]
                if required:
                    schema["required"] = required
            if not self.additional_properties:
                schema["additionalProperties"] = False

        return schema


@dataclass
class Tool:
    """Definition of a tool that can be called by an LLM."""

    name: str
    description: str
    handler: Callable
    parameters: List[ToolParameter] = field(default_factory=list)

    # Metadata
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    is_destructive: bool = False
    timeout_seconds: float = 30.0

    # Execution settings
    async_execution: bool = True
    max_retries: int = 0
    retry_delay_seconds: float = 1.0

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        try:
            if asyncio.iscoroutinefunction(self.handler):
                return await asyncio.wait_for(
                    self.handler(**kwargs),
                    timeout=self.timeout_seconds,
                )
            else:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.handler, **kwargs),
                    timeout=self.timeout_seconds,
                )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool '{self.name}' execution timed out after {self.timeout_seconds}s")
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution error: {e}")
            raise


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]

    # Execution state
    executed: bool = False
    result: Optional["ToolResult"] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        arguments = data.get("function", {}).get("arguments", "{}")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        return cls(
            id=data.get("id", ""),
            name=data.get("function", {}).get("name", ""),
            arguments=arguments,
        )


@dataclass
class ToolResult:
    """Result of tool execution."""

    tool_call_id: str
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None

    # Timing
    execution_time_ms: float = 0.0

    def to_message_content(self) -> str:
        """Convert to message content for LLM."""
        if self.success:
            if isinstance(self.output, (dict, list)):
                return json.dumps(self.output, indent=2)
            return str(self.output) if self.output is not None else "Success"
        else:
            return f"Error: {self.error}"


def _python_type_to_parameter_type(python_type: Type) -> ParameterType:
    """Convert Python type to ParameterType."""
    type_mapping = {
        str: ParameterType.STRING,
        int: ParameterType.INTEGER,
        float: ParameterType.NUMBER,
        bool: ParameterType.BOOLEAN,
        list: ParameterType.ARRAY,
        dict: ParameterType.OBJECT,
        type(None): ParameterType.NULL,
    }

    # Handle Optional types
    origin = getattr(python_type, "__origin__", None)
    if origin is Union:
        args = python_type.__args__
        # Filter out None type
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_parameter_type(non_none_args[0])

    return type_mapping.get(python_type, ParameterType.STRING)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    tags: Optional[List[str]] = None,
    requires_confirmation: bool = False,
    is_destructive: bool = False,
    timeout_seconds: float = 30.0,
):
    """
    Decorator to create a Tool from a function.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        category: Category for organization
        tags: Tags for filtering
        requires_confirmation: Whether to confirm before execution
        is_destructive: Whether the tool has destructive effects
        timeout_seconds: Execution timeout

    Example:
        @tool(description="Search for information")
        async def search(query: str, max_results: int = 10) -> list:
            '''Search the knowledge base.'''
            ...
    """
    def decorator(func: Callable) -> Tool:
        # Get function metadata
        func_name = name or func.__name__
        func_description = description or func.__doc__ or ""

        # Get type hints
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
        sig = inspect.signature(func)

        # Build parameters
        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            param_type_enum = _python_type_to_parameter_type(param_type)

            # Get default value
            has_default = param.default is not inspect.Parameter.empty
            default_value = param.default if has_default else None
            is_required = not has_default

            # Check for Optional
            origin = getattr(param_type, "__origin__", None)
            if origin is Union and type(None) in param_type.__args__:
                is_required = False

            parameters.append(ToolParameter(
                name=param_name,
                type=param_type_enum,
                description="",  # Could parse from docstring
                required=is_required,
                default=default_value,
            ))

        return Tool(
            name=func_name,
            description=func_description.strip(),
            handler=func,
            parameters=parameters,
            category=category,
            tags=tags or [],
            requires_confirmation=requires_confirmation,
            is_destructive=is_destructive,
            timeout_seconds=timeout_seconds,
        )

    return decorator


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

        # Track by category
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        if tool.name not in self._categories[tool.category]:
            self._categories[tool.category].append(tool.name)

        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            tool = self._tools.pop(name)
            if tool.category in self._categories:
                self._categories[tool.category].remove(name)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Tool]:
        """List tools with optional filtering."""
        tools = list(self._tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]

        return tools

    def list_categories(self) -> List[str]:
        """List all categories."""
        return list(self._categories.keys())

    def to_openai_format(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI format."""
        tools = self.list_tools(category=category, tags=tags)
        return [tool.to_openai_format() for tool in tools]

    async def execute(
        self,
        tool_call: ToolCall,
    ) -> ToolResult:
        """Execute a tool call."""
        import time
        start_time = time.time()

        tool = self.get(tool_call.name)
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                error=f"Tool not found: {tool_call.name}",
            )

        try:
            output = await tool.execute(**tool_call.arguments)
            execution_time = (time.time() - start_time) * 1000

            result = ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=True,
                output=output,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            result = ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )

        tool_call.executed = True
        tool_call.result = result

        return result

    async def execute_all(
        self,
        tool_calls: List[ToolCall],
        parallel: bool = True,
    ) -> List[ToolResult]:
        """Execute multiple tool calls."""
        if parallel:
            tasks = [self.execute(tc) for tc in tool_calls]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for tc in tool_calls:
                result = await self.execute(tc)
                results.append(result)
            return results


# Built-in tools

@tool(
    name="get_current_time",
    description="Get the current date and time",
    category="utility",
)
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in the specified timezone."""
    from datetime import datetime
    try:
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
    except ImportError:
        now = datetime.utcnow()
    return now.isoformat()


@tool(
    name="calculate",
    description="Perform mathematical calculations",
    category="utility",
)
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely."""
    import ast
    import operator

    # Safe operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }

    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_expr(node.operand))
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")

    try:
        tree = ast.parse(expression, mode='eval')
        return eval_expr(tree.body)
    except Exception as e:
        raise ValueError(f"Invalid expression: {expression}. Error: {e}")


@tool(
    name="format_json",
    description="Format JSON data for display",
    category="utility",
)
def format_json(data: dict, indent: int = 2) -> str:
    """Format a dictionary as pretty-printed JSON."""
    return json.dumps(data, indent=indent, ensure_ascii=False)


# Create default registry with built-in tools
default_registry = ToolRegistry()
default_registry.register(get_current_time)
default_registry.register(calculate)
default_registry.register(format_json)


__all__ = [
    "ParameterType",
    "ToolParameter",
    "Tool",
    "ToolCall",
    "ToolResult",
    "tool",
    "ToolRegistry",
    "default_registry",
]
