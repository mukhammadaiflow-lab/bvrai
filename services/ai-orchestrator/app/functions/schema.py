"""Function schema definitions."""

from typing import Optional, List, Dict, Any, Callable, Union
from pydantic import BaseModel, Field
from enum import Enum


class ParameterType(str, Enum):
    """Function parameter types."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class FunctionParameter(BaseModel):
    """Function parameter definition."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    items: Optional[Dict[str, Any]] = None  # For array types
    properties: Optional[Dict[str, Any]] = None  # For object types

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum

        if self.type == ParameterType.ARRAY and self.items:
            schema["items"] = self.items

        if self.type == ParameterType.OBJECT and self.properties:
            schema["properties"] = self.properties

        if self.default is not None:
            schema["default"] = self.default

        return schema


class FunctionSchema(BaseModel):
    """Function schema for LLM function calling."""
    name: str = Field(..., description="Function name")
    description: str = Field(..., description="Function description")
    parameters: List[FunctionParameter] = Field(default_factory=list)
    returns: Optional[str] = Field(None, description="Return type description")
    requires_confirmation: bool = Field(False, description="Requires user confirmation")
    timeout_seconds: int = Field(30, description="Execution timeout")
    tags: List[str] = Field(default_factory=list, description="Function tags")

    def to_openai_tool(self) -> Dict[str, Any]:
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
                },
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class FunctionCall(BaseModel):
    """Represents a function call from the LLM."""
    id: str = Field(..., description="Call ID")
    name: str = Field(..., description="Function name")
    arguments: Dict[str, Any] = Field(default_factory=dict)


class FunctionResult(BaseModel):
    """Result of a function execution."""
    call_id: str
    name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_openai_message(self) -> Dict[str, Any]:
        """Convert to OpenAI function result message."""
        content = self.result if self.success else f"Error: {self.error}"
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": str(content),
        }

    def to_anthropic_message(self) -> Dict[str, Any]:
        """Convert to Anthropic tool result."""
        return {
            "type": "tool_result",
            "tool_use_id": self.call_id,
            "content": str(self.result) if self.success else f"Error: {self.error}",
            "is_error": not self.success,
        }


class FunctionDefinition(BaseModel):
    """Complete function definition with handler."""
    schema: FunctionSchema
    handler: Optional[Callable] = None
    is_async: bool = True

    class Config:
        arbitrary_types_allowed = True
