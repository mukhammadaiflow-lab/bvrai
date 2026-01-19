"""
Tool Execution Module

This module provides tool/function execution capabilities for the agent runtime,
including built-in tools, custom tool registration, and execution management.
"""

import asyncio
import inspect
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

from .base import (
    FunctionDefinition,
    FunctionCall,
    FunctionExecutionError,
    ConversationContext,
)


logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution."""

    call_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class Tool(ABC):
    """Abstract base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for parameters."""
        pass

    @abstractmethod
    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional[ConversationContext] = None,
    ) -> Any:
        """Execute the tool."""
        pass

    def to_definition(self) -> FunctionDefinition:
        """Convert to function definition."""
        return FunctionDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class EndCallTool(Tool):
    """Tool to end the current call."""

    @property
    def name(self) -> str:
        return "end_call"

    @property
    def description(self) -> str:
        return "End the current call. Use when the conversation is complete or the customer wants to hang up."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for ending the call",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of the call",
                },
            },
            "required": ["reason"],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional[ConversationContext] = None,
    ) -> Any:
        """Execute end call."""
        return {
            "action": "end_call",
            "reason": arguments.get("reason", "conversation complete"),
            "summary": arguments.get("summary"),
        }


class TransferCallTool(Tool):
    """Tool to transfer the call to another party."""

    def __init__(self, transfer_targets: Optional[Dict[str, str]] = None):
        """
        Initialize tool.

        Args:
            transfer_targets: Map of target names to phone numbers/URIs
        """
        self.transfer_targets = transfer_targets or {}

    @property
    def name(self) -> str:
        return "transfer_call"

    @property
    def description(self) -> str:
        targets = ", ".join(self.transfer_targets.keys()) if self.transfer_targets else "any department"
        return f"Transfer the call to another party. Available targets: {targets}"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Transfer target (department or person name)",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for transfer",
                },
                "warm_transfer": {
                    "type": "boolean",
                    "description": "Whether to stay on the line during transfer",
                    "default": True,
                },
            },
            "required": ["target", "reason"],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional[ConversationContext] = None,
    ) -> Any:
        """Execute transfer."""
        target = arguments.get("target", "")
        target_number = self.transfer_targets.get(target, target)

        return {
            "action": "transfer_call",
            "target": target,
            "target_number": target_number,
            "reason": arguments.get("reason"),
            "warm_transfer": arguments.get("warm_transfer", True),
        }


class HoldCallTool(Tool):
    """Tool to put the call on hold."""

    @property
    def name(self) -> str:
        return "hold_call"

    @property
    def description(self) -> str:
        return "Put the caller on hold. Use when you need to look something up or consult with a colleague."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for putting on hold",
                },
                "estimated_wait_seconds": {
                    "type": "integer",
                    "description": "Estimated wait time in seconds",
                    "minimum": 5,
                    "maximum": 300,
                },
            },
            "required": ["reason"],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional[ConversationContext] = None,
    ) -> Any:
        """Execute hold."""
        return {
            "action": "hold_call",
            "reason": arguments.get("reason"),
            "estimated_wait_seconds": arguments.get("estimated_wait_seconds", 60),
        }


class CollectDTMFTool(Tool):
    """Tool to collect DTMF input from the caller."""

    @property
    def name(self) -> str:
        return "collect_dtmf"

    @property
    def description(self) -> str:
        return "Collect numeric input from the caller using their phone keypad. Use for things like PIN verification, menu selection, or account numbers."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Message to tell the caller before collecting input",
                },
                "num_digits": {
                    "type": "integer",
                    "description": "Expected number of digits",
                    "minimum": 1,
                    "maximum": 20,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout for input collection",
                    "default": 10,
                },
            },
            "required": ["prompt", "num_digits"],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional[ConversationContext] = None,
    ) -> Any:
        """Execute DTMF collection."""
        return {
            "action": "collect_dtmf",
            "prompt": arguments.get("prompt"),
            "num_digits": arguments.get("num_digits"),
            "timeout_seconds": arguments.get("timeout_seconds", 10),
        }


class ScheduleAppointmentTool(Tool):
    """Tool to schedule an appointment."""

    @property
    def name(self) -> str:
        return "schedule_appointment"

    @property
    def description(self) -> str:
        return "Schedule an appointment for the customer. Requires date, time, and purpose."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date of appointment (YYYY-MM-DD)",
                },
                "time": {
                    "type": "string",
                    "description": "Time of appointment (HH:MM in 24-hour format)",
                },
                "purpose": {
                    "type": "string",
                    "description": "Purpose of the appointment",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Duration in minutes",
                    "default": 30,
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes for the appointment",
                },
            },
            "required": ["date", "time", "purpose"],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional[ConversationContext] = None,
    ) -> Any:
        """Execute appointment scheduling."""
        # In production, this would call the scheduling system
        return {
            "success": True,
            "appointment_id": f"apt_{uuid.uuid4().hex[:12]}",
            "date": arguments.get("date"),
            "time": arguments.get("time"),
            "purpose": arguments.get("purpose"),
            "duration_minutes": arguments.get("duration_minutes", 30),
            "confirmation": f"Appointment scheduled for {arguments.get('date')} at {arguments.get('time')}",
        }


class SendSMSTool(Tool):
    """Tool to send an SMS message."""

    @property
    def name(self) -> str:
        return "send_sms"

    @property
    def description(self) -> str:
        return "Send an SMS message to the customer's phone number."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message content to send",
                    "maxLength": 1600,
                },
                "phone_number": {
                    "type": "string",
                    "description": "Phone number to send to (optional, defaults to caller's number)",
                },
            },
            "required": ["message"],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional[ConversationContext] = None,
    ) -> Any:
        """Execute SMS send."""
        phone = arguments.get("phone_number")
        if not phone and context:
            phone = context.caller_id

        # In production, this would call SMS service
        return {
            "success": True,
            "message_id": f"sms_{uuid.uuid4().hex[:12]}",
            "phone_number": phone,
            "message_preview": arguments.get("message", "")[:50],
        }


class SendEmailTool(Tool):
    """Tool to send an email."""

    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return "Send an email to the customer."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Email subject",
                },
                "body": {
                    "type": "string",
                    "description": "Email body content",
                },
                "to_email": {
                    "type": "string",
                    "description": "Recipient email address",
                },
            },
            "required": ["subject", "body", "to_email"],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional[ConversationContext] = None,
    ) -> Any:
        """Execute email send."""
        # In production, this would call email service
        return {
            "success": True,
            "email_id": f"email_{uuid.uuid4().hex[:12]}",
            "to": arguments.get("to_email"),
            "subject": arguments.get("subject"),
        }


class LookupAccountTool(Tool):
    """Tool to look up customer account information."""

    def __init__(
        self,
        lookup_callback: Optional[
            Callable[[str], Coroutine[Any, Any, Dict[str, Any]]]
        ] = None,
    ):
        """
        Initialize tool.

        Args:
            lookup_callback: Async function to perform account lookup
        """
        self.lookup_callback = lookup_callback

    @property
    def name(self) -> str:
        return "lookup_account"

    @property
    def description(self) -> str:
        return "Look up customer account information by phone number, email, or account ID."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "Customer's phone number",
                },
                "email": {
                    "type": "string",
                    "description": "Customer's email address",
                },
                "account_id": {
                    "type": "string",
                    "description": "Customer's account ID",
                },
            },
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional[ConversationContext] = None,
    ) -> Any:
        """Execute account lookup."""
        if self.lookup_callback:
            identifier = (
                arguments.get("account_id") or
                arguments.get("phone_number") or
                arguments.get("email") or
                (context.caller_id if context else None)
            )
            if identifier:
                return await self.lookup_callback(identifier)

        # Mock response
        return {
            "found": True,
            "account_id": "ACC123456",
            "name": "John Doe",
            "status": "active",
            "message": "Account information would be retrieved here",
        }


class ToolRegistry:
    """
    Registry for managing and executing tools.

    Provides:
    - Tool registration
    - Tool discovery
    - Tool execution with timeout and error handling
    """

    def __init__(self):
        """Initialize registry."""
        self._tools: Dict[str, Tool] = {}
        self._custom_handlers: Dict[str, Callable] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_function(
        self,
        name: str,
        description: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a function as a tool.

        Args:
            name: Function name
            description: Function description
            handler: Async function handler
            parameters: JSON Schema for parameters (auto-generated if None)
        """
        if parameters is None:
            parameters = self._generate_parameters_schema(handler)

        self._custom_handlers[name] = handler

        # Create wrapper tool
        class CustomTool(Tool):
            _name = name
            _description = description
            _parameters = parameters
            _handler = handler

            @property
            def name(self) -> str:
                return self._name

            @property
            def description(self) -> str:
                return self._description

            @property
            def parameters(self) -> Dict[str, Any]:
                return self._parameters

            async def execute(
                self,
                arguments: Dict[str, Any],
                context: Optional[ConversationContext] = None,
            ) -> Any:
                return await self._handler(**arguments)

        self._tools[name] = CustomTool()
        logger.debug(f"Registered function tool: {name}")

    def _generate_parameters_schema(
        self,
        handler: Callable,
    ) -> Dict[str, Any]:
        """Generate JSON Schema from function signature."""
        sig = inspect.signature(handler)
        type_hints = get_type_hints(handler) if hasattr(handler, "__annotations__") else {}

        properties = {}
        required = []

        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name == "context":
                continue

            param_type = type_hints.get(param_name, str)
            json_type = type_mapping.get(param_type, "string")

            properties[param_name] = {"type": json_type}

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            self._custom_handlers.pop(name, None)
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_definitions(self) -> List[FunctionDefinition]:
        """Get all tool definitions."""
        return [tool.to_definition() for tool in self._tools.values()]

    async def execute(
        self,
        call: FunctionCall,
        context: Optional[ConversationContext] = None,
        timeout_seconds: float = 30.0,
    ) -> ToolResult:
        """
        Execute a tool call.

        Args:
            call: Function call to execute
            context: Conversation context
            timeout_seconds: Execution timeout

        Returns:
            Tool result
        """
        start_time = time.time()

        tool = self._tools.get(call.name)
        if not tool:
            return ToolResult(
                call_id=call.id,
                tool_name=call.name,
                success=False,
                error=f"Unknown tool: {call.name}",
            )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool.execute(call.arguments, context),
                timeout=timeout_seconds,
            )

            execution_time_ms = (time.time() - start_time) * 1000

            return ToolResult(
                call_id=call.id,
                tool_name=call.name,
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                call_id=call.id,
                tool_name=call.name,
                success=False,
                error=f"Tool execution timed out after {timeout_seconds}s",
                execution_time_ms=timeout_seconds * 1000,
            )

        except Exception as e:
            logger.exception(f"Tool {call.name} execution failed: {e}")
            return ToolResult(
                call_id=call.id,
                tool_name=call.name,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def execute_batch(
        self,
        calls: List[FunctionCall],
        context: Optional[ConversationContext] = None,
        timeout_seconds: float = 30.0,
        parallel: bool = True,
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls.

        Args:
            calls: Function calls to execute
            context: Conversation context
            timeout_seconds: Execution timeout per call
            parallel: Execute in parallel if True

        Returns:
            List of tool results
        """
        if parallel:
            tasks = [
                self.execute(call, context, timeout_seconds)
                for call in calls
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for call in calls:
                result = await self.execute(call, context, timeout_seconds)
                results.append(result)
            return results


def create_default_tools(
    transfer_targets: Optional[Dict[str, str]] = None,
) -> ToolRegistry:
    """
    Create a tool registry with default tools.

    Args:
        transfer_targets: Transfer target mappings

    Returns:
        Configured tool registry
    """
    registry = ToolRegistry()

    # Register built-in tools
    registry.register(EndCallTool())
    registry.register(TransferCallTool(transfer_targets))
    registry.register(HoldCallTool())
    registry.register(CollectDTMFTool())
    registry.register(ScheduleAppointmentTool())
    registry.register(SendSMSTool())
    registry.register(SendEmailTool())
    registry.register(LookupAccountTool())

    return registry


__all__ = [
    "ToolResult",
    "Tool",
    "EndCallTool",
    "TransferCallTool",
    "HoldCallTool",
    "CollectDTMFTool",
    "ScheduleAppointmentTool",
    "SendSMSTool",
    "SendEmailTool",
    "LookupAccountTool",
    "ToolRegistry",
    "create_default_tools",
]
