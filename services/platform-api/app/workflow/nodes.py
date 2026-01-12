"""
Workflow Nodes

All node type implementations:
- Flow control nodes
- Action nodes
- Logic nodes
- AI nodes
- Integration nodes
"""

from typing import Optional, Dict, Any, List, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import aiohttp
import json
import re
import logging

from .engine import (
    NodeExecutor, NodeConfig, NodeResult, ExecutionContext,
    NodeType, NodePort, Position,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Flow Control Nodes
# =============================================================================

class StartNodeExecutor(NodeExecutor):
    """Start node - entry point of workflow."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Initialize workflow execution."""
        # Copy input data to variables
        for key, value in context.input_data.items():
            context.set_variable(key, value)

        return NodeResult(
            success=True,
            output={"started": True},
            next_port="default",
        )


class EndNodeExecutor(NodeExecutor):
    """End node - terminates workflow."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Finalize workflow execution."""
        # Collect output based on configuration
        output_mapping = self.node.config.get("output_mapping", {})

        for output_key, var_name in output_mapping.items():
            context.output[output_key] = context.get_variable(var_name)

        return NodeResult(
            success=True,
            output=context.output,
            should_continue=False,
        )


class ConditionNodeExecutor(NodeExecutor):
    """
    Conditional branching node.

    Evaluates condition and routes to appropriate output.
    """

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Evaluate condition and branch."""
        condition = self.node.config.get("condition", "")
        true_port = self.node.config.get("true_port", "true")
        false_port = self.node.config.get("false_port", "false")

        # Resolve variables in condition
        resolved_condition = context.resolve_expression(condition)

        # Evaluate condition
        result = self._evaluate(resolved_condition, context)

        return NodeResult(
            success=True,
            output={"condition_result": result},
            next_port=true_port if result else false_port,
        )

    def _evaluate(self, condition: str, context: ExecutionContext) -> bool:
        """Evaluate condition expression."""
        try:
            # Handle comparison operators
            operators = ['==', '!=', '>=', '<=', '>', '<', ' in ', ' not in ']

            for op in operators:
                if op in condition:
                    parts = condition.split(op)
                    if len(parts) == 2:
                        left = self._resolve_value(parts[0].strip(), context)
                        right = self._resolve_value(parts[1].strip(), context)

                        if op == '==':
                            return left == right
                        elif op == '!=':
                            return left != right
                        elif op == '>=':
                            return float(left) >= float(right)
                        elif op == '<=':
                            return float(left) <= float(right)
                        elif op == '>':
                            return float(left) > float(right)
                        elif op == '<':
                            return float(left) < float(right)
                        elif ' in ' in op:
                            return left in right
                        elif ' not in ' in op:
                            return left not in right

            # Boolean evaluation
            value = self._resolve_value(condition, context)
            return bool(value)

        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False

    def _resolve_value(self, value: str, context: ExecutionContext) -> Any:
        """Resolve value from string."""
        value = value.strip()

        # Check for string literals
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]

        # Check for numbers
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Check for boolean
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False

        # Try as variable
        return context.get_variable(value, value)


class SwitchNodeExecutor(NodeExecutor):
    """
    Multi-way branching node.

    Routes based on value matching.
    """

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Evaluate switch and route."""
        switch_var = self.node.config.get("variable", "")
        cases = self.node.config.get("cases", {})
        default_port = self.node.config.get("default_port", "default")

        # Get value
        value = context.get_variable(switch_var)
        if value is None:
            value = context.resolve_expression(switch_var)

        # Find matching case
        str_value = str(value).lower()
        for case_value, port in cases.items():
            if str(case_value).lower() == str_value:
                return NodeResult(
                    success=True,
                    output={"matched_case": case_value},
                    next_port=port,
                )

        return NodeResult(
            success=True,
            output={"matched_case": None},
            next_port=default_port,
        )


class ParallelNodeExecutor(NodeExecutor):
    """
    Parallel execution node.

    Executes multiple branches concurrently.
    """

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Mark start of parallel execution."""
        branch_count = self.node.config.get("branch_count", 2)

        return NodeResult(
            success=True,
            output={"branches": branch_count},
            variable_updates={"_parallel_branch_count": branch_count},
            next_port="branch_0",  # Start with first branch
        )


class MergeNodeExecutor(NodeExecutor):
    """
    Merge node for parallel branches.

    Waits for all branches to complete.
    """

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Merge parallel branches."""
        required_branches = context.get_variable("_parallel_branch_count", 1)
        completed = context.get_variable("_parallel_completed", 0) + 1

        if completed < required_branches:
            return NodeResult(
                success=True,
                output={"waiting": True, "completed": completed},
                variable_updates={"_parallel_completed": completed},
                should_wait=True,
                wait_duration_ms=100,
            )

        # All branches completed
        return NodeResult(
            success=True,
            output={"merged": True, "branch_count": completed},
            variable_updates={
                "_parallel_branch_count": 0,
                "_parallel_completed": 0,
            },
            next_port="default",
        )


class LoopNodeExecutor(NodeExecutor):
    """
    Loop node for iterations.

    Supports for-each and while loops.
    """

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute loop iteration."""
        loop_type = self.node.config.get("loop_type", "count")
        loop_var = self.node.config.get("loop_variable", "_loop_index")
        body_port = self.node.config.get("body_port", "body")
        exit_port = self.node.config.get("exit_port", "exit")

        current_index = context.get_variable(loop_var, 0)

        if loop_type == "count":
            max_count = self.node.config.get("count", 10)

            if current_index < max_count:
                return NodeResult(
                    success=True,
                    output={"iteration": current_index},
                    variable_updates={loop_var: current_index + 1},
                    next_port=body_port,
                )

        elif loop_type == "foreach":
            collection_var = self.node.config.get("collection", "")
            collection = context.get_variable(collection_var, [])
            item_var = self.node.config.get("item_variable", "item")

            if current_index < len(collection):
                return NodeResult(
                    success=True,
                    output={"iteration": current_index, "item": collection[current_index]},
                    variable_updates={
                        loop_var: current_index + 1,
                        item_var: collection[current_index],
                    },
                    next_port=body_port,
                )

        elif loop_type == "while":
            condition = self.node.config.get("condition", "false")
            resolved = context.resolve_expression(condition)

            if resolved.lower() in ("true", "1", "yes"):
                return NodeResult(
                    success=True,
                    output={"iteration": current_index},
                    variable_updates={loop_var: current_index + 1},
                    next_port=body_port,
                )

        # Exit loop
        return NodeResult(
            success=True,
            output={"loop_complete": True, "total_iterations": current_index},
            variable_updates={loop_var: 0},
            next_port=exit_port,
        )


class WaitNodeExecutor(NodeExecutor):
    """Wait/delay node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Wait for specified duration."""
        duration_ms = self.node.config.get("duration_ms", 1000)
        wait_for_event = self.node.config.get("wait_for_event", "")

        if wait_for_event:
            # Event-based wait would be handled by engine
            return NodeResult(
                success=True,
                output={"waiting_for": wait_for_event},
                should_wait=True,
                wait_duration_ms=duration_ms,
            )

        return NodeResult(
            success=True,
            output={"waited_ms": duration_ms},
            should_wait=True,
            wait_duration_ms=duration_ms,
        )


class GotoNodeExecutor(NodeExecutor):
    """Goto node - jump to specific node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Jump to target node."""
        target_node = self.node.config.get("target_node", "")

        if not target_node:
            return NodeResult(
                success=False,
                error="No target node specified",
            )

        return NodeResult(
            success=True,
            output={"goto": target_node},
            next_node=target_node,
        )


# =============================================================================
# Voice/Call Action Nodes
# =============================================================================

class SpeakNodeExecutor(NodeExecutor):
    """Speak node - text-to-speech output."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Generate speech output."""
        text = self.node.config.get("text", "")
        ssml = self.node.config.get("ssml", "")
        voice_id = self.node.config.get("voice_id", "")
        speed = self.node.config.get("speed", 1.0)
        pitch = self.node.config.get("pitch", 1.0)

        # Resolve variables
        resolved_text = context.resolve_expression(text)
        resolved_ssml = context.resolve_expression(ssml) if ssml else ""

        return NodeResult(
            success=True,
            output={
                "action": "speak",
                "text": resolved_text,
                "ssml": resolved_ssml,
                "voice_id": voice_id,
                "speed": speed,
                "pitch": pitch,
            },
            context_updates={"last_spoken": resolved_text},
        )


class ListenNodeExecutor(NodeExecutor):
    """Listen node - speech-to-text input."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Listen for user input."""
        timeout_ms = self.node.config.get("timeout_ms", 10000)
        language = self.node.config.get("language", "en")
        output_var = self.node.config.get("output_variable", "user_input")

        return NodeResult(
            success=True,
            output={
                "action": "listen",
                "timeout_ms": timeout_ms,
                "language": language,
                "output_variable": output_var,
            },
            should_wait=True,
            wait_duration_ms=timeout_ms,
        )


class GatherNodeExecutor(NodeExecutor):
    """
    Gather node - collect specific input.

    Combines speak and listen with validation.
    """

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Gather input from user."""
        prompt = self.node.config.get("prompt", "")
        input_type = self.node.config.get("input_type", "speech")  # speech, dtmf
        num_digits = self.node.config.get("num_digits", 0)
        timeout_ms = self.node.config.get("timeout_ms", 10000)
        validation = self.node.config.get("validation", "")
        output_var = self.node.config.get("output_variable", "gathered_input")

        resolved_prompt = context.resolve_expression(prompt)

        return NodeResult(
            success=True,
            output={
                "action": "gather",
                "prompt": resolved_prompt,
                "input_type": input_type,
                "num_digits": num_digits,
                "timeout_ms": timeout_ms,
                "validation": validation,
                "output_variable": output_var,
            },
            should_wait=True,
            wait_duration_ms=timeout_ms,
        )


class TransferNodeExecutor(NodeExecutor):
    """Transfer node - transfer call."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Transfer call to target."""
        transfer_type = self.node.config.get("transfer_type", "blind")
        destination = self.node.config.get("destination", "")
        destination_type = self.node.config.get("destination_type", "phone")  # phone, sip, agent
        announcement = self.node.config.get("announcement", "")

        resolved_dest = context.resolve_expression(destination)
        resolved_announce = context.resolve_expression(announcement) if announcement else ""

        return NodeResult(
            success=True,
            output={
                "action": "transfer",
                "transfer_type": transfer_type,
                "destination": resolved_dest,
                "destination_type": destination_type,
                "announcement": resolved_announce,
            },
            should_continue=False,  # Transfer typically ends workflow
        )


class HangupNodeExecutor(NodeExecutor):
    """Hangup node - end call."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Hang up the call."""
        reason = self.node.config.get("reason", "normal")
        message = self.node.config.get("message", "")

        resolved_message = context.resolve_expression(message) if message else ""

        return NodeResult(
            success=True,
            output={
                "action": "hangup",
                "reason": reason,
                "message": resolved_message,
            },
            should_continue=False,
        )


class RecordNodeExecutor(NodeExecutor):
    """Record node - start/stop recording."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Control recording."""
        action = self.node.config.get("action", "start")  # start, stop, pause, resume
        recording_id_var = self.node.config.get("recording_id_variable", "recording_id")
        max_duration_ms = self.node.config.get("max_duration_ms", 300000)
        silence_timeout_ms = self.node.config.get("silence_timeout_ms", 5000)

        return NodeResult(
            success=True,
            output={
                "action": f"record_{action}",
                "max_duration_ms": max_duration_ms,
                "silence_timeout_ms": silence_timeout_ms,
            },
            variable_updates={recording_id_var: f"rec_{context.execution_id}"} if action == "start" else {},
        )


class PlayNodeExecutor(NodeExecutor):
    """Play node - play audio file."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Play audio file."""
        audio_url = self.node.config.get("audio_url", "")
        audio_id = self.node.config.get("audio_id", "")
        loop = self.node.config.get("loop", False)

        resolved_url = context.resolve_expression(audio_url) if audio_url else ""

        return NodeResult(
            success=True,
            output={
                "action": "play",
                "audio_url": resolved_url,
                "audio_id": audio_id,
                "loop": loop,
            },
        )


# =============================================================================
# Logic Nodes
# =============================================================================

class SetVariableNodeExecutor(NodeExecutor):
    """Set variable node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Set one or more variables."""
        assignments = self.node.config.get("assignments", {})
        variable = self.node.config.get("variable", "")
        value = self.node.config.get("value", "")

        updates = {}

        # Single assignment (legacy)
        if variable:
            resolved_value = context.resolve_expression(value)
            updates[variable] = self._parse_value(resolved_value)

        # Multiple assignments
        for var_name, var_value in assignments.items():
            resolved = context.resolve_expression(str(var_value))
            updates[var_name] = self._parse_value(resolved)

        return NodeResult(
            success=True,
            output={"updated_variables": list(updates.keys())},
            variable_updates=updates,
        )

    def _parse_value(self, value: str) -> Any:
        """Parse value to appropriate type."""
        if value is None:
            return None

        str_value = str(value)

        # Try JSON parse
        try:
            return json.loads(str_value)
        except:
            pass

        # Try number
        try:
            if '.' in str_value:
                return float(str_value)
            return int(str_value)
        except:
            pass

        # Return as string
        return str_value


class FunctionNodeExecutor(NodeExecutor):
    """
    Function node - execute custom function.

    Supports JavaScript-like expressions.
    """

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute function."""
        code = self.node.config.get("code", "")
        output_var = self.node.config.get("output_variable", "result")

        if not code:
            return NodeResult(
                success=True,
                output=None,
            )

        try:
            # Create safe execution context
            safe_globals = {
                "__builtins__": {},
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
            }

            # Add context variables
            safe_locals = dict(context.variables)
            safe_locals["input"] = context.input_data
            safe_locals["output"] = context.output

            # Execute code
            exec(code, safe_globals, safe_locals)

            # Get result
            result = safe_locals.get("result", safe_locals.get(output_var))

            return NodeResult(
                success=True,
                output=result,
                variable_updates={output_var: result} if result is not None else {},
            )

        except Exception as e:
            logger.error(f"Function execution error: {e}")
            return NodeResult(
                success=False,
                error=str(e),
            )


class HttpRequestNodeExecutor(NodeExecutor):
    """HTTP request node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Make HTTP request."""
        method = self.node.config.get("method", "GET")
        url = self.node.config.get("url", "")
        headers = self.node.config.get("headers", {})
        body = self.node.config.get("body", {})
        timeout_seconds = self.node.config.get("timeout_seconds", 30)
        output_var = self.node.config.get("output_variable", "http_response")

        # Resolve variables
        resolved_url = context.resolve_expression(url)
        resolved_headers = {
            k: context.resolve_expression(v) for k, v in headers.items()
        }
        resolved_body = self._resolve_body(body, context)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method.upper(),
                    url=resolved_url,
                    headers=resolved_headers,
                    json=resolved_body if method.upper() != "GET" else None,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                ) as response:
                    response_data = {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "body": await response.text(),
                    }

                    # Try to parse JSON
                    try:
                        response_data["json"] = await response.json()
                    except:
                        pass

                    return NodeResult(
                        success=response.status < 400,
                        output=response_data,
                        variable_updates={output_var: response_data},
                        next_port="success" if response.status < 400 else "error",
                    )

        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            return NodeResult(
                success=False,
                error=str(e),
                next_port="error",
            )

    def _resolve_body(self, body: Any, context: ExecutionContext) -> Any:
        """Recursively resolve body variables."""
        if isinstance(body, str):
            return context.resolve_expression(body)
        elif isinstance(body, dict):
            return {k: self._resolve_body(v, context) for k, v in body.items()}
        elif isinstance(body, list):
            return [self._resolve_body(item, context) for item in body]
        return body


class WebhookNodeExecutor(NodeExecutor):
    """Webhook node - send webhook event."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Send webhook."""
        url = self.node.config.get("url", "")
        event = self.node.config.get("event", "workflow.event")
        payload = self.node.config.get("payload", {})
        async_send = self.node.config.get("async", True)

        resolved_url = context.resolve_expression(url)
        resolved_payload = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": context.workflow_id,
            "execution_id": context.execution_id,
            "data": {k: context.resolve_expression(str(v)) for k, v in payload.items()},
        }

        if async_send:
            # Fire and forget
            asyncio.create_task(self._send_webhook(resolved_url, resolved_payload))
            return NodeResult(
                success=True,
                output={"webhook_sent": True, "async": True},
            )
        else:
            # Wait for response
            result = await self._send_webhook(resolved_url, resolved_payload)
            return NodeResult(
                success=result.get("success", False),
                output=result,
            )

    async def _send_webhook(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook request."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    return {
                        "success": response.status < 400,
                        "status": response.status,
                    }
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return {"success": False, "error": str(e)}


# =============================================================================
# AI Nodes
# =============================================================================

class LLMPromptNodeExecutor(NodeExecutor):
    """LLM prompt node - send prompt to LLM."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute LLM prompt."""
        prompt = self.node.config.get("prompt", "")
        system_prompt = self.node.config.get("system_prompt", "")
        model = self.node.config.get("model", "gpt-4-turbo")
        temperature = self.node.config.get("temperature", 0.7)
        max_tokens = self.node.config.get("max_tokens", 500)
        output_var = self.node.config.get("output_variable", "llm_response")

        resolved_prompt = context.resolve_expression(prompt)
        resolved_system = context.resolve_expression(system_prompt)

        # In real implementation, this would call the LLM API
        # For now, return placeholder
        return NodeResult(
            success=True,
            output={
                "action": "llm_prompt",
                "prompt": resolved_prompt,
                "system_prompt": resolved_system,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            variable_updates={output_var: "{{LLM_RESPONSE}}"},
        )


class IntentDetectNodeExecutor(NodeExecutor):
    """Intent detection node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Detect intent from text."""
        text = self.node.config.get("text", "{{user_input}}")
        intents = self.node.config.get("intents", [])
        confidence_threshold = self.node.config.get("confidence_threshold", 0.7)
        output_var = self.node.config.get("output_variable", "detected_intent")

        resolved_text = context.resolve_expression(text)

        # In real implementation, this would call NLU service
        return NodeResult(
            success=True,
            output={
                "action": "intent_detect",
                "text": resolved_text,
                "intents": intents,
            },
            variable_updates={output_var: "{{DETECTED_INTENT}}"},
        )


class EntityExtractNodeExecutor(NodeExecutor):
    """Entity extraction node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Extract entities from text."""
        text = self.node.config.get("text", "{{user_input}}")
        entity_types = self.node.config.get("entity_types", [])
        output_var = self.node.config.get("output_variable", "entities")

        resolved_text = context.resolve_expression(text)

        return NodeResult(
            success=True,
            output={
                "action": "entity_extract",
                "text": resolved_text,
                "entity_types": entity_types,
            },
            variable_updates={output_var: {}},
        )


class SentimentNodeExecutor(NodeExecutor):
    """Sentiment analysis node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Analyze sentiment of text."""
        text = self.node.config.get("text", "{{user_input}}")
        output_var = self.node.config.get("output_variable", "sentiment")

        resolved_text = context.resolve_expression(text)

        return NodeResult(
            success=True,
            output={
                "action": "sentiment_analyze",
                "text": resolved_text,
            },
            variable_updates={
                output_var: {"label": "neutral", "score": 0.5}
            },
        )


# =============================================================================
# Integration Nodes
# =============================================================================

class CRMLookupNodeExecutor(NodeExecutor):
    """CRM lookup node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Look up record in CRM."""
        crm_type = self.node.config.get("crm_type", "salesforce")
        object_type = self.node.config.get("object_type", "contact")
        lookup_field = self.node.config.get("lookup_field", "phone")
        lookup_value = self.node.config.get("lookup_value", "")
        output_var = self.node.config.get("output_variable", "crm_record")

        resolved_value = context.resolve_expression(lookup_value)

        return NodeResult(
            success=True,
            output={
                "action": "crm_lookup",
                "crm_type": crm_type,
                "object_type": object_type,
                "lookup_field": lookup_field,
                "lookup_value": resolved_value,
            },
            variable_updates={output_var: None},
        )


class DatabaseNodeExecutor(NodeExecutor):
    """Database query node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute database operation."""
        operation = self.node.config.get("operation", "query")  # query, insert, update, delete
        query = self.node.config.get("query", "")
        parameters = self.node.config.get("parameters", {})
        output_var = self.node.config.get("output_variable", "db_result")

        resolved_query = context.resolve_expression(query)
        resolved_params = {
            k: context.resolve_expression(str(v))
            for k, v in parameters.items()
        }

        return NodeResult(
            success=True,
            output={
                "action": f"db_{operation}",
                "query": resolved_query,
                "parameters": resolved_params,
            },
            variable_updates={output_var: []},
        )


class QueueNodeExecutor(NodeExecutor):
    """Queue node - add to call queue."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Add call to queue."""
        queue_name = self.node.config.get("queue_name", "default")
        priority = self.node.config.get("priority", 5)
        skills = self.node.config.get("skills", [])
        timeout_seconds = self.node.config.get("timeout_seconds", 300)

        return NodeResult(
            success=True,
            output={
                "action": "queue",
                "queue_name": queue_name,
                "priority": priority,
                "skills": skills,
                "timeout_seconds": timeout_seconds,
            },
            should_wait=True,
            wait_duration_ms=timeout_seconds * 1000,
        )


class SMSNodeExecutor(NodeExecutor):
    """SMS node - send SMS message."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Send SMS message."""
        to_number = self.node.config.get("to_number", "")
        message = self.node.config.get("message", "")
        from_number = self.node.config.get("from_number", "")

        resolved_to = context.resolve_expression(to_number)
        resolved_message = context.resolve_expression(message)

        return NodeResult(
            success=True,
            output={
                "action": "send_sms",
                "to": resolved_to,
                "from": from_number,
                "message": resolved_message,
            },
        )


class EmailNodeExecutor(NodeExecutor):
    """Email node - send email."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Send email."""
        to_email = self.node.config.get("to_email", "")
        subject = self.node.config.get("subject", "")
        body = self.node.config.get("body", "")
        html_body = self.node.config.get("html_body", "")

        resolved_to = context.resolve_expression(to_email)
        resolved_subject = context.resolve_expression(subject)
        resolved_body = context.resolve_expression(body)
        resolved_html = context.resolve_expression(html_body) if html_body else ""

        return NodeResult(
            success=True,
            output={
                "action": "send_email",
                "to": resolved_to,
                "subject": resolved_subject,
                "body": resolved_body,
                "html_body": resolved_html,
            },
        )


class SubflowNodeExecutor(NodeExecutor):
    """Subflow node - execute nested workflow."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute subflow."""
        subflow_id = self.node.config.get("subflow_id", "")
        input_mapping = self.node.config.get("input_mapping", {})
        output_mapping = self.node.config.get("output_mapping", {})

        # Map inputs
        subflow_input = {}
        for subflow_var, parent_var in input_mapping.items():
            subflow_input[subflow_var] = context.get_variable(parent_var)

        return NodeResult(
            success=True,
            output={
                "action": "execute_subflow",
                "subflow_id": subflow_id,
                "input": subflow_input,
                "output_mapping": output_mapping,
            },
        )


# =============================================================================
# Node Registry
# =============================================================================

NODE_EXECUTORS: Dict[NodeType, Type[NodeExecutor]] = {
    # Flow control
    NodeType.START: StartNodeExecutor,
    NodeType.END: EndNodeExecutor,
    NodeType.CONDITION: ConditionNodeExecutor,
    NodeType.SWITCH: SwitchNodeExecutor,
    NodeType.PARALLEL: ParallelNodeExecutor,
    NodeType.MERGE: MergeNodeExecutor,
    NodeType.LOOP: LoopNodeExecutor,
    NodeType.WAIT: WaitNodeExecutor,
    NodeType.GOTO: GotoNodeExecutor,

    # Voice/Call actions
    NodeType.SPEAK: SpeakNodeExecutor,
    NodeType.LISTEN: ListenNodeExecutor,
    NodeType.GATHER: GatherNodeExecutor,
    NodeType.TRANSFER: TransferNodeExecutor,
    NodeType.HANGUP: HangupNodeExecutor,
    NodeType.RECORD: RecordNodeExecutor,
    NodeType.PLAY: PlayNodeExecutor,

    # Logic
    NodeType.SET_VARIABLE: SetVariableNodeExecutor,
    NodeType.FUNCTION: FunctionNodeExecutor,
    NodeType.HTTP_REQUEST: HttpRequestNodeExecutor,
    NodeType.WEBHOOK: WebhookNodeExecutor,

    # AI
    NodeType.LLM_PROMPT: LLMPromptNodeExecutor,
    NodeType.INTENT_DETECT: IntentDetectNodeExecutor,
    NodeType.ENTITY_EXTRACT: EntityExtractNodeExecutor,
    NodeType.SENTIMENT: SentimentNodeExecutor,

    # Integration
    NodeType.CRM_LOOKUP: CRMLookupNodeExecutor,
    NodeType.DATABASE: DatabaseNodeExecutor,
    NodeType.QUEUE: QueueNodeExecutor,
    NodeType.SMS: SMSNodeExecutor,
    NodeType.EMAIL: EmailNodeExecutor,
    NodeType.SUBFLOW: SubflowNodeExecutor,
}


def register_node_executors(engine) -> None:
    """Register all node executors with engine."""
    for node_type, executor_class in NODE_EXECUTORS.items():
        engine.register_executor(node_type, executor_class)


def create_node(
    node_type: NodeType,
    name: str = "",
    config: Optional[Dict[str, Any]] = None,
    position: Optional[Position] = None,
) -> NodeConfig:
    """Create a new node configuration."""
    return NodeConfig(
        node_type=node_type,
        name=name or node_type.value,
        config=config or {},
        position=position or Position(),
    )
