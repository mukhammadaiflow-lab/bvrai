"""Conversation flow execution engine."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
import httpx

from app.flow.node import FlowNode, NodeType, CollectConfig
from app.flow.state import FlowState, FlowStatus, StateStore


logger = structlog.get_logger()


@dataclass
class FlowConfig:
    """Configuration for flow engine."""
    max_steps: int = 100  # Max steps per execution
    step_timeout_seconds: float = 30.0
    ai_orchestrator_url: str = "http://ai-orchestrator:8085"
    tts_service_url: str = "http://tts-service:8083"
    enable_barge_in: bool = True
    default_voice_id: str = "default"


@dataclass
class FlowResult:
    """Result of flow execution."""
    messages: List[str]
    actions: List[Dict[str, Any]]
    status: FlowStatus
    current_node: Optional[str]
    waiting_for_input: bool
    error: Optional[str] = None


class FlowEngine:
    """
    Executes conversation flows.

    The flow engine interprets flow definitions and
    executes them step by step, handling user input,
    conditions, and function calls.
    """

    def __init__(self, config: Optional[FlowConfig] = None):
        self.config = config or FlowConfig()
        self._flows: Dict[str, Dict[str, FlowNode]] = {}  # flow_id -> nodes
        self._state_store = StateStore()
        self._function_handlers: Dict[str, Callable] = {}
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        """Start the flow engine."""
        self._http_client = httpx.AsyncClient(
            timeout=self.config.step_timeout_seconds
        )
        logger.info("flow_engine_started")

    async def stop(self) -> None:
        """Stop the flow engine."""
        if self._http_client:
            await self._http_client.aclose()
        logger.info("flow_engine_stopped")

    def register_flow(self, flow_id: str, nodes: List[FlowNode]) -> None:
        """
        Register a flow definition.

        Args:
            flow_id: Unique flow identifier
            nodes: List of flow nodes
        """
        node_map = {node.id: node for node in nodes}
        self._flows[flow_id] = node_map

        logger.info(
            "flow_registered",
            flow_id=flow_id,
            node_count=len(nodes),
        )

    def register_function(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """Register a function handler."""
        self._function_handlers[name] = handler

    async def start_flow(
        self,
        flow_id: str,
        session_id: str,
        initial_variables: Optional[Dict[str, Any]] = None,
    ) -> FlowResult:
        """
        Start a new flow for a session.

        Args:
            flow_id: Flow to start
            session_id: Session identifier
            initial_variables: Initial variable values

        Returns:
            Flow execution result
        """
        if flow_id not in self._flows:
            return FlowResult(
                messages=[],
                actions=[],
                status=FlowStatus.FAILED,
                current_node=None,
                waiting_for_input=False,
                error=f"Flow not found: {flow_id}",
            )

        nodes = self._flows[flow_id]

        # Find start node
        start_node = None
        for node in nodes.values():
            if node.type == NodeType.START:
                start_node = node
                break

        if not start_node:
            # Use first node
            start_node = list(nodes.values())[0]

        # Create state
        state = FlowState(
            flow_id=flow_id,
            session_id=session_id,
        )

        if initial_variables:
            for key, value in initial_variables.items():
                state.set_variable(key, value)

        # Start execution
        state.status = FlowStatus.RUNNING
        state.transition_to(start_node.id, trigger="start")

        # Execute until waiting or complete
        result = await self._execute_until_wait(state, nodes)

        # Save state
        await self._state_store.save(state)

        return result

    async def continue_flow(
        self,
        session_id: str,
        user_input: str,
    ) -> FlowResult:
        """
        Continue a flow with user input.

        Args:
            session_id: Session identifier
            user_input: User's input text

        Returns:
            Flow execution result
        """
        # Load state
        state = await self._state_store.load(session_id)
        if not state:
            return FlowResult(
                messages=["I'm sorry, I lost track of our conversation."],
                actions=[],
                status=FlowStatus.FAILED,
                current_node=None,
                waiting_for_input=False,
                error="State not found",
            )

        if state.flow_id not in self._flows:
            return FlowResult(
                messages=[],
                actions=[],
                status=FlowStatus.FAILED,
                current_node=state.current_node_id,
                waiting_for_input=False,
                error=f"Flow not found: {state.flow_id}",
            )

        nodes = self._flows[state.flow_id]

        # Process input if collecting slot
        if state.pending_slot:
            await self._process_slot_input(state, user_input, nodes)
        else:
            # Store as last_user_input
            state.set_variable("last_user_input", user_input)

        # Continue execution
        state.status = FlowStatus.RUNNING
        result = await self._execute_until_wait(state, nodes)

        # Save state
        await self._state_store.save(state)

        return result

    async def _execute_until_wait(
        self,
        state: FlowState,
        nodes: Dict[str, FlowNode],
    ) -> FlowResult:
        """Execute flow until waiting for input or complete."""
        steps = 0

        while steps < self.config.max_steps:
            steps += 1

            if not state.current_node_id:
                state.status = FlowStatus.COMPLETED
                break

            node = nodes.get(state.current_node_id)
            if not node:
                state.set_error(f"Node not found: {state.current_node_id}")
                break

            # Execute node
            should_wait = await self._execute_node(state, node, nodes)

            if should_wait:
                state.status = FlowStatus.WAITING
                break

            # Check for completion
            if state.status in (FlowStatus.COMPLETED, FlowStatus.FAILED, FlowStatus.TRANSFERRED):
                break

            # Move to next node
            next_node_id = node.get_next_node(state.get_context())
            if next_node_id:
                state.transition_to(next_node_id, trigger="auto")
            else:
                state.status = FlowStatus.COMPLETED
                break

        return FlowResult(
            messages=state.flush_messages(),
            actions=state.flush_actions(),
            status=state.status,
            current_node=state.current_node_id,
            waiting_for_input=(state.status == FlowStatus.WAITING),
            error=state.error_message,
        )

    async def _execute_node(
        self,
        state: FlowState,
        node: FlowNode,
        nodes: Dict[str, FlowNode],
    ) -> bool:
        """
        Execute a single node.

        Returns True if should wait for input.
        """
        logger.debug(
            "executing_node",
            session_id=state.session_id,
            node_id=node.id,
            node_type=node.type.value,
        )

        # Call on_enter hook
        if node.on_enter:
            await self._call_function(node.on_enter, state)

        if node.type == NodeType.START:
            # Start node - just continue
            return False

        elif node.type == NodeType.MESSAGE:
            # Send message
            message = node.get_message(state.get_context())
            if message:
                state.add_message(message)
            return False

        elif node.type == NodeType.COLLECT:
            # Collect user input
            return await self._handle_collect(state, node)

        elif node.type == NodeType.CONDITION:
            # Condition is handled by get_next_node
            return False

        elif node.type == NodeType.FUNCTION:
            # Execute function
            await self._handle_function(state, node)
            return False

        elif node.type == NodeType.TRANSFER:
            # Transfer to human
            state.add_action({
                "type": "transfer",
                "target": node.transfer_target,
                "message": node.transfer_message,
            })
            state.status = FlowStatus.TRANSFERRED
            return False

        elif node.type == NodeType.END:
            # End flow
            if node.message:
                state.add_message(node.get_message(state.get_context()))
            state.status = FlowStatus.COMPLETED
            return False

        elif node.type == NodeType.WAIT:
            # Wait for input
            if node.message:
                state.add_message(node.get_message(state.get_context()))
            return True

        elif node.type == NodeType.LOOP:
            # Handle loop
            loop_id = f"loop_{node.id}"
            iteration = state.increment_loop(loop_id)

            if state.is_loop_exceeded(loop_id):
                logger.warning(
                    "loop_exceeded",
                    node_id=node.id,
                    iterations=iteration,
                )
                state.reset_loop(loop_id)
                # Skip to next after loop
                return False

            return False

        return False

    async def _handle_collect(
        self,
        state: FlowState,
        node: FlowNode,
    ) -> bool:
        """Handle collect node - request user input."""
        if not node.collect:
            return False

        config = node.collect
        attempts = state.record_slot_attempt(config.slot_name)

        if attempts > config.max_attempts:
            # Max attempts reached
            state.add_message("I'm having trouble understanding. Let me try a different approach.")
            state.pending_slot = None
            return False

        # Set pending slot
        state.pending_slot = config.slot_name

        # Send prompt
        prompt = config.prompt or f"Please provide your {config.slot_name}."
        state.add_message(prompt)

        return True

    async def _process_slot_input(
        self,
        state: FlowState,
        user_input: str,
        nodes: Dict[str, FlowNode],
    ) -> None:
        """Process user input for a pending slot."""
        slot_name = state.pending_slot
        if not slot_name:
            return

        # Get current node's collect config
        node = nodes.get(state.current_node_id)
        if not node or not node.collect:
            state.pending_slot = None
            return

        config = node.collect

        # Validate input
        if config.validation_regex:
            import re
            if not re.match(config.validation_regex, user_input):
                # Invalid input
                validation_msg = config.validation_message or f"That doesn't look like a valid {config.slot_name}. Please try again."
                state.add_message(validation_msg)
                return  # Don't clear pending_slot, will retry

        # Store value
        state.set_variable(slot_name, user_input)
        state.pending_slot = None

        # Confirmation if needed
        if config.confirm:
            state.add_message(f"Got it, {user_input}. Is that correct?")
            state.set_variable(f"{slot_name}_pending_confirm", True)

        # Move to next node
        next_node_id = node.get_next_node(state.get_context())
        if next_node_id:
            state.transition_to(next_node_id, trigger="input")

    async def _handle_function(
        self,
        state: FlowState,
        node: FlowNode,
    ) -> None:
        """Handle function node - execute function."""
        if not node.function_name:
            return

        result = await self._call_function(
            node.function_name,
            state,
            **node.function_args,
        )

        # Store result if configured
        if node.store_result:
            state.set_variable(node.store_result, result)

        # Check for special result actions
        if isinstance(result, dict):
            if result.get("action") == "transfer":
                state.add_action(result)
                state.status = FlowStatus.TRANSFERRED
            elif result.get("action") == "end_call":
                state.status = FlowStatus.COMPLETED
            elif result.get("message"):
                state.add_message(result["message"])

    async def _call_function(
        self,
        function_name: str,
        state: FlowState,
        **kwargs,
    ) -> Any:
        """Call a registered function or remote function."""
        # Check local handlers first
        if function_name in self._function_handlers:
            handler = self._function_handlers[function_name]
            return await handler(state=state, **kwargs)

        # Call AI orchestrator for function execution
        if self._http_client:
            try:
                response = await self._http_client.post(
                    f"{self.config.ai_orchestrator_url}/v1/functions/execute",
                    json={
                        "function_name": function_name,
                        "arguments": kwargs,
                        "context": {
                            "session_id": state.session_id,
                            "variables": state.variables,
                        },
                    },
                )
                response.raise_for_status()
                return response.json().get("result")

            except Exception as e:
                logger.error(
                    "function_call_failed",
                    function=function_name,
                    error=str(e),
                )
                return {"error": str(e)}

        return None

    async def get_state(self, session_id: str) -> Optional[FlowState]:
        """Get current state for a session."""
        return await self._state_store.load(session_id)

    async def end_flow(self, session_id: str) -> bool:
        """End a flow early."""
        state = await self._state_store.load(session_id)
        if state:
            state.status = FlowStatus.COMPLETED
            await self._state_store.save(state)
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "registered_flows": len(self._flows),
            "registered_functions": len(self._function_handlers),
            "flows": list(self._flows.keys()),
        }
