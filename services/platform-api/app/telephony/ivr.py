"""IVR (Interactive Voice Response) system."""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class IVRAction(str, Enum):
    """Actions available in IVR flows."""
    PLAY = "play"  # Play audio/TTS
    GATHER = "gather"  # Gather DTMF input
    RECORD = "record"  # Record caller's voice
    TRANSFER = "transfer"  # Transfer call
    DIAL = "dial"  # Dial out
    HANGUP = "hangup"  # End call
    PAUSE = "pause"  # Pause for duration
    GOTO = "goto"  # Go to another node
    MENU = "menu"  # Present menu options
    VOICEMAIL = "voicemail"  # Send to voicemail
    CALLBACK = "callback"  # Execute custom callback
    CONDITION = "condition"  # Conditional branching
    SET_VARIABLE = "set_variable"  # Set a variable
    API_CALL = "api_call"  # Make external API call


class GatherInputType(str, Enum):
    """Types of input for gather action."""
    DTMF = "dtmf"
    SPEECH = "speech"
    DTMF_SPEECH = "dtmf_speech"


@dataclass
class IVRNode:
    """A node in an IVR flow."""
    node_id: str
    action: IVRAction
    parameters: Dict[str, Any] = field(default_factory=dict)
    next_node: Optional[str] = None
    on_error: Optional[str] = None
    conditions: Dict[str, str] = field(default_factory=dict)  # condition -> next_node
    timeout_seconds: float = 10.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "action": self.action.value,
            "parameters": self.parameters,
            "next_node": self.next_node,
            "on_error": self.on_error,
            "conditions": self.conditions,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "metadata": self.metadata,
        }


@dataclass
class IVRMenu:
    """A menu definition for IVR."""
    menu_id: str
    prompt_text: Optional[str] = None
    prompt_audio_url: Optional[str] = None
    options: Dict[str, str] = field(default_factory=dict)  # key -> label
    option_nodes: Dict[str, str] = field(default_factory=dict)  # key -> node_id
    invalid_input_node: Optional[str] = None
    timeout_node: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "menu_id": self.menu_id,
            "prompt_text": self.prompt_text,
            "prompt_audio_url": self.prompt_audio_url,
            "options": self.options,
            "option_nodes": self.option_nodes,
            "invalid_input_node": self.invalid_input_node,
            "timeout_node": self.timeout_node,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class IVRFlow:
    """Complete IVR flow definition."""
    flow_id: str
    name: str
    version: str = "1.0"
    start_node: str = "start"
    nodes: Dict[str, IVRNode] = field(default_factory=dict)
    menus: Dict[str, IVRMenu] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flow_id": self.flow_id,
            "name": self.name,
            "version": self.version,
            "start_node": self.start_node,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "menus": {k: v.to_dict() for k, v in self.menus.items()},
            "variables": self.variables,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "metadata": self.metadata,
        }


@dataclass
class IVRSession:
    """An active IVR session for a call."""
    session_id: str
    flow_id: str
    call_id: str
    current_node: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "flow_id": self.flow_id,
            "call_id": self.call_id,
            "current_node": self.current_node,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "variables": self.variables,
            "history": self.history,
            "is_active": self.is_active,
        }


class IVREngine:
    """
    Executes IVR flows for calls.

    Usage:
        engine = IVREngine()

        # Register handlers
        engine.on_play(tts_handler)
        engine.on_gather(dtmf_handler)
        engine.on_transfer(transfer_handler)

        # Start flow
        session = await engine.start_flow(flow, call_id)

        # Process events
        await engine.handle_input(session.session_id, "1")
    """

    def __init__(self):
        self._flows: Dict[str, IVRFlow] = {}
        self._sessions: Dict[str, IVRSession] = {}
        self._handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()

    def register_flow(self, flow: IVRFlow) -> None:
        """Register an IVR flow."""
        self._flows[flow.flow_id] = flow
        logger.info(f"IVR flow registered: {flow.flow_id}")

    def get_flow(self, flow_id: str) -> Optional[IVRFlow]:
        """Get a flow by ID."""
        return self._flows.get(flow_id)

    def on_action(
        self,
        action: IVRAction,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """Register handler for an action type."""
        self._handlers[action.value] = handler

    def on_play(self, handler: Callable) -> None:
        """Register play handler."""
        self.on_action(IVRAction.PLAY, handler)

    def on_gather(self, handler: Callable) -> None:
        """Register gather handler."""
        self.on_action(IVRAction.GATHER, handler)

    def on_record(self, handler: Callable) -> None:
        """Register record handler."""
        self.on_action(IVRAction.RECORD, handler)

    def on_transfer(self, handler: Callable) -> None:
        """Register transfer handler."""
        self.on_action(IVRAction.TRANSFER, handler)

    def on_dial(self, handler: Callable) -> None:
        """Register dial handler."""
        self.on_action(IVRAction.DIAL, handler)

    def on_hangup(self, handler: Callable) -> None:
        """Register hangup handler."""
        self.on_action(IVRAction.HANGUP, handler)

    async def start_flow(
        self,
        flow_id: str,
        call_id: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Optional[IVRSession]:
        """Start an IVR flow for a call."""
        flow = self._flows.get(flow_id)
        if not flow or not flow.is_active:
            logger.error(f"Flow not found or inactive: {flow_id}")
            return None

        import uuid
        session = IVRSession(
            session_id=f"ivr_{uuid.uuid4().hex[:12]}",
            flow_id=flow_id,
            call_id=call_id,
            current_node=flow.start_node,
            variables={**flow.variables, **(variables or {})},
        )

        async with self._lock:
            self._sessions[session.session_id] = session

        logger.info(f"IVR session started: {session.session_id} for call {call_id}")

        # Execute start node
        await self._execute_current_node(session)

        return session

    async def end_session(
        self,
        session_id: str,
        reason: str = "completed",
    ) -> Optional[IVRSession]:
        """End an IVR session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            session.is_active = False
            session.ended_at = datetime.utcnow()
            session.history.append({
                "action": "session_ended",
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            })

        logger.info(f"IVR session ended: {session_id} - {reason}")
        return session

    async def handle_input(
        self,
        session_id: str,
        input_value: str,
        input_type: str = "dtmf",
    ) -> bool:
        """Handle user input during IVR session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.is_active:
                return False

        flow = self._flows.get(session.flow_id)
        if not flow:
            return False

        current_node = flow.nodes.get(session.current_node)
        if not current_node:
            return False

        # Store input in variables
        session.variables["last_input"] = input_value
        session.variables["last_input_type"] = input_type

        # Record in history
        session.history.append({
            "node": session.current_node,
            "action": "input_received",
            "input": input_value,
            "input_type": input_type,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Determine next node based on input
        next_node = None

        # Check menu options
        if current_node.action == IVRAction.MENU:
            menu_id = current_node.parameters.get("menu_id")
            menu = flow.menus.get(menu_id) if menu_id else None

            if menu and input_value in menu.option_nodes:
                next_node = menu.option_nodes[input_value]
            elif menu and menu.invalid_input_node:
                # Check retries
                retry_key = f"{session.current_node}_menu"
                retries = session.retry_counts.get(retry_key, 0)
                if retries < menu.max_retries:
                    session.retry_counts[retry_key] = retries + 1
                    next_node = menu.invalid_input_node
                else:
                    next_node = menu.timeout_node

        # Check gather conditions
        elif current_node.action == IVRAction.GATHER:
            # Check conditions
            for condition, target_node in current_node.conditions.items():
                if self._evaluate_condition(condition, input_value, session.variables):
                    next_node = target_node
                    break

            # Default to next_node
            if not next_node:
                next_node = current_node.next_node

        # Move to next node
        if next_node:
            session.current_node = next_node
            await self._execute_current_node(session)
            return True

        return False

    async def handle_timeout(
        self,
        session_id: str,
    ) -> bool:
        """Handle timeout during IVR session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.is_active:
                return False

        flow = self._flows.get(session.flow_id)
        if not flow:
            return False

        current_node = flow.nodes.get(session.current_node)
        if not current_node:
            return False

        # Check for menu timeout
        if current_node.action == IVRAction.MENU:
            menu_id = current_node.parameters.get("menu_id")
            menu = flow.menus.get(menu_id) if menu_id else None

            if menu and menu.timeout_node:
                session.current_node = menu.timeout_node
                await self._execute_current_node(session)
                return True

        # Check for on_error handler
        if current_node.on_error:
            session.current_node = current_node.on_error
            await self._execute_current_node(session)
            return True

        return False

    async def _execute_current_node(
        self,
        session: IVRSession,
    ) -> None:
        """Execute the current node in a session."""
        flow = self._flows.get(session.flow_id)
        if not flow:
            return

        node = flow.nodes.get(session.current_node)
        if not node:
            logger.error(f"Node not found: {session.current_node}")
            return

        logger.debug(f"Executing IVR node: {node.node_id} ({node.action.value})")

        # Record in history
        session.history.append({
            "node": node.node_id,
            "action": node.action.value,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Execute action handler
        handler = self._handlers.get(node.action.value)
        if handler:
            try:
                result = await handler(session, node, flow)
                session.variables["last_result"] = result
            except Exception as e:
                logger.error(f"IVR action handler error: {e}")
                if node.on_error:
                    session.current_node = node.on_error
                    await self._execute_current_node(session)
                return

        # Handle special actions
        if node.action == IVRAction.HANGUP:
            await self.end_session(session.session_id, "hangup")
            return

        elif node.action == IVRAction.GOTO:
            target = node.parameters.get("target")
            if target:
                session.current_node = target
                await self._execute_current_node(session)
            return

        elif node.action == IVRAction.SET_VARIABLE:
            var_name = node.parameters.get("name")
            var_value = node.parameters.get("value")
            if var_name:
                session.variables[var_name] = var_value
            if node.next_node:
                session.current_node = node.next_node
                await self._execute_current_node(session)
            return

        elif node.action == IVRAction.CONDITION:
            # Evaluate conditions and branch
            for condition, target_node in node.conditions.items():
                if self._evaluate_condition(condition, None, session.variables):
                    session.current_node = target_node
                    await self._execute_current_node(session)
                    return
            # Default path
            if node.next_node:
                session.current_node = node.next_node
                await self._execute_current_node(session)
            return

        elif node.action == IVRAction.PAUSE:
            duration = node.parameters.get("duration", 1.0)
            await asyncio.sleep(duration)
            if node.next_node:
                session.current_node = node.next_node
                await self._execute_current_node(session)
            return

        # For gather/menu actions, wait for input
        # For other actions, auto-advance to next node
        if node.action not in [IVRAction.GATHER, IVRAction.MENU, IVRAction.RECORD]:
            if node.next_node:
                session.current_node = node.next_node
                # Small delay to prevent tight loops
                await asyncio.sleep(0.1)
                await self._execute_current_node(session)

    def _evaluate_condition(
        self,
        condition: str,
        input_value: Optional[str],
        variables: Dict[str, Any],
    ) -> bool:
        """Evaluate a condition expression."""
        # Simple condition evaluation
        # Format: "variable == value" or "input == value"

        if "==" in condition:
            parts = condition.split("==")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip().strip("'\"")

                if left == "input":
                    return input_value == right
                else:
                    return str(variables.get(left, "")) == right

        elif "!=" in condition:
            parts = condition.split("!=")
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip().strip("'\"")

                if left == "input":
                    return input_value != right
                else:
                    return str(variables.get(left, "")) != right

        elif condition == "true" or condition == "default":
            return True

        return False

    async def get_session(
        self,
        session_id: str,
    ) -> Optional[IVRSession]:
        """Get an IVR session."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_session_by_call(
        self,
        call_id: str,
    ) -> Optional[IVRSession]:
        """Get session for a call."""
        async with self._lock:
            for session in self._sessions.values():
                if session.call_id == call_id and session.is_active:
                    return session
        return None


class IVRBuilder:
    """
    Fluent builder for creating IVR flows.

    Usage:
        builder = IVRBuilder("greeting_flow", "Customer Greeting")

        flow = (builder
            .play("start", "Welcome to our service.")
            .menu("main_menu",
                prompt="Press 1 for sales, 2 for support.",
                options={"1": "sales", "2": "support"})
            .play("sales", "Connecting to sales.")
            .transfer("sales_transfer", "+15551234567", next_node="end")
            .play("support", "Connecting to support.")
            .transfer("support_transfer", "+15559876543", next_node="end")
            .hangup("end")
            .build())
    """

    def __init__(self, flow_id: str, name: str):
        self.flow = IVRFlow(
            flow_id=flow_id,
            name=name,
        )
        self._first_node: Optional[str] = None

    def _add_node(
        self,
        node_id: str,
        action: IVRAction,
        parameters: Dict[str, Any],
        next_node: Optional[str] = None,
        **kwargs,
    ) -> "IVRBuilder":
        """Add a node to the flow."""
        node = IVRNode(
            node_id=node_id,
            action=action,
            parameters=parameters,
            next_node=next_node,
            **kwargs,
        )
        self.flow.nodes[node_id] = node

        if not self._first_node:
            self._first_node = node_id
            self.flow.start_node = node_id

        return self

    def play(
        self,
        node_id: str,
        text: Optional[str] = None,
        audio_url: Optional[str] = None,
        next_node: Optional[str] = None,
    ) -> "IVRBuilder":
        """Add a play action node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.PLAY,
            parameters={"text": text, "audio_url": audio_url},
            next_node=next_node,
        )

    def gather(
        self,
        node_id: str,
        prompt_text: Optional[str] = None,
        num_digits: int = 1,
        timeout: float = 10.0,
        input_type: GatherInputType = GatherInputType.DTMF,
        conditions: Optional[Dict[str, str]] = None,
        next_node: Optional[str] = None,
    ) -> "IVRBuilder":
        """Add a gather action node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.GATHER,
            parameters={
                "prompt_text": prompt_text,
                "num_digits": num_digits,
                "input_type": input_type.value,
            },
            timeout_seconds=timeout,
            conditions=conditions or {},
            next_node=next_node,
        )

    def menu(
        self,
        node_id: str,
        prompt: str,
        options: Dict[str, str],  # key -> next_node_id
        option_labels: Optional[Dict[str, str]] = None,
        invalid_node: Optional[str] = None,
        timeout_node: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 10.0,
    ) -> "IVRBuilder":
        """Add a menu node."""
        # Create menu definition
        menu_id = f"menu_{node_id}"
        menu = IVRMenu(
            menu_id=menu_id,
            prompt_text=prompt,
            options=option_labels or {k: f"Option {k}" for k in options},
            option_nodes=options,
            invalid_input_node=invalid_node,
            timeout_node=timeout_node,
            max_retries=max_retries,
            timeout_seconds=timeout,
        )
        self.flow.menus[menu_id] = menu

        return self._add_node(
            node_id=node_id,
            action=IVRAction.MENU,
            parameters={"menu_id": menu_id},
            timeout_seconds=timeout,
        )

    def record(
        self,
        node_id: str,
        prompt_text: Optional[str] = None,
        max_duration: float = 60.0,
        beep: bool = True,
        next_node: Optional[str] = None,
    ) -> "IVRBuilder":
        """Add a record action node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.RECORD,
            parameters={
                "prompt_text": prompt_text,
                "max_duration": max_duration,
                "beep": beep,
            },
            next_node=next_node,
        )

    def transfer(
        self,
        node_id: str,
        destination: str,
        transfer_type: str = "blind",
        next_node: Optional[str] = None,
    ) -> "IVRBuilder":
        """Add a transfer action node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.TRANSFER,
            parameters={
                "destination": destination,
                "transfer_type": transfer_type,
            },
            next_node=next_node,
        )

    def dial(
        self,
        node_id: str,
        number: str,
        caller_id: Optional[str] = None,
        timeout: float = 30.0,
        next_node: Optional[str] = None,
    ) -> "IVRBuilder":
        """Add a dial action node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.DIAL,
            parameters={
                "number": number,
                "caller_id": caller_id,
            },
            timeout_seconds=timeout,
            next_node=next_node,
        )

    def hangup(
        self,
        node_id: str,
    ) -> "IVRBuilder":
        """Add a hangup node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.HANGUP,
            parameters={},
        )

    def goto(
        self,
        node_id: str,
        target: str,
    ) -> "IVRBuilder":
        """Add a goto node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.GOTO,
            parameters={"target": target},
        )

    def pause(
        self,
        node_id: str,
        duration: float,
        next_node: Optional[str] = None,
    ) -> "IVRBuilder":
        """Add a pause node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.PAUSE,
            parameters={"duration": duration},
            next_node=next_node,
        )

    def voicemail(
        self,
        node_id: str,
        mailbox_id: str,
        greeting_text: Optional[str] = None,
        next_node: Optional[str] = None,
    ) -> "IVRBuilder":
        """Add a voicemail node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.VOICEMAIL,
            parameters={
                "mailbox_id": mailbox_id,
                "greeting_text": greeting_text,
            },
            next_node=next_node,
        )

    def condition(
        self,
        node_id: str,
        conditions: Dict[str, str],  # condition -> next_node
        default_node: Optional[str] = None,
    ) -> "IVRBuilder":
        """Add a conditional branching node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.CONDITION,
            parameters={},
            conditions=conditions,
            next_node=default_node,
        )

    def set_variable(
        self,
        node_id: str,
        name: str,
        value: Any,
        next_node: Optional[str] = None,
    ) -> "IVRBuilder":
        """Add a set variable node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.SET_VARIABLE,
            parameters={"name": name, "value": value},
            next_node=next_node,
        )

    def callback(
        self,
        node_id: str,
        callback_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        next_node: Optional[str] = None,
    ) -> "IVRBuilder":
        """Add a custom callback node."""
        return self._add_node(
            node_id=node_id,
            action=IVRAction.CALLBACK,
            parameters={
                "callback_name": callback_name,
                "callback_params": parameters or {},
            },
            next_node=next_node,
        )

    def link(
        self,
        from_node: str,
        to_node: str,
    ) -> "IVRBuilder":
        """Link two nodes together."""
        if from_node in self.flow.nodes:
            self.flow.nodes[from_node].next_node = to_node
        return self

    def set_start_node(
        self,
        node_id: str,
    ) -> "IVRBuilder":
        """Set the start node."""
        self.flow.start_node = node_id
        return self

    def set_variable_default(
        self,
        name: str,
        value: Any,
    ) -> "IVRBuilder":
        """Set a default variable for the flow."""
        self.flow.variables[name] = value
        return self

    def build(self) -> IVRFlow:
        """Build and return the IVR flow."""
        self.flow.updated_at = datetime.utcnow()
        return self.flow
