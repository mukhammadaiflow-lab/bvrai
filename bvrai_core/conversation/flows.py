"""
Dialog Flow Definition and Management

This module provides structures for defining conversation flows
with nodes, transitions, conditions, and actions.
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from .context import ConversationContext
from .intents import Intent
from .slots import Slot


logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of dialog nodes."""

    MESSAGE = "message"          # Send a message
    QUESTION = "question"        # Ask a question (expect response)
    SLOT_FILL = "slot_fill"      # Fill a specific slot
    CONDITION = "condition"      # Branch based on condition
    ACTION = "action"            # Execute an action
    TRANSFER = "transfer"        # Transfer to human/queue
    SUBFLOW = "subflow"          # Enter another flow
    END = "end"                  # End the flow
    WAIT = "wait"                # Wait for external event


class ConditionOperator(str, Enum):
    """Operators for conditions."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_OR_EQUAL = "gte"
    LESS_OR_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"          # Regex match
    IN = "in"                    # Value in list
    NOT_IN = "not_in"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"


@dataclass
class FlowCondition:
    """Condition for branching."""

    # What to check
    variable: str  # Context variable or slot name
    operator: ConditionOperator = ConditionOperator.EQUALS
    value: Any = None

    # For complex conditions
    and_conditions: List["FlowCondition"] = field(default_factory=list)
    or_conditions: List["FlowCondition"] = field(default_factory=list)

    def evaluate(self, context: ConversationContext) -> bool:
        """Evaluate the condition against context."""
        # Get the value to check
        actual_value = self._get_value(context)

        # Check main condition
        result = self._check_operator(actual_value)

        # AND conditions
        if result and self.and_conditions:
            for cond in self.and_conditions:
                if not cond.evaluate(context):
                    return False

        # OR conditions
        if not result and self.or_conditions:
            for cond in self.or_conditions:
                if cond.evaluate(context):
                    return True

        return result

    def _get_value(self, context: ConversationContext) -> Any:
        """Get the value to check from context."""
        # Try slot first
        value = context.get_slot(self.variable)
        if value is not None:
            return value

        # Try context variable
        return context.get(self.variable)

    def _check_operator(self, actual: Any) -> bool:
        """Check the operator condition."""
        if self.operator == ConditionOperator.EXISTS:
            return actual is not None

        if self.operator == ConditionOperator.NOT_EXISTS:
            return actual is None

        if self.operator == ConditionOperator.IS_EMPTY:
            return not actual

        if self.operator == ConditionOperator.IS_NOT_EMPTY:
            return bool(actual)

        if actual is None:
            return False

        expected = self.value

        if self.operator == ConditionOperator.EQUALS:
            return actual == expected

        if self.operator == ConditionOperator.NOT_EQUALS:
            return actual != expected

        if self.operator == ConditionOperator.GREATER_THAN:
            return actual > expected

        if self.operator == ConditionOperator.LESS_THAN:
            return actual < expected

        if self.operator == ConditionOperator.GREATER_OR_EQUAL:
            return actual >= expected

        if self.operator == ConditionOperator.LESS_OR_EQUAL:
            return actual <= expected

        if self.operator == ConditionOperator.CONTAINS:
            return expected in str(actual)

        if self.operator == ConditionOperator.NOT_CONTAINS:
            return expected not in str(actual)

        if self.operator == ConditionOperator.STARTS_WITH:
            return str(actual).startswith(str(expected))

        if self.operator == ConditionOperator.ENDS_WITH:
            return str(actual).endswith(str(expected))

        if self.operator == ConditionOperator.MATCHES:
            import re
            return bool(re.match(str(expected), str(actual)))

        if self.operator == ConditionOperator.IN:
            return actual in (expected if isinstance(expected, list) else [expected])

        if self.operator == ConditionOperator.NOT_IN:
            return actual not in (expected if isinstance(expected, list) else [expected])

        return False


@dataclass
class FlowTransition:
    """Transition between nodes."""

    target_node_id: str

    # Conditions for this transition
    condition: Optional[FlowCondition] = None

    # Intent-based transition
    on_intent: Optional[str] = None

    # Priority (for multiple matching transitions)
    priority: int = 0

    # Is this the default transition?
    is_default: bool = False


@dataclass
class FlowAction:
    """Action to execute in a node."""

    # Action type
    action_type: str  # webhook, api_call, set_variable, etc.
    action_name: str = ""

    # Parameters
    params: Dict[str, Any] = field(default_factory=dict)

    # Handler function
    handler: Optional[Callable] = None

    # Result handling
    result_variable: Optional[str] = None  # Where to store result
    on_error: str = "continue"  # continue, retry, fail

    async def execute(self, context: ConversationContext) -> Any:
        """Execute the action."""
        if self.handler:
            result = self.handler(context, **self.params)
            if hasattr(result, '__await__'):
                result = await result
            return result

        # Built-in action types
        if self.action_type == "set_variable":
            context.set(
                self.params.get("name", ""),
                self.params.get("value"),
            )
            return True

        if self.action_type == "clear_slot":
            context.clear_slot(self.params.get("slot", ""))
            return True

        if self.action_type == "webhook":
            # Placeholder for webhook execution
            return {"status": "ok"}

        return None


@dataclass
class DialogNode:
    """A node in a dialog flow."""

    # Identification
    node_id: str
    name: str = ""
    node_type: NodeType = NodeType.MESSAGE

    # Content
    message: str = ""
    messages: List[str] = field(default_factory=list)  # For random selection
    ssml: Optional[str] = None

    # For QUESTION type
    expected_intents: List[str] = field(default_factory=list)

    # For SLOT_FILL type
    slot: Optional[Slot] = None

    # For CONDITION type
    condition: Optional[FlowCondition] = None

    # For ACTION type
    actions: List[FlowAction] = field(default_factory=list)

    # For TRANSFER type
    transfer_target: Optional[str] = None
    transfer_message: Optional[str] = None

    # For SUBFLOW type
    subflow_id: Optional[str] = None

    # Transitions
    transitions: List[FlowTransition] = field(default_factory=list)

    # Behavior
    wait_for_response: bool = True
    timeout_seconds: int = 30
    max_no_input: int = 2
    no_input_message: str = "I didn't hear anything. "

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_message(self) -> str:
        """Get the message to display."""
        if self.messages:
            import random
            return random.choice(self.messages)
        return self.message

    def get_next_node(
        self,
        context: ConversationContext,
        matched_intent: Optional[str] = None,
    ) -> Optional[str]:
        """Determine the next node based on context."""
        # Sort transitions by priority
        sorted_transitions = sorted(
            self.transitions,
            key=lambda t: t.priority,
            reverse=True,
        )

        default_transition = None

        for transition in sorted_transitions:
            if transition.is_default:
                default_transition = transition
                continue

            # Check intent match
            if transition.on_intent:
                if matched_intent == transition.on_intent:
                    return transition.target_node_id

            # Check condition
            elif transition.condition:
                if transition.condition.evaluate(context):
                    return transition.target_node_id

            # No condition = always matches
            elif not transition.on_intent:
                return transition.target_node_id

        # Use default if available
        if default_transition:
            return default_transition.target_node_id

        return None


@dataclass
class DialogFlow:
    """A complete dialog flow."""

    # Identification
    flow_id: str
    name: str = ""
    description: str = ""

    # Nodes
    nodes: Dict[str, DialogNode] = field(default_factory=dict)
    start_node_id: str = ""

    # Associated intents (what triggers this flow)
    trigger_intents: List[str] = field(default_factory=list)

    # Required slots for this flow
    required_slots: List[Slot] = field(default_factory=list)

    # Flow-level settings
    max_turns: int = 50
    timeout_seconds: int = 300
    allow_interruption: bool = True

    # Entry/exit actions
    on_enter: List[FlowAction] = field(default_factory=list)
    on_exit: List[FlowAction] = field(default_factory=list)

    # Error handling
    error_node_id: Optional[str] = None
    fallback_node_id: Optional[str] = None

    # Metadata
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: DialogNode) -> None:
        """Add a node to the flow."""
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[DialogNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_start_node(self) -> Optional[DialogNode]:
        """Get the start node."""
        if self.start_node_id:
            return self.nodes.get(self.start_node_id)
        # Return first node if no start specified
        if self.nodes:
            return list(self.nodes.values())[0]
        return None


class FlowBuilder:
    """
    Builder for creating dialog flows programmatically.
    """

    def __init__(self, flow_id: Optional[str] = None, name: str = ""):
        self._flow = DialogFlow(
            flow_id=flow_id or str(uuid.uuid4()),
            name=name,
        )
        self._current_node: Optional[DialogNode] = None
        self._node_counter = 0

    def set_name(self, name: str) -> "FlowBuilder":
        """Set flow name."""
        self._flow.name = name
        return self

    def set_description(self, description: str) -> "FlowBuilder":
        """Set flow description."""
        self._flow.description = description
        return self

    def trigger_on(self, *intents: str) -> "FlowBuilder":
        """Set trigger intents."""
        self._flow.trigger_intents.extend(intents)
        return self

    def require_slot(self, slot: Slot) -> "FlowBuilder":
        """Add a required slot."""
        self._flow.required_slots.append(slot)
        return self

    def on_enter(self, action: FlowAction) -> "FlowBuilder":
        """Add entry action."""
        self._flow.on_enter.append(action)
        return self

    def on_exit(self, action: FlowAction) -> "FlowBuilder":
        """Add exit action."""
        self._flow.on_exit.append(action)
        return self

    def _get_node_id(self, name: Optional[str] = None) -> str:
        """Generate a node ID."""
        self._node_counter += 1
        if name:
            return f"{name}_{self._node_counter}"
        return f"node_{self._node_counter}"

    def add_message(
        self,
        message: str,
        node_id: Optional[str] = None,
        **kwargs,
    ) -> "FlowBuilder":
        """Add a message node."""
        node = DialogNode(
            node_id=node_id or self._get_node_id("message"),
            node_type=NodeType.MESSAGE,
            message=message,
            wait_for_response=False,
            **kwargs,
        )
        self._add_node(node)
        return self

    def add_question(
        self,
        message: str,
        expected_intents: Optional[List[str]] = None,
        node_id: Optional[str] = None,
        **kwargs,
    ) -> "FlowBuilder":
        """Add a question node."""
        node = DialogNode(
            node_id=node_id or self._get_node_id("question"),
            node_type=NodeType.QUESTION,
            message=message,
            expected_intents=expected_intents or [],
            wait_for_response=True,
            **kwargs,
        )
        self._add_node(node)
        return self

    def add_slot_fill(
        self,
        slot: Slot,
        node_id: Optional[str] = None,
        **kwargs,
    ) -> "FlowBuilder":
        """Add a slot filling node."""
        node = DialogNode(
            node_id=node_id or self._get_node_id(f"slot_{slot.name}"),
            node_type=NodeType.SLOT_FILL,
            slot=slot,
            message=slot.prompt,
            wait_for_response=True,
            **kwargs,
        )
        self._add_node(node)
        return self

    def add_condition(
        self,
        condition: FlowCondition,
        node_id: Optional[str] = None,
        **kwargs,
    ) -> "FlowBuilder":
        """Add a condition node."""
        node = DialogNode(
            node_id=node_id or self._get_node_id("condition"),
            node_type=NodeType.CONDITION,
            condition=condition,
            wait_for_response=False,
            **kwargs,
        )
        self._add_node(node)
        return self

    def add_action(
        self,
        action: FlowAction,
        node_id: Optional[str] = None,
        **kwargs,
    ) -> "FlowBuilder":
        """Add an action node."""
        node = DialogNode(
            node_id=node_id or self._get_node_id("action"),
            node_type=NodeType.ACTION,
            actions=[action],
            wait_for_response=False,
            **kwargs,
        )
        self._add_node(node)
        return self

    def add_transfer(
        self,
        target: str,
        message: Optional[str] = None,
        node_id: Optional[str] = None,
        **kwargs,
    ) -> "FlowBuilder":
        """Add a transfer node."""
        node = DialogNode(
            node_id=node_id or self._get_node_id("transfer"),
            node_type=NodeType.TRANSFER,
            transfer_target=target,
            transfer_message=message,
            message=message or "Transferring you now...",
            wait_for_response=False,
            **kwargs,
        )
        self._add_node(node)
        return self

    def add_end(
        self,
        message: Optional[str] = None,
        node_id: Optional[str] = None,
        **kwargs,
    ) -> "FlowBuilder":
        """Add an end node."""
        node = DialogNode(
            node_id=node_id or self._get_node_id("end"),
            node_type=NodeType.END,
            message=message or "",
            wait_for_response=False,
            **kwargs,
        )
        self._add_node(node)
        return self

    def _add_node(self, node: DialogNode) -> None:
        """Add node to flow and link from previous."""
        # Set as start node if first
        if not self._flow.start_node_id:
            self._flow.start_node_id = node.node_id

        # Add transition from previous node
        if self._current_node:
            self._current_node.transitions.append(
                FlowTransition(target_node_id=node.node_id, is_default=True)
            )

        self._flow.add_node(node)
        self._current_node = node

    def transition_to(
        self,
        target_node_id: str,
        condition: Optional[FlowCondition] = None,
        on_intent: Optional[str] = None,
    ) -> "FlowBuilder":
        """Add a transition from current node."""
        if self._current_node:
            self._current_node.transitions.append(
                FlowTransition(
                    target_node_id=target_node_id,
                    condition=condition,
                    on_intent=on_intent,
                )
            )
        return self

    def go_to(self, node_id: str) -> "FlowBuilder":
        """Set current node for adding transitions."""
        node = self._flow.get_node(node_id)
        if node:
            self._current_node = node
        return self

    def build(self) -> DialogFlow:
        """Build and return the flow."""
        return self._flow


__all__ = [
    "NodeType",
    "ConditionOperator",
    "FlowCondition",
    "FlowTransition",
    "FlowAction",
    "DialogNode",
    "DialogFlow",
    "FlowBuilder",
]
