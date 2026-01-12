"""Flow builder for creating conversation flows programmatically."""

from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
import json
import yaml

from app.flow.node import FlowNode, NodeType, CollectConfig, ConditionBranch


class FlowBuilder:
    """
    Builder for creating conversation flows.

    Provides a fluent API for constructing flows
    without directly manipulating node objects.

    Example:
        flow = (
            FlowBuilder("greeting_flow")
            .start("start")
            .message("greeting", "Hello! How can I help you today?")
            .collect("name", slot_name="customer_name", prompt="May I have your name?")
            .message("confirm", "Nice to meet you, {customer_name}!")
            .condition("intent_check")
                .when("intent == 'schedule'", "schedule_flow")
                .when("intent == 'cancel'", "cancel_flow")
                .otherwise("general_help")
            .end("goodbye", "Thank you for calling!")
            .build()
        )
    """

    def __init__(self, flow_id: str):
        self.flow_id = flow_id
        self._nodes: List[FlowNode] = []
        self._current_node: Optional[FlowNode] = None
        self._node_map: Dict[str, FlowNode] = {}

    def start(self, node_id: str = "start") -> "FlowBuilder":
        """Add start node."""
        node = FlowNode(
            id=node_id,
            type=NodeType.START,
            name="Start",
        )
        self._add_node(node)
        return self

    def message(
        self,
        node_id: str,
        message: str,
        ssml: Optional[str] = None,
        interruptible: bool = True,
    ) -> "FlowBuilder":
        """Add message node."""
        node = FlowNode(
            id=node_id,
            type=NodeType.MESSAGE,
            message=message,
            message_ssml=ssml,
            interruptible=interruptible,
        )
        self._add_node(node)
        return self

    def collect(
        self,
        node_id: str,
        slot_name: str,
        slot_type: str = "string",
        prompt: Optional[str] = None,
        validation_regex: Optional[str] = None,
        validation_message: Optional[str] = None,
        max_attempts: int = 3,
        required: bool = True,
        confirm: bool = False,
    ) -> "FlowBuilder":
        """Add collect node for gathering user input."""
        node = FlowNode(
            id=node_id,
            type=NodeType.COLLECT,
            collect=CollectConfig(
                slot_name=slot_name,
                slot_type=slot_type,
                prompt=prompt or f"Please provide your {slot_name}.",
                validation_regex=validation_regex,
                validation_message=validation_message,
                max_attempts=max_attempts,
                required=required,
                confirm=confirm,
            ),
        )
        self._add_node(node)
        return self

    def condition(self, node_id: str) -> "ConditionBuilder":
        """Add condition node (returns ConditionBuilder)."""
        node = FlowNode(
            id=node_id,
            type=NodeType.CONDITION,
            branches=[],
        )
        self._add_node(node)
        return ConditionBuilder(self, node)

    def function(
        self,
        node_id: str,
        function_name: str,
        args: Optional[Dict[str, Any]] = None,
        store_result: Optional[str] = None,
    ) -> "FlowBuilder":
        """Add function call node."""
        node = FlowNode(
            id=node_id,
            type=NodeType.FUNCTION,
            function_name=function_name,
            function_args=args or {},
            store_result=store_result,
        )
        self._add_node(node)
        return self

    def transfer(
        self,
        node_id: str,
        target: str,
        message: Optional[str] = None,
    ) -> "FlowBuilder":
        """Add transfer node."""
        node = FlowNode(
            id=node_id,
            type=NodeType.TRANSFER,
            transfer_target=target,
            transfer_message=message or f"Transferring you to {target}.",
        )
        self._add_node(node)
        return self

    def wait(
        self,
        node_id: str,
        message: Optional[str] = None,
    ) -> "FlowBuilder":
        """Add wait node (waits for user input)."""
        node = FlowNode(
            id=node_id,
            type=NodeType.WAIT,
            message=message,
        )
        self._add_node(node)
        return self

    def end(
        self,
        node_id: str = "end",
        message: Optional[str] = None,
    ) -> "FlowBuilder":
        """Add end node."""
        node = FlowNode(
            id=node_id,
            type=NodeType.END,
            message=message,
        )
        self._add_node(node)
        return self

    def goto(self, target_node_id: str) -> "FlowBuilder":
        """Set next node for current node."""
        if self._current_node:
            self._current_node.next_node = target_node_id
        return self

    def on_enter(self, function_name: str) -> "FlowBuilder":
        """Set on_enter hook for current node."""
        if self._current_node:
            self._current_node.on_enter = function_name
        return self

    def on_exit(self, function_name: str) -> "FlowBuilder":
        """Set on_exit hook for current node."""
        if self._current_node:
            self._current_node.on_exit = function_name
        return self

    def metadata(self, key: str, value: Any) -> "FlowBuilder":
        """Add metadata to current node."""
        if self._current_node:
            self._current_node.metadata[key] = value
        return self

    def _add_node(self, node: FlowNode) -> None:
        """Add node and set up linking."""
        # Link from previous node
        if self._current_node and self._current_node.type != NodeType.CONDITION:
            if not self._current_node.next_node:
                self._current_node.next_node = node.id

        self._nodes.append(node)
        self._node_map[node.id] = node
        self._current_node = node

    def build(self) -> List[FlowNode]:
        """Build and return the flow nodes."""
        return self._nodes

    def to_dict(self) -> Dict[str, Any]:
        """Export flow as dictionary."""
        return {
            "flow_id": self.flow_id,
            "nodes": [node.to_dict() for node in self._nodes],
        }

    def to_json(self) -> str:
        """Export flow as JSON."""
        return json.dumps(self.to_dict(), indent=2)

    def to_yaml(self) -> str:
        """Export flow as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowBuilder":
        """Create builder from dictionary."""
        builder = cls(data["flow_id"])
        builder._nodes = [FlowNode.from_dict(n) for n in data["nodes"]]
        builder._node_map = {n.id: n for n in builder._nodes}
        return builder

    @classmethod
    def from_json(cls, json_str: str) -> "FlowBuilder":
        """Create builder from JSON."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "FlowBuilder":
        """Create builder from YAML."""
        return cls.from_dict(yaml.safe_load(yaml_str))


class ConditionBuilder:
    """Builder for condition node branches."""

    def __init__(self, parent: FlowBuilder, node: FlowNode):
        self._parent = parent
        self._node = node
        self._priority = 0

    def when(
        self,
        condition: str,
        target_node: str,
        priority: Optional[int] = None,
    ) -> "ConditionBuilder":
        """Add a condition branch."""
        self._node.branches.append(
            ConditionBranch(
                condition=condition,
                target_node=target_node,
                priority=priority if priority is not None else self._priority,
            )
        )
        self._priority += 1
        return self

    def otherwise(self, target_node: str) -> FlowBuilder:
        """Set default branch and return to parent builder."""
        self._node.default_branch = target_node
        return self._parent

    def end_condition(self) -> FlowBuilder:
        """End condition without default (use with caution)."""
        return self._parent


# Pre-built flow templates
class FlowTemplates:
    """Common flow templates."""

    @staticmethod
    def greeting_flow(agent_name: str = "Assistant") -> FlowBuilder:
        """Create a basic greeting flow."""
        return (
            FlowBuilder("greeting")
            .start()
            .message("greet", f"Hello! I'm {agent_name}. How can I help you today?")
            .wait("listen")
            .end()
        )

    @staticmethod
    def appointment_booking_flow() -> FlowBuilder:
        """Create an appointment booking flow."""
        return (
            FlowBuilder("appointment_booking")
            .start()
            .message("intro", "I'd be happy to help you schedule an appointment.")
            .collect(
                "get_name",
                slot_name="customer_name",
                prompt="May I have your name, please?",
            )
            .collect(
                "get_phone",
                slot_name="customer_phone",
                prompt="And your phone number?",
                validation_regex=r"^\d{10}$",
                validation_message="Please provide a 10-digit phone number.",
            )
            .function(
                "check_slots",
                function_name="check_availability",
                args={"date": "today"},
                store_result="available_slots",
            )
            .message("show_slots", "We have the following times available: {available_slots}")
            .collect(
                "get_time",
                slot_name="preferred_time",
                prompt="Which time works best for you?",
            )
            .function(
                "book",
                function_name="book_appointment",
                store_result="booking_result",
            )
            .message("confirm", "Great! Your appointment is confirmed for {preferred_time}. {booking_result.confirmation_message}")
            .end("goodbye", "Thank you for booking with us. Goodbye!")
        )

    @staticmethod
    def support_flow() -> FlowBuilder:
        """Create a customer support flow."""
        return (
            FlowBuilder("support")
            .start()
            .message("intro", "I understand you need some help. Let me get some information.")
            .collect(
                "get_issue",
                slot_name="issue_description",
                prompt="Please describe the issue you're experiencing.",
            )
            .function(
                "classify",
                function_name="classify_issue",
                store_result="issue_category",
            )
            .condition("route_issue")
                .when("issue_category == 'billing'", "billing_help")
                .when("issue_category == 'technical'", "technical_help")
                .when("issue_category == 'urgent'", "urgent_transfer")
                .otherwise("general_help")
            .message("billing_help", "Let me look up your billing information.")
            .message("technical_help", "I'll help troubleshoot this technical issue.")
            .transfer("urgent_transfer", "support_manager", "This requires immediate attention. Connecting you with a manager.")
            .message("general_help", "Let me see how I can assist you with that.")
            .end()
        )

    @staticmethod
    def outbound_call_flow() -> FlowBuilder:
        """Create an outbound calling flow."""
        return (
            FlowBuilder("outbound_call")
            .start()
            .message("intro", "Hello, this is {agent_name} calling from {company_name}.")
            .wait("confirm_person", "Am I speaking with {customer_name}?")
            .condition("confirm_identity")
                .when("last_user_input contains 'yes'", "continue_call")
                .when("last_user_input contains 'no'", "wrong_person")
                .otherwise("clarify")
            .message("wrong_person", "I apologize for the inconvenience. Have a good day.")
            .end("end_wrong")
            .message("clarify", "I'm looking for {customer_name}. Is this the right number?")
            .goto("confirm_identity")
            .message("continue_call", "Great! I'm calling regarding {call_purpose}.")
            .wait("listen_response")
            .end("goodbye", "Thank you for your time. Have a great day!")
        )
