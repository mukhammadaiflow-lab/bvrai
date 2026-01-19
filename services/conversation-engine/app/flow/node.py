"""Flow node definitions."""

from typing import Optional, Dict, List, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import re


class NodeType(str, Enum):
    """Types of flow nodes."""
    START = "start"
    MESSAGE = "message"  # Send a message
    COLLECT = "collect"  # Collect user input
    CONDITION = "condition"  # Branch based on condition
    FUNCTION = "function"  # Execute function
    TRANSFER = "transfer"  # Transfer to human
    END = "end"
    WAIT = "wait"  # Wait for user input
    LOOP = "loop"  # Loop until condition
    PARALLEL = "parallel"  # Execute multiple branches


@dataclass
class CollectConfig:
    """Configuration for input collection."""
    slot_name: str
    slot_type: str = "string"
    prompt: Optional[str] = None
    validation_regex: Optional[str] = None
    validation_message: Optional[str] = None
    max_attempts: int = 3
    required: bool = True
    confirm: bool = False
    examples: List[str] = field(default_factory=list)


@dataclass
class ConditionBranch:
    """A branch in a condition node."""
    condition: str  # Expression to evaluate
    target_node: str  # Node to go to if true
    priority: int = 0  # Higher priority evaluated first


@dataclass
class FlowNode:
    """
    A node in the conversation flow.

    Represents a single step in the conversation,
    with transitions to other nodes.
    """

    id: str
    type: NodeType
    name: Optional[str] = None

    # For MESSAGE nodes
    message: Optional[str] = None
    message_ssml: Optional[str] = None  # SSML for TTS
    interruptible: bool = True

    # For COLLECT nodes
    collect: Optional[CollectConfig] = None

    # For CONDITION nodes
    branches: List[ConditionBranch] = field(default_factory=list)
    default_branch: Optional[str] = None

    # For FUNCTION nodes
    function_name: Optional[str] = None
    function_args: Dict[str, Any] = field(default_factory=dict)
    store_result: Optional[str] = None  # Variable to store result

    # For TRANSFER nodes
    transfer_target: Optional[str] = None  # Department or number
    transfer_message: Optional[str] = None

    # Transitions
    next_node: Optional[str] = None  # Default next node

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Hooks
    on_enter: Optional[str] = None  # Function to call on enter
    on_exit: Optional[str] = None  # Function to call on exit

    def get_message(self, context: Dict[str, Any]) -> str:
        """Get message with variable substitution."""
        if not self.message:
            return ""

        # Simple variable substitution: {variable_name}
        result = self.message

        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def evaluate_condition(
        self,
        condition: str,
        context: Dict[str, Any],
    ) -> bool:
        """
        Evaluate a condition expression.

        Supports:
        - Comparison: slot.name == "value"
        - Exists: slot.name exists
        - Contains: slot.name contains "value"
        - Regex: slot.name matches "pattern"
        - Numeric: slot.value > 10
        """
        # Parse condition
        condition = condition.strip()

        # Exists check
        if condition.endswith(" exists"):
            var_name = condition.replace(" exists", "").strip()
            return self._get_value(var_name, context) is not None

        # Not exists
        if condition.endswith(" not exists"):
            var_name = condition.replace(" not exists", "").strip()
            return self._get_value(var_name, context) is None

        # Contains
        if " contains " in condition:
            parts = condition.split(" contains ")
            var_name = parts[0].strip()
            search = parts[1].strip().strip('"\'')
            value = self._get_value(var_name, context)
            return search.lower() in str(value).lower() if value else False

        # Matches (regex)
        if " matches " in condition:
            parts = condition.split(" matches ")
            var_name = parts[0].strip()
            pattern = parts[1].strip().strip('"\'')
            value = self._get_value(var_name, context)
            if value:
                return bool(re.match(pattern, str(value)))
            return False

        # Comparisons
        for op in ["==", "!=", ">=", "<=", ">", "<"]:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    left = self._get_value(parts[0].strip(), context)
                    right_str = parts[1].strip().strip('"\'')

                    # Try numeric comparison
                    try:
                        left_num = float(left) if left else 0
                        right_num = float(right_str)
                        return self._compare(left_num, op, right_num)
                    except (ValueError, TypeError):
                        # String comparison
                        left_str = str(left) if left else ""
                        return self._compare(left_str, op, right_str)

        # Boolean-like values
        if condition.lower() in ("true", "yes", "1"):
            return True
        if condition.lower() in ("false", "no", "0"):
            return False

        # Variable as boolean
        value = self._get_value(condition, context)
        return bool(value)

    def _get_value(self, path: str, context: Dict[str, Any]) -> Any:
        """Get value from context using dot notation."""
        parts = path.split(".")
        current = context

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

        return current

    def _compare(self, left: Any, op: str, right: Any) -> bool:
        """Compare two values."""
        if op == "==":
            return left == right
        elif op == "!=":
            return left != right
        elif op == ">":
            return left > right
        elif op == "<":
            return left < right
        elif op == ">=":
            return left >= right
        elif op == "<=":
            return left <= right
        return False

    def get_next_node(self, context: Dict[str, Any]) -> Optional[str]:
        """Determine next node based on type and conditions."""
        if self.type == NodeType.CONDITION:
            # Evaluate branches in priority order
            sorted_branches = sorted(
                self.branches,
                key=lambda b: b.priority,
                reverse=True,
            )

            for branch in sorted_branches:
                if self.evaluate_condition(branch.condition, context):
                    return branch.target_node

            return self.default_branch

        return self.next_node

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "message": self.message,
            "next_node": self.next_node,
            "collect": {
                "slot_name": self.collect.slot_name,
                "slot_type": self.collect.slot_type,
                "required": self.collect.required,
            } if self.collect else None,
            "branches": [
                {"condition": b.condition, "target": b.target_node}
                for b in self.branches
            ] if self.branches else None,
            "function_name": self.function_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowNode":
        """Create from dictionary."""
        node = cls(
            id=data["id"],
            type=NodeType(data["type"]),
            name=data.get("name"),
            message=data.get("message"),
            next_node=data.get("next_node"),
            function_name=data.get("function_name"),
            metadata=data.get("metadata", {}),
        )

        # Parse collect config
        if data.get("collect"):
            node.collect = CollectConfig(**data["collect"])

        # Parse branches
        if data.get("branches"):
            node.branches = [
                ConditionBranch(
                    condition=b["condition"],
                    target_node=b["target"],
                    priority=b.get("priority", 0),
                )
                for b in data["branches"]
            ]
            node.default_branch = data.get("default_branch")

        return node
