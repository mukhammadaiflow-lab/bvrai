"""
Flow Validator.

Validates flow structure, connections, and node configurations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from ..config import NodeType, get_settings
from ..models import (
    Flow,
    NodeInstance,
    Connection,
    ValidationResult,
    ValidationIssue,
)
from ..nodes import get_node_registry

logger = logging.getLogger(__name__)


class FlowValidator:
    """
    Validates flow structure and configuration.

    Checks:
    - Structural integrity (connections, orphans)
    - Node configuration (required properties)
    - Logic flow (unreachable nodes, infinite loops)
    - Resource limits
    """

    def __init__(self):
        """Initialize validator."""
        self.settings = get_settings()
        self.registry = get_node_registry()

    def validate(self, flow: Flow) -> ValidationResult:
        """
        Validate a complete flow.

        Args:
            flow: Flow to validate

        Returns:
            ValidationResult with issues found
        """
        issues: List[ValidationIssue] = []

        # Structural validation
        issues.extend(self._validate_structure(flow))

        # Node validation
        issues.extend(self._validate_nodes(flow))

        # Connection validation
        issues.extend(self._validate_connections(flow))

        # Logic validation
        issues.extend(self._validate_logic(flow))

        # Resource limits
        issues.extend(self._validate_limits(flow))

        # Determine if valid (no errors)
        valid = all(i.severity != "error" for i in issues)

        return ValidationResult(
            valid=valid,
            issues=issues,
            checked_at=datetime.utcnow(),
        )

    def _validate_structure(self, flow: Flow) -> List[ValidationIssue]:
        """Validate basic flow structure."""
        issues = []

        # Must have at least one trigger node
        trigger_nodes = [
            n for n in flow.nodes
            if n.type in [
                NodeType.INCOMING_CALL,
                NodeType.OUTBOUND_CALL,
                NodeType.WEBHOOK,
                NodeType.SCHEDULE,
                NodeType.EVENT,
            ]
        ]

        if not trigger_nodes:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message="Flow must have at least one trigger node",
                )
            )

        # Check for duplicate node IDs
        node_ids = [n.id for n in flow.nodes]
        seen_ids: Set[str] = set()
        for node_id in node_ids:
            if node_id in seen_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Duplicate node ID: {node_id}",
                        node_id=node_id,
                    )
                )
            seen_ids.add(node_id)

        # Check for duplicate connection IDs
        connection_ids = [c.id for c in flow.connections]
        seen_conn_ids: Set[str] = set()
        for conn_id in connection_ids:
            if conn_id in seen_conn_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Duplicate connection ID: {conn_id}",
                        connection_id=conn_id,
                    )
                )
            seen_conn_ids.add(conn_id)

        return issues

    def _validate_nodes(self, flow: Flow) -> List[ValidationIssue]:
        """Validate individual nodes."""
        issues = []

        for node in flow.nodes:
            # Get node definition
            node_def = self.registry.get(node.type)

            if not node_def:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Unknown node type: {node.type.value}",
                        node_id=node.id,
                    )
                )
                continue

            # Check required properties
            for prop in node_def.properties:
                if prop.required:
                    value = node.data.get(prop.name)
                    if value is None or value == "":
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                message=f"Required property '{prop.name}' is missing",
                                node_id=node.id,
                                property_name=prop.name,
                            )
                        )

            # Check for disabled nodes with active connections
            if node.disabled:
                outgoing = [
                    c for c in flow.connections if c.source_node_id == node.id
                ]
                if outgoing:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            message="Disabled node has outgoing connections",
                            node_id=node.id,
                        )
                    )

        return issues

    def _validate_connections(self, flow: Flow) -> List[ValidationIssue]:
        """Validate connections between nodes."""
        issues = []
        node_ids = {n.id for n in flow.nodes}

        for conn in flow.connections:
            # Check source node exists
            if conn.source_node_id not in node_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Connection source node not found: {conn.source_node_id}",
                        connection_id=conn.id,
                    )
                )

            # Check target node exists
            if conn.target_node_id not in node_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Connection target node not found: {conn.target_node_id}",
                        connection_id=conn.id,
                    )
                )

            # Check for self-connections
            if conn.source_node_id == conn.target_node_id:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message="Node has a connection to itself",
                        node_id=conn.source_node_id,
                        connection_id=conn.id,
                    )
                )

            # Validate port compatibility (if definitions available)
            source_node = next(
                (n for n in flow.nodes if n.id == conn.source_node_id), None
            )
            target_node = next(
                (n for n in flow.nodes if n.id == conn.target_node_id), None
            )

            if source_node and target_node:
                source_def = self.registry.get(source_node.type)
                target_def = self.registry.get(target_node.type)

                if source_def and target_def:
                    # Check source port exists
                    source_port = next(
                        (p for p in source_def.outputs if p.id == conn.source_port_id),
                        None,
                    )
                    if not source_port:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                message=f"Invalid source port: {conn.source_port_id}",
                                connection_id=conn.id,
                            )
                        )

                    # Check target port exists
                    target_port = next(
                        (p for p in target_def.inputs if p.id == conn.target_port_id),
                        None,
                    )
                    if not target_port:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                message=f"Invalid target port: {conn.target_port_id}",
                                connection_id=conn.id,
                            )
                        )

        return issues

    def _validate_logic(self, flow: Flow) -> List[ValidationIssue]:
        """Validate flow logic and reachability."""
        issues = []

        # Build adjacency graph
        graph: Dict[str, List[str]] = {n.id: [] for n in flow.nodes}
        for conn in flow.connections:
            if conn.source_node_id in graph:
                graph[conn.source_node_id].append(conn.target_node_id)

        # Find trigger nodes
        trigger_ids = {
            n.id for n in flow.nodes
            if n.type in [
                NodeType.INCOMING_CALL,
                NodeType.OUTBOUND_CALL,
                NodeType.WEBHOOK,
                NodeType.SCHEDULE,
                NodeType.EVENT,
            ]
        }

        # BFS to find reachable nodes from triggers
        reachable: Set[str] = set()
        queue = list(trigger_ids)

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            queue.extend(graph.get(current, []))

        # Find unreachable nodes
        all_node_ids = {n.id for n in flow.nodes}
        unreachable = all_node_ids - reachable

        for node_id in unreachable:
            # Skip comment nodes (they're decorative)
            node = next(n for n in flow.nodes if n.id == node_id)
            if node.type == NodeType.COMMENT:
                continue

            issues.append(
                ValidationIssue(
                    severity="warning",
                    message="Node is not reachable from any trigger",
                    node_id=node_id,
                )
            )

        # Check for potential infinite loops
        issues.extend(self._detect_loops(flow, graph))

        return issues

    def _detect_loops(
        self,
        flow: Flow,
        graph: Dict[str, List[str]],
    ) -> List[ValidationIssue]:
        """Detect potential infinite loops."""
        issues = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node_id: str, path: List[str]) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in graph.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor, path + [node_id]):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    # Check if there's a loop node (intentional loop)
                    node = next(n for n in flow.nodes if n.id == neighbor)
                    if node.type != NodeType.LOOP:
                        issues.append(
                            ValidationIssue(
                                severity="warning",
                                message="Potential infinite loop detected (consider using Loop node)",
                                node_id=neighbor,
                            )
                        )
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in graph:
            if node_id not in visited:
                dfs(node_id, [])

        return issues

    def _validate_limits(self, flow: Flow) -> List[ValidationIssue]:
        """Validate resource limits."""
        issues = []
        canvas_config = self.settings.canvas

        # Check node count
        if len(flow.nodes) > canvas_config.max_nodes_per_flow:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Flow exceeds maximum nodes ({len(flow.nodes)} > {canvas_config.max_nodes_per_flow})",
                )
            )

        # Check connections per node
        for node in flow.nodes:
            incoming = len([
                c for c in flow.connections if c.target_node_id == node.id
            ])
            outgoing = len([
                c for c in flow.connections if c.source_node_id == node.id
            ])
            total = incoming + outgoing

            if total > canvas_config.max_connections_per_node:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=f"Node has too many connections ({total} > {canvas_config.max_connections_per_node})",
                        node_id=node.id,
                    )
                )

        return issues

    def quick_validate(self, flow: Flow) -> bool:
        """
        Quick validation for basic errors.

        Returns True if flow has no critical errors.
        """
        # Check for triggers
        has_trigger = any(
            n.type in [
                NodeType.INCOMING_CALL,
                NodeType.OUTBOUND_CALL,
                NodeType.WEBHOOK,
                NodeType.SCHEDULE,
                NodeType.EVENT,
            ]
            for n in flow.nodes
        )

        if not has_trigger:
            return False

        # Check connection validity
        node_ids = {n.id for n in flow.nodes}
        for conn in flow.connections:
            if (
                conn.source_node_id not in node_ids
                or conn.target_node_id not in node_ids
            ):
                return False

        return True
