"""
Node Registry.

Manages registration and lookup of node types.
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional

from ..config import NodeType, NodeCategory
from ..models import NodeDefinition
from .definitions import ALL_NODES

logger = logging.getLogger(__name__)


class NodeRegistry:
    """
    Registry for node type definitions.

    Provides lookup and filtering of available node types.
    """

    def __init__(self):
        """Initialize registry with all node definitions."""
        self._nodes: Dict[NodeType, NodeDefinition] = {}
        self._by_category: Dict[NodeCategory, List[NodeDefinition]] = {}

        # Register all built-in nodes
        for node_def in ALL_NODES:
            self.register(node_def)

        logger.info(f"Registered {len(self._nodes)} node types")

    def register(self, node_def: NodeDefinition) -> None:
        """Register a node definition."""
        if node_def.type in self._nodes:
            logger.warning(f"Overwriting existing node type: {node_def.type}")

        self._nodes[node_def.type] = node_def

        # Index by category
        if node_def.category not in self._by_category:
            self._by_category[node_def.category] = []
        self._by_category[node_def.category].append(node_def)

    def get(self, node_type: NodeType) -> Optional[NodeDefinition]:
        """Get node definition by type."""
        return self._nodes.get(node_type)

    def get_by_name(self, type_name: str) -> Optional[NodeDefinition]:
        """Get node definition by type name string."""
        try:
            node_type = NodeType(type_name)
            return self.get(node_type)
        except ValueError:
            return None

    def list_all(self) -> List[NodeDefinition]:
        """List all registered node definitions."""
        return list(self._nodes.values())

    def list_by_category(self, category: NodeCategory) -> List[NodeDefinition]:
        """List nodes in a specific category."""
        return self._by_category.get(category, [])

    def get_categories(self) -> List[NodeCategory]:
        """Get all categories with registered nodes."""
        return list(self._by_category.keys())

    def search(self, query: str) -> List[NodeDefinition]:
        """Search nodes by name or description."""
        query = query.lower()
        results = []

        for node_def in self._nodes.values():
            if (
                query in node_def.name.lower()
                or query in node_def.description.lower()
                or query in node_def.type.value.lower()
            ):
                results.append(node_def)

        return results

    def to_catalog(self) -> Dict[str, List[Dict]]:
        """
        Export registry as a catalog organized by category.

        Returns:
            Dict mapping category names to lists of node definitions
        """
        catalog = {}

        for category in NodeCategory:
            nodes = self.list_by_category(category)
            if nodes:
                catalog[category.value] = [n.to_dict() for n in nodes]

        return catalog

    def validate_node_type(self, type_name: str) -> bool:
        """Check if a node type is valid."""
        try:
            NodeType(type_name)
            return True
        except ValueError:
            return False


# Singleton instance
_registry: Optional[NodeRegistry] = None


@lru_cache
def get_node_registry() -> NodeRegistry:
    """Get the singleton node registry."""
    global _registry
    if _registry is None:
        _registry = NodeRegistry()
    return _registry
