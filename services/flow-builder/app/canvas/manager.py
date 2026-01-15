"""
Canvas Manager.

Manages flow CRUD operations, versioning, and state.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from ..config import FlowStatus, NodeType, get_settings
from ..models import (
    Flow,
    FlowMetadata,
    FlowVariable,
    NodeInstance,
    Connection,
    ValidationResult,
)
from .validator import FlowValidator

logger = logging.getLogger(__name__)


class CanvasManager:
    """
    Manages flow lifecycle and operations.

    Features:
    - Flow CRUD operations
    - Version history
    - Auto-save
    - Template application
    """

    def __init__(self):
        """Initialize canvas manager."""
        self.settings = get_settings()
        self.validator = FlowValidator()

        # In-memory storage (replace with database in production)
        self._flows: Dict[str, Flow] = {}
        self._versions: Dict[str, List[Flow]] = {}

    async def create_flow(
        self,
        name: str,
        user_id: str,
        description: str = "",
        template_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Flow:
        """
        Create a new flow.

        Args:
            name: Flow name
            user_id: Creating user ID
            description: Optional description
            template_id: Optional template to apply
            tags: Optional tags

        Returns:
            Created Flow
        """
        flow_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Initialize with template or empty
        if template_id:
            flow = await self._apply_template(flow_id, template_id)
            flow.name = name
            flow.metadata.description = description
        else:
            flow = Flow(
                id=flow_id,
                name=name,
                status=FlowStatus.DRAFT,
                nodes=[],
                connections=[],
                variables=[],
                metadata=FlowMetadata(
                    created_at=now,
                    updated_at=now,
                    created_by=user_id,
                    updated_by=user_id,
                    version=1,
                    tags=tags or [],
                    description=description,
                ),
            )

        # Store
        self._flows[flow_id] = flow
        self._versions[flow_id] = [flow]

        logger.info(f"Created flow: {flow_id} ({name})")
        return flow

    async def get_flow(self, flow_id: str) -> Optional[Flow]:
        """Get a flow by ID."""
        return self._flows.get(flow_id)

    async def list_flows(
        self,
        user_id: Optional[str] = None,
        status: Optional[FlowStatus] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[Flow], int]:
        """
        List flows with filtering.

        Returns:
            Tuple of (flows, total_count)
        """
        flows = list(self._flows.values())

        # Filter by user
        if user_id:
            flows = [f for f in flows if f.metadata.created_by == user_id]

        # Filter by status
        if status:
            flows = [f for f in flows if f.status == status]

        # Filter by tags
        if tags:
            flows = [
                f for f in flows
                if any(t in f.metadata.tags for t in tags)
            ]

        # Search
        if search:
            search_lower = search.lower()
            flows = [
                f for f in flows
                if (
                    search_lower in f.name.lower()
                    or search_lower in f.metadata.description.lower()
                )
            ]

        # Sort by updated_at descending
        flows.sort(key=lambda f: f.metadata.updated_at, reverse=True)

        # Paginate
        total = len(flows)
        start = (page - 1) * page_size
        end = start + page_size
        flows = flows[start:end]

        return flows, total

    async def update_flow(
        self,
        flow_id: str,
        user_id: str,
        updates: Dict[str, Any],
        create_version: bool = True,
    ) -> Optional[Flow]:
        """
        Update a flow.

        Args:
            flow_id: Flow to update
            user_id: Updating user
            updates: Fields to update
            create_version: Whether to create a new version

        Returns:
            Updated Flow or None if not found
        """
        flow = self._flows.get(flow_id)
        if not flow:
            return None

        # Create version before update
        if create_version:
            await self._create_version(flow)

        now = datetime.utcnow()

        # Update fields
        if "name" in updates:
            flow.name = updates["name"]

        if "status" in updates:
            flow.status = FlowStatus(updates["status"])

        if "nodes" in updates:
            flow.nodes = [
                self._dict_to_node(n) for n in updates["nodes"]
            ]

        if "connections" in updates:
            flow.connections = [
                self._dict_to_connection(c) for c in updates["connections"]
            ]

        if "variables" in updates:
            flow.variables = [
                self._dict_to_variable(v) for v in updates["variables"]
            ]

        if "settings" in updates:
            flow.settings = updates["settings"]

        if "viewport" in updates:
            flow.viewport = updates["viewport"]

        if "tags" in updates:
            flow.metadata.tags = updates["tags"]

        if "description" in updates:
            flow.metadata.description = updates["description"]

        # Update metadata
        flow.metadata.updated_at = now
        flow.metadata.updated_by = user_id
        flow.metadata.version += 1

        logger.info(f"Updated flow: {flow_id} (v{flow.metadata.version})")
        return flow

    async def delete_flow(self, flow_id: str) -> bool:
        """Delete a flow."""
        if flow_id not in self._flows:
            return False

        del self._flows[flow_id]
        self._versions.pop(flow_id, None)

        logger.info(f"Deleted flow: {flow_id}")
        return True

    async def validate_flow(self, flow_id: str) -> ValidationResult:
        """Validate a flow."""
        flow = self._flows.get(flow_id)
        if not flow:
            return ValidationResult(
                valid=False,
                issues=[{"severity": "error", "message": "Flow not found"}],
            )

        return self.validator.validate(flow)

    async def duplicate_flow(
        self,
        flow_id: str,
        user_id: str,
        new_name: Optional[str] = None,
    ) -> Optional[Flow]:
        """
        Duplicate a flow.

        Args:
            flow_id: Flow to duplicate
            user_id: User creating the duplicate
            new_name: Optional new name

        Returns:
            New Flow or None if source not found
        """
        source = self._flows.get(flow_id)
        if not source:
            return None

        new_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Deep copy with new IDs
        new_flow = Flow(
            id=new_id,
            name=new_name or f"{source.name} (Copy)",
            status=FlowStatus.DRAFT,
            nodes=[
                NodeInstance(
                    id=str(uuid.uuid4()),
                    type=n.type,
                    position=n.position.copy(),
                    data=n.data.copy(),
                    label=n.label,
                    disabled=n.disabled,
                    notes=n.notes,
                )
                for n in source.nodes
            ],
            connections=[],  # Rebuild below
            variables=[
                FlowVariable(
                    name=v.name,
                    data_type=v.data_type,
                    scope=v.scope,
                    default_value=v.default_value,
                    description=v.description,
                )
                for v in source.variables
            ],
            metadata=FlowMetadata(
                created_at=now,
                updated_at=now,
                created_by=user_id,
                updated_by=user_id,
                version=1,
                tags=source.metadata.tags.copy(),
                description=source.metadata.description,
            ),
            settings=source.settings.copy(),
            viewport=source.viewport.copy(),
        )

        # Rebuild connections with new node IDs
        old_to_new = {
            old.id: new.id
            for old, new in zip(source.nodes, new_flow.nodes)
        }

        for conn in source.connections:
            new_flow.connections.append(
                Connection(
                    id=str(uuid.uuid4()),
                    source_node_id=old_to_new.get(conn.source_node_id, conn.source_node_id),
                    source_port_id=conn.source_port_id,
                    target_node_id=old_to_new.get(conn.target_node_id, conn.target_node_id),
                    target_port_id=conn.target_port_id,
                    type=conn.type,
                    label=conn.label,
                    condition=conn.condition,
                )
            )

        # Store
        self._flows[new_id] = new_flow
        self._versions[new_id] = [new_flow]

        logger.info(f"Duplicated flow: {flow_id} -> {new_id}")
        return new_flow

    async def get_versions(
        self,
        flow_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get version history for a flow."""
        versions = self._versions.get(flow_id, [])

        return [
            {
                "version": i + 1,
                "updated_at": v.metadata.updated_at.isoformat(),
                "updated_by": v.metadata.updated_by,
                "node_count": len(v.nodes),
            }
            for i, v in enumerate(reversed(versions[-limit:]))
        ]

    async def restore_version(
        self,
        flow_id: str,
        version: int,
        user_id: str,
    ) -> Optional[Flow]:
        """Restore a previous version of a flow."""
        versions = self._versions.get(flow_id, [])

        if version < 1 or version > len(versions):
            return None

        # Get the version to restore
        old_flow = versions[version - 1]
        current = self._flows.get(flow_id)

        if not current:
            return None

        # Create version of current state
        await self._create_version(current)

        # Restore old state
        now = datetime.utcnow()
        current.nodes = old_flow.nodes
        current.connections = old_flow.connections
        current.variables = old_flow.variables
        current.settings = old_flow.settings
        current.metadata.updated_at = now
        current.metadata.updated_by = user_id
        current.metadata.version += 1

        logger.info(f"Restored flow {flow_id} to version {version}")
        return current

    async def _create_version(self, flow: Flow) -> None:
        """Create a version snapshot."""
        if flow.id not in self._versions:
            self._versions[flow.id] = []

        versions = self._versions[flow.id]

        # Limit version history
        max_versions = self.settings.storage.max_versions
        if len(versions) >= max_versions:
            versions.pop(0)

        # Deep copy current state
        import copy
        versions.append(copy.deepcopy(flow))

    async def _apply_template(
        self,
        flow_id: str,
        template_id: str,
    ) -> Flow:
        """Apply a template to create a flow."""
        # Get template (placeholder - would load from templates)
        template = self._get_template(template_id)

        if not template:
            # Return empty flow if template not found
            now = datetime.utcnow()
            return Flow(
                id=flow_id,
                name="New Flow",
                status=FlowStatus.DRAFT,
                nodes=[],
                connections=[],
                variables=[],
                metadata=FlowMetadata(
                    created_at=now,
                    updated_at=now,
                    created_by="",
                    updated_by="",
                    version=1,
                    tags=[],
                    description="",
                ),
            )

        # Apply template with new IDs
        # (implementation would copy template structure)
        return template

    def _get_template(self, template_id: str) -> Optional[Flow]:
        """Get a template by ID (placeholder)."""
        # Would load from template storage
        return None

    def _dict_to_node(self, data: Dict[str, Any]) -> NodeInstance:
        """Convert dictionary to NodeInstance."""
        return NodeInstance(
            id=data.get("id", str(uuid.uuid4())),
            type=NodeType(data["type"]),
            position=data.get("position", {"x": 0, "y": 0}),
            data=data.get("data", {}),
            label=data.get("label"),
            disabled=data.get("disabled", False),
            notes=data.get("notes"),
        )

    def _dict_to_connection(self, data: Dict[str, Any]) -> Connection:
        """Convert dictionary to Connection."""
        from ..config import ConnectionType

        return Connection(
            id=data.get("id", str(uuid.uuid4())),
            source_node_id=data["source"],
            source_port_id=data.get("sourceHandle", "output"),
            target_node_id=data["target"],
            target_port_id=data.get("targetHandle", "input"),
            type=ConnectionType(data.get("type", "default")),
            label=data.get("label"),
            condition=data.get("condition"),
        )

    def _dict_to_variable(self, data: Dict[str, Any]) -> FlowVariable:
        """Convert dictionary to FlowVariable."""
        from ..config import DataType, VariableScope

        return FlowVariable(
            name=data["name"],
            data_type=DataType(data.get("data_type", "string")),
            scope=VariableScope(data.get("scope", "flow")),
            default_value=data.get("default_value"),
            description=data.get("description", ""),
        )
