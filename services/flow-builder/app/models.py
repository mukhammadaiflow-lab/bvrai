"""
Data Models for Flow Builder Service.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, Field

from .config import (
    NodeType,
    NodeCategory,
    ConnectionType,
    FlowStatus,
    ExecutionMode,
    VariableScope,
    DataType,
)


# =============================================================================
# Node Models
# =============================================================================


@dataclass
class PortDefinition:
    """Definition of a node input/output port."""

    id: str
    name: str
    data_type: DataType
    required: bool = False
    multiple: bool = False  # Can accept multiple connections
    default_value: Any = None
    description: str = ""


@dataclass
class NodeProperty:
    """Definition of a configurable node property."""

    name: str
    data_type: DataType
    required: bool = False
    default_value: Any = None
    description: str = ""
    options: Optional[List[Dict[str, Any]]] = None  # For select/enum
    validation: Optional[Dict[str, Any]] = None


@dataclass
class NodeDefinition:
    """Definition of a node type."""

    type: NodeType
    category: NodeCategory
    name: str
    description: str
    icon: str

    # Ports
    inputs: List[PortDefinition] = field(default_factory=list)
    outputs: List[PortDefinition] = field(default_factory=list)

    # Properties
    properties: List[NodeProperty] = field(default_factory=list)

    # Behavior
    is_async: bool = False
    supports_timeout: bool = True
    supports_retry: bool = True
    max_instances: int = -1  # -1 = unlimited

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "category": self.category.value,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "inputs": [
                {
                    "id": i.id,
                    "name": i.name,
                    "data_type": i.data_type.value,
                    "required": i.required,
                    "multiple": i.multiple,
                }
                for i in self.inputs
            ],
            "outputs": [
                {
                    "id": o.id,
                    "name": o.name,
                    "data_type": o.data_type.value,
                }
                for o in self.outputs
            ],
            "properties": [
                {
                    "name": p.name,
                    "data_type": p.data_type.value,
                    "required": p.required,
                    "default_value": p.default_value,
                    "description": p.description,
                    "options": p.options,
                }
                for p in self.properties
            ],
        }


@dataclass
class NodeInstance:
    """Instance of a node in a flow."""

    id: str
    type: NodeType
    position: Dict[str, float]  # {x, y}
    data: Dict[str, Any]  # Property values
    label: Optional[str] = None
    disabled: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "position": self.position,
            "data": self.data,
            "label": self.label,
            "disabled": self.disabled,
            "notes": self.notes,
        }


# =============================================================================
# Connection Models
# =============================================================================


@dataclass
class Connection:
    """Connection between two nodes."""

    id: str
    source_node_id: str
    source_port_id: str
    target_node_id: str
    target_port_id: str
    type: ConnectionType = ConnectionType.DEFAULT
    label: Optional[str] = None
    condition: Optional[str] = None  # For conditional edges

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source_node_id,
            "sourceHandle": self.source_port_id,
            "target": self.target_node_id,
            "targetHandle": self.target_port_id,
            "type": self.type.value,
            "label": self.label,
            "condition": self.condition,
        }


# =============================================================================
# Flow Models
# =============================================================================


@dataclass
class FlowVariable:
    """Variable definition for a flow."""

    name: str
    data_type: DataType
    scope: VariableScope
    default_value: Any = None
    description: str = ""


@dataclass
class FlowMetadata:
    """Metadata for a flow."""

    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    version: int = 1
    tags: List[str] = field(default_factory=list)
    description: str = ""
    thumbnail: Optional[str] = None


@dataclass
class Flow:
    """Complete flow definition."""

    id: str
    name: str
    status: FlowStatus
    nodes: List[NodeInstance]
    connections: List[Connection]
    variables: List[FlowVariable]
    metadata: FlowMetadata
    settings: Dict[str, Any] = field(default_factory=dict)

    # Canvas state
    viewport: Dict[str, float] = field(
        default_factory=lambda: {"x": 0, "y": 0, "zoom": 1}
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [c.to_dict() for c in self.connections],
            "variables": [
                {
                    "name": v.name,
                    "data_type": v.data_type.value,
                    "scope": v.scope.value,
                    "default_value": v.default_value,
                }
                for v in self.variables
            ],
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "created_by": self.metadata.created_by,
                "version": self.metadata.version,
                "tags": self.metadata.tags,
                "description": self.metadata.description,
            },
            "settings": self.settings,
            "viewport": self.viewport,
        }


# =============================================================================
# Execution Models
# =============================================================================


@dataclass
class ExecutionContext:
    """Context for flow execution."""

    execution_id: str
    flow_id: str
    mode: ExecutionMode
    started_at: datetime

    # Variables
    global_vars: Dict[str, Any] = field(default_factory=dict)
    flow_vars: Dict[str, Any] = field(default_factory=dict)
    session_vars: Dict[str, Any] = field(default_factory=dict)

    # Call context (if voice call)
    call_sid: Optional[str] = None
    caller_number: Optional[str] = None
    called_number: Optional[str] = None

    # Tracking
    current_node_id: Optional[str] = None
    visited_nodes: List[str] = field(default_factory=list)
    execution_path: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class NodeExecutionResult:
    """Result from executing a single node."""

    node_id: str
    success: bool
    output_data: Dict[str, Any] = field(default_factory=dict)
    next_port_id: Optional[str] = None  # Which output port to follow
    error: Optional[str] = None
    duration_ms: float = 0.0
    logs: List[str] = field(default_factory=list)


@dataclass
class FlowExecutionResult:
    """Result from executing a complete flow."""

    execution_id: str
    flow_id: str
    success: bool
    started_at: datetime
    completed_at: datetime

    # Results
    final_output: Dict[str, Any] = field(default_factory=dict)
    node_results: List[NodeExecutionResult] = field(default_factory=list)

    # Stats
    nodes_executed: int = 0
    total_duration_ms: float = 0.0

    # Errors
    error: Optional[str] = None
    error_node_id: Optional[str] = None


# =============================================================================
# Validation Models
# =============================================================================


@dataclass
class ValidationIssue:
    """A validation issue found in a flow."""

    severity: str  # error, warning, info
    message: str
    node_id: Optional[str] = None
    connection_id: Optional[str] = None
    property_name: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of flow validation."""

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


# =============================================================================
# API Request/Response Models
# =============================================================================


class CreateFlowRequest(BaseModel):
    """Request to create a new flow."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    template_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class UpdateFlowRequest(BaseModel):
    """Request to update a flow."""

    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[FlowStatus] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    connections: Optional[List[Dict[str, Any]]] = None
    variables: Optional[List[Dict[str, Any]]] = None
    settings: Optional[Dict[str, Any]] = None
    viewport: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None


class FlowResponse(BaseModel):
    """Response with flow data."""

    id: str
    name: str
    status: str
    nodes: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    variables: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    settings: Dict[str, Any]
    viewport: Dict[str, float]


class FlowListResponse(BaseModel):
    """Response with list of flows."""

    flows: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


class ExecuteFlowRequest(BaseModel):
    """Request to execute a flow."""

    mode: ExecutionMode = ExecutionMode.DRY_RUN
    input_variables: Dict[str, Any] = Field(default_factory=dict)
    mock_call: Optional[Dict[str, Any]] = None


class ExecuteFlowResponse(BaseModel):
    """Response from flow execution."""

    execution_id: str
    success: bool
    output: Dict[str, Any]
    node_results: List[Dict[str, Any]]
    duration_ms: float
    error: Optional[str] = None


class ValidateFlowResponse(BaseModel):
    """Response from flow validation."""

    valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]


class NodeListResponse(BaseModel):
    """Response with available node types."""

    nodes: List[Dict[str, Any]]
    categories: List[str]


class TemplateResponse(BaseModel):
    """Response with flow template."""

    id: str
    name: str
    description: str
    category: str
    preview_image: Optional[str]
    flow_data: Dict[str, Any]


# =============================================================================
# WebSocket Models
# =============================================================================


class CanvasAction(BaseModel):
    """Canvas action for real-time collaboration."""

    action: str  # add_node, move_node, delete_node, add_connection, etc.
    payload: Dict[str, Any]
    user_id: str
    timestamp: float


class CanvasUpdate(BaseModel):
    """Canvas update broadcast to collaborators."""

    flow_id: str
    action: str
    payload: Dict[str, Any]
    user_id: str
    timestamp: float


# =============================================================================
# Export
# =============================================================================


__all__ = [
    # Node
    "PortDefinition",
    "NodeProperty",
    "NodeDefinition",
    "NodeInstance",
    # Connection
    "Connection",
    # Flow
    "FlowVariable",
    "FlowMetadata",
    "Flow",
    # Execution
    "ExecutionContext",
    "NodeExecutionResult",
    "FlowExecutionResult",
    # Validation
    "ValidationIssue",
    "ValidationResult",
    # API
    "CreateFlowRequest",
    "UpdateFlowRequest",
    "FlowResponse",
    "FlowListResponse",
    "ExecuteFlowRequest",
    "ExecuteFlowResponse",
    "ValidateFlowResponse",
    "NodeListResponse",
    "TemplateResponse",
    # WebSocket
    "CanvasAction",
    "CanvasUpdate",
]
