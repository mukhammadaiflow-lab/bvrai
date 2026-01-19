"""
Workflows API Routes

Handles:
- Workflow CRUD operations
- Workflow execution
- Node management
- Templates
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db_session,
    get_current_user,
    get_current_tenant,
    UserContext,
    TenantContext,
    require_permissions,
    require_feature,
    get_pagination,
    PaginationParams,
)

router = APIRouter(prefix="/workflows")


# ============================================================================
# Schemas
# ============================================================================

class WorkflowStatus(str, Enum):
    """Workflow status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    """Execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(str, Enum):
    """Node types."""
    START = "start"
    END = "end"
    SPEAK = "speak"
    LISTEN = "listen"
    LLM_PROMPT = "llm_prompt"
    CONDITION = "condition"
    SWITCH = "switch"
    HTTP_REQUEST = "http_request"
    SET_VARIABLE = "set_variable"
    TRANSFER = "transfer"
    HANGUP = "hangup"
    WAIT = "wait"
    RECORD = "record"
    PLAY_AUDIO = "play_audio"
    DTMF = "dtmf"
    WEBHOOK = "webhook"
    FUNCTION = "function"


class NodeConfig(BaseModel):
    """Node configuration."""
    id: str
    type: NodeType
    name: str
    config: Dict[str, Any] = {}
    position: Dict[str, float] = {"x": 0, "y": 0}
    next_nodes: List[str] = []
    conditions: List[Dict[str, Any]] = []


class WorkflowCreate(BaseModel):
    """Create workflow request."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    agent_id: Optional[UUID] = None
    trigger: str = "inbound_call"
    nodes: List[NodeConfig] = []
    variables: Dict[str, Any] = {}
    settings: Dict[str, Any] = {}


class WorkflowUpdate(BaseModel):
    """Update workflow request."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    agent_id: Optional[UUID] = None
    trigger: Optional[str] = None
    nodes: Optional[List[NodeConfig]] = None
    variables: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None
    status: Optional[WorkflowStatus] = None


class WorkflowResponse(BaseModel):
    """Workflow response."""
    id: str
    name: str
    description: Optional[str]
    agent_id: Optional[str]
    trigger: str
    status: WorkflowStatus
    nodes: List[NodeConfig]
    variables: Dict[str, Any]
    settings: Dict[str, Any]
    version: int
    execution_count: int
    last_executed_at: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class WorkflowListResponse(BaseModel):
    """List response with pagination."""
    workflows: List[WorkflowResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ExecutionResponse(BaseModel):
    """Execution response."""
    id: str
    workflow_id: str
    call_id: Optional[str]
    status: ExecutionStatus
    current_node: Optional[str]
    variables: Dict[str, Any]
    execution_path: List[str]
    error: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: Optional[int]


class ExecutionListResponse(BaseModel):
    """Execution list response."""
    executions: List[ExecutionResponse]
    total: int
    page: int
    page_size: int


class ExecuteWorkflowRequest(BaseModel):
    """Execute workflow request."""
    call_id: Optional[str] = None
    variables: Dict[str, Any] = {}
    context: Dict[str, Any] = {}


class WorkflowTemplate(BaseModel):
    """Workflow template."""
    id: str
    name: str
    description: str
    category: str
    nodes: List[NodeConfig]
    variables: Dict[str, Any]


class WorkflowStats(BaseModel):
    """Workflow statistics."""
    total_workflows: int
    active_workflows: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_duration_ms: float


# ============================================================================
# CRUD Operations
# ============================================================================

@router.post("", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    data: WorkflowCreate,
    user: UserContext = Depends(require_permissions("workflows:create")),
    tenant: TenantContext = Depends(require_feature("workflows")),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new workflow."""
    from app.workflow import WorkflowBuilder, WorkflowEngine

    # Build workflow definition
    builder = WorkflowBuilder(data.name)
    builder.description = data.description
    builder.trigger = data.trigger

    for node in data.nodes:
        builder.add_node(
            node_id=node.id,
            node_type=node.type.value,
            name=node.name,
            config=node.config,
            position=node.position,
        )

    for node in data.nodes:
        for next_node in node.next_nodes:
            builder.connect(node.id, next_node)

    definition = builder.build()

    # Save workflow
    engine = WorkflowEngine()
    workflow_id = await engine.save_workflow(
        tenant_id=tenant.tenant_id,
        definition=definition,
        agent_id=str(data.agent_id) if data.agent_id else None,
    )

    return WorkflowResponse(
        id=workflow_id,
        name=data.name,
        description=data.description,
        agent_id=str(data.agent_id) if data.agent_id else None,
        trigger=data.trigger,
        status=WorkflowStatus.DRAFT,
        nodes=data.nodes,
        variables=data.variables,
        settings=data.settings,
        version=1,
        execution_count=0,
        last_executed_at=None,
        created_at=datetime.utcnow(),
        updated_at=None,
    )


@router.get("", response_model=WorkflowListResponse)
async def list_workflows(
    status_filter: Optional[WorkflowStatus] = Query(None, alias="status"),
    agent_id: Optional[UUID] = None,
    search: Optional[str] = None,
    pagination: PaginationParams = Depends(get_pagination),
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List all workflows."""
    from app.workflow import WorkflowEngine

    engine = WorkflowEngine()

    # For now, return mock data structure
    workflows = []
    total = 0

    return WorkflowListResponse(
        workflows=workflows,
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=(total + pagination.page_size - 1) // pagination.page_size,
    )


@router.get("/stats", response_model=WorkflowStats)
async def get_workflow_stats(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get workflow statistics."""
    return WorkflowStats(
        total_workflows=0,
        active_workflows=0,
        total_executions=0,
        successful_executions=0,
        failed_executions=0,
        average_duration_ms=0,
    )


@router.get("/templates", response_model=List[WorkflowTemplate])
async def list_templates(
    category: Optional[str] = None,
    user: UserContext = Depends(get_current_user),
):
    """List available workflow templates."""
    from app.workflow import WorkflowTemplates

    templates = WorkflowTemplates.list_templates()

    if category:
        templates = [t for t in templates if t.get("category") == category]

    return [
        WorkflowTemplate(
            id=t["id"],
            name=t["name"],
            description=t["description"],
            category=t["category"],
            nodes=[],
            variables={},
        )
        for t in templates
    ]


@router.post("/from-template/{template_id}", response_model=WorkflowResponse)
async def create_from_template(
    template_id: str,
    name: str = Query(..., min_length=1),
    user: UserContext = Depends(require_permissions("workflows:create")),
    tenant: TenantContext = Depends(require_feature("workflows")),
    db: AsyncSession = Depends(get_db_session),
):
    """Create workflow from template."""
    from app.workflow import WorkflowFactory

    try:
        workflow = WorkflowFactory.from_template(template_id, name)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found: {template_id}",
        )

    return WorkflowResponse(
        id=workflow.id,
        name=workflow.name,
        description=workflow.description,
        agent_id=None,
        trigger="inbound_call",
        status=WorkflowStatus.DRAFT,
        nodes=[],
        variables={},
        settings={},
        version=1,
        execution_count=0,
        last_executed_at=None,
        created_at=datetime.utcnow(),
        updated_at=None,
    )


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get workflow by ID."""
    from app.workflow import WorkflowEngine

    engine = WorkflowEngine()

    # Retrieve workflow
    workflow = await engine.get_workflow(str(workflow_id), tenant.tenant_id)

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    return workflow


@router.patch("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: UUID,
    data: WorkflowUpdate,
    user: UserContext = Depends(require_permissions("workflows:update")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Update a workflow."""
    from app.workflow import WorkflowEngine

    engine = WorkflowEngine()

    workflow = await engine.update_workflow(
        workflow_id=str(workflow_id),
        tenant_id=tenant.tenant_id,
        **data.model_dump(exclude_unset=True),
    )

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    return workflow


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: UUID,
    user: UserContext = Depends(require_permissions("workflows:delete")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a workflow."""
    from app.workflow import WorkflowEngine

    engine = WorkflowEngine()
    deleted = await engine.delete_workflow(str(workflow_id), tenant.tenant_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )


# ============================================================================
# Execution
# ============================================================================

@router.post("/{workflow_id}/execute", response_model=ExecutionResponse)
async def execute_workflow(
    workflow_id: UUID,
    data: ExecuteWorkflowRequest,
    user: UserContext = Depends(require_permissions("workflows:execute")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Execute a workflow."""
    from app.workflow import WorkflowEngine, ExecutionContext

    engine = WorkflowEngine()

    # Create execution context
    context = ExecutionContext(
        workflow_id=str(workflow_id),
        call_id=data.call_id,
        tenant_id=tenant.tenant_id,
        variables=data.variables,
    )

    try:
        result = await engine.execute(str(workflow_id), context)

        return ExecutionResponse(
            id=result.execution_id,
            workflow_id=str(workflow_id),
            call_id=data.call_id,
            status=ExecutionStatus(result.status),
            current_node=result.current_node,
            variables=result.variables,
            execution_path=result.execution_path,
            error=result.error,
            started_at=result.started_at,
            completed_at=result.completed_at,
            duration_ms=result.duration_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}",
        )


@router.get("/{workflow_id}/executions", response_model=ExecutionListResponse)
async def list_executions(
    workflow_id: UUID,
    status_filter: Optional[ExecutionStatus] = Query(None, alias="status"),
    pagination: PaginationParams = Depends(get_pagination),
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List workflow executions."""
    from app.workflow import WorkflowEngine

    engine = WorkflowEngine()

    executions = await engine.list_executions(
        workflow_id=str(workflow_id),
        tenant_id=tenant.tenant_id,
        status=status_filter.value if status_filter else None,
        offset=pagination.offset,
        limit=pagination.limit,
    )

    return ExecutionListResponse(
        executions=[
            ExecutionResponse(
                id=e.execution_id,
                workflow_id=str(workflow_id),
                call_id=e.call_id,
                status=ExecutionStatus(e.status),
                current_node=e.current_node,
                variables=e.variables,
                execution_path=e.execution_path,
                error=e.error,
                started_at=e.started_at,
                completed_at=e.completed_at,
                duration_ms=e.duration_ms,
            )
            for e in executions
        ],
        total=len(executions),
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get("/{workflow_id}/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution(
    workflow_id: UUID,
    execution_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get execution details."""
    from app.workflow import WorkflowEngine

    engine = WorkflowEngine()

    execution = await engine.get_execution(str(execution_id))

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found",
        )

    return ExecutionResponse(
        id=execution.execution_id,
        workflow_id=str(workflow_id),
        call_id=execution.call_id,
        status=ExecutionStatus(execution.status),
        current_node=execution.current_node,
        variables=execution.variables,
        execution_path=execution.execution_path,
        error=execution.error,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        duration_ms=execution.duration_ms,
    )


@router.post("/{workflow_id}/executions/{execution_id}/cancel", response_model=ExecutionResponse)
async def cancel_execution(
    workflow_id: UUID,
    execution_id: UUID,
    user: UserContext = Depends(require_permissions("workflows:execute")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Cancel a running execution."""
    from app.workflow import WorkflowEngine

    engine = WorkflowEngine()

    execution = await engine.cancel_execution(str(execution_id))

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found",
        )

    return ExecutionResponse(
        id=execution.execution_id,
        workflow_id=str(workflow_id),
        call_id=execution.call_id,
        status=ExecutionStatus.CANCELLED,
        current_node=execution.current_node,
        variables=execution.variables,
        execution_path=execution.execution_path,
        error="Cancelled by user",
        started_at=execution.started_at,
        completed_at=datetime.utcnow(),
        duration_ms=execution.duration_ms,
    )


# ============================================================================
# Status Management
# ============================================================================

@router.post("/{workflow_id}/activate", response_model=WorkflowResponse)
async def activate_workflow(
    workflow_id: UUID,
    user: UserContext = Depends(require_permissions("workflows:update")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Activate a workflow."""
    from app.workflow import WorkflowEngine

    engine = WorkflowEngine()

    workflow = await engine.update_workflow(
        workflow_id=str(workflow_id),
        tenant_id=tenant.tenant_id,
        status="active",
    )

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    return workflow


@router.post("/{workflow_id}/pause", response_model=WorkflowResponse)
async def pause_workflow(
    workflow_id: UUID,
    user: UserContext = Depends(require_permissions("workflows:update")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Pause a workflow."""
    from app.workflow import WorkflowEngine

    engine = WorkflowEngine()

    workflow = await engine.update_workflow(
        workflow_id=str(workflow_id),
        tenant_id=tenant.tenant_id,
        status="paused",
    )

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    return workflow


# ============================================================================
# Versioning
# ============================================================================

@router.post("/{workflow_id}/duplicate", response_model=WorkflowResponse)
async def duplicate_workflow(
    workflow_id: UUID,
    name: str = Query(..., min_length=1),
    user: UserContext = Depends(require_permissions("workflows:create")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Duplicate a workflow."""
    from app.workflow import WorkflowEngine

    engine = WorkflowEngine()

    # Get original
    original = await engine.get_workflow(str(workflow_id), tenant.tenant_id)
    if not original:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    # Create copy
    new_workflow = await engine.duplicate_workflow(
        workflow_id=str(workflow_id),
        tenant_id=tenant.tenant_id,
        new_name=name,
    )

    return new_workflow
