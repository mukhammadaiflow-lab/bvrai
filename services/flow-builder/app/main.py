"""
Flow Builder Service.

Visual flow builder backend for creating AI voice agent workflows.

API Endpoints:
- Flows: CRUD operations for voice agent flows
- Nodes: Available node types and configurations
- Templates: Pre-built flow templates
- Execution: Run and test flows
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import FlowStatus, ExecutionMode, get_settings
from .models import (
    CreateFlowRequest,
    UpdateFlowRequest,
    FlowResponse,
    FlowListResponse,
    ExecuteFlowRequest,
    ExecuteFlowResponse,
    ValidateFlowResponse,
    NodeListResponse,
)
from .canvas import CanvasManager, FlowValidator
from .executor import FlowExecutor
from .nodes import get_node_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Service metadata
SERVICE_NAME = "flow-builder"
SERVICE_VERSION = "1.0.0"
START_TIME = time.time()

# Global instances
canvas_manager: Optional[CanvasManager] = None
flow_executor: Optional[FlowExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global canvas_manager, flow_executor

    logger.info(f"Starting {SERVICE_NAME} v{SERVICE_VERSION}")

    # Initialize components
    canvas_manager = CanvasManager()
    flow_executor = FlowExecutor()

    logger.info("Flow Builder initialized successfully")

    yield

    logger.info("Shutting down Flow Builder")


# Create FastAPI app
app = FastAPI(
    title="Flow Builder Service",
    description="Visual flow builder backend for AI voice agent workflows",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

# CORS middleware - use shared secure configuration
try:
    from bvrai_core.security.cors import get_cors_middleware_config
    app.add_middleware(CORSMiddleware, **get_cors_middleware_config())
except ImportError:
    import os
    env = os.getenv("ENVIRONMENT", "development")
    origins = (
        ["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"]
        if env == "development"
        else os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") or ["https://app.bvrai.com"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )


# =============================================================================
# Health & Info
# =============================================================================


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "uptime_seconds": time.time() - START_TIME,
    }


@app.get("/info")
async def get_info() -> Dict[str, Any]:
    """Get service information."""
    registry = get_node_registry()
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "node_types": len(registry.list_all()),
        "categories": [c.value for c in registry.get_categories()],
    }


# =============================================================================
# Flows API
# =============================================================================


@app.post("/flows", response_model=FlowResponse)
async def create_flow(request: CreateFlowRequest) -> Dict[str, Any]:
    """Create a new flow."""
    flow = await canvas_manager.create_flow(
        name=request.name,
        user_id="user-1",  # TODO: Get from auth
        description=request.description or "",
        template_id=request.template_id,
        tags=request.tags,
    )

    return flow.to_dict()


@app.get("/flows", response_model=FlowListResponse)
async def list_flows(
    status: Optional[str] = None,
    tags: Optional[str] = None,
    search: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> Dict[str, Any]:
    """List flows with filtering."""
    status_filter = FlowStatus(status) if status else None
    tags_filter = tags.split(",") if tags else None

    flows, total = await canvas_manager.list_flows(
        status=status_filter,
        tags=tags_filter,
        search=search,
        page=page,
        page_size=page_size,
    )

    return {
        "flows": [f.to_dict() for f in flows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.get("/flows/{flow_id}", response_model=FlowResponse)
async def get_flow(flow_id: str) -> Dict[str, Any]:
    """Get a flow by ID."""
    flow = await canvas_manager.get_flow(flow_id)

    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")

    return flow.to_dict()


@app.put("/flows/{flow_id}", response_model=FlowResponse)
async def update_flow(flow_id: str, request: UpdateFlowRequest) -> Dict[str, Any]:
    """Update a flow."""
    updates = request.model_dump(exclude_unset=True)

    flow = await canvas_manager.update_flow(
        flow_id=flow_id,
        user_id="user-1",  # TODO: Get from auth
        updates=updates,
    )

    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")

    return flow.to_dict()


@app.delete("/flows/{flow_id}")
async def delete_flow(flow_id: str) -> Dict[str, Any]:
    """Delete a flow."""
    success = await canvas_manager.delete_flow(flow_id)

    if not success:
        raise HTTPException(status_code=404, detail="Flow not found")

    return {"deleted": True, "flow_id": flow_id}


@app.post("/flows/{flow_id}/duplicate", response_model=FlowResponse)
async def duplicate_flow(
    flow_id: str,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Duplicate a flow."""
    flow = await canvas_manager.duplicate_flow(
        flow_id=flow_id,
        user_id="user-1",  # TODO: Get from auth
        new_name=name,
    )

    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")

    return flow.to_dict()


# =============================================================================
# Validation & Execution
# =============================================================================


@app.post("/flows/{flow_id}/validate", response_model=ValidateFlowResponse)
async def validate_flow(flow_id: str) -> Dict[str, Any]:
    """Validate a flow."""
    result = await canvas_manager.validate_flow(flow_id)

    return {
        "valid": result.valid,
        "errors": [
            {
                "severity": i.severity,
                "message": i.message,
                "node_id": i.node_id,
                "property": i.property_name,
            }
            for i in result.errors
        ],
        "warnings": [
            {
                "severity": i.severity,
                "message": i.message,
                "node_id": i.node_id,
            }
            for i in result.warnings
        ],
    }


@app.post("/flows/{flow_id}/execute", response_model=ExecuteFlowResponse)
async def execute_flow(
    flow_id: str,
    request: ExecuteFlowRequest,
) -> Dict[str, Any]:
    """Execute a flow (dry-run or production)."""
    flow = await canvas_manager.get_flow(flow_id)

    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")

    result = await flow_executor.execute(
        flow=flow,
        mode=request.mode,
        input_vars=request.input_variables,
        mock_call=request.mock_call,
    )

    return {
        "execution_id": result.execution_id,
        "success": result.success,
        "output": result.final_output,
        "node_results": [
            {
                "node_id": nr.node_id,
                "success": nr.success,
                "output": nr.output_data,
                "duration_ms": nr.duration_ms,
                "error": nr.error,
                "logs": nr.logs,
            }
            for nr in result.node_results
        ],
        "duration_ms": result.total_duration_ms,
        "error": result.error,
    }


# =============================================================================
# Version History
# =============================================================================


@app.get("/flows/{flow_id}/versions")
async def get_flow_versions(
    flow_id: str,
    limit: int = Query(default=10, ge=1, le=50),
) -> Dict[str, Any]:
    """Get version history for a flow."""
    versions = await canvas_manager.get_versions(flow_id, limit)

    if not versions:
        raise HTTPException(status_code=404, detail="Flow not found")

    return {"flow_id": flow_id, "versions": versions}


@app.post("/flows/{flow_id}/versions/{version}/restore")
async def restore_flow_version(flow_id: str, version: int) -> Dict[str, Any]:
    """Restore a previous version of a flow."""
    flow = await canvas_manager.restore_version(
        flow_id=flow_id,
        version=version,
        user_id="user-1",  # TODO: Get from auth
    )

    if not flow:
        raise HTTPException(status_code=404, detail="Version not found")

    return {"restored": True, "current_version": flow.metadata.version}


# =============================================================================
# Nodes API
# =============================================================================


@app.get("/nodes", response_model=NodeListResponse)
async def list_nodes(
    category: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """List available node types."""
    registry = get_node_registry()

    if category:
        from .config import NodeCategory

        try:
            cat = NodeCategory(category)
            nodes = registry.list_by_category(cat)
        except ValueError:
            nodes = []
    elif search:
        nodes = registry.search(search)
    else:
        nodes = registry.list_all()

    return {
        "nodes": [n.to_dict() for n in nodes],
        "categories": [c.value for c in registry.get_categories()],
    }


@app.get("/nodes/catalog")
async def get_node_catalog() -> Dict[str, Any]:
    """Get complete node catalog organized by category."""
    registry = get_node_registry()
    return registry.to_catalog()


@app.get("/nodes/{node_type}")
async def get_node_type(node_type: str) -> Dict[str, Any]:
    """Get details for a specific node type."""
    registry = get_node_registry()
    node_def = registry.get_by_name(node_type)

    if not node_def:
        raise HTTPException(status_code=404, detail="Node type not found")

    return node_def.to_dict()


# =============================================================================
# Templates API
# =============================================================================


# Predefined templates
TEMPLATES = [
    {
        "id": "simple-ivr",
        "name": "Simple IVR",
        "description": "Basic interactive voice response with menu options",
        "category": "ivr",
        "preview_image": None,
    },
    {
        "id": "appointment-booking",
        "name": "Appointment Booking",
        "description": "Book appointments with calendar integration",
        "category": "booking",
        "preview_image": None,
    },
    {
        "id": "lead-qualification",
        "name": "Lead Qualification",
        "description": "Qualify leads with AI-powered conversation",
        "category": "sales",
        "preview_image": None,
    },
    {
        "id": "customer-support",
        "name": "Customer Support",
        "description": "Handle support inquiries with knowledge base",
        "category": "support",
        "preview_image": None,
    },
    {
        "id": "survey",
        "name": "Survey",
        "description": "Conduct phone surveys and collect responses",
        "category": "research",
        "preview_image": None,
    },
]


@app.get("/templates")
async def list_templates(
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """List available flow templates."""
    templates = TEMPLATES

    if category:
        templates = [t for t in templates if t["category"] == category]

    return {"templates": templates}


@app.get("/templates/{template_id}")
async def get_template(template_id: str) -> Dict[str, Any]:
    """Get a template by ID."""
    template = next((t for t in TEMPLATES if t["id"] == template_id), None)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Would include full flow_data in real implementation
    return {**template, "flow_data": {"nodes": [], "connections": []}}


# =============================================================================
# WebSocket for Real-Time Collaboration
# =============================================================================


# Active WebSocket connections per flow
active_connections: Dict[str, List[WebSocket]] = {}


@app.websocket("/ws/flows/{flow_id}")
async def websocket_flow(websocket: WebSocket, flow_id: str):
    """
    WebSocket for real-time flow collaboration.

    Protocol:
    - Client sends: {"action": "move_node", "payload": {...}, "user_id": "..."}
    - Server broadcasts to all collaborators
    """
    await websocket.accept()

    # Add to active connections
    if flow_id not in active_connections:
        active_connections[flow_id] = []
    active_connections[flow_id].append(websocket)

    logger.info(f"WebSocket connected to flow: {flow_id}")

    try:
        while True:
            data = await websocket.receive_json()

            # Broadcast to all other connections
            for conn in active_connections.get(flow_id, []):
                if conn != websocket:
                    await conn.send_json({
                        "flow_id": flow_id,
                        "action": data.get("action"),
                        "payload": data.get("payload"),
                        "user_id": data.get("user_id"),
                        "timestamp": time.time(),
                    })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from flow: {flow_id}")

    finally:
        if flow_id in active_connections:
            active_connections[flow_id].remove(websocket)
            if not active_connections[flow_id]:
                del active_connections[flow_id]


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
