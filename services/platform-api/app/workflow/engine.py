"""
Workflow Engine Core

Visual flow orchestration engine:
- Flow definition and execution
- Node-based processing
- State management
- Event-driven execution
"""

from typing import Optional, Dict, Any, List, Callable, Set, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import asyncio
import json
import logging
import copy

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"


class NodeType(str, Enum):
    """Types of workflow nodes."""
    # Flow control
    START = "start"
    END = "end"
    CONDITION = "condition"
    SWITCH = "switch"
    PARALLEL = "parallel"
    MERGE = "merge"
    LOOP = "loop"
    WAIT = "wait"
    GOTO = "goto"

    # Actions
    SPEAK = "speak"
    LISTEN = "listen"
    GATHER = "gather"
    TRANSFER = "transfer"
    HANGUP = "hangup"
    RECORD = "record"
    PLAY = "play"

    # Logic
    SET_VARIABLE = "set_variable"
    FUNCTION = "function"
    HTTP_REQUEST = "http_request"
    WEBHOOK = "webhook"

    # AI
    LLM_PROMPT = "llm_prompt"
    INTENT_DETECT = "intent_detect"
    ENTITY_EXTRACT = "entity_extract"
    SENTIMENT = "sentiment"

    # Integration
    CRM_LOOKUP = "crm_lookup"
    DATABASE = "database"
    QUEUE = "queue"
    SMS = "sms"
    EMAIL = "email"

    # Custom
    SUBFLOW = "subflow"
    CUSTOM = "custom"


class ExecutionMode(str, Enum):
    """Workflow execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    EVENT_DRIVEN = "event_driven"


@dataclass
class Position:
    """Node position in visual editor."""
    x: float = 0.0
    y: float = 0.0


@dataclass
class NodePort:
    """Connection port on a node."""
    port_id: str
    name: str
    port_type: str = "output"  # input, output
    data_type: str = "any"  # any, string, number, boolean, object
    required: bool = False
    multiple: bool = False


@dataclass
class Connection:
    """Connection between nodes."""
    connection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = ""
    source_port: str = "default"
    target_node_id: str = ""
    target_port: str = "default"
    label: str = ""
    condition: Optional[str] = None
    priority: int = 0


@dataclass
class NodeConfig:
    """Configuration for a workflow node."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.CUSTOM
    name: str = ""
    description: str = ""

    # Visual
    position: Position = field(default_factory=Position)
    color: str = ""
    icon: str = ""

    # Ports
    input_ports: List[NodePort] = field(default_factory=list)
    output_ports: List[NodePort] = field(default_factory=list)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Execution
    timeout_ms: int = 30000
    retry_count: int = 0
    retry_delay_ms: int = 1000
    async_execution: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class NodeResult:
    """Result from node execution."""
    success: bool = True
    output: Any = None
    error: Optional[str] = None
    next_node: Optional[str] = None
    next_port: str = "default"

    # Execution info
    execution_time_ms: float = 0.0
    retries: int = 0

    # Data updates
    variable_updates: Dict[str, Any] = field(default_factory=dict)
    context_updates: Dict[str, Any] = field(default_factory=dict)

    # Flow control
    should_continue: bool = True
    should_pause: bool = False
    should_wait: bool = False
    wait_duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "next_node": self.next_node,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class ExecutionContext:
    """Context for workflow execution."""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    tenant_id: str = ""

    # Call context
    call_id: Optional[str] = None
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None

    # Variables
    variables: Dict[str, Any] = field(default_factory=dict)
    global_variables: Dict[str, Any] = field(default_factory=dict)

    # Input data
    input_data: Dict[str, Any] = field(default_factory=dict)

    # State
    current_node_id: Optional[str] = None
    visited_nodes: List[str] = field(default_factory=list)
    execution_path: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None

    # Results
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)

    def set_variable(self, name: str, value: Any) -> None:
        """Set variable value."""
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get variable value."""
        # Check local variables first, then global
        if name in self.variables:
            return self.variables[name]
        return self.global_variables.get(name, default)

    def resolve_expression(self, expression: str) -> Any:
        """Resolve expression with variable substitution."""
        if not expression:
            return expression

        # Simple variable substitution {{variable}}
        import re
        pattern = r'\{\{(\w+(?:\.\w+)*)\}\}'

        def replace(match):
            var_path = match.group(1)
            parts = var_path.split('.')

            value = self.get_variable(parts[0])
            for part in parts[1:]:
                if isinstance(value, dict):
                    value = value.get(part)
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return match.group(0)

            return str(value) if value is not None else ""

        return re.sub(pattern, replace, str(expression))


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    # Structure
    nodes: Dict[str, NodeConfig] = field(default_factory=dict)
    connections: List[Connection] = field(default_factory=list)
    start_node_id: Optional[str] = None

    # Configuration
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_execution_time_ms: int = 300000  # 5 minutes
    max_iterations: int = 1000

    # Variables
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    default_variables: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    tenant_id: str = ""
    is_active: bool = True
    is_template: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_node(self, node: NodeConfig) -> None:
        """Add node to workflow."""
        self.nodes[node.node_id] = node
        self.updated_at = datetime.utcnow()

    def remove_node(self, node_id: str) -> bool:
        """Remove node from workflow."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            # Remove related connections
            self.connections = [
                c for c in self.connections
                if c.source_node_id != node_id and c.target_node_id != node_id
            ]
            self.updated_at = datetime.utcnow()
            return True
        return False

    def add_connection(self, connection: Connection) -> None:
        """Add connection to workflow."""
        self.connections.append(connection)
        self.updated_at = datetime.utcnow()

    def get_outgoing_connections(self, node_id: str) -> List[Connection]:
        """Get connections from a node."""
        return [c for c in self.connections if c.source_node_id == node_id]

    def get_incoming_connections(self, node_id: str) -> List[Connection]:
        """Get connections to a node."""
        return [c for c in self.connections if c.target_node_id == node_id]

    def validate(self) -> List[str]:
        """Validate workflow definition."""
        errors = []

        # Check for start node
        if not self.start_node_id:
            errors.append("No start node defined")
        elif self.start_node_id not in self.nodes:
            errors.append(f"Start node {self.start_node_id} not found")

        # Check all connections reference valid nodes
        for conn in self.connections:
            if conn.source_node_id not in self.nodes:
                errors.append(f"Connection source {conn.source_node_id} not found")
            if conn.target_node_id not in self.nodes:
                errors.append(f"Connection target {conn.target_node_id} not found")

        # Check for unreachable nodes
        reachable = set()
        if self.start_node_id:
            self._find_reachable(self.start_node_id, reachable)

        for node_id in self.nodes:
            if node_id not in reachable:
                errors.append(f"Node {node_id} is unreachable")

        return errors

    def _find_reachable(self, node_id: str, reachable: Set[str]) -> None:
        """Find all reachable nodes from a starting node."""
        if node_id in reachable:
            return
        reachable.add(node_id)
        for conn in self.get_outgoing_connections(node_id):
            self._find_reachable(conn.target_node_id, reachable)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "nodes": {nid: {
                "node_id": n.node_id,
                "node_type": n.node_type.value,
                "name": n.name,
                "config": n.config,
                "position": {"x": n.position.x, "y": n.position.y},
            } for nid, n in self.nodes.items()},
            "connections": [{
                "connection_id": c.connection_id,
                "source_node_id": c.source_node_id,
                "target_node_id": c.target_node_id,
                "label": c.label,
            } for c in self.connections],
            "start_node_id": self.start_node_id,
            "is_active": self.is_active,
        }


class WorkflowExecution:
    """
    Manages workflow execution state.

    Features:
    - State persistence
    - Pause/resume support
    - Error recovery
    """

    def __init__(
        self,
        workflow: WorkflowDefinition,
        context: Optional[ExecutionContext] = None,
    ):
        self.workflow = workflow
        self.context = context or ExecutionContext(workflow_id=workflow.workflow_id)
        self.status = WorkflowStatus.PENDING

        # Execution state
        self._current_node: Optional[NodeConfig] = None
        self._iteration_count: int = 0
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Execution history
        self._history: List[Dict[str, Any]] = []

    @property
    def execution_id(self) -> str:
        return self.context.execution_id

    def on(self, event: str, handler: Callable) -> None:
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    async def _emit(self, event: str, data: Any = None) -> None:
        """Emit event."""
        for handler in self._event_handlers.get(event, []):
            try:
                result = handler(event, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def pause(self) -> None:
        """Pause execution."""
        if self.status == WorkflowStatus.RUNNING:
            self.status = WorkflowStatus.PAUSED
            self._record_event("paused")

    def resume(self) -> None:
        """Resume execution."""
        if self.status == WorkflowStatus.PAUSED:
            self.status = WorkflowStatus.RUNNING
            self._record_event("resumed")

    def cancel(self) -> None:
        """Cancel execution."""
        self.status = WorkflowStatus.CANCELLED
        self._end_time = datetime.utcnow()
        self._record_event("cancelled")

    def _record_event(self, event: str, data: Any = None) -> None:
        """Record execution event."""
        self._history.append({
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "node_id": self._current_node.node_id if self._current_node else None,
            "data": data,
        })

    def get_state(self) -> Dict[str, Any]:
        """Get current execution state."""
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow.workflow_id,
            "status": self.status.value,
            "current_node": self._current_node.node_id if self._current_node else None,
            "iteration_count": self._iteration_count,
            "variables": self.context.variables,
            "visited_nodes": self.context.visited_nodes,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore execution from state."""
        self.status = WorkflowStatus(state.get("status", "pending"))
        self._iteration_count = state.get("iteration_count", 0)
        self.context.variables = state.get("variables", {})
        self.context.visited_nodes = state.get("visited_nodes", [])

        current_node_id = state.get("current_node")
        if current_node_id:
            self._current_node = self.workflow.nodes.get(current_node_id)


class WorkflowEngine:
    """
    Core workflow execution engine.

    Features:
    - Node-based execution
    - Parallel and sequential flows
    - Error handling and recovery
    - Event emission
    """

    def __init__(self):
        self._node_executors: Dict[NodeType, Type["NodeExecutor"]] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self._workflows: Dict[str, WorkflowDefinition] = {}

        # Register default executors
        self._register_default_executors()

    def _register_default_executors(self) -> None:
        """Register default node executors."""
        # Will be populated by nodes.py
        pass

    def register_executor(
        self,
        node_type: NodeType,
        executor_class: Type["NodeExecutor"],
    ) -> None:
        """Register node executor."""
        self._node_executors[node_type] = executor_class

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register workflow definition."""
        self._workflows[workflow.workflow_id] = workflow

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow by ID."""
        return self._workflows.get(workflow_id)

    async def execute(
        self,
        workflow: WorkflowDefinition,
        input_data: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        """Execute workflow."""
        # Create execution context
        ctx = context or ExecutionContext(
            workflow_id=workflow.workflow_id,
            variables=copy.deepcopy(workflow.default_variables),
        )
        ctx.input_data = input_data or {}
        ctx.started_at = datetime.utcnow()

        # Create execution
        execution = WorkflowExecution(workflow, ctx)
        self._executions[execution.execution_id] = execution

        try:
            # Validate workflow
            errors = workflow.validate()
            if errors:
                raise ValueError(f"Invalid workflow: {', '.join(errors)}")

            # Start execution
            execution.status = WorkflowStatus.RUNNING
            execution._start_time = datetime.utcnow()
            await execution._emit("start", {"workflow_id": workflow.workflow_id})

            # Execute from start node
            result = await self._execute_from_node(
                execution,
                workflow.start_node_id,
            )

            # Complete execution
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                execution._end_time = datetime.utcnow()

            await execution._emit("complete", {"result": result})

            return {
                "execution_id": execution.execution_id,
                "status": execution.status.value,
                "output": ctx.output,
                "variables": ctx.variables,
                "execution_path": ctx.execution_path,
            }

        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            execution.status = WorkflowStatus.FAILED
            execution._end_time = datetime.utcnow()
            await execution._emit("error", {"error": str(e)})

            return {
                "execution_id": execution.execution_id,
                "status": WorkflowStatus.FAILED.value,
                "error": str(e),
            }

    async def _execute_from_node(
        self,
        execution: WorkflowExecution,
        node_id: str,
    ) -> NodeResult:
        """Execute workflow starting from a specific node."""
        workflow = execution.workflow
        ctx = execution.context

        while node_id and execution.status == WorkflowStatus.RUNNING:
            # Check limits
            execution._iteration_count += 1
            if execution._iteration_count > workflow.max_iterations:
                raise RuntimeError(f"Max iterations ({workflow.max_iterations}) exceeded")

            # Check timeout
            elapsed = (datetime.utcnow() - execution._start_time).total_seconds() * 1000
            if elapsed > workflow.max_execution_time_ms:
                raise TimeoutError(f"Workflow timeout exceeded")

            # Get node
            node = workflow.nodes.get(node_id)
            if not node:
                raise ValueError(f"Node not found: {node_id}")

            execution._current_node = node
            ctx.current_node_id = node_id
            ctx.visited_nodes.append(node_id)
            ctx.last_activity = datetime.utcnow()

            await execution._emit("node_start", {"node_id": node_id})

            # Execute node
            result = await self._execute_node(execution, node)

            # Record result
            ctx.node_results[node_id] = result
            ctx.execution_path.append({
                "node_id": node_id,
                "node_type": node.node_type.value,
                "result": result.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            })

            # Apply variable updates
            ctx.variables.update(result.variable_updates)

            await execution._emit("node_complete", {
                "node_id": node_id,
                "result": result.to_dict(),
            })

            # Handle flow control
            if not result.should_continue:
                break

            if result.should_pause:
                execution.status = WorkflowStatus.PAUSED
                break

            if result.should_wait:
                await asyncio.sleep(result.wait_duration_ms / 1000)

            # Determine next node
            if result.next_node:
                node_id = result.next_node
            else:
                node_id = await self._get_next_node(execution, node, result)

        return result

    async def _execute_node(
        self,
        execution: WorkflowExecution,
        node: NodeConfig,
    ) -> NodeResult:
        """Execute a single node."""
        import time
        start_time = time.time()

        try:
            # Get executor
            executor_class = self._node_executors.get(node.node_type)
            if not executor_class:
                # Use default executor
                executor_class = DefaultNodeExecutor

            executor = executor_class(node)

            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    executor.execute(execution.context),
                    timeout=node.timeout_ms / 1000,
                )
            except asyncio.TimeoutError:
                result = NodeResult(
                    success=False,
                    error=f"Node timeout after {node.timeout_ms}ms",
                )

            result.execution_time_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            logger.error(f"Node execution error: {e}")
            return NodeResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _get_next_node(
        self,
        execution: WorkflowExecution,
        current_node: NodeConfig,
        result: NodeResult,
    ) -> Optional[str]:
        """Determine next node based on connections."""
        workflow = execution.workflow
        ctx = execution.context

        connections = workflow.get_outgoing_connections(current_node.node_id)
        if not connections:
            return None

        # Sort by priority
        connections.sort(key=lambda c: c.priority, reverse=True)

        # Find matching connection
        for conn in connections:
            # Check port match
            if conn.source_port != result.next_port:
                continue

            # Check condition
            if conn.condition:
                condition_result = self._evaluate_condition(conn.condition, ctx)
                if not condition_result:
                    continue

            return conn.target_node_id

        # Default to first connection if no match
        if connections:
            return connections[0].target_node_id

        return None

    def _evaluate_condition(
        self,
        condition: str,
        ctx: ExecutionContext,
    ) -> bool:
        """Evaluate condition expression."""
        try:
            # Resolve variables in condition
            resolved = ctx.resolve_expression(condition)

            # Simple evaluation (in production, use safe eval)
            # This is simplified - real implementation would use
            # a proper expression parser
            if resolved.lower() in ("true", "1", "yes"):
                return True
            if resolved.lower() in ("false", "0", "no"):
                return False

            # Try numeric comparison
            try:
                return bool(eval(resolved, {"__builtins__": {}}, {}))
            except:
                return False

        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False

    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution by ID."""
        return self._executions.get(execution_id)

    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
    ) -> List[WorkflowExecution]:
        """List executions with optional filters."""
        executions = list(self._executions.values())

        if workflow_id:
            executions = [e for e in executions if e.workflow.workflow_id == workflow_id]

        if status:
            executions = [e for e in executions if e.status == status]

        return executions

    def cleanup_executions(self, max_age_hours: int = 24) -> int:
        """Clean up old executions."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = []

        for exec_id, execution in self._executions.items():
            if execution._end_time and execution._end_time < cutoff:
                to_remove.append(exec_id)

        for exec_id in to_remove:
            del self._executions[exec_id]

        return len(to_remove)


class NodeExecutor(ABC):
    """Abstract base for node executors."""

    def __init__(self, node: NodeConfig):
        self.node = node

    @abstractmethod
    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute the node."""
        pass


class DefaultNodeExecutor(NodeExecutor):
    """Default executor for unknown node types."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Pass through execution."""
        return NodeResult(
            success=True,
            output=None,
            next_port="default",
        )


# Singleton engine instance
_engine_instance: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get singleton workflow engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = WorkflowEngine()
    return _engine_instance
