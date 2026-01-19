"""
Workflow Engine Module

Complete workflow orchestration system:
- Visual flow definition
- Node-based execution
- Variable management
- Expression evaluation
"""

# Engine
from app.workflow.engine import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStatus,
    NodeConfig,
    NodeResult,
    NodeType,
    NodePort,
    Connection,
    Position,
    ExecutionContext,
    ExecutionMode,
    NodeExecutor,
    DefaultNodeExecutor,
    get_workflow_engine,
)

# Nodes
from app.workflow.nodes import (
    # Flow control
    StartNodeExecutor,
    EndNodeExecutor,
    ConditionNodeExecutor,
    SwitchNodeExecutor,
    ParallelNodeExecutor,
    MergeNodeExecutor,
    LoopNodeExecutor,
    WaitNodeExecutor,
    GotoNodeExecutor,
    # Voice actions
    SpeakNodeExecutor,
    ListenNodeExecutor,
    GatherNodeExecutor,
    TransferNodeExecutor,
    HangupNodeExecutor,
    RecordNodeExecutor,
    PlayNodeExecutor,
    # Logic
    SetVariableNodeExecutor,
    FunctionNodeExecutor,
    HttpRequestNodeExecutor,
    WebhookNodeExecutor,
    # AI
    LLMPromptNodeExecutor,
    IntentDetectNodeExecutor,
    EntityExtractNodeExecutor,
    SentimentNodeExecutor,
    # Integration
    CRMLookupNodeExecutor,
    DatabaseNodeExecutor,
    QueueNodeExecutor,
    SMSNodeExecutor,
    EmailNodeExecutor,
    SubflowNodeExecutor,
    # Registry
    NODE_EXECUTORS,
    register_node_executors,
    create_node,
)

# Builder
from app.workflow.builder import (
    WorkflowBuilder,
    WorkflowTemplates,
    WorkflowFactory,
    create_workflow,
    from_template,
)

# Variables
from app.workflow.variables import (
    VariableType,
    VariableScope,
    VariableDefinition,
    Variable,
    VariableStore,
    ExpressionEvaluator,
    VariableTransformer,
    get_variable_store,
    get_expression_evaluator,
    get_variable_transformer,
)

__all__ = [
    # Engine
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowExecution",
    "WorkflowStatus",
    "NodeConfig",
    "NodeResult",
    "NodeType",
    "NodePort",
    "Connection",
    "Position",
    "ExecutionContext",
    "ExecutionMode",
    "NodeExecutor",
    "DefaultNodeExecutor",
    "get_workflow_engine",
    # Node executors
    "StartNodeExecutor",
    "EndNodeExecutor",
    "ConditionNodeExecutor",
    "SwitchNodeExecutor",
    "ParallelNodeExecutor",
    "MergeNodeExecutor",
    "LoopNodeExecutor",
    "WaitNodeExecutor",
    "GotoNodeExecutor",
    "SpeakNodeExecutor",
    "ListenNodeExecutor",
    "GatherNodeExecutor",
    "TransferNodeExecutor",
    "HangupNodeExecutor",
    "RecordNodeExecutor",
    "PlayNodeExecutor",
    "SetVariableNodeExecutor",
    "FunctionNodeExecutor",
    "HttpRequestNodeExecutor",
    "WebhookNodeExecutor",
    "LLMPromptNodeExecutor",
    "IntentDetectNodeExecutor",
    "EntityExtractNodeExecutor",
    "SentimentNodeExecutor",
    "CRMLookupNodeExecutor",
    "DatabaseNodeExecutor",
    "QueueNodeExecutor",
    "SMSNodeExecutor",
    "EmailNodeExecutor",
    "SubflowNodeExecutor",
    "NODE_EXECUTORS",
    "register_node_executors",
    "create_node",
    # Builder
    "WorkflowBuilder",
    "WorkflowTemplates",
    "WorkflowFactory",
    "create_workflow",
    "from_template",
    # Variables
    "VariableType",
    "VariableScope",
    "VariableDefinition",
    "Variable",
    "VariableStore",
    "ExpressionEvaluator",
    "VariableTransformer",
    "get_variable_store",
    "get_expression_evaluator",
    "get_variable_transformer",
]
