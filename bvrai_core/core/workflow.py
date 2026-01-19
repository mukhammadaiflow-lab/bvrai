"""
Advanced Workflow Engine
========================

Enterprise-grade workflow engine for orchestrating complex multi-step
processes with support for:
- Parallel and sequential execution
- Conditional branching
- Error handling and compensation
- Long-running workflows with persistence
- Sub-workflows
- Human tasks and approvals
- Event-driven triggers
- Saga pattern support

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import partial, wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")
StepResult = TypeVar("StepResult")


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowState(str, Enum):
    """Workflow execution states"""

    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    TIMED_OUT = "timed_out"


class StepState(str, Enum):
    """Individual step states"""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    CANCELLED = "cancelled"


class StepType(str, Enum):
    """Types of workflow steps"""

    TASK = "task"                    # Automated task
    HUMAN = "human"                  # Human task/approval
    DECISION = "decision"            # Conditional branching
    PARALLEL = "parallel"            # Parallel execution
    SUBPROCESS = "subprocess"        # Sub-workflow
    WAIT = "wait"                    # Wait for event/time
    LOOP = "loop"                    # Loop over collection
    FORK_JOIN = "fork_join"          # Fork and join
    SERVICE_CALL = "service_call"    # External service
    COMPENSATION = "compensation"    # Compensation action


class TriggerType(str, Enum):
    """Workflow trigger types"""

    MANUAL = "manual"
    EVENT = "event"
    SCHEDULE = "schedule"
    API = "api"
    SIGNAL = "signal"


class RetryStrategy(str, Enum):
    """Retry strategies"""

    NONE = "none"
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class RetryConfig(BaseModel):
    """Retry configuration for steps"""

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    multiplier: float = 2.0
    retryable_errors: List[str] = Field(default_factory=list)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.strategy == RetryStrategy.NONE:
            return 0
        elif self.strategy == RetryStrategy.FIXED:
            return self.base_delay_seconds
        elif self.strategy == RetryStrategy.LINEAR:
            return min(
                self.base_delay_seconds * attempt,
                self.max_delay_seconds
            )
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            return min(
                self.base_delay_seconds * (self.multiplier ** (attempt - 1)),
                self.max_delay_seconds
            )
        return self.base_delay_seconds


class TimeoutConfig(BaseModel):
    """Timeout configuration"""

    step_timeout_seconds: float = 300.0
    workflow_timeout_seconds: float = 86400.0  # 24 hours
    idle_timeout_seconds: float = 3600.0       # 1 hour


class WorkflowConfig(BaseModel):
    """Workflow configuration"""

    name: str
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = Field(default_factory=list)

    # Execution
    retry: RetryConfig = Field(default_factory=RetryConfig)
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)
    max_concurrent_steps: int = 10

    # Persistence
    persist_state: bool = True
    checkpoint_interval_seconds: float = 30.0

    # Error handling
    fail_fast: bool = False
    enable_compensation: bool = True

    # Metadata
    created_by: Optional[str] = None
    organization_id: Optional[str] = None


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class WorkflowContext:
    """
    Context passed through workflow execution.

    Contains all data needed for workflow steps to execute,
    including input data, intermediate results, and metadata.
    """

    workflow_id: str = field(default_factory=lambda: str(uuid4()))
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    parent_workflow_id: Optional[str] = None
    parent_step_id: Optional[str] = None

    # Data
    input_data: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    organization_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None

    # State
    current_step: Optional[str] = None
    execution_path: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable value"""
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a variable value"""
        self.variables[key] = value
        self.updated_at = datetime.utcnow()

    def get_step_result(self, step_id: str, default: Any = None) -> Any:
        """Get result from a completed step"""
        return self.step_results.get(step_id, default)

    def set_step_result(self, step_id: str, result: Any) -> None:
        """Store step result"""
        self.step_results[step_id] = result
        self.updated_at = datetime.utcnow()

    def add_error(self, step_id: str, error: Exception) -> None:
        """Record an error"""
        self.errors.append({
            "step_id": step_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "traceback": traceback.format_exc()
        })

    def clone(self) -> "WorkflowContext":
        """Create a deep copy of the context"""
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dict"""
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "parent_workflow_id": self.parent_workflow_id,
            "parent_step_id": self.parent_step_id,
            "input_data": self.input_data,
            "variables": self.variables,
            "step_results": self.step_results,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "current_step": self.current_step,
            "execution_path": self.execution_path,
            "errors": self.errors,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class StepResult:
    """Result of a workflow step execution"""

    step_id: str
    state: StepState
    output: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.state == StepState.COMPLETED


@dataclass
class WorkflowResult:
    """Result of a workflow execution"""

    workflow_id: str
    execution_id: str
    state: WorkflowState
    output: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    context: Optional[WorkflowContext] = None


# =============================================================================
# WORKFLOW CONDITIONS
# =============================================================================


class WorkflowCondition(ABC):
    """Abstract base class for workflow conditions"""

    @abstractmethod
    def evaluate(self, context: WorkflowContext) -> bool:
        """Evaluate the condition"""
        pass

    def __and__(self, other: "WorkflowCondition") -> "AndCondition":
        return AndCondition(self, other)

    def __or__(self, other: "WorkflowCondition") -> "OrCondition":
        return OrCondition(self, other)

    def __invert__(self) -> "NotCondition":
        return NotCondition(self)


class AndCondition(WorkflowCondition):
    """Logical AND of two conditions"""

    def __init__(self, left: WorkflowCondition, right: WorkflowCondition):
        self.left = left
        self.right = right

    def evaluate(self, context: WorkflowContext) -> bool:
        return self.left.evaluate(context) and self.right.evaluate(context)


class OrCondition(WorkflowCondition):
    """Logical OR of two conditions"""

    def __init__(self, left: WorkflowCondition, right: WorkflowCondition):
        self.left = left
        self.right = right

    def evaluate(self, context: WorkflowContext) -> bool:
        return self.left.evaluate(context) or self.right.evaluate(context)


class NotCondition(WorkflowCondition):
    """Logical NOT of a condition"""

    def __init__(self, condition: WorkflowCondition):
        self.condition = condition

    def evaluate(self, context: WorkflowContext) -> bool:
        return not self.condition.evaluate(context)


class ExpressionCondition(WorkflowCondition):
    """Condition based on a Python expression"""

    def __init__(self, expression: str):
        self.expression = expression

    def evaluate(self, context: WorkflowContext) -> bool:
        # Create safe evaluation context
        eval_context = {
            "input": context.input_data,
            "vars": context.variables,
            "results": context.step_results,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "any": any,
            "all": all,
        }
        try:
            return bool(eval(self.expression, {"__builtins__": {}}, eval_context))
        except Exception:
            return False


class VariableCondition(WorkflowCondition):
    """Condition based on variable comparison"""

    def __init__(
        self,
        variable: str,
        operator: str,
        value: Any
    ):
        self.variable = variable
        self.operator = operator
        self.value = value

    def evaluate(self, context: WorkflowContext) -> bool:
        var_value = context.get(self.variable)

        if self.operator == "==":
            return var_value == self.value
        elif self.operator == "!=":
            return var_value != self.value
        elif self.operator == ">":
            return var_value > self.value
        elif self.operator == ">=":
            return var_value >= self.value
        elif self.operator == "<":
            return var_value < self.value
        elif self.operator == "<=":
            return var_value <= self.value
        elif self.operator == "in":
            return var_value in self.value
        elif self.operator == "not_in":
            return var_value not in self.value
        elif self.operator == "contains":
            return self.value in var_value
        elif self.operator == "is_null":
            return var_value is None
        elif self.operator == "is_not_null":
            return var_value is not None
        elif self.operator == "matches":
            import re
            return bool(re.match(self.value, str(var_value)))

        return False


class StepResultCondition(WorkflowCondition):
    """Condition based on step result"""

    def __init__(
        self,
        step_id: str,
        expected_state: StepState = StepState.COMPLETED,
        result_check: Optional[Callable[[Any], bool]] = None
    ):
        self.step_id = step_id
        self.expected_state = expected_state
        self.result_check = result_check

    def evaluate(self, context: WorkflowContext) -> bool:
        result = context.get_step_result(self.step_id)

        if result is None:
            return False

        if self.result_check:
            return self.result_check(result)

        return True


class AlwaysTrueCondition(WorkflowCondition):
    """Condition that always returns True"""

    def evaluate(self, context: WorkflowContext) -> bool:
        return True


class AlwaysFalseCondition(WorkflowCondition):
    """Condition that always returns False"""

    def evaluate(self, context: WorkflowContext) -> bool:
        return False


# Convenience functions for creating conditions
def when(expression: str) -> ExpressionCondition:
    """Create expression condition"""
    return ExpressionCondition(expression)


def var(variable: str) -> "VariableConditionBuilder":
    """Create variable condition builder"""
    return VariableConditionBuilder(variable)


class VariableConditionBuilder:
    """Builder for variable conditions"""

    def __init__(self, variable: str):
        self.variable = variable

    def eq(self, value: Any) -> VariableCondition:
        return VariableCondition(self.variable, "==", value)

    def ne(self, value: Any) -> VariableCondition:
        return VariableCondition(self.variable, "!=", value)

    def gt(self, value: Any) -> VariableCondition:
        return VariableCondition(self.variable, ">", value)

    def gte(self, value: Any) -> VariableCondition:
        return VariableCondition(self.variable, ">=", value)

    def lt(self, value: Any) -> VariableCondition:
        return VariableCondition(self.variable, "<", value)

    def lte(self, value: Any) -> VariableCondition:
        return VariableCondition(self.variable, "<=", value)

    def in_(self, values: List[Any]) -> VariableCondition:
        return VariableCondition(self.variable, "in", values)

    def not_in(self, values: List[Any]) -> VariableCondition:
        return VariableCondition(self.variable, "not_in", values)

    def contains(self, value: Any) -> VariableCondition:
        return VariableCondition(self.variable, "contains", value)

    def is_null(self) -> VariableCondition:
        return VariableCondition(self.variable, "is_null", None)

    def is_not_null(self) -> VariableCondition:
        return VariableCondition(self.variable, "is_not_null", None)

    def matches(self, pattern: str) -> VariableCondition:
        return VariableCondition(self.variable, "matches", pattern)


# =============================================================================
# WORKFLOW STEP
# =============================================================================


class WorkflowStep(ABC):
    """
    Abstract base class for workflow steps.

    Each step represents a unit of work in a workflow.
    """

    def __init__(
        self,
        step_id: str,
        name: str = "",
        description: str = "",
        step_type: StepType = StepType.TASK,
        retry_config: Optional[RetryConfig] = None,
        timeout_seconds: float = 300.0,
        condition: Optional[WorkflowCondition] = None,
        compensation: Optional["WorkflowStep"] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.step_id = step_id
        self.name = name or step_id
        self.description = description
        self.step_type = step_type
        self.retry_config = retry_config or RetryConfig()
        self.timeout_seconds = timeout_seconds
        self.condition = condition
        self.compensation = compensation
        self.metadata = metadata or {}

        self._state = StepState.PENDING
        self._logger = structlog.get_logger(f"workflow.step.{step_id}")

    @property
    def state(self) -> StepState:
        return self._state

    @state.setter
    def state(self, value: StepState) -> None:
        self._state = value

    def should_execute(self, context: WorkflowContext) -> bool:
        """Check if step should execute based on condition"""
        if self.condition is None:
            return True
        return self.condition.evaluate(context)

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the step"""
        pass

    async def compensate(self, context: WorkflowContext) -> None:
        """Execute compensation for this step"""
        if self.compensation:
            await self.compensation.execute(context)


class TaskStep(WorkflowStep):
    """Task step that executes a function"""

    def __init__(
        self,
        step_id: str,
        task: Callable[[WorkflowContext], Awaitable[Any]],
        **kwargs
    ):
        super().__init__(step_id, step_type=StepType.TASK, **kwargs)
        self.task = task

    async def execute(self, context: WorkflowContext) -> Any:
        return await self.task(context)


class DecisionStep(WorkflowStep):
    """Decision step for conditional branching"""

    def __init__(
        self,
        step_id: str,
        branches: Dict[str, Tuple[WorkflowCondition, str]],
        default_branch: Optional[str] = None,
        **kwargs
    ):
        super().__init__(step_id, step_type=StepType.DECISION, **kwargs)
        self.branches = branches
        self.default_branch = default_branch

    async def execute(self, context: WorkflowContext) -> str:
        """Execute and return the next step ID"""
        for branch_name, (condition, next_step) in self.branches.items():
            if condition.evaluate(context):
                return next_step

        if self.default_branch:
            return self.default_branch

        raise ValueError(f"No matching branch in decision step {self.step_id}")


class ParallelStep(WorkflowStep):
    """Step that executes multiple steps in parallel"""

    def __init__(
        self,
        step_id: str,
        steps: List[WorkflowStep],
        wait_all: bool = True,
        **kwargs
    ):
        super().__init__(step_id, step_type=StepType.PARALLEL, **kwargs)
        self.steps = steps
        self.wait_all = wait_all

    async def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute all steps in parallel"""
        tasks = {}
        for step in self.steps:
            if step.should_execute(context):
                tasks[step.step_id] = asyncio.create_task(step.execute(context))

        results = {}
        if self.wait_all:
            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for (step_id, _), result in zip(tasks.items(), gathered):
                if isinstance(result, Exception):
                    raise result
                results[step_id] = result
        else:
            done, pending = await asyncio.wait(
                tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                for step_id, t in tasks.items():
                    if t == task:
                        results[step_id] = task.result()

            for task in pending:
                task.cancel()

        return results


class WaitStep(WorkflowStep):
    """Step that waits for a duration or event"""

    def __init__(
        self,
        step_id: str,
        duration_seconds: Optional[float] = None,
        wait_for_event: Optional[str] = None,
        **kwargs
    ):
        super().__init__(step_id, step_type=StepType.WAIT, **kwargs)
        self.duration_seconds = duration_seconds
        self.wait_for_event = wait_for_event
        self._event_received = asyncio.Event()
        self._event_data: Dict[str, Any] = {}

    async def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        if self.duration_seconds:
            await asyncio.sleep(self.duration_seconds)
            return {"waited_seconds": self.duration_seconds}
        elif self.wait_for_event:
            await self._event_received.wait()
            return {"event": self.wait_for_event, "data": self._event_data}
        return {}

    def signal_event(self, data: Dict[str, Any]) -> None:
        """Signal that the expected event has occurred"""
        self._event_data = data
        self._event_received.set()


class LoopStep(WorkflowStep):
    """Step that loops over a collection"""

    def __init__(
        self,
        step_id: str,
        collection_variable: str,
        item_variable: str,
        body: WorkflowStep,
        max_iterations: int = 1000,
        parallel: bool = False,
        **kwargs
    ):
        super().__init__(step_id, step_type=StepType.LOOP, **kwargs)
        self.collection_variable = collection_variable
        self.item_variable = item_variable
        self.body = body
        self.max_iterations = max_iterations
        self.parallel = parallel

    async def execute(self, context: WorkflowContext) -> List[Any]:
        collection = context.get(self.collection_variable, [])
        results = []

        if self.parallel:
            async def execute_iteration(item):
                iter_context = context.clone()
                iter_context.set(self.item_variable, item)
                return await self.body.execute(iter_context)

            tasks = [execute_iteration(item) for item in collection[:self.max_iterations]]
            results = await asyncio.gather(*tasks)
        else:
            for i, item in enumerate(collection):
                if i >= self.max_iterations:
                    break
                context.set(self.item_variable, item)
                result = await self.body.execute(context)
                results.append(result)

        return results


class SubProcessStep(WorkflowStep):
    """Step that executes a sub-workflow"""

    def __init__(
        self,
        step_id: str,
        workflow: "Workflow",
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        super().__init__(step_id, step_type=StepType.SUBPROCESS, **kwargs)
        self.workflow = workflow
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}

    async def execute(self, context: WorkflowContext) -> Any:
        # Create sub-context with mapped inputs
        sub_context = WorkflowContext(
            parent_workflow_id=context.workflow_id,
            parent_step_id=self.step_id,
            organization_id=context.organization_id,
            user_id=context.user_id,
            correlation_id=context.correlation_id,
            trace_id=context.trace_id
        )

        # Map inputs
        for target, source in self.input_mapping.items():
            sub_context.set(target, context.get(source))

        # Execute sub-workflow
        result = await self.workflow.execute(sub_context)

        # Map outputs back
        for target, source in self.output_mapping.items():
            context.set(target, sub_context.get(source))

        return result


class ServiceCallStep(WorkflowStep):
    """Step that calls an external service"""

    def __init__(
        self,
        step_id: str,
        service_name: str,
        method: str,
        params_builder: Callable[[WorkflowContext], Dict[str, Any]],
        result_handler: Optional[Callable[[Any, WorkflowContext], None]] = None,
        **kwargs
    ):
        super().__init__(step_id, step_type=StepType.SERVICE_CALL, **kwargs)
        self.service_name = service_name
        self.method = method
        self.params_builder = params_builder
        self.result_handler = result_handler
        self._service_client: Any = None

    def set_service_client(self, client: Any) -> None:
        self._service_client = client

    async def execute(self, context: WorkflowContext) -> Any:
        if not self._service_client:
            raise RuntimeError(f"Service client not set for {self.service_name}")

        params = self.params_builder(context)
        method = getattr(self._service_client, self.method)
        result = await method(**params)

        if self.result_handler:
            self.result_handler(result, context)

        return result


class HumanTaskStep(WorkflowStep):
    """Step that requires human interaction"""

    def __init__(
        self,
        step_id: str,
        task_definition: Dict[str, Any],
        assignee: Optional[str] = None,
        due_date: Optional[datetime] = None,
        form_fields: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        super().__init__(step_id, step_type=StepType.HUMAN, **kwargs)
        self.task_definition = task_definition
        self.assignee = assignee
        self.due_date = due_date
        self.form_fields = form_fields or []
        self._completion_event = asyncio.Event()
        self._task_result: Dict[str, Any] = {}

    async def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        # Wait for human to complete the task
        await self._completion_event.wait()
        return self._task_result

    def complete_task(self, result: Dict[str, Any]) -> None:
        """Complete the human task with result"""
        self._task_result = result
        self._completion_event.set()


# =============================================================================
# WORKFLOW DEFINITION
# =============================================================================


class Workflow:
    """
    Workflow definition and execution.

    Workflows are composed of steps that execute in a defined order,
    with support for parallel execution, branching, and error handling.

    Usage:
        workflow = Workflow("order-processing")

        workflow.add_step(TaskStep("validate", validate_order))
        workflow.add_step(TaskStep("process", process_order))
        workflow.add_step(TaskStep("notify", send_notification))

        workflow.set_transitions([
            ("validate", "process"),
            ("process", "notify")
        ])

        result = await workflow.execute(context)
    """

    def __init__(
        self,
        name: str,
        config: Optional[WorkflowConfig] = None
    ):
        self.name = name
        self.config = config or WorkflowConfig(name=name)
        self._steps: Dict[str, WorkflowStep] = {}
        self._transitions: Dict[str, List[str]] = defaultdict(list)
        self._start_step: Optional[str] = None
        self._end_steps: Set[str] = set()
        self._state = WorkflowState.CREATED
        self._logger = structlog.get_logger(f"workflow.{name}")
        self._execution_hooks: Dict[str, List[Callable]] = defaultdict(list)

    @property
    def state(self) -> WorkflowState:
        return self._state

    @state.setter
    def state(self, value: WorkflowState) -> None:
        old_state = self._state
        self._state = value
        self._logger.info("workflow_state_changed", old_state=old_state, new_state=value)

    # -------------------------------------------------------------------------
    # Definition Methods
    # -------------------------------------------------------------------------

    def add_step(self, step: WorkflowStep) -> "Workflow":
        """Add a step to the workflow"""
        self._steps[step.step_id] = step
        return self

    def remove_step(self, step_id: str) -> "Workflow":
        """Remove a step from the workflow"""
        if step_id in self._steps:
            del self._steps[step_id]
            self._transitions.pop(step_id, None)
            for transitions in self._transitions.values():
                if step_id in transitions:
                    transitions.remove(step_id)
        return self

    def add_transition(self, from_step: str, to_step: str) -> "Workflow":
        """Add a transition between steps"""
        self._transitions[from_step].append(to_step)
        return self

    def set_transitions(
        self,
        transitions: List[Tuple[str, str]]
    ) -> "Workflow":
        """Set all transitions at once"""
        self._transitions.clear()
        for from_step, to_step in transitions:
            self._transitions[from_step].append(to_step)
        return self

    def set_start(self, step_id: str) -> "Workflow":
        """Set the starting step"""
        if step_id not in self._steps:
            raise ValueError(f"Step {step_id} not found in workflow")
        self._start_step = step_id
        return self

    def set_end(self, *step_ids: str) -> "Workflow":
        """Set the ending steps"""
        for step_id in step_ids:
            if step_id not in self._steps:
                raise ValueError(f"Step {step_id} not found in workflow")
            self._end_steps.add(step_id)
        return self

    # -------------------------------------------------------------------------
    # Hook Methods
    # -------------------------------------------------------------------------

    def on_step_start(
        self,
        handler: Callable[[str, WorkflowContext], Awaitable[None]]
    ) -> "Workflow":
        """Register handler for step start events"""
        self._execution_hooks["step_start"].append(handler)
        return self

    def on_step_complete(
        self,
        handler: Callable[[str, Any, WorkflowContext], Awaitable[None]]
    ) -> "Workflow":
        """Register handler for step completion events"""
        self._execution_hooks["step_complete"].append(handler)
        return self

    def on_step_error(
        self,
        handler: Callable[[str, Exception, WorkflowContext], Awaitable[None]]
    ) -> "Workflow":
        """Register handler for step error events"""
        self._execution_hooks["step_error"].append(handler)
        return self

    def on_workflow_complete(
        self,
        handler: Callable[[WorkflowContext], Awaitable[None]]
    ) -> "Workflow":
        """Register handler for workflow completion"""
        self._execution_hooks["workflow_complete"].append(handler)
        return self

    # -------------------------------------------------------------------------
    # Execution Methods
    # -------------------------------------------------------------------------

    async def execute(
        self,
        context: Optional[WorkflowContext] = None
    ) -> WorkflowResult:
        """
        Execute the workflow.

        Args:
            context: Workflow context with input data

        Returns:
            WorkflowResult with execution details
        """
        context = context or WorkflowContext()
        context.workflow_id = str(uuid4())
        context.started_at = datetime.utcnow()

        self.state = WorkflowState.RUNNING
        start_time = time.time()

        step_results: Dict[str, StepResult] = {}
        current_step_id = self._start_step or self._find_start_step()

        self._logger.info(
            "workflow_started",
            workflow_id=context.workflow_id,
            start_step=current_step_id
        )

        try:
            while current_step_id:
                step = self._steps.get(current_step_id)
                if not step:
                    raise ValueError(f"Step {current_step_id} not found")

                context.current_step = current_step_id
                context.execution_path.append(current_step_id)

                # Check if step should execute
                if not step.should_execute(context):
                    step.state = StepState.SKIPPED
                    current_step_id = self._get_next_step(current_step_id, step, context)
                    continue

                # Execute step with retry
                result = await self._execute_step_with_retry(step, context)
                step_results[current_step_id] = result

                if not result.success:
                    if self.config.fail_fast:
                        raise RuntimeError(
                            f"Step {current_step_id} failed: {result.error}"
                        )

                # Store result
                context.set_step_result(current_step_id, result.output)

                # Get next step
                if current_step_id in self._end_steps:
                    current_step_id = None
                else:
                    current_step_id = self._get_next_step(
                        current_step_id, step, context
                    )

            self.state = WorkflowState.COMPLETED
            duration_ms = (time.time() - start_time) * 1000

            # Call completion hooks
            for handler in self._execution_hooks.get("workflow_complete", []):
                await handler(context)

            self._logger.info(
                "workflow_completed",
                workflow_id=context.workflow_id,
                duration_ms=duration_ms
            )

            return WorkflowResult(
                workflow_id=context.workflow_id,
                execution_id=context.execution_id,
                state=WorkflowState.COMPLETED,
                output=context.variables,
                started_at=context.started_at,
                completed_at=datetime.utcnow(),
                duration_ms=duration_ms,
                step_results=step_results,
                context=context
            )

        except Exception as e:
            self.state = WorkflowState.FAILED
            duration_ms = (time.time() - start_time) * 1000

            self._logger.error(
                "workflow_failed",
                workflow_id=context.workflow_id,
                error=str(e),
                current_step=context.current_step
            )

            # Run compensation if enabled
            if self.config.enable_compensation:
                await self._run_compensation(context)

            return WorkflowResult(
                workflow_id=context.workflow_id,
                execution_id=context.execution_id,
                state=WorkflowState.FAILED,
                error=str(e),
                started_at=context.started_at,
                completed_at=datetime.utcnow(),
                duration_ms=duration_ms,
                step_results=step_results,
                context=context
            )

    async def _execute_step_with_retry(
        self,
        step: WorkflowStep,
        context: WorkflowContext
    ) -> StepResult:
        """Execute a step with retry logic"""
        start_time = time.time()
        retries = 0
        last_error: Optional[Exception] = None

        # Call step start hooks
        for handler in self._execution_hooks.get("step_start", []):
            await handler(step.step_id, context)

        while retries <= step.retry_config.max_retries:
            try:
                step.state = StepState.RUNNING

                # Execute with timeout
                output = await asyncio.wait_for(
                    step.execute(context),
                    timeout=step.timeout_seconds
                )

                step.state = StepState.COMPLETED
                duration_ms = (time.time() - start_time) * 1000

                # Call step complete hooks
                for handler in self._execution_hooks.get("step_complete", []):
                    await handler(step.step_id, output, context)

                return StepResult(
                    step_id=step.step_id,
                    state=StepState.COMPLETED,
                    output=output,
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    duration_ms=duration_ms,
                    retries=retries
                )

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Step {step.step_id} timed out")
                retries += 1
                self._logger.warning(
                    "step_timeout",
                    step_id=step.step_id,
                    retry=retries
                )

            except Exception as e:
                last_error = e
                retries += 1
                self._logger.warning(
                    "step_error",
                    step_id=step.step_id,
                    error=str(e),
                    retry=retries
                )

                context.add_error(step.step_id, e)

                # Call step error hooks
                for handler in self._execution_hooks.get("step_error", []):
                    await handler(step.step_id, e, context)

            if retries <= step.retry_config.max_retries:
                delay = step.retry_config.get_delay(retries)
                await asyncio.sleep(delay)

        step.state = StepState.FAILED
        duration_ms = (time.time() - start_time) * 1000

        return StepResult(
            step_id=step.step_id,
            state=StepState.FAILED,
            error=str(last_error) if last_error else "Unknown error",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            retries=retries
        )

    async def _run_compensation(self, context: WorkflowContext) -> None:
        """Run compensation for completed steps in reverse order"""
        self.state = WorkflowState.COMPENSATING
        self._logger.info("running_compensation", workflow_id=context.workflow_id)

        # Get completed steps in reverse order
        completed_steps = [
            step_id for step_id in reversed(context.execution_path)
            if self._steps.get(step_id) and self._steps[step_id].state == StepState.COMPLETED
        ]

        for step_id in completed_steps:
            step = self._steps[step_id]
            if step.compensation:
                try:
                    step.state = StepState.COMPENSATING
                    await step.compensate(context)
                    step.state = StepState.COMPENSATED
                except Exception as e:
                    self._logger.error(
                        "compensation_failed",
                        step_id=step_id,
                        error=str(e)
                    )

        self.state = WorkflowState.COMPENSATED

    def _find_start_step(self) -> str:
        """Find the start step if not explicitly set"""
        if not self._steps:
            raise ValueError("Workflow has no steps")

        # Find step with no incoming transitions
        all_targets = set()
        for targets in self._transitions.values():
            all_targets.update(targets)

        for step_id in self._steps:
            if step_id not in all_targets:
                return step_id

        return list(self._steps.keys())[0]

    def _get_next_step(
        self,
        current_step_id: str,
        step: WorkflowStep,
        context: WorkflowContext
    ) -> Optional[str]:
        """Get the next step to execute"""
        # Handle decision steps
        if isinstance(step, DecisionStep):
            result = context.get_step_result(current_step_id)
            if result:
                return result

        # Follow transitions
        next_steps = self._transitions.get(current_step_id, [])

        if not next_steps:
            return None

        if len(next_steps) == 1:
            return next_steps[0]

        # Multiple transitions - use conditions if available
        for next_step_id in next_steps:
            next_step = self._steps.get(next_step_id)
            if next_step and next_step.should_execute(context):
                return next_step_id

        return next_steps[0]


# =============================================================================
# WORKFLOW BUILDER
# =============================================================================


class WorkflowBuilder:
    """
    Fluent builder for creating workflows.

    Usage:
        workflow = (WorkflowBuilder("order-processing")
            .task("validate", validate_order)
            .task("process", process_order)
            .task("notify", send_notification)
            .transition("validate", "process")
            .transition("process", "notify")
            .build())
    """

    def __init__(self, name: str, config: Optional[WorkflowConfig] = None):
        self._name = name
        self._config = config or WorkflowConfig(name=name)
        self._steps: List[WorkflowStep] = []
        self._transitions: List[Tuple[str, str]] = []
        self._start_step: Optional[str] = None
        self._end_steps: List[str] = []

    def task(
        self,
        step_id: str,
        task: Callable[[WorkflowContext], Awaitable[Any]],
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a task step"""
        self._steps.append(TaskStep(step_id, task, **kwargs))
        return self

    def decision(
        self,
        step_id: str,
        branches: Dict[str, Tuple[WorkflowCondition, str]],
        default: Optional[str] = None,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a decision step"""
        self._steps.append(DecisionStep(step_id, branches, default, **kwargs))
        return self

    def parallel(
        self,
        step_id: str,
        steps: List[WorkflowStep],
        wait_all: bool = True,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a parallel execution step"""
        self._steps.append(ParallelStep(step_id, steps, wait_all, **kwargs))
        return self

    def wait(
        self,
        step_id: str,
        duration_seconds: Optional[float] = None,
        wait_for_event: Optional[str] = None,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a wait step"""
        self._steps.append(WaitStep(step_id, duration_seconds, wait_for_event, **kwargs))
        return self

    def loop(
        self,
        step_id: str,
        collection: str,
        item_var: str,
        body: WorkflowStep,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a loop step"""
        self._steps.append(LoopStep(step_id, collection, item_var, body, **kwargs))
        return self

    def subprocess(
        self,
        step_id: str,
        workflow: Workflow,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a subprocess step"""
        self._steps.append(SubProcessStep(step_id, workflow, **kwargs))
        return self

    def human_task(
        self,
        step_id: str,
        task_definition: Dict[str, Any],
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a human task step"""
        self._steps.append(HumanTaskStep(step_id, task_definition, **kwargs))
        return self

    def transition(self, from_step: str, to_step: str) -> "WorkflowBuilder":
        """Add a transition"""
        self._transitions.append((from_step, to_step))
        return self

    def start(self, step_id: str) -> "WorkflowBuilder":
        """Set the start step"""
        self._start_step = step_id
        return self

    def end(self, *step_ids: str) -> "WorkflowBuilder":
        """Set the end steps"""
        self._end_steps.extend(step_ids)
        return self

    def build(self) -> Workflow:
        """Build the workflow"""
        workflow = Workflow(self._name, self._config)

        for step in self._steps:
            workflow.add_step(step)

        workflow.set_transitions(self._transitions)

        if self._start_step:
            workflow.set_start(self._start_step)

        if self._end_steps:
            workflow.set_end(*self._end_steps)

        return workflow


# =============================================================================
# WORKFLOW ENGINE
# =============================================================================


class WorkflowEngine:
    """
    Central workflow execution engine.

    Manages workflow registration, execution, persistence, and monitoring.

    Usage:
        engine = WorkflowEngine()
        await engine.start()

        workflow_id = await engine.start_workflow("order-processing", input_data)
        status = await engine.get_workflow_status(workflow_id)

        await engine.stop()
    """

    def __init__(
        self,
        max_concurrent_workflows: int = 100,
        persistence_enabled: bool = True
    ):
        self._workflows: Dict[str, Workflow] = {}
        self._executions: Dict[str, Tuple[Workflow, WorkflowContext, asyncio.Task]] = {}
        self._results: Dict[str, WorkflowResult] = {}
        self._max_concurrent = max_concurrent_workflows
        self._semaphore = asyncio.Semaphore(max_concurrent_workflows)
        self._persistence_enabled = persistence_enabled
        self._running = False
        self._logger = structlog.get_logger("workflow_engine")
        self._metrics = {
            "workflows_started": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "total_duration_ms": 0.0
        }

    async def start(self) -> None:
        """Start the workflow engine"""
        self._running = True
        self._logger.info("workflow_engine_started")

    async def stop(self) -> None:
        """Stop the workflow engine"""
        # Cancel all running executions
        for execution_id, (_, _, task) in self._executions.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._running = False
        self._logger.info("workflow_engine_stopped")

    def register_workflow(self, workflow: Workflow) -> None:
        """Register a workflow definition"""
        self._workflows[workflow.name] = workflow
        self._logger.info("workflow_registered", name=workflow.name)

    def unregister_workflow(self, name: str) -> None:
        """Unregister a workflow definition"""
        if name in self._workflows:
            del self._workflows[name]
            self._logger.info("workflow_unregistered", name=name)

    async def start_workflow(
        self,
        workflow_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        **context_kwargs
    ) -> str:
        """
        Start a workflow execution.

        Args:
            workflow_name: Name of the registered workflow
            input_data: Input data for the workflow
            **context_kwargs: Additional context parameters

        Returns:
            Execution ID
        """
        if not self._running:
            raise RuntimeError("Workflow engine is not running")

        workflow = self._workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        # Create context
        context = WorkflowContext(
            input_data=input_data or {},
            **context_kwargs
        )

        # Start execution
        async with self._semaphore:
            self._metrics["workflows_started"] += 1

            task = asyncio.create_task(workflow.execute(context))
            self._executions[context.execution_id] = (workflow, context, task)

            # Set up completion callback
            task.add_done_callback(
                partial(self._on_execution_complete, context.execution_id)
            )

            self._logger.info(
                "workflow_execution_started",
                workflow=workflow_name,
                execution_id=context.execution_id
            )

            return context.execution_id

    def _on_execution_complete(
        self,
        execution_id: str,
        task: asyncio.Task
    ) -> None:
        """Handle workflow execution completion"""
        try:
            result = task.result()
            self._results[execution_id] = result

            if result.state == WorkflowState.COMPLETED:
                self._metrics["workflows_completed"] += 1
            else:
                self._metrics["workflows_failed"] += 1

            self._metrics["total_duration_ms"] += result.duration_ms

            self._logger.info(
                "workflow_execution_complete",
                execution_id=execution_id,
                state=result.state.value,
                duration_ms=result.duration_ms
            )

        except asyncio.CancelledError:
            self._metrics["workflows_failed"] += 1
            self._logger.warning(
                "workflow_execution_cancelled",
                execution_id=execution_id
            )

        except Exception as e:
            self._metrics["workflows_failed"] += 1
            self._logger.error(
                "workflow_execution_error",
                execution_id=execution_id,
                error=str(e)
            )

        finally:
            self._executions.pop(execution_id, None)

    async def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow execution"""
        # Check completed results
        if execution_id in self._results:
            result = self._results[execution_id]
            return {
                "execution_id": execution_id,
                "state": result.state.value,
                "output": result.output,
                "error": result.error,
                "duration_ms": result.duration_ms
            }

        # Check running executions
        if execution_id in self._executions:
            workflow, context, _ = self._executions[execution_id]
            return {
                "execution_id": execution_id,
                "state": workflow.state.value,
                "current_step": context.current_step,
                "execution_path": context.execution_path
            }

        return None

    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow"""
        if execution_id in self._executions:
            _, _, task = self._executions[execution_id]
            task.cancel()
            return True
        return False

    async def signal_workflow(
        self,
        execution_id: str,
        signal_name: str,
        data: Dict[str, Any]
    ) -> bool:
        """Send a signal to a waiting workflow"""
        if execution_id not in self._executions:
            return False

        workflow, context, _ = self._executions[execution_id]
        current_step = context.current_step

        if current_step:
            step = workflow._steps.get(current_step)
            if isinstance(step, WaitStep) and step.wait_for_event == signal_name:
                step.signal_event(data)
                return True

        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get workflow engine metrics"""
        return {
            **self._metrics,
            "registered_workflows": len(self._workflows),
            "active_executions": len(self._executions),
            "completed_executions": len(self._results)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def workflow(name: str, **config_kwargs) -> WorkflowBuilder:
    """Create a workflow builder"""
    config = WorkflowConfig(name=name, **config_kwargs)
    return WorkflowBuilder(name, config)


def task(
    step_id: str,
    func: Optional[Callable[[WorkflowContext], Awaitable[Any]]] = None,
    **kwargs
) -> Union[TaskStep, Callable]:
    """Create a task step or decorator"""
    if func:
        return TaskStep(step_id, func, **kwargs)

    def decorator(f: Callable[[WorkflowContext], Awaitable[Any]]) -> TaskStep:
        return TaskStep(step_id, f, name=f.__name__, **kwargs)

    return decorator
