"""
Workflow Builder

Visual flow builder for creating workflows:
- Fluent API for workflow construction
- Visual editor support
- Template generation
- Import/export
"""

from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import copy

from .engine import (
    WorkflowDefinition, NodeConfig, Connection, Position,
    NodeType, NodePort, ExecutionMode, WorkflowEngine,
)
from .nodes import create_node, NODE_EXECUTORS


class WorkflowBuilder:
    """
    Fluent builder for constructing workflows.

    Features:
    - Chainable API
    - Auto-connection
    - Validation
    - Visual layout
    """

    def __init__(self, name: str = "", tenant_id: str = ""):
        self._workflow = WorkflowDefinition(
            name=name,
            tenant_id=tenant_id,
        )
        self._last_node_id: Optional[str] = None
        self._node_stack: List[str] = []
        self._auto_position = True
        self._position_x = 100
        self._position_y = 100
        self._grid_spacing = 200

    def _next_position(self) -> Position:
        """Get next auto-layout position."""
        pos = Position(x=self._position_x, y=self._position_y)
        self._position_x += self._grid_spacing
        if self._position_x > 1000:
            self._position_x = 100
            self._position_y += self._grid_spacing
        return pos

    # Basic configuration
    def with_name(self, name: str) -> "WorkflowBuilder":
        """Set workflow name."""
        self._workflow.name = name
        return self

    def with_description(self, description: str) -> "WorkflowBuilder":
        """Set workflow description."""
        self._workflow.description = description
        return self

    def with_version(self, version: str) -> "WorkflowBuilder":
        """Set workflow version."""
        self._workflow.version = version
        return self

    def with_execution_mode(self, mode: ExecutionMode) -> "WorkflowBuilder":
        """Set execution mode."""
        self._workflow.execution_mode = mode
        return self

    def with_max_execution_time(self, ms: int) -> "WorkflowBuilder":
        """Set max execution time in milliseconds."""
        self._workflow.max_execution_time_ms = ms
        return self

    def with_default_variable(self, name: str, value: Any) -> "WorkflowBuilder":
        """Add default variable."""
        self._workflow.default_variables[name] = value
        return self

    def with_metadata(self, key: str, value: Any) -> "WorkflowBuilder":
        """Add metadata."""
        self._workflow.metadata[key] = value
        return self

    def with_tag(self, tag: str) -> "WorkflowBuilder":
        """Add tag."""
        if tag not in self._workflow.tags:
            self._workflow.tags.append(tag)
        return self

    # Node creation
    def add_node(
        self,
        node_type: NodeType,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        position: Optional[Position] = None,
        node_id: Optional[str] = None,
    ) -> "WorkflowBuilder":
        """Add a node to the workflow."""
        node = NodeConfig(
            node_id=node_id or str(uuid.uuid4()),
            node_type=node_type,
            name=name or node_type.value,
            config=config or {},
            position=position or (self._next_position() if self._auto_position else Position()),
        )

        self._workflow.add_node(node)

        # Auto-connect to last node
        if self._last_node_id:
            self.connect(self._last_node_id, node.node_id)

        self._last_node_id = node.node_id
        return self

    # Flow control nodes
    def start(self, name: str = "Start") -> "WorkflowBuilder":
        """Add start node."""
        self.add_node(NodeType.START, name)
        self._workflow.start_node_id = self._last_node_id
        return self

    def end(self, name: str = "End", output_mapping: Optional[Dict[str, str]] = None) -> "WorkflowBuilder":
        """Add end node."""
        return self.add_node(
            NodeType.END,
            name,
            config={"output_mapping": output_mapping or {}},
        )

    def condition(
        self,
        condition: str,
        name: str = "Condition",
        true_port: str = "true",
        false_port: str = "false",
    ) -> "WorkflowBuilder":
        """Add condition node."""
        return self.add_node(
            NodeType.CONDITION,
            name,
            config={
                "condition": condition,
                "true_port": true_port,
                "false_port": false_port,
            },
        )

    def switch(
        self,
        variable: str,
        cases: Dict[str, str],
        name: str = "Switch",
        default_port: str = "default",
    ) -> "WorkflowBuilder":
        """Add switch node."""
        return self.add_node(
            NodeType.SWITCH,
            name,
            config={
                "variable": variable,
                "cases": cases,
                "default_port": default_port,
            },
        )

    def loop(
        self,
        loop_type: str = "count",
        count: int = 10,
        collection: str = "",
        condition: str = "",
        name: str = "Loop",
    ) -> "WorkflowBuilder":
        """Add loop node."""
        return self.add_node(
            NodeType.LOOP,
            name,
            config={
                "loop_type": loop_type,
                "count": count,
                "collection": collection,
                "condition": condition,
            },
        )

    def wait(
        self,
        duration_ms: int = 1000,
        event: str = "",
        name: str = "Wait",
    ) -> "WorkflowBuilder":
        """Add wait node."""
        return self.add_node(
            NodeType.WAIT,
            name,
            config={
                "duration_ms": duration_ms,
                "wait_for_event": event,
            },
        )

    def goto(self, target_node: str, name: str = "Goto") -> "WorkflowBuilder":
        """Add goto node."""
        return self.add_node(
            NodeType.GOTO,
            name,
            config={"target_node": target_node},
        )

    # Voice action nodes
    def speak(
        self,
        text: str,
        voice_id: str = "",
        speed: float = 1.0,
        name: str = "Speak",
    ) -> "WorkflowBuilder":
        """Add speak node."""
        return self.add_node(
            NodeType.SPEAK,
            name,
            config={
                "text": text,
                "voice_id": voice_id,
                "speed": speed,
            },
        )

    def listen(
        self,
        output_variable: str = "user_input",
        timeout_ms: int = 10000,
        language: str = "en",
        name: str = "Listen",
    ) -> "WorkflowBuilder":
        """Add listen node."""
        return self.add_node(
            NodeType.LISTEN,
            name,
            config={
                "output_variable": output_variable,
                "timeout_ms": timeout_ms,
                "language": language,
            },
        )

    def gather(
        self,
        prompt: str,
        output_variable: str = "gathered_input",
        input_type: str = "speech",
        timeout_ms: int = 10000,
        name: str = "Gather",
    ) -> "WorkflowBuilder":
        """Add gather node."""
        return self.add_node(
            NodeType.GATHER,
            name,
            config={
                "prompt": prompt,
                "output_variable": output_variable,
                "input_type": input_type,
                "timeout_ms": timeout_ms,
            },
        )

    def transfer(
        self,
        destination: str,
        transfer_type: str = "blind",
        destination_type: str = "phone",
        announcement: str = "",
        name: str = "Transfer",
    ) -> "WorkflowBuilder":
        """Add transfer node."""
        return self.add_node(
            NodeType.TRANSFER,
            name,
            config={
                "destination": destination,
                "transfer_type": transfer_type,
                "destination_type": destination_type,
                "announcement": announcement,
            },
        )

    def hangup(
        self,
        reason: str = "normal",
        message: str = "",
        name: str = "Hangup",
    ) -> "WorkflowBuilder":
        """Add hangup node."""
        return self.add_node(
            NodeType.HANGUP,
            name,
            config={
                "reason": reason,
                "message": message,
            },
        )

    def record(
        self,
        action: str = "start",
        max_duration_ms: int = 300000,
        name: str = "Record",
    ) -> "WorkflowBuilder":
        """Add record node."""
        return self.add_node(
            NodeType.RECORD,
            name,
            config={
                "action": action,
                "max_duration_ms": max_duration_ms,
            },
        )

    def play(
        self,
        audio_url: str = "",
        audio_id: str = "",
        loop: bool = False,
        name: str = "Play",
    ) -> "WorkflowBuilder":
        """Add play audio node."""
        return self.add_node(
            NodeType.PLAY,
            name,
            config={
                "audio_url": audio_url,
                "audio_id": audio_id,
                "loop": loop,
            },
        )

    # Logic nodes
    def set_variable(
        self,
        variable: str,
        value: Any,
        name: str = "Set Variable",
    ) -> "WorkflowBuilder":
        """Add set variable node."""
        return self.add_node(
            NodeType.SET_VARIABLE,
            name,
            config={
                "variable": variable,
                "value": value,
            },
        )

    def set_variables(
        self,
        assignments: Dict[str, Any],
        name: str = "Set Variables",
    ) -> "WorkflowBuilder":
        """Add multi-variable assignment node."""
        return self.add_node(
            NodeType.SET_VARIABLE,
            name,
            config={"assignments": assignments},
        )

    def function(
        self,
        code: str,
        output_variable: str = "result",
        name: str = "Function",
    ) -> "WorkflowBuilder":
        """Add function node."""
        return self.add_node(
            NodeType.FUNCTION,
            name,
            config={
                "code": code,
                "output_variable": output_variable,
            },
        )

    def http_request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
        output_variable: str = "http_response",
        name: str = "HTTP Request",
    ) -> "WorkflowBuilder":
        """Add HTTP request node."""
        return self.add_node(
            NodeType.HTTP_REQUEST,
            name,
            config={
                "url": url,
                "method": method,
                "headers": headers or {},
                "body": body or {},
                "output_variable": output_variable,
            },
        )

    def webhook(
        self,
        url: str,
        event: str = "workflow.event",
        payload: Optional[Dict[str, Any]] = None,
        async_send: bool = True,
        name: str = "Webhook",
    ) -> "WorkflowBuilder":
        """Add webhook node."""
        return self.add_node(
            NodeType.WEBHOOK,
            name,
            config={
                "url": url,
                "event": event,
                "payload": payload or {},
                "async": async_send,
            },
        )

    # AI nodes
    def llm_prompt(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        output_variable: str = "llm_response",
        name: str = "LLM Prompt",
    ) -> "WorkflowBuilder":
        """Add LLM prompt node."""
        return self.add_node(
            NodeType.LLM_PROMPT,
            name,
            config={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "model": model,
                "temperature": temperature,
                "output_variable": output_variable,
            },
        )

    def intent_detect(
        self,
        text: str = "{{user_input}}",
        intents: Optional[List[str]] = None,
        output_variable: str = "detected_intent",
        name: str = "Intent Detection",
    ) -> "WorkflowBuilder":
        """Add intent detection node."""
        return self.add_node(
            NodeType.INTENT_DETECT,
            name,
            config={
                "text": text,
                "intents": intents or [],
                "output_variable": output_variable,
            },
        )

    def entity_extract(
        self,
        text: str = "{{user_input}}",
        entity_types: Optional[List[str]] = None,
        output_variable: str = "entities",
        name: str = "Entity Extraction",
    ) -> "WorkflowBuilder":
        """Add entity extraction node."""
        return self.add_node(
            NodeType.ENTITY_EXTRACT,
            name,
            config={
                "text": text,
                "entity_types": entity_types or [],
                "output_variable": output_variable,
            },
        )

    def sentiment(
        self,
        text: str = "{{user_input}}",
        output_variable: str = "sentiment",
        name: str = "Sentiment Analysis",
    ) -> "WorkflowBuilder":
        """Add sentiment analysis node."""
        return self.add_node(
            NodeType.SENTIMENT,
            name,
            config={
                "text": text,
                "output_variable": output_variable,
            },
        )

    # Integration nodes
    def crm_lookup(
        self,
        crm_type: str = "salesforce",
        object_type: str = "contact",
        lookup_field: str = "phone",
        lookup_value: str = "",
        output_variable: str = "crm_record",
        name: str = "CRM Lookup",
    ) -> "WorkflowBuilder":
        """Add CRM lookup node."""
        return self.add_node(
            NodeType.CRM_LOOKUP,
            name,
            config={
                "crm_type": crm_type,
                "object_type": object_type,
                "lookup_field": lookup_field,
                "lookup_value": lookup_value,
                "output_variable": output_variable,
            },
        )

    def database(
        self,
        operation: str = "query",
        query: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        output_variable: str = "db_result",
        name: str = "Database",
    ) -> "WorkflowBuilder":
        """Add database node."""
        return self.add_node(
            NodeType.DATABASE,
            name,
            config={
                "operation": operation,
                "query": query,
                "parameters": parameters or {},
                "output_variable": output_variable,
            },
        )

    def queue(
        self,
        queue_name: str = "default",
        priority: int = 5,
        skills: Optional[List[str]] = None,
        timeout_seconds: int = 300,
        name: str = "Queue",
    ) -> "WorkflowBuilder":
        """Add queue node."""
        return self.add_node(
            NodeType.QUEUE,
            name,
            config={
                "queue_name": queue_name,
                "priority": priority,
                "skills": skills or [],
                "timeout_seconds": timeout_seconds,
            },
        )

    def sms(
        self,
        to_number: str,
        message: str,
        from_number: str = "",
        name: str = "Send SMS",
    ) -> "WorkflowBuilder":
        """Add SMS node."""
        return self.add_node(
            NodeType.SMS,
            name,
            config={
                "to_number": to_number,
                "message": message,
                "from_number": from_number,
            },
        )

    def email(
        self,
        to_email: str,
        subject: str,
        body: str,
        html_body: str = "",
        name: str = "Send Email",
    ) -> "WorkflowBuilder":
        """Add email node."""
        return self.add_node(
            NodeType.EMAIL,
            name,
            config={
                "to_email": to_email,
                "subject": subject,
                "body": body,
                "html_body": html_body,
            },
        )

    def subflow(
        self,
        subflow_id: str,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        name: str = "Subflow",
    ) -> "WorkflowBuilder":
        """Add subflow node."""
        return self.add_node(
            NodeType.SUBFLOW,
            name,
            config={
                "subflow_id": subflow_id,
                "input_mapping": input_mapping or {},
                "output_mapping": output_mapping or {},
            },
        )

    # Connection management
    def connect(
        self,
        source_node_id: str,
        target_node_id: str,
        source_port: str = "default",
        target_port: str = "default",
        label: str = "",
        condition: Optional[str] = None,
    ) -> "WorkflowBuilder":
        """Add connection between nodes."""
        connection = Connection(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_port=source_port,
            target_port=target_port,
            label=label,
            condition=condition,
        )
        self._workflow.add_connection(connection)
        return self

    def connect_true(self, target_node_id: str) -> "WorkflowBuilder":
        """Connect from condition true port."""
        if self._last_node_id:
            self.connect(self._last_node_id, target_node_id, source_port="true")
        return self

    def connect_false(self, target_node_id: str) -> "WorkflowBuilder":
        """Connect from condition false port."""
        if self._last_node_id:
            self.connect(self._last_node_id, target_node_id, source_port="false")
        return self

    # Building
    def branch(self) -> "WorkflowBuilder":
        """Save current position for branching."""
        if self._last_node_id:
            self._node_stack.append(self._last_node_id)
        return self

    def merge(self, target_node_id: Optional[str] = None) -> "WorkflowBuilder":
        """Return to branch point."""
        if self._node_stack:
            self._last_node_id = self._node_stack.pop()
        if target_node_id:
            self.connect(self._last_node_id, target_node_id)
        return self

    def from_node(self, node_id: str) -> "WorkflowBuilder":
        """Continue from specific node."""
        self._last_node_id = node_id
        return self

    def validate(self) -> List[str]:
        """Validate workflow."""
        return self._workflow.validate()

    def build(self, validate: bool = True) -> WorkflowDefinition:
        """Build and return workflow."""
        if validate:
            errors = self.validate()
            if errors:
                raise ValueError(f"Invalid workflow: {', '.join(errors)}")
        return self._workflow

    def to_json(self) -> str:
        """Export workflow as JSON."""
        return json.dumps(self._workflow.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "WorkflowBuilder":
        """Import workflow from JSON."""
        data = json.loads(json_str)
        builder = cls(name=data.get("name", ""))
        builder._workflow.workflow_id = data.get("workflow_id", str(uuid.uuid4()))
        builder._workflow.description = data.get("description", "")
        builder._workflow.version = data.get("version", "1.0.0")
        builder._workflow.start_node_id = data.get("start_node_id")
        builder._workflow.is_active = data.get("is_active", True)

        # Import nodes
        for node_id, node_data in data.get("nodes", {}).items():
            node = NodeConfig(
                node_id=node_id,
                node_type=NodeType(node_data.get("node_type", "custom")),
                name=node_data.get("name", ""),
                config=node_data.get("config", {}),
                position=Position(
                    x=node_data.get("position", {}).get("x", 0),
                    y=node_data.get("position", {}).get("y", 0),
                ),
            )
            builder._workflow.nodes[node_id] = node

        # Import connections
        for conn_data in data.get("connections", []):
            conn = Connection(
                connection_id=conn_data.get("connection_id", str(uuid.uuid4())),
                source_node_id=conn_data.get("source_node_id", ""),
                target_node_id=conn_data.get("target_node_id", ""),
                label=conn_data.get("label", ""),
            )
            builder._workflow.connections.append(conn)

        return builder


class WorkflowTemplates:
    """
    Pre-built workflow templates.

    Common patterns for voice agent workflows.
    """

    @staticmethod
    def simple_greeting() -> WorkflowBuilder:
        """Simple greeting workflow."""
        return (
            WorkflowBuilder("Simple Greeting")
            .with_description("Basic greeting and response workflow")
            .start()
            .speak("Hello! How can I help you today?")
            .listen(output_variable="user_input")
            .llm_prompt(
                prompt="User said: {{user_input}}. Provide a helpful response.",
                output_variable="response",
            )
            .speak("{{response}}")
            .end()
        )

    @staticmethod
    def customer_service_flow() -> WorkflowBuilder:
        """Customer service workflow with intent routing."""
        builder = WorkflowBuilder("Customer Service")
        builder.with_description("Customer service with intent-based routing")

        # Start
        builder.start()
        builder.speak("Thank you for calling. How may I assist you today?")
        builder.listen(output_variable="user_input")
        builder.intent_detect(
            text="{{user_input}}",
            intents=["billing", "technical_support", "sales", "general"],
            output_variable="intent",
        )

        # Intent switch
        builder.switch(
            variable="intent",
            cases={
                "billing": "billing_branch",
                "technical_support": "tech_branch",
                "sales": "sales_branch",
            },
            default_port="general_branch",
        )

        # Create branches (simplified - in real use you'd build full subflows)
        current = builder._last_node_id

        # Billing branch
        builder.from_node(current)
        builder.add_node(
            NodeType.SPEAK,
            "Billing Response",
            config={"text": "I'll connect you with our billing department."},
            node_id="billing_branch",
        )
        builder.transfer("{{billing_number}}", name="Transfer Billing")

        # Technical branch
        builder.from_node(current)
        builder.add_node(
            NodeType.SPEAK,
            "Tech Response",
            config={"text": "I'll help you with technical support."},
            node_id="tech_branch",
        )

        # Sales branch
        builder.from_node(current)
        builder.add_node(
            NodeType.SPEAK,
            "Sales Response",
            config={"text": "I'll connect you with sales."},
            node_id="sales_branch",
        )
        builder.transfer("{{sales_number}}", name="Transfer Sales")

        # General branch
        builder.from_node(current)
        builder.add_node(
            NodeType.SPEAK,
            "General Response",
            config={"text": "Let me help you with that."},
            node_id="general_branch",
        )

        return builder

    @staticmethod
    def appointment_scheduler() -> WorkflowBuilder:
        """Appointment scheduling workflow."""
        return (
            WorkflowBuilder("Appointment Scheduler")
            .with_description("Schedule appointments with validation")
            .start()
            .speak("I can help you schedule an appointment. What day would you like?")
            .gather(
                prompt="Please say the date for your appointment.",
                output_variable="date",
                timeout_ms=15000,
            )
            .speak("And what time works best for you?")
            .gather(
                prompt="Please say your preferred time.",
                output_variable="time",
                timeout_ms=15000,
            )
            .speak("Let me confirm: You'd like an appointment on {{date}} at {{time}}. Is that correct?")
            .gather(
                prompt="Please say yes or no.",
                output_variable="confirmation",
                timeout_ms=10000,
            )
            .condition(
                condition="confirmation == 'yes'",
                true_port="confirmed",
                false_port="retry",
            )
            .branch()
            .add_node(
                NodeType.SPEAK,
                "Confirmation",
                config={"text": "Your appointment has been scheduled. You'll receive a confirmation shortly."},
                node_id="confirmed",
            )
            .webhook(
                url="{{appointment_webhook}}",
                event="appointment.scheduled",
                payload={"date": "{{date}}", "time": "{{time}}"},
            )
            .end()
        )

    @staticmethod
    def ivr_menu() -> WorkflowBuilder:
        """IVR menu with DTMF input."""
        return (
            WorkflowBuilder("IVR Menu")
            .with_description("Interactive voice response menu")
            .start()
            .speak(
                "Welcome. Press 1 for sales, 2 for support, 3 for billing, "
                "or stay on the line to speak with an agent."
            )
            .gather(
                prompt="",
                output_variable="selection",
                input_type="dtmf",
                timeout_ms=10000,
            )
            .switch(
                variable="selection",
                cases={
                    "1": "sales_queue",
                    "2": "support_queue",
                    "3": "billing_queue",
                },
                default_port="agent_queue",
            )
        )

    @staticmethod
    def outbound_survey() -> WorkflowBuilder:
        """Outbound survey workflow."""
        builder = WorkflowBuilder("Outbound Survey")
        builder.with_description("Conduct automated phone surveys")

        builder.start()
        builder.speak("Hello, this is a brief survey. Your feedback helps us improve.")

        # Question loop
        builder.set_variable("question_index", 0)
        builder.set_variable("responses", [])

        builder.loop(
            loop_type="count",
            count=3,
            name="Question Loop",
        )
        builder.branch()

        # Ask question
        builder.speak("{{questions[question_index]}}")
        builder.gather(
            prompt="",
            output_variable="answer",
            timeout_ms=15000,
        )

        # Store response
        builder.function(
            code="""
responses.append({
    'question': question_index,
    'answer': answer
})
question_index += 1
result = responses
""",
            output_variable="responses",
        )

        # Loop back
        builder.goto("Question Loop")

        # End
        builder.merge()
        builder.from_node(builder._workflow.nodes["Question Loop"].node_id)
        builder.speak("Thank you for your feedback. Have a great day!")
        builder.webhook(
            url="{{survey_webhook}}",
            event="survey.completed",
            payload={"responses": "{{responses}}"},
        )
        builder.end()

        return builder

    @staticmethod
    def lead_qualifier() -> WorkflowBuilder:
        """Lead qualification workflow."""
        return (
            WorkflowBuilder("Lead Qualifier")
            .with_description("Qualify leads through discovery questions")
            .with_default_variable("lead_score", 0)
            .start()
            .speak("Hi, I'm calling to learn about your business needs. Do you have a few minutes?")
            .gather(
                prompt="",
                output_variable="availability",
                timeout_ms=10000,
            )
            .condition(
                condition="availability contains 'yes'",
                true_port="continue",
                false_port="reschedule",
            )
            .branch()
            # Continue path
            .speak("Great! What's your biggest challenge right now?")
            .gather(prompt="", output_variable="challenge", timeout_ms=20000)
            .speak("And what's your timeline for addressing this?")
            .gather(prompt="", output_variable="timeline", timeout_ms=15000)
            .speak("Finally, do you have budget allocated for a solution?")
            .gather(prompt="", output_variable="budget", timeout_ms=15000)
            # Score lead
            .function(
                code="""
score = 0
if 'urgent' in challenge.lower():
    score += 30
if 'month' in timeline.lower() or 'week' in timeline.lower():
    score += 30
if 'yes' in budget.lower():
    score += 40
result = score
""",
                output_variable="lead_score",
            )
            .condition(
                condition="lead_score >= 70",
                true_port="qualified",
                false_port="nurture",
            )
        )


class WorkflowFactory:
    """
    Factory for creating and managing workflows.
    """

    def __init__(self):
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._templates: Dict[str, Callable[[], WorkflowBuilder]] = {
            "simple_greeting": WorkflowTemplates.simple_greeting,
            "customer_service": WorkflowTemplates.customer_service_flow,
            "appointment_scheduler": WorkflowTemplates.appointment_scheduler,
            "ivr_menu": WorkflowTemplates.ivr_menu,
            "outbound_survey": WorkflowTemplates.outbound_survey,
            "lead_qualifier": WorkflowTemplates.lead_qualifier,
        }

    def create_from_template(
        self,
        template_name: str,
        name: Optional[str] = None,
    ) -> WorkflowDefinition:
        """Create workflow from template."""
        template_func = self._templates.get(template_name)
        if not template_func:
            raise ValueError(f"Unknown template: {template_name}")

        builder = template_func()
        if name:
            builder.with_name(name)

        workflow = builder.build(validate=False)  # Templates might be incomplete
        self._workflows[workflow.workflow_id] = workflow
        return workflow

    def register_template(
        self,
        name: str,
        template_func: Callable[[], WorkflowBuilder],
    ) -> None:
        """Register custom template."""
        self._templates[name] = template_func

    def save(self, workflow: WorkflowDefinition) -> None:
        """Save workflow."""
        self._workflows[workflow.workflow_id] = workflow

    def get(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow by ID."""
        return self._workflows.get(workflow_id)

    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self._templates.keys())

    def list_workflows(self) -> List[WorkflowDefinition]:
        """List all workflows."""
        return list(self._workflows.values())

    def delete(self, workflow_id: str) -> bool:
        """Delete workflow."""
        return self._workflows.pop(workflow_id, None) is not None


# Convenience functions
def create_workflow(name: str = "") -> WorkflowBuilder:
    """Create new workflow builder."""
    return WorkflowBuilder(name)


def from_template(template_name: str) -> WorkflowBuilder:
    """Create workflow from template name."""
    templates = {
        "simple_greeting": WorkflowTemplates.simple_greeting,
        "customer_service": WorkflowTemplates.customer_service_flow,
        "appointment_scheduler": WorkflowTemplates.appointment_scheduler,
        "ivr_menu": WorkflowTemplates.ivr_menu,
        "outbound_survey": WorkflowTemplates.outbound_survey,
        "lead_qualifier": WorkflowTemplates.lead_qualifier,
    }

    template_func = templates.get(template_name)
    if not template_func:
        raise ValueError(f"Unknown template: {template_name}")

    return template_func()
