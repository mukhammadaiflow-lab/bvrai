"""
Dialog Flow Generator Module

This module automatically generates conversation flows based on
business information, services, and industry patterns.
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
)

from .base import (
    BusinessInfo,
    BusinessCategory,
    ServiceInfo,
)


logger = logging.getLogger(__name__)


class FlowType(str, Enum):
    """Types of conversation flows."""
    GREETING = "greeting"
    FAQ = "faq"
    SERVICE_INQUIRY = "service_inquiry"
    SCHEDULING = "scheduling"
    QUOTE_REQUEST = "quote_request"
    COMPLAINT = "complaint"
    TRANSFER = "transfer"
    CLOSING = "closing"
    AFTER_HOURS = "after_hours"
    LEAD_QUALIFICATION = "lead_qualification"
    ORDER_TAKING = "order_taking"


@dataclass
class FlowNode:
    """Node in a dialog flow."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = "message"  # message, question, condition, action, transfer, end

    # Content
    message: str = ""
    question: str = ""
    slot_to_fill: Optional[str] = None

    # Branching
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    next_node: Optional[str] = None
    branches: Dict[str, str] = field(default_factory=dict)

    # Actions
    action: Optional[str] = None
    action_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "message": self.message,
            "question": self.question,
            "slot_to_fill": self.slot_to_fill,
            "next_node": self.next_node,
            "branches": self.branches,
            "action": self.action,
        }


@dataclass
class FlowTemplate:
    """Template for a dialog flow."""

    name: str = ""
    description: str = ""
    type: FlowType = FlowType.GREETING

    # Trigger
    trigger_intents: List[str] = field(default_factory=list)
    trigger_keywords: List[str] = field(default_factory=list)

    # Nodes
    nodes: List[FlowNode] = field(default_factory=list)
    entry_node: str = ""

    # Required slots
    required_slots: List[str] = field(default_factory=list)

    # Metadata
    is_interruptible: bool = True
    priority: int = 0


@dataclass
class GeneratedFlow:
    """Generated dialog flow."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Structure
    nodes: Dict[str, FlowNode] = field(default_factory=dict)
    entry_node_id: str = ""

    # Triggers
    trigger_intents: List[str] = field(default_factory=list)
    trigger_keywords: List[str] = field(default_factory=list)

    # Slots
    required_slots: List[str] = field(default_factory=list)

    # Settings
    is_interruptible: bool = True
    priority: int = 0

    # Source
    generated_from: str = ""  # template name or "custom"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "entry_node_id": self.entry_node_id,
            "trigger_intents": self.trigger_intents,
            "required_slots": self.required_slots,
        }


# Common flow templates
COMMON_FLOWS: Dict[FlowType, FlowTemplate] = {
    FlowType.GREETING: FlowTemplate(
        name="Standard Greeting",
        description="Greet callers and determine their needs",
        type=FlowType.GREETING,
        trigger_intents=["greeting", "start"],
        nodes=[
            FlowNode(
                id="greet",
                type="message",
                message="Hello, thank you for calling {business_name}. My name is {agent_name}. How can I help you today?",
                next_node="listen",
            ),
            FlowNode(
                id="listen",
                type="question",
                question="",  # Open-ended, waiting for response
                slot_to_fill="initial_intent",
            ),
        ],
        entry_node="greet",
    ),

    FlowType.AFTER_HOURS: FlowTemplate(
        name="After Hours",
        description="Handle calls outside business hours",
        type=FlowType.AFTER_HOURS,
        trigger_intents=["greeting"],
        nodes=[
            FlowNode(
                id="after_hours_greeting",
                type="message",
                message="Thank you for calling {business_name}. We're currently closed. Our regular hours are {business_hours}.",
                next_node="offer_options",
            ),
            FlowNode(
                id="offer_options",
                type="question",
                question="Would you like to leave a message, or is this an emergency?",
                slot_to_fill="after_hours_choice",
                branches={
                    "message": "take_message",
                    "emergency": "emergency_transfer",
                    "default": "provide_hours",
                },
            ),
            FlowNode(
                id="take_message",
                type="action",
                action="collect_voicemail",
                message="Please leave your name, number, and a brief message, and someone will return your call during business hours.",
            ),
            FlowNode(
                id="emergency_transfer",
                type="transfer",
                message="I'll connect you to our emergency line now.",
                action="transfer",
                action_params={"destination": "emergency"},
            ),
            FlowNode(
                id="provide_hours",
                type="message",
                message="You can reach us during our regular hours: {business_hours}. Is there anything else I can help with?",
            ),
        ],
        entry_node="after_hours_greeting",
    ),

    FlowType.FAQ: FlowTemplate(
        name="FAQ Handler",
        description="Answer frequently asked questions",
        type=FlowType.FAQ,
        trigger_intents=["question", "faq"],
        nodes=[
            FlowNode(
                id="identify_question",
                type="action",
                action="match_faq",
                next_node="answer_question",
            ),
            FlowNode(
                id="answer_question",
                type="message",
                message="{faq_answer}",
                next_node="check_followup",
            ),
            FlowNode(
                id="check_followup",
                type="question",
                question="Does that answer your question? Is there anything else I can help with?",
                branches={
                    "yes": "closing",
                    "no": "clarify",
                    "more_questions": "identify_question",
                },
            ),
            FlowNode(
                id="clarify",
                type="message",
                message="I'm sorry, let me try to help with that differently. Can you tell me more about what you're looking for?",
                next_node="identify_question",
            ),
            FlowNode(
                id="closing",
                type="message",
                message="Great! Is there anything else I can help you with today?",
            ),
        ],
        entry_node="identify_question",
    ),

    FlowType.SCHEDULING: FlowTemplate(
        name="Appointment Scheduling",
        description="Schedule appointments or services",
        type=FlowType.SCHEDULING,
        trigger_intents=["schedule_appointment", "book", "make_appointment"],
        trigger_keywords=["appointment", "schedule", "book", "available"],
        required_slots=["service_type", "preferred_date", "preferred_time", "contact_name", "contact_phone"],
        nodes=[
            FlowNode(
                id="confirm_scheduling",
                type="message",
                message="I'd be happy to help you schedule an appointment.",
                next_node="get_service",
            ),
            FlowNode(
                id="get_service",
                type="question",
                question="What type of service are you looking to schedule?",
                slot_to_fill="service_type",
                next_node="get_date",
            ),
            FlowNode(
                id="get_date",
                type="question",
                question="What day works best for you?",
                slot_to_fill="preferred_date",
                next_node="get_time",
            ),
            FlowNode(
                id="get_time",
                type="question",
                question="And what time would you prefer? Morning or afternoon?",
                slot_to_fill="preferred_time",
                next_node="get_name",
            ),
            FlowNode(
                id="get_name",
                type="question",
                question="May I have your name for the appointment?",
                slot_to_fill="contact_name",
                next_node="get_phone",
            ),
            FlowNode(
                id="get_phone",
                type="question",
                question="And a phone number where we can reach you?",
                slot_to_fill="contact_phone",
                next_node="confirm_appointment",
            ),
            FlowNode(
                id="confirm_appointment",
                type="message",
                message="Let me confirm: You'd like to schedule {service_type} on {preferred_date} at {preferred_time}. The appointment is for {contact_name}, and we can reach you at {contact_phone}. Is that all correct?",
                next_node="finalize",
            ),
            FlowNode(
                id="finalize",
                type="action",
                action="create_appointment",
                next_node="appointment_created",
            ),
            FlowNode(
                id="appointment_created",
                type="message",
                message="Your appointment has been scheduled. You'll receive a confirmation. Is there anything else I can help with?",
            ),
        ],
        entry_node="confirm_scheduling",
    ),

    FlowType.QUOTE_REQUEST: FlowTemplate(
        name="Quote Request",
        description="Handle quote and estimate requests",
        type=FlowType.QUOTE_REQUEST,
        trigger_intents=["get_quote", "estimate", "pricing"],
        trigger_keywords=["quote", "estimate", "price", "cost", "how much"],
        required_slots=["service_needed", "contact_info"],
        nodes=[
            FlowNode(
                id="start_quote",
                type="message",
                message="I'd be happy to help you get a quote.",
                next_node="get_service_details",
            ),
            FlowNode(
                id="get_service_details",
                type="question",
                question="Can you tell me what service or work you're looking for?",
                slot_to_fill="service_needed",
                next_node="get_details",
            ),
            FlowNode(
                id="get_details",
                type="question",
                question="Can you describe the specifics of what you need?",
                slot_to_fill="project_details",
                next_node="get_contact",
            ),
            FlowNode(
                id="get_contact",
                type="question",
                question="What's the best way to reach you with the quote? Can I get your name and phone number?",
                slot_to_fill="contact_info",
                next_node="confirm_quote_request",
            ),
            FlowNode(
                id="confirm_quote_request",
                type="message",
                message="I've got all the information I need. We'll prepare a quote for {service_needed} and contact you at {contact_info}. Is there anything else you'd like to add?",
                next_node="submit_quote",
            ),
            FlowNode(
                id="submit_quote",
                type="action",
                action="create_quote_request",
                next_node="quote_submitted",
            ),
            FlowNode(
                id="quote_submitted",
                type="message",
                message="Your quote request has been submitted. Someone will be in touch shortly. Is there anything else I can help with?",
            ),
        ],
        entry_node="start_quote",
    ),

    FlowType.COMPLAINT: FlowTemplate(
        name="Complaint Handler",
        description="Handle customer complaints with empathy",
        type=FlowType.COMPLAINT,
        trigger_intents=["complaint", "problem", "issue", "frustrated"],
        trigger_keywords=["complaint", "problem", "issue", "frustrated", "upset", "disappointed", "angry"],
        required_slots=["complaint_description", "customer_name", "customer_contact"],
        nodes=[
            FlowNode(
                id="acknowledge",
                type="message",
                message="I'm sorry to hear you're experiencing an issue. I want to make sure we address this properly. Can you tell me what happened?",
                next_node="get_complaint",
            ),
            FlowNode(
                id="get_complaint",
                type="question",
                question="",
                slot_to_fill="complaint_description",
                next_node="empathize",
            ),
            FlowNode(
                id="empathize",
                type="message",
                message="I completely understand how frustrating that must be, and I apologize for this experience. Let me make sure we get this resolved for you.",
                next_node="get_customer_info",
            ),
            FlowNode(
                id="get_customer_info",
                type="question",
                question="Can I get your name and the best number to reach you?",
                slot_to_fill="customer_name",
                next_node="resolution_options",
            ),
            FlowNode(
                id="resolution_options",
                type="question",
                question="Would you like me to have a manager call you back, or would you prefer to speak with someone right now?",
                slot_to_fill="resolution_preference",
                branches={
                    "callback": "schedule_callback",
                    "now": "transfer_to_manager",
                },
            ),
            FlowNode(
                id="schedule_callback",
                type="message",
                message="I'll make sure a manager calls you back as soon as possible. Is there a particular time that works best?",
                next_node="confirm_callback",
            ),
            FlowNode(
                id="transfer_to_manager",
                type="transfer",
                message="I'll connect you with a manager right now. Please hold for just a moment.",
                action="transfer",
                action_params={"destination": "manager"},
            ),
            FlowNode(
                id="confirm_callback",
                type="message",
                message="We will call you back. We truly value your business and want to make this right. Is there anything else I can help with in the meantime?",
            ),
        ],
        entry_node="acknowledge",
        is_interruptible=False,
        priority=10,  # High priority
    ),

    FlowType.TRANSFER: FlowTemplate(
        name="Transfer Request",
        description="Handle requests to speak with someone specific",
        type=FlowType.TRANSFER,
        trigger_intents=["transfer", "speak_to_human", "speak_to_person"],
        trigger_keywords=["transfer", "speak to", "talk to", "real person", "human", "someone"],
        nodes=[
            FlowNode(
                id="understand_transfer",
                type="question",
                question="I'd be happy to connect you with someone. Is there a specific person or department you'd like to speak with?",
                slot_to_fill="transfer_target",
                branches={
                    "specific_person": "lookup_person",
                    "department": "lookup_department",
                    "anyone": "general_transfer",
                },
            ),
            FlowNode(
                id="lookup_person",
                type="action",
                action="find_team_member",
                next_node="transfer_call",
            ),
            FlowNode(
                id="lookup_department",
                type="action",
                action="get_department_number",
                next_node="transfer_call",
            ),
            FlowNode(
                id="general_transfer",
                type="message",
                message="I'll connect you with the next available representative.",
                next_node="transfer_call",
            ),
            FlowNode(
                id="transfer_call",
                type="transfer",
                message="I'm transferring you now. Please hold.",
                action="transfer",
            ),
        ],
        entry_node="understand_transfer",
    ),

    FlowType.CLOSING: FlowTemplate(
        name="Call Closing",
        description="Close the conversation gracefully",
        type=FlowType.CLOSING,
        trigger_intents=["goodbye", "done", "that's all"],
        trigger_keywords=["goodbye", "bye", "thanks", "that's all", "nothing else"],
        nodes=[
            FlowNode(
                id="check_complete",
                type="question",
                question="Is there anything else I can help you with today?",
                branches={
                    "yes": "continue",
                    "no": "close",
                },
            ),
            FlowNode(
                id="continue",
                type="message",
                message="Of course, what else can I help with?",
            ),
            FlowNode(
                id="close",
                type="message",
                message="Thank you for calling {business_name}. Have a wonderful {time_of_day}!",
                next_node="end",
            ),
            FlowNode(
                id="end",
                type="end",
            ),
        ],
        entry_node="check_complete",
    ),

    FlowType.LEAD_QUALIFICATION: FlowTemplate(
        name="Lead Qualification",
        description="Qualify incoming leads",
        type=FlowType.LEAD_QUALIFICATION,
        trigger_intents=["interested", "inquiry", "learn_more"],
        required_slots=["lead_name", "lead_phone", "lead_interest", "lead_timeline"],
        nodes=[
            FlowNode(
                id="start_qualification",
                type="message",
                message="I'd love to learn more about what you're looking for so we can best help you.",
                next_node="get_interest",
            ),
            FlowNode(
                id="get_interest",
                type="question",
                question="What specifically are you interested in?",
                slot_to_fill="lead_interest",
                next_node="get_timeline",
            ),
            FlowNode(
                id="get_timeline",
                type="question",
                question="And when are you looking to move forward with this?",
                slot_to_fill="lead_timeline",
                next_node="get_lead_info",
            ),
            FlowNode(
                id="get_lead_info",
                type="question",
                question="Can I get your name and the best number to reach you?",
                slot_to_fill="lead_contact",
                next_node="qualify_lead",
            ),
            FlowNode(
                id="qualify_lead",
                type="action",
                action="score_lead",
                next_node="handle_qualified",
            ),
            FlowNode(
                id="handle_qualified",
                type="message",
                message="Thank you for that information. Based on what you've shared, I think we can definitely help. A member of our team will reach out to you shortly. Is there anything else you'd like to know in the meantime?",
            ),
        ],
        entry_node="start_qualification",
    ),
}


class FlowLibrary:
    """Library of flow templates."""

    def __init__(self):
        self._templates: Dict[str, FlowTemplate] = {}
        self._load_common_flows()

    def _load_common_flows(self):
        """Load common flow templates."""
        for flow_type, template in COMMON_FLOWS.items():
            self._templates[template.name] = template

    def get_template(self, name: str) -> Optional[FlowTemplate]:
        """Get template by name."""
        return self._templates.get(name)

    def get_templates_by_type(self, flow_type: FlowType) -> List[FlowTemplate]:
        """Get all templates of a type."""
        return [t for t in self._templates.values() if t.type == flow_type]

    def add_template(self, template: FlowTemplate) -> None:
        """Add a custom template."""
        self._templates[template.name] = template

    def list_templates(self) -> List[str]:
        """List all template names."""
        return list(self._templates.keys())


class FlowGenerator:
    """
    Generates dialog flows based on business information.

    Creates complete conversation flows for scheduling, inquiries,
    complaints, and other common scenarios.
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        enable_llm_generation: bool = False,
    ):
        """
        Initialize flow generator.

        Args:
            llm_provider: Optional LLM for enhanced generation
            enable_llm_generation: Whether to use LLM generation
        """
        self._llm_provider = llm_provider
        self._enable_llm = enable_llm_generation
        self._library = FlowLibrary()

    async def generate(
        self,
        business_info: BusinessInfo,
        requested_flows: Optional[List[FlowType]] = None,
    ) -> List[GeneratedFlow]:
        """
        Generate flows for the business.

        Args:
            business_info: Business information
            requested_flows: Specific flows to generate

        Returns:
            List of generated flows
        """
        flows = []

        # Determine which flows to generate
        if requested_flows:
            flow_types = requested_flows
        else:
            flow_types = self._determine_needed_flows(business_info)

        # Generate each flow type
        for flow_type in flow_types:
            template = COMMON_FLOWS.get(flow_type)
            if template:
                flow = self._generate_from_template(template, business_info)
                flows.append(flow)

        # Generate service-specific flows
        service_flows = self._generate_service_flows(business_info)
        flows.extend(service_flows)

        # Generate FAQ flows
        faq_flows = self._generate_faq_flows(business_info)
        flows.extend(faq_flows)

        # Industry-specific flows
        industry_flows = self._generate_industry_flows(business_info)
        flows.extend(industry_flows)

        return flows

    def _determine_needed_flows(self, business_info: BusinessInfo) -> List[FlowType]:
        """Determine which flows are needed based on business info."""
        flows = [
            FlowType.GREETING,
            FlowType.CLOSING,
            FlowType.COMPLAINT,
            FlowType.TRANSFER,
        ]

        # Add after hours if hours are defined
        if business_info.hours:
            flows.append(FlowType.AFTER_HOURS)

        # Add scheduling if services require booking
        if any(s.booking_required for s in business_info.services):
            flows.append(FlowType.SCHEDULING)

        # Add quote if services have estimates
        if any(s.price_type == "estimate" or s.price_type == "quote" for s in business_info.services):
            flows.append(FlowType.QUOTE_REQUEST)

        # Add FAQ if FAQs exist
        if business_info.faqs:
            flows.append(FlowType.FAQ)

        # Industry-specific flows
        if business_info.category in [BusinessCategory.REAL_ESTATE, BusinessCategory.AUTO_DEALERSHIP]:
            flows.append(FlowType.LEAD_QUALIFICATION)

        return flows

    def _generate_from_template(
        self,
        template: FlowTemplate,
        business_info: BusinessInfo,
    ) -> GeneratedFlow:
        """Generate a flow from a template."""
        flow = GeneratedFlow(
            name=f"{business_info.name} - {template.name}",
            description=template.description,
            trigger_intents=list(template.trigger_intents),
            trigger_keywords=list(template.trigger_keywords),
            required_slots=list(template.required_slots),
            is_interruptible=template.is_interruptible,
            priority=template.priority,
            generated_from=template.name,
        )

        # Copy and customize nodes
        for node in template.nodes:
            customized_node = FlowNode(
                id=node.id,
                type=node.type,
                message=self._customize_message(node.message, business_info),
                question=self._customize_message(node.question, business_info),
                slot_to_fill=node.slot_to_fill,
                conditions=list(node.conditions),
                next_node=node.next_node,
                branches=dict(node.branches),
                action=node.action,
                action_params=dict(node.action_params),
            )
            flow.nodes[customized_node.id] = customized_node

        flow.entry_node_id = template.entry_node

        return flow

    def _customize_message(self, message: str, business_info: BusinessInfo) -> str:
        """Customize message with business information."""
        if not message:
            return message

        replacements = {
            "{business_name}": business_info.name,
            "{business_hours}": self._format_hours(business_info),
            "{phone}": business_info.contact.phone or "",
            "{email}": business_info.contact.email or "",
            "{address}": business_info.contact.address or "",
            "{website}": business_info.contact.website or "",
        }

        for placeholder, value in replacements.items():
            message = message.replace(placeholder, value)

        return message

    def _format_hours(self, business_info: BusinessInfo) -> str:
        """Format business hours for speech."""
        if not business_info.hours:
            return "regular business hours"

        # Simplified format for voice
        return "Monday through Friday, 9 AM to 5 PM"

    def _generate_service_flows(self, business_info: BusinessInfo) -> List[GeneratedFlow]:
        """Generate flows for specific services."""
        flows = []

        for service in business_info.services:
            if service.booking_required:
                flow = self._generate_service_booking_flow(service, business_info)
                flows.append(flow)

        return flows

    def _generate_service_booking_flow(
        self,
        service: ServiceInfo,
        business_info: BusinessInfo,
    ) -> GeneratedFlow:
        """Generate a booking flow for a specific service."""
        flow = GeneratedFlow(
            name=f"Book {service.name}",
            description=f"Schedule {service.name}",
            trigger_intents=[f"book_{self._slugify(service.name)}", "schedule"],
            trigger_keywords=[service.name.lower(), "book", "schedule"],
            required_slots=["contact_name", "contact_phone", "preferred_date", "preferred_time"],
        )

        # Build nodes
        nodes = [
            FlowNode(
                id="start",
                type="message",
                message=f"I'd be happy to help you schedule {service.name}.",
                next_node="get_date",
            ),
            FlowNode(
                id="get_date",
                type="question",
                question="What day works best for you?",
                slot_to_fill="preferred_date",
                next_node="get_time",
            ),
            FlowNode(
                id="get_time",
                type="question",
                question="And what time would you prefer?",
                slot_to_fill="preferred_time",
                next_node="get_name",
            ),
            FlowNode(
                id="get_name",
                type="question",
                question="May I have your name?",
                slot_to_fill="contact_name",
                next_node="get_phone",
            ),
            FlowNode(
                id="get_phone",
                type="question",
                question="And a phone number where we can reach you?",
                slot_to_fill="contact_phone",
                next_node="confirm",
            ),
            FlowNode(
                id="confirm",
                type="message",
                message=f"I have you scheduled for {service.name} on {{preferred_date}} at {{preferred_time}}. We'll send a confirmation to {{contact_phone}}. Is there anything else you need?",
                next_node="end",
            ),
        ]

        for node in nodes:
            flow.nodes[node.id] = node
        flow.entry_node_id = "start"

        return flow

    def _generate_faq_flows(self, business_info: BusinessInfo) -> List[GeneratedFlow]:
        """Generate flows for FAQ handling."""
        if not business_info.faqs:
            return []

        # Create a single FAQ handler flow that routes to answers
        flow = GeneratedFlow(
            name="FAQ Handler",
            description="Answer common questions",
            trigger_intents=["question", "faq", "ask"],
        )

        # Build FAQ response nodes
        nodes = [
            FlowNode(
                id="start",
                type="action",
                action="match_faq",
                next_node="respond",
            ),
            FlowNode(
                id="respond",
                type="message",
                message="{matched_faq_answer}",
                next_node="followup",
            ),
            FlowNode(
                id="followup",
                type="question",
                question="Does that answer your question?",
                branches={
                    "yes": "closing",
                    "no": "offer_help",
                },
            ),
            FlowNode(
                id="offer_help",
                type="message",
                message="I'm sorry, let me try to help differently. Would you like me to connect you with someone who can help?",
                next_node="transfer_check",
            ),
            FlowNode(
                id="closing",
                type="message",
                message="Great! Is there anything else I can help with?",
            ),
        ]

        for node in nodes:
            flow.nodes[node.id] = node
        flow.entry_node_id = "start"

        return [flow]

    def _generate_industry_flows(self, business_info: BusinessInfo) -> List[GeneratedFlow]:
        """Generate industry-specific flows."""
        flows = []

        if business_info.category == BusinessCategory.HEALTHCARE:
            flows.extend(self._generate_healthcare_flows(business_info))
        elif business_info.category == BusinessCategory.DENTAL:
            flows.extend(self._generate_dental_flows(business_info))
        elif business_info.category == BusinessCategory.PLUMBING:
            flows.extend(self._generate_home_service_flows(business_info))
        elif business_info.category == BusinessCategory.HVAC:
            flows.extend(self._generate_home_service_flows(business_info))
        elif business_info.category == BusinessCategory.AUTO_DEALERSHIP:
            flows.extend(self._generate_auto_flows(business_info))
        elif business_info.category == BusinessCategory.RESTAURANT:
            flows.extend(self._generate_restaurant_flows(business_info))

        return flows

    def _generate_healthcare_flows(self, business_info: BusinessInfo) -> List[GeneratedFlow]:
        """Generate healthcare-specific flows."""
        flows = []

        # New patient flow
        new_patient_flow = GeneratedFlow(
            name="New Patient Registration",
            description="Handle new patient inquiries",
            trigger_intents=["new_patient", "first_time"],
            trigger_keywords=["new patient", "first time", "never been"],
            required_slots=["patient_name", "date_of_birth", "insurance_provider", "contact_phone"],
        )

        nodes = [
            FlowNode(
                id="welcome",
                type="message",
                message=f"Welcome! We're happy to have you as a new patient at {business_info.name}.",
                next_node="get_info",
            ),
            FlowNode(
                id="get_info",
                type="question",
                question="To get you set up, I'll need some information. Can I start with your name?",
                slot_to_fill="patient_name",
                next_node="get_dob",
            ),
            FlowNode(
                id="get_dob",
                type="question",
                question="And your date of birth?",
                slot_to_fill="date_of_birth",
                next_node="get_insurance",
            ),
            FlowNode(
                id="get_insurance",
                type="question",
                question="Do you have insurance? If so, who is your insurance provider?",
                slot_to_fill="insurance_provider",
                next_node="schedule_new",
            ),
            FlowNode(
                id="schedule_new",
                type="message",
                message="Thank you. Let me help you schedule your first appointment. What day works best for you?",
                next_node="collect_date",
            ),
        ]

        for node in nodes:
            new_patient_flow.nodes[node.id] = node
        new_patient_flow.entry_node_id = "welcome"
        flows.append(new_patient_flow)

        # Prescription refill flow
        refill_flow = GeneratedFlow(
            name="Prescription Refill",
            description="Handle prescription refill requests",
            trigger_intents=["prescription_refill", "refill"],
            trigger_keywords=["prescription", "refill", "medication"],
        )

        refill_nodes = [
            FlowNode(
                id="start",
                type="message",
                message="I can help you with a prescription refill request.",
                next_node="get_name",
            ),
            FlowNode(
                id="get_name",
                type="question",
                question="What is the patient's name?",
                slot_to_fill="patient_name",
                next_node="get_medication",
            ),
            FlowNode(
                id="get_medication",
                type="question",
                question="Which medication do you need refilled?",
                slot_to_fill="medication_name",
                next_node="get_pharmacy",
            ),
            FlowNode(
                id="get_pharmacy",
                type="question",
                question="Which pharmacy should we send it to?",
                slot_to_fill="pharmacy",
                next_node="submit",
            ),
            FlowNode(
                id="submit",
                type="message",
                message="I've submitted the refill request. The doctor will review it and send it to your pharmacy. Is there anything else I can help with?",
            ),
        ]

        for node in refill_nodes:
            refill_flow.nodes[node.id] = node
        refill_flow.entry_node_id = "start"
        flows.append(refill_flow)

        return flows

    def _generate_dental_flows(self, business_info: BusinessInfo) -> List[GeneratedFlow]:
        """Generate dental-specific flows."""
        flows = []

        # Emergency dental flow
        emergency_flow = GeneratedFlow(
            name="Dental Emergency",
            description="Handle dental emergencies",
            trigger_intents=["emergency", "dental_emergency"],
            trigger_keywords=["emergency", "pain", "broken tooth", "knocked out"],
            priority=10,
        )

        nodes = [
            FlowNode(
                id="assess",
                type="message",
                message="I'm sorry to hear you're experiencing a dental emergency. Let me help you.",
                next_node="get_issue",
            ),
            FlowNode(
                id="get_issue",
                type="question",
                question="Can you describe what's happening?",
                slot_to_fill="emergency_description",
                next_node="urgent_check",
            ),
            FlowNode(
                id="urgent_check",
                type="condition",
                conditions=[
                    {"check": "contains_keywords", "keywords": ["knocked out", "severe bleeding", "trauma"], "next": "immediate_help"},
                ],
                next_node="schedule_urgent",
            ),
            FlowNode(
                id="immediate_help",
                type="message",
                message="For immediate dental trauma, please keep the area clean. If a tooth was knocked out, try to keep it moist. Let me connect you with our emergency line right away.",
                next_node="transfer_emergency",
            ),
            FlowNode(
                id="schedule_urgent",
                type="message",
                message="I understand you're in discomfort. Let me get you scheduled for an urgent appointment today. What's your name?",
                next_node="collect_info",
            ),
        ]

        for node in nodes:
            emergency_flow.nodes[node.id] = node
        emergency_flow.entry_node_id = "assess"
        flows.append(emergency_flow)

        return flows

    def _generate_home_service_flows(self, business_info: BusinessInfo) -> List[GeneratedFlow]:
        """Generate flows for home service businesses."""
        flows = []

        # Emergency service flow
        emergency_flow = GeneratedFlow(
            name="Emergency Service",
            description="Handle emergency service requests",
            trigger_intents=["emergency", "urgent"],
            trigger_keywords=["emergency", "urgent", "leak", "flood", "no heat", "no ac"],
            priority=10,
        )

        service_type = "plumbing" if business_info.category == BusinessCategory.PLUMBING else "HVAC"

        nodes = [
            FlowNode(
                id="assess",
                type="message",
                message=f"I understand you have an urgent {service_type} issue. Let me help you right away.",
                next_node="get_problem",
            ),
            FlowNode(
                id="get_problem",
                type="question",
                question="What's happening?",
                slot_to_fill="problem_description",
                next_node="get_address",
            ),
            FlowNode(
                id="get_address",
                type="question",
                question="What's your address?",
                slot_to_fill="service_address",
                next_node="get_contact",
            ),
            FlowNode(
                id="get_contact",
                type="question",
                question="And a phone number where the technician can reach you?",
                slot_to_fill="contact_phone",
                next_node="dispatch",
            ),
            FlowNode(
                id="dispatch",
                type="action",
                action="dispatch_technician",
                next_node="confirm_dispatch",
            ),
            FlowNode(
                id="confirm_dispatch",
                type="message",
                message="I've dispatched a technician to your location. They should arrive within the hour. We'll call you with an ETA. Is there anything else you need?",
            ),
        ]

        for node in nodes:
            emergency_flow.nodes[node.id] = node
        emergency_flow.entry_node_id = "assess"
        flows.append(emergency_flow)

        return flows

    def _generate_auto_flows(self, business_info: BusinessInfo) -> List[GeneratedFlow]:
        """Generate auto dealership flows."""
        flows = []

        # Test drive scheduling flow
        test_drive_flow = GeneratedFlow(
            name="Schedule Test Drive",
            description="Schedule a vehicle test drive",
            trigger_intents=["test_drive", "schedule_test_drive"],
            trigger_keywords=["test drive", "try", "drive"],
            required_slots=["vehicle_interest", "preferred_date", "contact_name", "contact_phone"],
        )

        nodes = [
            FlowNode(
                id="start",
                type="message",
                message="I'd be happy to help you schedule a test drive!",
                next_node="get_vehicle",
            ),
            FlowNode(
                id="get_vehicle",
                type="question",
                question="Which vehicle are you interested in test driving?",
                slot_to_fill="vehicle_interest",
                next_node="get_date",
            ),
            FlowNode(
                id="get_date",
                type="question",
                question="When would you like to come in?",
                slot_to_fill="preferred_date",
                next_node="get_name",
            ),
            FlowNode(
                id="get_name",
                type="question",
                question="May I have your name?",
                slot_to_fill="contact_name",
                next_node="get_phone",
            ),
            FlowNode(
                id="get_phone",
                type="question",
                question="And a phone number?",
                slot_to_fill="contact_phone",
                next_node="confirm",
            ),
            FlowNode(
                id="confirm",
                type="message",
                message="You're all set! We have you scheduled to test drive the {vehicle_interest} on {preferred_date}. We'll have it ready and waiting for you. Looking forward to seeing you!",
            ),
        ]

        for node in nodes:
            test_drive_flow.nodes[node.id] = node
        test_drive_flow.entry_node_id = "start"
        flows.append(test_drive_flow)

        return flows

    def _generate_restaurant_flows(self, business_info: BusinessInfo) -> List[GeneratedFlow]:
        """Generate restaurant-specific flows."""
        flows = []

        # Reservation flow
        reservation_flow = GeneratedFlow(
            name="Make Reservation",
            description="Handle restaurant reservations",
            trigger_intents=["reservation", "book_table"],
            trigger_keywords=["reservation", "table", "book", "party"],
            required_slots=["party_size", "reservation_date", "reservation_time", "guest_name", "contact_phone"],
        )

        nodes = [
            FlowNode(
                id="start",
                type="message",
                message=f"I'd be happy to help you make a reservation at {business_info.name}!",
                next_node="get_party_size",
            ),
            FlowNode(
                id="get_party_size",
                type="question",
                question="How many people will be in your party?",
                slot_to_fill="party_size",
                next_node="get_date",
            ),
            FlowNode(
                id="get_date",
                type="question",
                question="What date were you thinking?",
                slot_to_fill="reservation_date",
                next_node="get_time",
            ),
            FlowNode(
                id="get_time",
                type="question",
                question="And what time?",
                slot_to_fill="reservation_time",
                next_node="check_availability",
            ),
            FlowNode(
                id="check_availability",
                type="action",
                action="check_availability",
                next_node="get_name",
            ),
            FlowNode(
                id="get_name",
                type="question",
                question="And the name for the reservation?",
                slot_to_fill="guest_name",
                next_node="get_phone",
            ),
            FlowNode(
                id="get_phone",
                type="question",
                question="And a contact number?",
                slot_to_fill="contact_phone",
                next_node="confirm",
            ),
            FlowNode(
                id="confirm",
                type="message",
                message="You're all set! I have a table for {party_size} on {reservation_date} at {reservation_time} under the name {guest_name}. We look forward to seeing you!",
            ),
        ]

        for node in nodes:
            reservation_flow.nodes[node.id] = node
        reservation_flow.entry_node_id = "start"
        flows.append(reservation_flow)

        return flows

    def _slugify(self, text: str) -> str:
        """Convert text to slug format."""
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', '_', text)
        return text.strip('_')


__all__ = [
    "FlowGenerator",
    "FlowTemplate",
    "GeneratedFlow",
    "FlowLibrary",
    "FlowType",
    "FlowNode",
    "COMMON_FLOWS",
]
