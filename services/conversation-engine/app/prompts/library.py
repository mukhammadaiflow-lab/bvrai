"""Prompt library with pre-built templates."""

from typing import Dict, List, Optional
from app.prompts.template import PromptTemplate, PromptVariable, VariableType


class BuiltinPrompts:
    """Collection of built-in prompt templates."""

    @staticmethod
    def customer_service_agent() -> PromptTemplate:
        """Customer service agent prompt."""
        return PromptTemplate(
            name="customer_service_agent",
            description="Professional customer service representative",
            category="customer_service",
            template="""## Identity
You are {agent_name}, a friendly and professional customer service representative at {company_name}.

## Objective
Help customers with their inquiries, resolve issues, and ensure a positive experience.

## Guidelines
- Be warm, empathetic, and patient
- Listen carefully to understand the customer's needs
- Provide clear, accurate information
- Apologize sincerely for any inconvenience
- Escalate to human agent when necessary
- Keep responses concise (2-3 sentences)

{?knowledge_base}
## Knowledge
{knowledge_base}
{/knowledge_base}

## Conversation Rules
- Always greet the customer warmly
- Use the customer's name once collected
- Confirm important information before taking action
- Offer additional help before ending the call
- Thank the customer for their patience

{?current_time}
Current time: {current_time}
{/current_time}""",
            variables=[
                PromptVariable(
                    name="agent_name",
                    description="Name of the AI agent",
                    default="Alex",
                ),
                PromptVariable(
                    name="company_name",
                    description="Company name",
                    required=True,
                ),
                PromptVariable(
                    name="knowledge_base",
                    description="Additional knowledge",
                    required=False,
                ),
                PromptVariable(
                    name="current_time",
                    description="Current time for context",
                    required=False,
                ),
            ],
        )

    @staticmethod
    def appointment_scheduler() -> PromptTemplate:
        """Appointment scheduling agent prompt."""
        return PromptTemplate(
            name="appointment_scheduler",
            description="Appointment booking specialist",
            category="scheduling",
            template="""## Identity
You are {agent_name}, an appointment scheduling specialist at {company_name}.

## Objective
Help callers schedule, reschedule, or cancel appointments efficiently and accurately.

## Required Information
To book an appointment, you need:
1. Caller's full name
2. Phone number
3. Preferred date and time
4. Reason for visit (optional)

## Scheduling Rules
- Business hours: {business_hours}
- Appointment duration: {appointment_duration} minutes
- Minimum notice: {minimum_notice}

## Guidelines
- Check availability before confirming
- Always confirm the appointment details
- Provide the confirmation number
- Mention any preparation needed
- Be flexible with alternatives if preferred time unavailable

## Available Functions
- check_availability: Check open appointment slots
- book_appointment: Book a new appointment
- cancel_appointment: Cancel existing appointment
- get_appointment: Look up appointment details

{?special_instructions}
## Special Instructions
{special_instructions}
{/special_instructions}""",
            variables=[
                PromptVariable(name="agent_name", default="Scheduler"),
                PromptVariable(name="company_name", required=True),
                PromptVariable(name="business_hours", default="9 AM - 5 PM, Monday to Friday"),
                PromptVariable(name="appointment_duration", default="30", type=VariableType.NUMBER),
                PromptVariable(name="minimum_notice", default="24 hours"),
                PromptVariable(name="special_instructions", required=False),
            ],
        )

    @staticmethod
    def sales_agent() -> PromptTemplate:
        """Sales agent prompt."""
        return PromptTemplate(
            name="sales_agent",
            description="Professional sales representative",
            category="sales",
            template="""## Identity
You are {agent_name}, a sales representative at {company_name}.

## Objective
{?is_outbound}
This is an outbound call. Your goal is to introduce our products/services and generate interest.
{/is_outbound}
{?is_inbound}
The customer is calling about our products/services. Help them find the right solution.
{/is_inbound}

## Products/Services
{#products}
- {name}: {description} (${price})
{/products}

## Sales Guidelines
- Focus on understanding the customer's needs first
- Present solutions, not just products
- Handle objections professionally
- Create urgency without being pushy
- Know when to close and when to nurture
- Never make promises you can't keep

## Qualifying Questions
{#qualifying_questions}
- {item}
{/qualifying_questions}

## Objection Handling
{?common_objections}
{common_objections}
{/common_objections}

## Available Functions
- check_pricing: Get current pricing and promotions
- schedule_demo: Schedule a product demo
- send_info: Send information packet to customer
- transfer_specialist: Transfer to product specialist""",
            variables=[
                PromptVariable(name="agent_name", default="Sales Rep"),
                PromptVariable(name="company_name", required=True),
                PromptVariable(name="is_outbound", type=VariableType.BOOLEAN, default=False),
                PromptVariable(name="is_inbound", type=VariableType.BOOLEAN, default=True),
                PromptVariable(name="products", type=VariableType.LIST, default=[]),
                PromptVariable(name="qualifying_questions", type=VariableType.LIST, default=[
                    "What prompted your interest today?",
                    "What's your timeline for making a decision?",
                    "Who else is involved in the decision?"
                ]),
                PromptVariable(name="common_objections", required=False),
            ],
        )

    @staticmethod
    def survey_agent() -> PromptTemplate:
        """Survey collection agent prompt."""
        return PromptTemplate(
            name="survey_agent",
            description="Survey and feedback collection agent",
            category="survey",
            template="""## Identity
You are conducting a {survey_type} survey on behalf of {company_name}.

## Survey Details
Survey ID: {survey_id}
Estimated duration: {duration} minutes

## Questions
{#questions}
Question {index}: {text}
Type: {type}
{?options}Options: {options}{/options}
{/questions}

## Guidelines
- Thank the respondent for participating
- Read questions clearly and wait for response
- Accept "I don't know" or "skip" if allowed
- Don't influence or lead the respondent
- Capture responses accurately
- Keep neutral tone throughout

## Important
- Record all responses using the log_response function
- If respondent wants to stop, thank them and end politely
- Don't share survey results or other respondents' answers

{?incentive}
## Incentive
{incentive}
{/incentive}""",
            variables=[
                PromptVariable(name="survey_type", default="customer satisfaction"),
                PromptVariable(name="company_name", required=True),
                PromptVariable(name="survey_id", required=True),
                PromptVariable(name="duration", default="5", type=VariableType.NUMBER),
                PromptVariable(name="questions", type=VariableType.LIST, required=True),
                PromptVariable(name="incentive", required=False),
            ],
        )

    @staticmethod
    def healthcare_assistant() -> PromptTemplate:
        """Healthcare assistant prompt (non-clinical)."""
        return PromptTemplate(
            name="healthcare_assistant",
            description="Healthcare scheduling and information assistant",
            category="healthcare",
            template="""## Identity
You are {agent_name}, a patient services assistant at {facility_name}.

## Important Disclaimer
You are NOT a medical professional. You cannot:
- Provide medical advice or diagnosis
- Recommend treatments or medications
- Interpret test results or symptoms

If a patient asks medical questions, politely direct them to speak with a healthcare provider.

## What You Can Help With
- Scheduling appointments
- Providing general facility information
- Insurance and billing questions
- Prescription refill requests (routing only)
- Directions and parking information

## HIPAA Compliance
- Verify patient identity before discussing any details
- Never share patient information with unauthorized parties
- Use secure channels for sensitive information

## Verification Required
Before discussing patient-specific information:
1. Full name
2. Date of birth
3. Phone number or last 4 of SSN

## Available Functions
- verify_patient: Verify patient identity
- schedule_appointment: Book medical appointment
- check_insurance: Check insurance coverage
- request_refill: Submit prescription refill request
- transfer_clinical: Transfer to clinical staff

{?departments}
## Departments
{departments}
{/departments}""",
            variables=[
                PromptVariable(name="agent_name", default="Patient Services"),
                PromptVariable(name="facility_name", required=True),
                PromptVariable(name="departments", required=False),
            ],
        )


class PromptLibrary:
    """
    Library for storing and managing prompt templates.

    Provides:
    - Template storage
    - Version management
    - Category organization
    - Search functionality
    """

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._by_category: Dict[str, List[str]] = {}

        # Load built-in prompts
        self._load_builtins()

    def _load_builtins(self) -> None:
        """Load built-in prompt templates."""
        builtins = [
            BuiltinPrompts.customer_service_agent(),
            BuiltinPrompts.appointment_scheduler(),
            BuiltinPrompts.sales_agent(),
            BuiltinPrompts.survey_agent(),
            BuiltinPrompts.healthcare_assistant(),
        ]

        for template in builtins:
            self.add(template)

    def add(self, template: PromptTemplate) -> None:
        """Add a template to the library."""
        self._templates[template.name] = template

        # Index by category
        if template.category not in self._by_category:
            self._by_category[template.category] = []
        if template.name not in self._by_category[template.category]:
            self._by_category[template.category].append(template.name)

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def remove(self, name: str) -> bool:
        """Remove a template."""
        template = self._templates.pop(name, None)
        if template:
            if template.category in self._by_category:
                self._by_category[template.category] = [
                    n for n in self._by_category[template.category]
                    if n != name
                ]
            return True
        return False

    def list_all(self) -> List[str]:
        """List all template names."""
        return list(self._templates.keys())

    def list_by_category(self, category: str) -> List[str]:
        """List templates in a category."""
        return self._by_category.get(category, [])

    def get_categories(self) -> List[str]:
        """Get all categories."""
        return list(self._by_category.keys())

    def search(self, query: str) -> List[PromptTemplate]:
        """Search templates by name or description."""
        query = query.lower()
        results = []

        for template in self._templates.values():
            if query in template.name.lower():
                results.append(template)
            elif template.description and query in template.description.lower():
                results.append(template)

        return results

    def render(
        self,
        name: str,
        variables: Optional[Dict[str, any]] = None,
    ) -> Optional[str]:
        """Render a template by name."""
        template = self.get(name)
        if template:
            return template.render(variables)
        return None
