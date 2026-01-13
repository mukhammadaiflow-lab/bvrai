"""
Prompt Generator Module

This module generates system prompts and prompt templates for AI voice agents
based on business information, persona, and conversation requirements.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from .base import (
    BusinessInfo,
    BusinessCategory,
    AgentConfig,
    AgentPersona,
    BehaviorConfig,
    ComplianceConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """Context for prompt generation."""

    # Business context
    business_name: str = ""
    business_description: str = ""
    industry: str = ""

    # Agent context
    agent_name: str = ""
    agent_role: str = ""
    personality_traits: List[str] = field(default_factory=list)

    # Operational context
    current_time: Optional[datetime] = None
    is_business_hours: bool = True
    caller_name: Optional[str] = None

    # Knowledge context
    available_services: List[str] = field(default_factory=list)
    available_products: List[str] = field(default_factory=list)

    # Session context
    conversation_history: str = ""
    detected_intent: Optional[str] = None
    collected_slots: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """Template for generating prompts."""

    name: str = ""
    template: str = ""
    variables: List[str] = field(default_factory=list)
    category: str = "general"

    def render(self, context: Dict[str, Any]) -> str:
        """Render template with context."""
        result = self.template
        for var in self.variables:
            placeholder = f"{{{var}}}"
            value = context.get(var, "")
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            result = result.replace(placeholder, str(value))
        return result


@dataclass
class SystemPrompt:
    """Complete system prompt for an agent."""

    # Core prompt
    base_prompt: str = ""

    # Sections
    identity_section: str = ""
    knowledge_section: str = ""
    behavior_section: str = ""
    constraints_section: str = ""
    examples_section: str = ""

    # Metadata
    version: str = "1.0"
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def compile(self) -> str:
        """Compile full system prompt."""
        sections = [
            self.base_prompt,
            self.identity_section,
            self.knowledge_section,
            self.behavior_section,
            self.constraints_section,
            self.examples_section,
        ]

        return "\n\n".join(s for s in sections if s.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compiled": self.compile(),
            "version": self.version,
            "generated_at": self.generated_at.isoformat(),
        }


class PromptBuilder:
    """Builder for constructing system prompts."""

    def __init__(self):
        self._sections: Dict[str, str] = {}
        self._variables: Dict[str, Any] = {}

    def add_section(self, name: str, content: str) -> "PromptBuilder":
        """Add a section to the prompt."""
        self._sections[name] = content
        return self

    def set_variable(self, name: str, value: Any) -> "PromptBuilder":
        """Set a template variable."""
        self._variables[name] = value
        return self

    def build(self) -> str:
        """Build the final prompt."""
        sections = []

        # Order sections
        section_order = [
            "identity",
            "role",
            "knowledge",
            "behavior",
            "communication",
            "constraints",
            "capabilities",
            "examples",
        ]

        for section_name in section_order:
            if section_name in self._sections:
                sections.append(self._sections[section_name])

        # Add any remaining sections
        for name, content in self._sections.items():
            if name not in section_order:
                sections.append(content)

        # Compile and substitute variables
        result = "\n\n".join(sections)

        for var_name, var_value in self._variables.items():
            placeholder = f"{{{var_name}}}"
            if isinstance(var_value, list):
                var_value = ", ".join(str(v) for v in var_value)
            result = result.replace(placeholder, str(var_value))

        return result

    def reset(self) -> "PromptBuilder":
        """Reset the builder."""
        self._sections.clear()
        self._variables.clear()
        return self


class PromptGenerator:
    """
    Generates system prompts for AI voice agents.

    Creates comprehensive, industry-appropriate prompts that define
    agent behavior, knowledge, and communication style.
    """

    # Industry-specific prompt additions
    INDUSTRY_PROMPTS = {
        BusinessCategory.HEALTHCARE: """
## Healthcare-Specific Guidelines
- Always maintain patient confidentiality and HIPAA compliance
- Never provide medical advice or diagnoses
- For medical emergencies, immediately advise calling 911
- Collect insurance information only when appropriate
- Be sensitive to health-related concerns and show empathy
- Refer clinical questions to healthcare providers
""",
        BusinessCategory.DENTAL: """
## Dental Practice Guidelines
- Maintain patient confidentiality
- Never provide dental advice or diagnoses
- For dental emergencies, offer to schedule urgent appointments
- Be reassuring about dental procedures when asked general questions
- Collect insurance information when scheduling
""",
        BusinessCategory.LEGAL: """
## Legal Practice Guidelines
- Maintain strict client confidentiality
- Never provide legal advice or opinions
- Clarify that you're an assistant, not an attorney
- For urgent legal matters, offer to connect with an attorney
- Collect case information only as needed for scheduling
""",
        BusinessCategory.INSURANCE: """
## Insurance Guidelines
- Never guarantee coverage or claim outcomes
- Clarify that coverage depends on policy terms
- Be helpful with general policy questions
- Direct complex coverage questions to licensed agents
- Collect information carefully for claim filing
""",
        BusinessCategory.PLUMBING: """
## Plumbing Service Guidelines
- Prioritize emergency calls appropriately
- Ask about the nature and severity of the issue
- Collect address and contact information for dispatch
- Provide realistic service windows
- For gas-related emergencies, advise calling gas company first
""",
        BusinessCategory.HVAC: """
## HVAC Service Guidelines
- Prioritize no-heat/no-cooling emergencies appropriately
- Ask about system type and issue symptoms
- Inquire about home comfort level (temperature)
- Collect address and contact for scheduling
- Mention maintenance plans when appropriate
""",
        BusinessCategory.AUTO_DEALERSHIP: """
## Auto Dealership Guidelines
- Be enthusiastic but not pushy about vehicles
- Have current inventory knowledge available
- Assist with test drive scheduling
- Provide general financing information without promises
- Connect serious buyers with sales representatives
""",
        BusinessCategory.RESTAURANT: """
## Restaurant Guidelines
- Be warm and welcoming
- Know current hours, specials, and menu highlights
- Handle reservations efficiently
- Be knowledgeable about dietary accommodations
- Know about private events and catering options
""",
        BusinessCategory.REAL_ESTATE: """
## Real Estate Guidelines
- Be helpful for both buyers and sellers
- Assist with scheduling property showings
- Provide general market information
- Connect interested parties with appropriate agents
- Never make promises about property values or offers
""",
    }

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        enable_llm_generation: bool = False,
    ):
        """
        Initialize prompt generator.

        Args:
            llm_provider: Optional LLM for enhanced generation
            enable_llm_generation: Whether to use LLM generation
        """
        self._llm_provider = llm_provider
        self._enable_llm = enable_llm_generation
        self._builder = PromptBuilder()

    async def generate(
        self,
        business_info: BusinessInfo,
        persona: AgentPersona,
        behavior: Optional[BehaviorConfig] = None,
        compliance: Optional[ComplianceConfig] = None,
    ) -> SystemPrompt:
        """
        Generate a complete system prompt.

        Args:
            business_info: Business information
            persona: Agent persona
            behavior: Behavior configuration
            compliance: Compliance configuration

        Returns:
            Generated system prompt
        """
        behavior = behavior or BehaviorConfig()
        compliance = compliance or ComplianceConfig()

        prompt = SystemPrompt()

        # Generate base prompt
        prompt.base_prompt = self._generate_base_prompt(business_info, persona)

        # Generate identity section
        prompt.identity_section = self._generate_identity_section(persona, business_info)

        # Generate knowledge section
        prompt.knowledge_section = self._generate_knowledge_section(business_info)

        # Generate behavior section
        prompt.behavior_section = self._generate_behavior_section(persona, behavior)

        # Generate constraints section
        prompt.constraints_section = self._generate_constraints_section(
            business_info, compliance
        )

        # Generate examples section
        prompt.examples_section = self._generate_examples_section(business_info, persona)

        # Enhance with LLM if enabled
        if self._enable_llm and self._llm_provider:
            prompt = await self._enhance_with_llm(prompt, business_info, persona)

        return prompt

    def _generate_base_prompt(
        self,
        business_info: BusinessInfo,
        persona: AgentPersona,
    ) -> str:
        """Generate the base system prompt."""
        return f"""You are {persona.name}, an AI voice assistant for {business_info.name}.

{business_info.name} is a {self._get_industry_description(business_info.category)} that {business_info.description}

Your role is to provide excellent phone support to callers, helping them with inquiries, scheduling, and general information about {business_info.name}.

You are speaking on the phone with a caller. Keep your responses conversational and natural - this is a voice conversation, not text. Speak clearly and at a natural pace."""

    def _generate_identity_section(
        self,
        persona: AgentPersona,
        business_info: BusinessInfo,
    ) -> str:
        """Generate the identity section."""
        traits_str = ", ".join(persona.personality_traits)

        return f"""## Your Identity

**Name:** {persona.name}
**Role:** {persona.role}

**Personality:** You are {traits_str}. {persona.background_story or ''}

**Communication Style:** Your communication style is {persona.communication_style}.
- Use natural, conversational language
- Be warm and engaging while maintaining professionalism
- Listen actively and acknowledge what callers say
- Use the caller's name when appropriate

**Voice:** Speak in a natural, {persona.communication_style} manner that reflects your personality."""

    def _generate_knowledge_section(self, business_info: BusinessInfo) -> str:
        """Generate the knowledge section."""
        sections = ["## Your Knowledge\n"]

        # Business overview
        sections.append(f"**About {business_info.name}:**")
        if business_info.about_us:
            sections.append(business_info.about_us)
        else:
            sections.append(business_info.description)

        # Services
        if business_info.services:
            sections.append("\n**Services Offered:**")
            for service in business_info.services[:10]:
                price_info = ""
                if service.price:
                    price_info = f" - ${service.price}"
                elif service.price_range:
                    price_info = f" - {service.price_range}"
                sections.append(f"- {service.name}: {service.description}{price_info}")

        # Products
        if business_info.products:
            sections.append("\n**Products:**")
            for product in business_info.products[:10]:
                sections.append(f"- {product.name}: {product.description}")

        # Contact information
        sections.append("\n**Contact Information:**")
        if business_info.contact.phone:
            sections.append(f"- Phone: {business_info.contact.phone}")
        if business_info.contact.email:
            sections.append(f"- Email: {business_info.contact.email}")
        if business_info.contact.website:
            sections.append(f"- Website: {business_info.contact.website}")
        if business_info.contact.address:
            address = f"{business_info.contact.address}"
            if business_info.contact.city:
                address += f", {business_info.contact.city}"
            if business_info.contact.state:
                address += f", {business_info.contact.state}"
            if business_info.contact.zip_code:
                address += f" {business_info.contact.zip_code}"
            sections.append(f"- Address: {address}")

        # Hours
        if business_info.hours:
            sections.append("\n**Business Hours:**")
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            for i, day in enumerate(days):
                hours = business_info.hours.get_day_hours(i)
                if hours:
                    if hours.is_closed:
                        sections.append(f"- {day}: Closed")
                    elif hours.open_time and hours.close_time:
                        sections.append(
                            f"- {day}: {hours.open_time.strftime('%I:%M %p')} - {hours.close_time.strftime('%I:%M %p')}"
                        )

        # FAQs
        if business_info.faqs:
            sections.append("\n**Common Questions:**")
            for faq in business_info.faqs[:10]:
                sections.append(f"Q: {faq.question}")
                sections.append(f"A: {faq.answer}\n")

        # Policies
        if business_info.policies:
            sections.append("\n**Important Policies:**")
            for policy in business_info.policies[:5]:
                sections.append(f"- {policy.name}: {policy.summary}")

        # Team
        if business_info.team_members:
            sections.append("\n**Team Members:**")
            for member in business_info.team_members[:10]:
                info = f"- {member.name}"
                if member.role:
                    info += f" ({member.role})"
                if member.specialties:
                    info += f" - Specializes in: {', '.join(member.specialties)}"
                sections.append(info)

        return "\n".join(sections)

    def _generate_behavior_section(
        self,
        persona: AgentPersona,
        behavior: BehaviorConfig,
    ) -> str:
        """Generate the behavior section."""
        sections = ["## Conversation Guidelines\n"]

        # Response style
        sections.append("**Response Style:**")
        sections.append(f"- Keep responses {behavior.response_length}")
        sections.append(f"- Maintain a {behavior.formality_level} tone")
        sections.append("- Be conversational - remember this is a phone call, not text")
        sections.append("- Pause naturally between sentences")

        # Active listening
        sections.append("\n**Active Listening:**")
        sections.append("- Acknowledge what the caller says before responding")
        sections.append("- Use brief acknowledgments like 'I understand' or 'I see'")
        if behavior.ask_for_clarification:
            sections.append("- Ask for clarification if something is unclear")

        # Information handling
        sections.append("\n**Information Handling:**")
        if behavior.confirm_important_info:
            sections.append("- Always confirm important details like names, dates, and phone numbers")
        if behavior.repeat_back_details:
            sections.append("- Repeat back key information to ensure accuracy")

        # Error handling
        sections.append("\n**When You Don't Know:**")
        sections.append("- Be honest if you don't have certain information")
        sections.append("- Offer to transfer to someone who can help")
        if behavior.apologize_for_errors:
            sections.append("- Apologize sincerely for any inconvenience")

        # Empathy
        if behavior.show_empathy:
            sections.append("\n**Showing Empathy:**")
            sections.append("- Acknowledge the caller's feelings and concerns")
            sections.append("- Show understanding, especially for complaints or frustrations")
            sections.append("- Use phrases like 'I understand how frustrating that must be'")

        # Common phrases
        if persona.common_phrases:
            sections.append("\n**Phrases You Use:**")
            for phrase in persona.common_phrases[:10]:
                sections.append(f"- \"{phrase}\"")

        return "\n".join(sections)

    def _generate_constraints_section(
        self,
        business_info: BusinessInfo,
        compliance: ComplianceConfig,
    ) -> str:
        """Generate the constraints section."""
        sections = ["## Important Constraints\n"]

        # Recording disclosure
        if compliance.recording_disclosure:
            sections.append("**Call Recording:**")
            sections.append(f"- If asked, inform callers: \"{compliance.recording_message}\"")

        # Things not to do
        sections.append("\n**Never Do:**")
        sections.append("- Never pretend to be a human if directly asked - be honest that you're an AI assistant")
        sections.append("- Never make promises you can't keep")
        sections.append("- Never share confidential information about other customers")
        sections.append("- Never argue with callers")
        sections.append("- Never use profanity or inappropriate language")

        # Industry-specific constraints
        industry_prompt = self.INDUSTRY_PROMPTS.get(business_info.category, "")
        if industry_prompt:
            sections.append(industry_prompt)

        # Compliance constraints
        if compliance.hipaa_compliant:
            sections.append("\n**HIPAA Compliance:**")
            sections.append("- Protect all patient health information")
            sections.append("- Never discuss patient information with unauthorized individuals")
            sections.append("- Verify caller identity before discussing sensitive information")

        if compliance.pci_compliant:
            sections.append("\n**PCI Compliance:**")
            sections.append("- Never ask for full credit card numbers")
            sections.append("- Do not store or repeat sensitive payment information")

        # Prohibited topics
        if compliance.prohibited_topics:
            sections.append("\n**Topics to Avoid:**")
            for topic in compliance.prohibited_topics:
                sections.append(f"- {topic}")

        # Required disclosures
        if compliance.required_disclosures:
            sections.append("\n**Required Disclosures:**")
            for disclosure in compliance.required_disclosures:
                sections.append(f"- {disclosure}")

        # Escalation
        sections.append("\n**When to Transfer to Human:**")
        sections.append("- When caller explicitly requests to speak with a human")
        sections.append("- For complex complaints or legal matters")
        sections.append("- When you've tried multiple times but can't help")
        sections.append("- For sensitive or emergency situations")

        return "\n".join(sections)

    def _generate_examples_section(
        self,
        business_info: BusinessInfo,
        persona: AgentPersona,
    ) -> str:
        """Generate conversation examples."""
        sections = ["## Example Conversations\n"]

        # Greeting example
        sections.append("**Opening a Call:**")
        if persona.sample_greetings:
            greeting = persona.sample_greetings[0].replace("{time_of_day}", "morning")
            sections.append(f'You: "{greeting}"')
        sections.append("")

        # General inquiry example
        sections.append("**Handling a General Inquiry:**")
        sections.append('Caller: "What services do you offer?"')
        if business_info.services:
            services = [s.name for s in business_info.services[:3]]
            sections.append(
                f'You: "We offer a variety of services including {", ".join(services)}, and more. '
                f'Is there a particular service you\'re interested in learning more about?"'
            )
        sections.append("")

        # Hours inquiry example
        sections.append("**Hours Inquiry:**")
        sections.append('Caller: "What are your hours?"')
        sections.append(
            'You: "We\'re open Monday through Friday from 9 AM to 5 PM. '
            'Were you looking to schedule an appointment or stop by?"'
        )
        sections.append("")

        # Handling frustration example
        sections.append("**Handling Caller Frustration:**")
        sections.append('Caller: "I\'ve been waiting for a callback for two days!"')
        sections.append(
            'You: "I\'m really sorry to hear you\'ve been waiting. That\'s not the experience we want '
            'for our customers. Let me look into this right away and make sure we get this resolved for you today. '
            'Can I get your name and contact number?"'
        )
        sections.append("")

        # Closing example
        sections.append("**Closing a Call:**")
        sections.append(
            f'You: "Is there anything else I can help you with today? ... '
            f'Thank you for calling {business_info.name}. Have a great day!"'
        )

        return "\n".join(sections)

    async def _enhance_with_llm(
        self,
        prompt: SystemPrompt,
        business_info: BusinessInfo,
        persona: AgentPersona,
    ) -> SystemPrompt:
        """Enhance prompt with LLM generation."""
        if not self._llm_provider:
            return prompt

        try:
            # Generate additional examples
            examples_prompt = f"""Generate 2 brief example conversations for a voice AI assistant for
{business_info.name} ({business_info.category.value}).

The assistant is named {persona.name} and is {', '.join(persona.personality_traits)}.

Show realistic phone conversations with:
1. A scheduling inquiry
2. Handling a complaint gracefully

Format each as:
**Scenario Name:**
Caller: "..."
You: "..."

Keep responses natural and conversational."""

            response = await self._llm_provider.generate(examples_prompt)
            if response and response.content:
                prompt.examples_section += f"\n\n{response.content}"

        except Exception as e:
            logger.warning(f"LLM prompt enhancement failed: {e}")

        return prompt

    def _get_industry_description(self, category: BusinessCategory) -> str:
        """Get industry description."""
        descriptions = {
            BusinessCategory.HEALTHCARE: "healthcare provider",
            BusinessCategory.DENTAL: "dental practice",
            BusinessCategory.LEGAL: "law firm",
            BusinessCategory.REAL_ESTATE: "real estate agency",
            BusinessCategory.PLUMBING: "plumbing company",
            BusinessCategory.HVAC: "heating and cooling company",
            BusinessCategory.AUTO_DEALERSHIP: "auto dealership",
            BusinessCategory.RESTAURANT: "restaurant",
            BusinessCategory.SALON: "salon",
            BusinessCategory.INSURANCE: "insurance agency",
        }
        return descriptions.get(category, "business")

    def generate_contextual_prompt(
        self,
        context: PromptContext,
        system_prompt: SystemPrompt,
    ) -> str:
        """
        Generate a contextual prompt for a specific conversation turn.

        Args:
            context: Current conversation context
            system_prompt: Base system prompt

        Returns:
            Contextual prompt for this turn
        """
        parts = [system_prompt.compile()]

        # Add current context
        parts.append("\n## Current Context\n")

        if context.caller_name:
            parts.append(f"Caller's name: {context.caller_name}")

        if not context.is_business_hours:
            parts.append("Note: Currently outside business hours")

        if context.detected_intent:
            parts.append(f"Caller's intent appears to be: {context.detected_intent}")

        if context.collected_slots:
            parts.append("Information collected:")
            for key, value in context.collected_slots.items():
                parts.append(f"  - {key}: {value}")

        return "\n".join(parts)


__all__ = [
    "PromptGenerator",
    "SystemPrompt",
    "PromptTemplate",
    "PromptContext",
    "PromptBuilder",
]
