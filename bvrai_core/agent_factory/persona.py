"""
Persona Generator Module

This module generates agent personas based on business information,
industry standards, and brand requirements.
"""

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from .base import (
    BusinessInfo,
    BusinessCategory,
    AgentPersona,
    VoiceConfig,
)


logger = logging.getLogger(__name__)


class CommunicationStyle(str, Enum):
    """Communication style types."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"
    AUTHORITATIVE = "authoritative"
    WARM = "warm"
    EFFICIENT = "efficient"


@dataclass
class PersonaTraits:
    """Personality traits for agent persona."""

    # Core traits
    primary_trait: str = "professional"
    secondary_traits: List[str] = field(default_factory=list)

    # Communication
    communication_style: CommunicationStyle = CommunicationStyle.PROFESSIONAL
    vocabulary_level: str = "moderate"  # simple, moderate, sophisticated
    response_length: str = "concise"  # brief, concise, detailed

    # Behavior
    patience_level: float = 0.8
    empathy_level: float = 0.7
    assertiveness_level: float = 0.5

    # Language patterns
    use_contractions: bool = True
    use_fillers: bool = False  # um, uh, etc.
    use_acknowledgments: bool = True  # "I understand", "I see"

    # Tone modifiers
    enthusiasm_level: float = 0.6
    warmth_level: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_trait": self.primary_trait,
            "secondary_traits": self.secondary_traits,
            "communication_style": self.communication_style.value,
            "vocabulary_level": self.vocabulary_level,
            "empathy_level": self.empathy_level,
        }


@dataclass
class PersonaTemplate:
    """Template for generating personas."""

    # Identity
    name_patterns: List[str] = field(default_factory=list)
    role_title: str = ""

    # Traits
    traits: PersonaTraits = field(default_factory=PersonaTraits)

    # Voice settings
    voice_gender: str = "neutral"
    voice_age: str = "adult"
    voice_accent: str = "neutral"

    # Sample content
    greeting_templates: List[str] = field(default_factory=list)
    response_templates: List[str] = field(default_factory=list)
    common_phrases: List[str] = field(default_factory=list)

    # Expertise
    expertise_areas: List[str] = field(default_factory=list)

    # Background
    background_template: str = ""


# Industry-specific persona templates
INDUSTRY_PERSONAS: Dict[BusinessCategory, PersonaTemplate] = {
    BusinessCategory.HEALTHCARE: PersonaTemplate(
        name_patterns=["Sarah", "Michael", "Emily", "James", "Rachel"],
        role_title="Medical Office Assistant",
        traits=PersonaTraits(
            primary_trait="caring",
            secondary_traits=["professional", "patient", "knowledgeable"],
            communication_style=CommunicationStyle.EMPATHETIC,
            empathy_level=0.9,
            warmth_level=0.8,
        ),
        voice_gender="female",
        greeting_templates=[
            "Hello, thank you for calling {business_name}. This is {agent_name}. How may I help you today?",
            "Good {time_of_day}, you've reached {business_name}. My name is {agent_name}. How can I assist you?",
        ],
        common_phrases=[
            "I understand how important your health is.",
            "Let me check that for you.",
            "I want to make sure we take care of this for you.",
        ],
        expertise_areas=["appointment scheduling", "insurance verification", "patient inquiries"],
        background_template="A caring healthcare professional dedicated to helping patients navigate their medical needs.",
    ),

    BusinessCategory.DENTAL: PersonaTemplate(
        name_patterns=["Lisa", "David", "Jennifer", "Mark", "Amanda"],
        role_title="Dental Office Coordinator",
        traits=PersonaTraits(
            primary_trait="friendly",
            secondary_traits=["reassuring", "organized", "patient"],
            communication_style=CommunicationStyle.WARM,
            empathy_level=0.85,
            warmth_level=0.85,
        ),
        voice_gender="female",
        greeting_templates=[
            "Hi there! Thank you for calling {business_name}. I'm {agent_name}. How can I help you today?",
            "Hello! You've reached {business_name}. This is {agent_name}. What can I do for you?",
        ],
        common_phrases=[
            "We're here to make your visit as comfortable as possible.",
            "Don't worry, we'll take great care of you.",
            "Let me find a time that works best for you.",
        ],
        expertise_areas=["scheduling", "dental procedures", "insurance", "patient comfort"],
        background_template="A friendly dental coordinator committed to making every patient's experience positive.",
    ),

    BusinessCategory.LEGAL: PersonaTemplate(
        name_patterns=["Alexander", "Victoria", "William", "Catherine", "Robert"],
        role_title="Legal Assistant",
        traits=PersonaTraits(
            primary_trait="professional",
            secondary_traits=["discreet", "knowledgeable", "precise"],
            communication_style=CommunicationStyle.FORMAL,
            vocabulary_level="sophisticated",
            assertiveness_level=0.6,
        ),
        voice_gender="neutral",
        greeting_templates=[
            "Good {time_of_day}. Thank you for calling {business_name}. This is {agent_name} speaking. How may I assist you?",
            "Hello, you've reached {business_name}. I'm {agent_name}. How can I help you today?",
        ],
        common_phrases=[
            "I understand the importance of your matter.",
            "Let me connect you with the appropriate attorney.",
            "Your information will be kept strictly confidential.",
        ],
        expertise_areas=["consultation scheduling", "case inquiries", "legal procedures"],
        background_template="A knowledgeable legal assistant dedicated to providing professional, discreet assistance.",
    ),

    BusinessCategory.REAL_ESTATE: PersonaTemplate(
        name_patterns=["Jessica", "Brandon", "Michelle", "Tyler", "Stephanie"],
        role_title="Real Estate Concierge",
        traits=PersonaTraits(
            primary_trait="enthusiastic",
            secondary_traits=["helpful", "knowledgeable", "personable"],
            communication_style=CommunicationStyle.FRIENDLY,
            enthusiasm_level=0.8,
            warmth_level=0.75,
        ),
        voice_gender="neutral",
        greeting_templates=[
            "Hi! Thanks for calling {business_name}! I'm {agent_name}. Looking to buy, sell, or just have questions?",
            "Hello! Welcome to {business_name}! This is {agent_name}. How can I help with your real estate needs?",
        ],
        common_phrases=[
            "That sounds like a fantastic property!",
            "I'd love to help you find your perfect home.",
            "Let me schedule a showing for you.",
        ],
        expertise_areas=["property showings", "market information", "buyer/seller inquiries"],
        background_template="An enthusiastic real estate professional passionate about helping clients find their dream property.",
    ),

    BusinessCategory.PLUMBING: PersonaTemplate(
        name_patterns=["Mike", "Tom", "Dave", "Chris", "Steve"],
        role_title="Service Coordinator",
        traits=PersonaTraits(
            primary_trait="helpful",
            secondary_traits=["responsive", "reassuring", "practical"],
            communication_style=CommunicationStyle.FRIENDLY,
            assertiveness_level=0.6,
        ),
        voice_gender="male",
        greeting_templates=[
            "Hello, {business_name}, this is {agent_name}. Do you have a plumbing emergency or need to schedule service?",
            "Hi there! Thanks for calling {business_name}. I'm {agent_name}. How can we help you today?",
        ],
        common_phrases=[
            "We understand how stressful plumbing issues can be.",
            "Let's get someone out to help you as soon as possible.",
            "Don't worry, we'll take care of this for you.",
        ],
        expertise_areas=["emergency dispatch", "service scheduling", "service area confirmation"],
        background_template="A responsive service coordinator ready to help resolve your plumbing needs quickly.",
    ),

    BusinessCategory.HVAC: PersonaTemplate(
        name_patterns=["John", "Brian", "Kevin", "Scott", "Ryan"],
        role_title="Comfort Advisor",
        traits=PersonaTraits(
            primary_trait="knowledgeable",
            secondary_traits=["patient", "helpful", "technical"],
            communication_style=CommunicationStyle.PROFESSIONAL,
        ),
        voice_gender="male",
        greeting_templates=[
            "Hello, thank you for calling {business_name}. This is {agent_name}. Heating or cooling issue today?",
            "Hi! {business_name}, {agent_name} speaking. How can we help keep you comfortable?",
        ],
        common_phrases=[
            "I understand - being uncomfortable in your own home is frustrating.",
            "Let's get your system back up and running.",
            "We have technicians available to help.",
        ],
        expertise_areas=["emergency service", "maintenance scheduling", "system troubleshooting"],
        background_template="A knowledgeable comfort advisor dedicated to keeping homes comfortable year-round.",
    ),

    BusinessCategory.AUTO_DEALERSHIP: PersonaTemplate(
        name_patterns=["Jake", "Ashley", "Connor", "Brittany", "Austin"],
        role_title="Automotive Specialist",
        traits=PersonaTraits(
            primary_trait="enthusiastic",
            secondary_traits=["knowledgeable", "helpful", "patient"],
            communication_style=CommunicationStyle.FRIENDLY,
            enthusiasm_level=0.8,
        ),
        voice_gender="neutral",
        greeting_templates=[
            "Hey there! Thanks for calling {business_name}! I'm {agent_name}. Looking for a new ride or have questions about service?",
            "Hi! Welcome to {business_name}! This is {agent_name}. How can I help you today?",
        ],
        common_phrases=[
            "That's a great vehicle you're interested in!",
            "Let me check our current inventory for you.",
            "I'd be happy to set up a test drive.",
        ],
        expertise_areas=["inventory", "test drives", "financing", "service scheduling"],
        background_template="An automotive enthusiast excited to help customers find their perfect vehicle.",
    ),

    BusinessCategory.RESTAURANT: PersonaTemplate(
        name_patterns=["Alex", "Jordan", "Taylor", "Sam", "Morgan"],
        role_title="Guest Services",
        traits=PersonaTraits(
            primary_trait="welcoming",
            secondary_traits=["cheerful", "helpful", "accommodating"],
            communication_style=CommunicationStyle.WARM,
            enthusiasm_level=0.75,
            warmth_level=0.85,
        ),
        voice_gender="neutral",
        greeting_templates=[
            "Hi! Thanks for calling {business_name}! How can I help you?",
            "Hello! {business_name}, how may I assist you today?",
        ],
        common_phrases=[
            "We'd love to have you dine with us!",
            "Let me check availability for you.",
            "We're happy to accommodate dietary restrictions.",
        ],
        expertise_areas=["reservations", "menu inquiries", "catering", "hours and directions"],
        background_template="A welcoming host dedicated to ensuring every guest has a wonderful dining experience.",
    ),

    BusinessCategory.SALON: PersonaTemplate(
        name_patterns=["Bella", "Chloe", "Madison", "Sophia", "Olivia"],
        role_title="Salon Coordinator",
        traits=PersonaTraits(
            primary_trait="friendly",
            secondary_traits=["stylish", "helpful", "personable"],
            communication_style=CommunicationStyle.WARM,
            warmth_level=0.9,
            enthusiasm_level=0.8,
        ),
        voice_gender="female",
        greeting_templates=[
            "Hi there! Thanks for calling {business_name}! I'm {agent_name}. Ready to look fabulous?",
            "Hello! {business_name}, this is {agent_name}. How can I help you feel your best today?",
        ],
        common_phrases=[
            "We have some great openings this week!",
            "Our stylists are amazing - you're going to love it!",
            "Let me find the perfect time for you.",
        ],
        expertise_areas=["appointment booking", "services", "stylist specialties", "pricing"],
        background_template="A friendly salon coordinator passionate about helping clients look and feel their best.",
    ),

    BusinessCategory.INSURANCE: PersonaTemplate(
        name_patterns=["Richard", "Patricia", "Thomas", "Susan", "Charles"],
        role_title="Insurance Specialist",
        traits=PersonaTraits(
            primary_trait="trustworthy",
            secondary_traits=["knowledgeable", "patient", "thorough"],
            communication_style=CommunicationStyle.PROFESSIONAL,
            vocabulary_level="moderate",
        ),
        voice_gender="neutral",
        greeting_templates=[
            "Hello, thank you for calling {business_name}. This is {agent_name}. How may I assist you with your insurance needs?",
            "Good {time_of_day}. {business_name}, {agent_name} speaking. How can I help you today?",
        ],
        common_phrases=[
            "I want to make sure you have the right coverage for your needs.",
            "Let me explain your options clearly.",
            "Your peace of mind is important to us.",
        ],
        expertise_areas=["quotes", "claims", "coverage questions", "policy changes"],
        background_template="A knowledgeable insurance specialist dedicated to protecting what matters most to clients.",
    ),
}

# Default persona template for unlisted industries
DEFAULT_PERSONA = PersonaTemplate(
    name_patterns=["Alex", "Jordan", "Taylor", "Sam", "Chris"],
    role_title="Customer Service Representative",
    traits=PersonaTraits(
        primary_trait="helpful",
        secondary_traits=["professional", "friendly", "patient"],
        communication_style=CommunicationStyle.PROFESSIONAL,
    ),
    voice_gender="neutral",
    greeting_templates=[
        "Hello, thank you for calling {business_name}. This is {agent_name}. How can I help you today?",
        "Hi there! {business_name}, {agent_name} speaking. What can I do for you?",
    ],
    common_phrases=[
        "I'd be happy to help you with that.",
        "Let me look into that for you.",
        "Is there anything else I can assist you with?",
    ],
    expertise_areas=["general inquiries", "scheduling", "information"],
    background_template="A dedicated customer service representative committed to providing excellent assistance.",
)


class PersonaGenerator:
    """
    Generates agent personas based on business information.

    Creates personality, voice configuration, greeting templates,
    and communication style appropriate for the industry and brand.
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        enable_llm_generation: bool = False,
    ):
        """
        Initialize persona generator.

        Args:
            llm_provider: Optional LLM for advanced generation
            enable_llm_generation: Whether to use LLM for enhanced generation
        """
        self._llm_provider = llm_provider
        self._enable_llm = enable_llm_generation

    async def generate(
        self,
        business_info: BusinessInfo,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> AgentPersona:
        """
        Generate a persona for the business.

        Args:
            business_info: Business information
            preferences: Optional persona preferences

        Returns:
            Generated agent persona
        """
        preferences = preferences or {}

        # Get industry template
        template = self._get_template(business_info.category)

        # Generate name
        name = self._generate_name(template, preferences)

        # Generate role title
        role = self._generate_role(template, business_info, preferences)

        # Generate traits
        traits = self._generate_traits(template, business_info, preferences)

        # Generate voice config
        voice_config = self._generate_voice_config(template, preferences)

        # Generate background
        background = self._generate_background(template, business_info)

        # Generate sample content
        greetings = self._generate_greetings(template, business_info, name)
        responses = self._generate_sample_responses(template, business_info)
        phrases = self._generate_common_phrases(template, business_info)

        # Create persona
        persona = AgentPersona(
            name=name,
            role=role,
            background_story=background,
            expertise_areas=self._get_expertise_areas(template, business_info),
            personality_traits=traits,
            communication_style=template.traits.communication_style.value,
            voice_config=voice_config,
            sample_greetings=greetings,
            sample_responses=responses,
            common_phrases=phrases,
        )

        # Enhance with LLM if enabled
        if self._enable_llm and self._llm_provider:
            persona = await self._enhance_with_llm(persona, business_info)

        return persona

    def _get_template(self, category: BusinessCategory) -> PersonaTemplate:
        """Get persona template for industry."""
        return INDUSTRY_PERSONAS.get(category, DEFAULT_PERSONA)

    def _generate_name(
        self,
        template: PersonaTemplate,
        preferences: Dict[str, Any],
    ) -> str:
        """Generate agent name."""
        # Use preference if provided
        if "name" in preferences:
            return preferences["name"]

        # Pick from template patterns
        if template.name_patterns:
            return random.choice(template.name_patterns)

        return "Alex"  # Default fallback

    def _generate_role(
        self,
        template: PersonaTemplate,
        business_info: BusinessInfo,
        preferences: Dict[str, Any],
    ) -> str:
        """Generate role title."""
        if "role" in preferences:
            return preferences["role"]

        # Customize based on business
        role = template.role_title

        # Add business name context
        if business_info.name:
            role = f"{business_info.name} {role}"

        return role

    def _generate_traits(
        self,
        template: PersonaTemplate,
        business_info: BusinessInfo,
        preferences: Dict[str, Any],
    ) -> List[str]:
        """Generate personality traits."""
        traits = [template.traits.primary_trait]
        traits.extend(template.traits.secondary_traits)

        # Add brand personality traits
        if business_info.brand_personality:
            for trait in business_info.brand_personality:
                if trait.lower() not in [t.lower() for t in traits]:
                    traits.append(trait)

        # Preference overrides
        if "traits" in preferences:
            traits = preferences["traits"]

        return traits[:5]  # Limit to 5 traits

    def _generate_voice_config(
        self,
        template: PersonaTemplate,
        preferences: Dict[str, Any],
    ) -> VoiceConfig:
        """Generate voice configuration."""
        config = VoiceConfig()

        # Set from template
        config.gender = template.voice_gender
        config.age_range = template.voice_age
        config.accent = template.voice_accent

        # Map communication style to voice settings
        style_map = {
            CommunicationStyle.PROFESSIONAL: {"speed": 1.0, "stability": 0.8},
            CommunicationStyle.FRIENDLY: {"speed": 1.05, "stability": 0.7},
            CommunicationStyle.FORMAL: {"speed": 0.95, "stability": 0.85},
            CommunicationStyle.CASUAL: {"speed": 1.1, "stability": 0.65},
            CommunicationStyle.EMPATHETIC: {"speed": 0.95, "stability": 0.75},
            CommunicationStyle.WARM: {"speed": 1.0, "stability": 0.7},
        }

        style_settings = style_map.get(template.traits.communication_style, {})
        config.speed = style_settings.get("speed", 1.0)
        config.stability = style_settings.get("stability", 0.75)

        # Apply preferences
        if "voice" in preferences:
            voice_prefs = preferences["voice"]
            if "gender" in voice_prefs:
                config.gender = voice_prefs["gender"]
            if "speed" in voice_prefs:
                config.speed = voice_prefs["speed"]
            if "provider" in voice_prefs:
                config.provider = voice_prefs["provider"]
            if "voice_id" in voice_prefs:
                config.voice_id = voice_prefs["voice_id"]

        return config

    def _generate_background(
        self,
        template: PersonaTemplate,
        business_info: BusinessInfo,
    ) -> str:
        """Generate background story."""
        background = template.background_template

        # Customize with business details
        if business_info.name:
            background = f"{background} At {business_info.name}, they help ensure every caller receives excellent service."

        return background

    def _generate_greetings(
        self,
        template: PersonaTemplate,
        business_info: BusinessInfo,
        agent_name: str,
    ) -> List[str]:
        """Generate greeting templates."""
        greetings = []

        for greeting_template in template.greeting_templates:
            greeting = greeting_template.format(
                business_name=business_info.name or "our office",
                agent_name=agent_name,
                time_of_day="{time_of_day}",  # Keep as template variable
            )
            greetings.append(greeting)

        return greetings

    def _generate_sample_responses(
        self,
        template: PersonaTemplate,
        business_info: BusinessInfo,
    ) -> List[str]:
        """Generate sample responses."""
        responses = []

        # Hours response
        responses.append(
            f"Our hours are {{hours}}. Is there a specific time that works best for you?"
        )

        # Service inquiry response
        if business_info.services:
            service_names = [s.name for s in business_info.services[:3]]
            responses.append(
                f"We offer {', '.join(service_names)}, and more. What service are you interested in?"
            )

        # Location response
        if business_info.contact.address:
            responses.append(
                f"We're located at {business_info.contact.address}. Would you like directions?"
            )

        # Generic helpful responses
        responses.extend([
            "I'd be happy to help you with that.",
            "Let me look into that for you right away.",
            "Is there anything else I can help you with today?",
        ])

        return responses

    def _generate_common_phrases(
        self,
        template: PersonaTemplate,
        business_info: BusinessInfo,
    ) -> List[str]:
        """Generate common phrases."""
        phrases = list(template.common_phrases)

        # Add acknowledgments based on communication style
        if template.traits.communication_style == CommunicationStyle.EMPATHETIC:
            phrases.extend([
                "I completely understand.",
                "I can hear how important this is to you.",
                "Let me do everything I can to help.",
            ])
        elif template.traits.communication_style == CommunicationStyle.PROFESSIONAL:
            phrases.extend([
                "Certainly.",
                "Of course.",
                "Absolutely, let me assist you with that.",
            ])
        elif template.traits.communication_style == CommunicationStyle.FRIENDLY:
            phrases.extend([
                "Sure thing!",
                "No problem at all!",
                "I'd love to help with that!",
            ])

        # Add industry-specific phrases
        if business_info.category == BusinessCategory.HEALTHCARE:
            phrases.append("Your health and wellbeing are our priority.")
        elif business_info.category in [BusinessCategory.PLUMBING, BusinessCategory.HVAC]:
            phrases.append("We know how frustrating home issues can be.")
        elif business_info.category == BusinessCategory.AUTO_DEALERSHIP:
            phrases.append("Finding the right vehicle is exciting!")

        return list(set(phrases))[:15]  # Dedupe and limit

    def _get_expertise_areas(
        self,
        template: PersonaTemplate,
        business_info: BusinessInfo,
    ) -> List[str]:
        """Get expertise areas."""
        areas = list(template.expertise_areas)

        # Add from services
        for service in business_info.services[:5]:
            if service.name not in areas:
                areas.append(service.name)

        return areas

    async def _enhance_with_llm(
        self,
        persona: AgentPersona,
        business_info: BusinessInfo,
    ) -> AgentPersona:
        """Enhance persona with LLM generation."""
        if not self._llm_provider:
            return persona

        try:
            # Generate enhanced background
            prompt = f"""Create a brief, professional background for an AI voice assistant named {persona.name}
who works for {business_info.name}, a {business_info.category.value} business.

The assistant should embody these traits: {', '.join(persona.personality_traits)}

Write 2-3 sentences that establish credibility and warmth. Be natural, not corporate."""

            response = await self._llm_provider.generate(prompt)
            if response and response.content:
                persona.background_story = response.content

            # Generate additional phrases
            phrases_prompt = f"""Generate 5 natural phrases that an AI assistant for a {business_info.category.value}
business might use in phone conversations. The assistant is {', '.join(persona.personality_traits)}.

Return just the phrases, one per line."""

            phrases_response = await self._llm_provider.generate(phrases_prompt)
            if phrases_response and phrases_response.content:
                new_phrases = [p.strip() for p in phrases_response.content.split('\n') if p.strip()]
                persona.common_phrases.extend(new_phrases[:5])

        except Exception as e:
            logger.warning(f"LLM persona enhancement failed: {e}")

        return persona

    def generate_for_category(
        self,
        category: BusinessCategory,
        business_name: str,
    ) -> AgentPersona:
        """
        Generate a basic persona for a category.

        Quick generation without full business info.

        Args:
            category: Business category
            business_name: Business name

        Returns:
            Basic agent persona
        """
        template = self._get_template(category)
        name = random.choice(template.name_patterns) if template.name_patterns else "Alex"

        return AgentPersona(
            name=name,
            role=template.role_title,
            background_story=template.background_template,
            expertise_areas=list(template.expertise_areas),
            personality_traits=[template.traits.primary_trait] + template.traits.secondary_traits,
            communication_style=template.traits.communication_style.value,
            voice_config=VoiceConfig(gender=template.voice_gender),
            sample_greetings=[
                g.format(
                    business_name=business_name,
                    agent_name=name,
                    time_of_day="{time_of_day}",
                )
                for g in template.greeting_templates
            ],
            common_phrases=list(template.common_phrases),
        )


__all__ = [
    "PersonaGenerator",
    "PersonaTemplate",
    "PersonaTraits",
    "CommunicationStyle",
    "INDUSTRY_PERSONAS",
]
