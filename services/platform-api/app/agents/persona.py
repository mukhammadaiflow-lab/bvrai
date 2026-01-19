"""
Persona Management

Agent personality and character:
- Persona definition
- Character traits
- Speaking styles
- Emotional states
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class PersonaTrait(str, Enum):
    """Personality traits."""
    # Warmth
    FRIENDLY = "friendly"
    WARM = "warm"
    PROFESSIONAL = "professional"
    FORMAL = "formal"
    CASUAL = "casual"

    # Energy
    ENERGETIC = "energetic"
    CALM = "calm"
    ENTHUSIASTIC = "enthusiastic"
    RESERVED = "reserved"

    # Communication
    CONCISE = "concise"
    DETAILED = "detailed"
    EMPATHETIC = "empathetic"
    DIRECT = "direct"

    # Expertise
    EXPERT = "expert"
    HELPFUL = "helpful"
    PATIENT = "patient"
    ENCOURAGING = "encouraging"


class SpeakingStyle(str, Enum):
    """Speaking styles."""
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"
    AUTHORITATIVE = "authoritative"


class EmotionalTone(str, Enum):
    """Emotional tones."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    CONCERNED = "concerned"
    APOLOGETIC = "apologetic"
    EXCITED = "excited"
    CALM = "calm"
    SERIOUS = "serious"
    SUPPORTIVE = "supportive"


@dataclass
class PersonaConfig:
    """Persona configuration."""
    persona_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Identity
    display_name: str = ""
    role: str = ""  # e.g., "Customer Service Representative"
    company: str = ""

    # Background
    backstory: str = ""
    expertise_areas: List[str] = field(default_factory=list)
    experience_years: int = 0

    # Traits
    primary_traits: List[PersonaTrait] = field(default_factory=list)
    speaking_style: SpeakingStyle = SpeakingStyle.PROFESSIONAL
    default_tone: EmotionalTone = EmotionalTone.NEUTRAL

    # Voice
    voice_id: str = ""
    voice_name: str = ""
    voice_gender: str = ""  # male, female, neutral
    voice_age: str = ""  # young, middle, mature

    # Language preferences
    primary_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: ["en"])
    formality_level: int = 5  # 1-10, where 10 is most formal

    # Communication style
    use_filler_words: bool = False
    use_contractions: bool = True
    average_sentence_length: str = "medium"  # short, medium, long
    vocabulary_level: str = "standard"  # simple, standard, technical

    # Boundaries
    topics_to_avoid: List[str] = field(default_factory=list)
    phrases_to_use: List[str] = field(default_factory=list)
    phrases_to_avoid: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def get_trait_description(self) -> str:
        """Get description of traits."""
        if not self.primary_traits:
            return "professional"
        return ", ".join(t.value for t in self.primary_traits)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "description": self.description,
            "display_name": self.display_name,
            "role": self.role,
            "company": self.company,
            "backstory": self.backstory,
            "expertise_areas": self.expertise_areas,
            "primary_traits": [t.value for t in self.primary_traits],
            "speaking_style": self.speaking_style.value,
            "default_tone": self.default_tone.value,
            "voice_id": self.voice_id,
            "primary_language": self.primary_language,
            "formality_level": self.formality_level,
        }


@dataclass
class Persona:
    """Complete persona definition."""
    config: PersonaConfig

    # Runtime state
    current_tone: EmotionalTone = EmotionalTone.NEUTRAL
    current_energy: float = 0.5  # 0-1

    def get_greeting(self) -> str:
        """Get persona-appropriate greeting."""
        name = self.config.display_name or "I"

        if self.config.speaking_style == SpeakingStyle.FORMAL:
            return f"Good day. My name is {name}, and I'll be assisting you today."
        elif self.config.speaking_style == SpeakingStyle.FRIENDLY:
            return f"Hi there! I'm {name}. How can I help you today?"
        elif self.config.speaking_style == SpeakingStyle.CASUAL:
            return f"Hey! {name} here. What can I do for you?"
        else:
            return f"Hello, I'm {name}. How may I assist you?"

    def get_farewell(self) -> str:
        """Get persona-appropriate farewell."""
        if self.config.speaking_style == SpeakingStyle.FORMAL:
            return "Thank you for contacting us. Have a pleasant day."
        elif self.config.speaking_style == SpeakingStyle.FRIENDLY:
            return "Thanks so much for calling! Take care!"
        elif self.config.speaking_style == SpeakingStyle.CASUAL:
            return "Alright, thanks! Bye for now!"
        else:
            return "Thank you for your call. Goodbye!"

    def get_apology(self) -> str:
        """Get persona-appropriate apology."""
        if PersonaTrait.EMPATHETIC in self.config.primary_traits:
            return "I completely understand how frustrating this must be for you. I sincerely apologize."
        elif self.config.speaking_style == SpeakingStyle.FORMAL:
            return "I apologize for any inconvenience this may have caused."
        else:
            return "I'm sorry about that. Let me help fix this for you."

    def adjust_response(self, text: str) -> str:
        """Adjust response based on persona style."""
        # Apply contractions preference
        if self.config.use_contractions:
            text = text.replace(" will not ", " won't ")
            text = text.replace(" cannot ", " can't ")
            text = text.replace(" do not ", " don't ")
            text = text.replace(" I am ", " I'm ")
            text = text.replace(" you are ", " you're ")
        else:
            text = text.replace("won't", "will not")
            text = text.replace("can't", "cannot")
            text = text.replace("don't", "do not")
            text = text.replace("I'm", "I am")
            text = text.replace("you're", "you are")

        return text

    def set_emotional_tone(self, tone: EmotionalTone) -> None:
        """Set current emotional tone."""
        self.current_tone = tone

    def get_system_prompt_section(self) -> str:
        """Generate system prompt section for persona."""
        parts = []

        # Identity
        if self.config.display_name:
            parts.append(f"You are {self.config.display_name}")
            if self.config.role:
                parts.append(f", a {self.config.role}")
            if self.config.company:
                parts.append(f" at {self.config.company}")
            parts.append(".")

        # Traits
        if self.config.primary_traits:
            traits = self.config.get_trait_description()
            parts.append(f" Your personality is {traits}.")

        # Speaking style
        style_desc = {
            SpeakingStyle.CONVERSATIONAL: "conversational and natural",
            SpeakingStyle.FORMAL: "formal and professional",
            SpeakingStyle.FRIENDLY: "warm and friendly",
            SpeakingStyle.PROFESSIONAL: "professional yet approachable",
            SpeakingStyle.TECHNICAL: "precise and technical",
            SpeakingStyle.CASUAL: "casual and relaxed",
            SpeakingStyle.EMPATHETIC: "empathetic and understanding",
            SpeakingStyle.AUTHORITATIVE: "confident and authoritative",
        }
        parts.append(f" Speak in a {style_desc.get(self.config.speaking_style, 'professional')} manner.")

        # Expertise
        if self.config.expertise_areas:
            areas = ", ".join(self.config.expertise_areas)
            parts.append(f" You have expertise in {areas}.")

        # Boundaries
        if self.config.topics_to_avoid:
            topics = ", ".join(self.config.topics_to_avoid)
            parts.append(f" Avoid discussing {topics}.")

        return "".join(parts)


class PersonaManager:
    """Manages agent personas."""

    def __init__(self):
        self._personas: Dict[str, Persona] = {}
        self._templates: Dict[str, PersonaConfig] = {}

    def create_persona(self, config: PersonaConfig) -> Persona:
        """Create persona from config."""
        persona = Persona(config=config)
        self._personas[config.persona_id] = persona
        return persona

    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """Get persona by ID."""
        return self._personas.get(persona_id)

    def update_persona(self, persona_id: str, updates: Dict[str, Any]) -> Optional[Persona]:
        """Update persona configuration."""
        persona = self._personas.get(persona_id)
        if not persona:
            return None

        config = persona.config
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        config.updated_at = datetime.utcnow()
        return persona

    def delete_persona(self, persona_id: str) -> bool:
        """Delete persona."""
        return self._personas.pop(persona_id, None) is not None

    def add_template(self, name: str, config: PersonaConfig) -> None:
        """Add persona template."""
        self._templates[name] = config

    def create_from_template(self, template_name: str) -> Optional[Persona]:
        """Create persona from template."""
        template = self._templates.get(template_name)
        if not template:
            return None

        # Clone template config
        import copy
        config = copy.deepcopy(template)
        config.persona_id = str(uuid.uuid4())
        config.created_at = datetime.utcnow()
        config.updated_at = datetime.utcnow()

        return self.create_persona(config)

    def get_default_templates(self) -> Dict[str, PersonaConfig]:
        """Get built-in persona templates."""
        return {
            "customer_service": PersonaConfig(
                name="Customer Service Agent",
                display_name="Alex",
                role="Customer Service Representative",
                primary_traits=[PersonaTrait.FRIENDLY, PersonaTrait.HELPFUL, PersonaTrait.PATIENT],
                speaking_style=SpeakingStyle.FRIENDLY,
                formality_level=5,
            ),
            "technical_support": PersonaConfig(
                name="Technical Support Agent",
                display_name="Jordan",
                role="Technical Support Specialist",
                primary_traits=[PersonaTrait.EXPERT, PersonaTrait.PATIENT, PersonaTrait.DETAILED],
                speaking_style=SpeakingStyle.TECHNICAL,
                formality_level=6,
                expertise_areas=["troubleshooting", "technical issues", "product features"],
            ),
            "sales": PersonaConfig(
                name="Sales Agent",
                display_name="Morgan",
                role="Sales Representative",
                primary_traits=[PersonaTrait.ENTHUSIASTIC, PersonaTrait.FRIENDLY, PersonaTrait.DIRECT],
                speaking_style=SpeakingStyle.CONVERSATIONAL,
                formality_level=4,
            ),
            "executive": PersonaConfig(
                name="Executive Assistant",
                display_name="Taylor",
                role="Executive Assistant",
                primary_traits=[PersonaTrait.PROFESSIONAL, PersonaTrait.CONCISE, PersonaTrait.DIRECT],
                speaking_style=SpeakingStyle.FORMAL,
                formality_level=8,
            ),
        }

    def list_personas(self) -> List[Persona]:
        """List all personas."""
        return list(self._personas.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "total_personas": len(self._personas),
            "templates": len(self._templates),
        }
