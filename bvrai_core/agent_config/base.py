"""
Agent Configuration Base Types Module

This module defines the core types and data structures for agent configuration
management including prompt templates, personality settings, and version control.
"""

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# =============================================================================
# Enums
# =============================================================================


class PersonalityTrait(str, Enum):
    """Personality traits for agent behavior."""

    # Tone traits
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CASUAL = "casual"
    WARM = "warm"
    AUTHORITATIVE = "authoritative"
    EMPATHETIC = "empathetic"
    ENTHUSIASTIC = "enthusiastic"
    CALM = "calm"
    ASSERTIVE = "assertive"

    # Communication style
    CONCISE = "concise"
    DETAILED = "detailed"
    CONVERSATIONAL = "conversational"
    DIRECT = "direct"
    DIPLOMATIC = "diplomatic"

    # Behavior traits
    PROACTIVE = "proactive"
    REACTIVE = "reactive"
    PATIENT = "patient"
    EFFICIENT = "efficient"
    THOROUGH = "thorough"


class TemplateCategory(str, Enum):
    """Categories for prompt templates."""

    SYSTEM_PROMPT = "system_prompt"
    GREETING = "greeting"
    FAREWELL = "farewell"
    TRANSFER = "transfer"
    HOLD = "hold"
    CLARIFICATION = "clarification"
    APOLOGY = "apology"
    CONFIRMATION = "confirmation"
    ERROR_RECOVERY = "error_recovery"
    OBJECTION_HANDLING = "objection_handling"
    SCHEDULING = "scheduling"
    INFORMATION = "information"
    CUSTOM = "custom"


class VariableType(str, Enum):
    """Types of template variables."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    LIST = "list"
    OBJECT = "object"
    ENUM = "enum"


class ConfigStatus(str, Enum):
    """Status of agent configuration."""

    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    TESTING = "testing"


class IndustryType(str, Enum):
    """Industry types for specialized agent behavior."""

    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"
    INSURANCE = "insurance"
    REAL_ESTATE = "real_estate"
    LEGAL = "legal"
    RETAIL = "retail"
    HOSPITALITY = "hospitality"
    AUTOMOTIVE = "automotive"
    TELECOMMUNICATIONS = "telecommunications"
    TECHNOLOGY = "technology"
    EDUCATION = "education"
    GOVERNMENT = "government"
    NONPROFIT = "nonprofit"
    PROFESSIONAL_SERVICES = "professional_services"
    MANUFACTURING = "manufacturing"
    CONSTRUCTION = "construction"
    UTILITIES = "utilities"
    TRANSPORTATION = "transportation"
    ENTERTAINMENT = "entertainment"
    FITNESS = "fitness"
    BEAUTY = "beauty"
    FOOD_SERVICE = "food_service"
    HOME_SERVICES = "home_services"
    CUSTOM = "custom"


class ComplianceMode(str, Enum):
    """Compliance modes for regulated industries."""

    NONE = "none"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOX = "sox"
    FERPA = "ferpa"
    GLBA = "glba"
    CUSTOM = "custom"


class EscalationTrigger(str, Enum):
    """Triggers for escalation to human agent."""

    EXPLICIT_REQUEST = "explicit_request"
    SENTIMENT_NEGATIVE = "sentiment_negative"
    REPEATED_FAILURE = "repeated_failure"
    COMPLEX_ISSUE = "complex_issue"
    HIGH_VALUE_CUSTOMER = "high_value_customer"
    COMPLIANCE_REQUIRED = "compliance_required"
    UNKNOWN_INTENT = "unknown_intent"
    TIMEOUT = "timeout"
    CUSTOM = "custom"


# =============================================================================
# Template Variables
# =============================================================================


@dataclass
class TemplateVariable:
    """Definition of a template variable."""

    name: str
    type: VariableType
    description: str = ""
    required: bool = True
    default_value: Optional[Any] = None

    # Validation
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None  # Regex pattern for validation
    enum_values: List[str] = field(default_factory=list)

    # Display
    label: str = ""
    placeholder: str = ""
    help_text: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.name.replace("_", " ").title()

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against this variable's constraints.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            if self.required and self.default_value is None:
                return False, f"Variable '{self.name}' is required"
            return True, None

        # Type validation
        if self.type == VariableType.STRING:
            if not isinstance(value, str):
                return False, f"Variable '{self.name}' must be a string"
            if self.min_length and len(value) < self.min_length:
                return False, f"Variable '{self.name}' must be at least {self.min_length} characters"
            if self.max_length and len(value) > self.max_length:
                return False, f"Variable '{self.name}' must be at most {self.max_length} characters"
            if self.pattern:
                if not re.match(self.pattern, value):
                    return False, f"Variable '{self.name}' does not match required pattern"

        elif self.type == VariableType.NUMBER:
            if not isinstance(value, (int, float)):
                return False, f"Variable '{self.name}' must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"Variable '{self.name}' must be at least {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Variable '{self.name}' must be at most {self.max_value}"

        elif self.type == VariableType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Variable '{self.name}' must be a boolean"

        elif self.type == VariableType.ENUM:
            if value not in self.enum_values:
                return False, f"Variable '{self.name}' must be one of: {', '.join(self.enum_values)}"

        elif self.type == VariableType.LIST:
            if not isinstance(value, list):
                return False, f"Variable '{self.name}' must be a list"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "required": self.required,
            "default_value": self.default_value,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "pattern": self.pattern,
            "enum_values": self.enum_values,
            "label": self.label,
            "placeholder": self.placeholder,
            "help_text": self.help_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateVariable":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type=VariableType(data["type"]),
            description=data.get("description", ""),
            required=data.get("required", True),
            default_value=data.get("default_value"),
            min_length=data.get("min_length"),
            max_length=data.get("max_length"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            pattern=data.get("pattern"),
            enum_values=data.get("enum_values", []),
            label=data.get("label", ""),
            placeholder=data.get("placeholder", ""),
            help_text=data.get("help_text", ""),
        )


# =============================================================================
# Prompt Templates
# =============================================================================


@dataclass
class PromptTemplate:
    """A prompt template with variable substitution."""

    id: str
    name: str
    category: TemplateCategory
    content: str

    # Organization
    organization_id: str

    # Metadata
    description: str = ""
    version: int = 1
    is_active: bool = True
    is_default: bool = False

    # Variables
    variables: List[TemplateVariable] = field(default_factory=list)

    # Industry targeting
    industries: List[IndustryType] = field(default_factory=list)

    # Tags for organization
    tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"tmpl_{uuid.uuid4().hex[:24]}"
        self._extract_variables_from_content()

    def _extract_variables_from_content(self) -> None:
        """Extract variable names from template content."""
        # Find all {{variable_name}} patterns
        pattern = r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}'
        found_vars = set(re.findall(pattern, self.content))

        # Create variables for any not already defined
        existing_names = {v.name for v in self.variables}
        for var_name in found_vars:
            if var_name not in existing_names:
                self.variables.append(TemplateVariable(
                    name=var_name,
                    type=VariableType.STRING,
                    description=f"Auto-detected variable: {var_name}",
                ))

    def get_variable_names(self) -> Set[str]:
        """Get all variable names in this template."""
        return {v.name for v in self.variables}

    def render(
        self,
        values: Dict[str, Any],
        strict: bool = True,
    ) -> str:
        """
        Render template with provided values.

        Args:
            values: Variable values to substitute
            strict: If True, raise error for missing required variables

        Returns:
            Rendered template string
        """
        # Validate all variables
        errors = []
        resolved_values = {}

        for var in self.variables:
            value = values.get(var.name, var.default_value)
            is_valid, error = var.validate(value)

            if not is_valid:
                if strict:
                    errors.append(error)
                else:
                    value = var.default_value if var.default_value is not None else ""

            resolved_values[var.name] = value

        if errors:
            raise ValueError(f"Template validation errors: {'; '.join(errors)}")

        # Perform substitution
        result = self.content
        for name, value in resolved_values.items():
            if value is not None:
                # Handle different value types
                if isinstance(value, list):
                    str_value = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    str_value = str(value)
                elif isinstance(value, datetime):
                    str_value = value.strftime("%Y-%m-%d %H:%M")
                else:
                    str_value = str(value)

                result = result.replace(f"{{{{{name}}}}}", str_value)

        return result

    def get_content_hash(self) -> str:
        """Get hash of template content for change detection."""
        content_bytes = self.content.encode("utf-8")
        return hashlib.sha256(content_bytes).hexdigest()[:16]

    def clone(self, new_name: Optional[str] = None) -> "PromptTemplate":
        """Create a clone of this template."""
        return PromptTemplate(
            id=f"tmpl_{uuid.uuid4().hex[:24]}",
            name=new_name or f"{self.name} (Copy)",
            category=self.category,
            content=self.content,
            organization_id=self.organization_id,
            description=self.description,
            version=1,
            is_active=False,
            is_default=False,
            variables=[
                TemplateVariable(
                    name=v.name,
                    type=v.type,
                    description=v.description,
                    required=v.required,
                    default_value=v.default_value,
                    min_length=v.min_length,
                    max_length=v.max_length,
                    min_value=v.min_value,
                    max_value=v.max_value,
                    pattern=v.pattern,
                    enum_values=v.enum_values.copy(),
                    label=v.label,
                    placeholder=v.placeholder,
                    help_text=v.help_text,
                )
                for v in self.variables
            ],
            industries=self.industries.copy(),
            tags=self.tags.copy(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "content": self.content,
            "organization_id": self.organization_id,
            "description": self.description,
            "version": self.version,
            "is_active": self.is_active,
            "is_default": self.is_default,
            "variables": [v.to_dict() for v in self.variables],
            "industries": [i.value for i in self.industries],
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            category=TemplateCategory(data["category"]),
            content=data["content"],
            organization_id=data["organization_id"],
            description=data.get("description", ""),
            version=data.get("version", 1),
            is_active=data.get("is_active", True),
            is_default=data.get("is_default", False),
            variables=[
                TemplateVariable.from_dict(v)
                for v in data.get("variables", [])
            ],
            industries=[
                IndustryType(i) for i in data.get("industries", [])
            ],
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            created_by=data.get("created_by"),
            updated_by=data.get("updated_by"),
        )


# =============================================================================
# Personality Configuration
# =============================================================================


@dataclass
class VoiceSettings:
    """Voice synthesis settings for agent."""

    provider: str = "elevenlabs"
    voice_id: str = ""
    voice_name: str = ""

    # Voice characteristics
    speed: float = 1.0  # 0.5 to 2.0
    pitch: float = 1.0  # 0.5 to 2.0
    volume: float = 1.0  # 0.0 to 1.0

    # Provider-specific
    stability: float = 0.5  # ElevenLabs
    similarity_boost: float = 0.75  # ElevenLabs
    style: float = 0.0  # ElevenLabs v2
    use_speaker_boost: bool = True  # ElevenLabs

    # Language
    language: str = "en-US"
    accent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
            "speed": self.speed,
            "pitch": self.pitch,
            "volume": self.volume,
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
            "language": self.language,
            "accent": self.accent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceSettings":
        """Create from dictionary."""
        return cls(
            provider=data.get("provider", "elevenlabs"),
            voice_id=data.get("voice_id", ""),
            voice_name=data.get("voice_name", ""),
            speed=data.get("speed", 1.0),
            pitch=data.get("pitch", 1.0),
            volume=data.get("volume", 1.0),
            stability=data.get("stability", 0.5),
            similarity_boost=data.get("similarity_boost", 0.75),
            style=data.get("style", 0.0),
            use_speaker_boost=data.get("use_speaker_boost", True),
            language=data.get("language", "en-US"),
            accent=data.get("accent"),
        )


@dataclass
class BehaviorSettings:
    """Behavioral settings for agent."""

    # Response behavior
    response_style: str = "conversational"  # conversational, formal, casual
    response_length: str = "moderate"  # brief, moderate, detailed
    use_filler_words: bool = True
    allow_interruptions: bool = True
    interruption_sensitivity: float = 0.5  # 0.0 to 1.0

    # Timing
    silence_timeout_seconds: int = 10
    max_silence_prompts: int = 3
    thinking_acknowledgment: bool = True

    # Call handling
    max_call_duration_seconds: int = 1800  # 30 minutes
    call_recording_enabled: bool = True
    voicemail_detection: bool = True
    voicemail_action: str = "leave_message"  # leave_message, hang_up, callback

    # Escalation
    escalation_enabled: bool = True
    escalation_triggers: List[EscalationTrigger] = field(default_factory=list)
    max_attempts_before_escalation: int = 3

    # Compliance
    compliance_mode: ComplianceMode = ComplianceMode.NONE
    required_disclosures: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.escalation_triggers:
            self.escalation_triggers = [
                EscalationTrigger.EXPLICIT_REQUEST,
                EscalationTrigger.SENTIMENT_NEGATIVE,
                EscalationTrigger.REPEATED_FAILURE,
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response_style": self.response_style,
            "response_length": self.response_length,
            "use_filler_words": self.use_filler_words,
            "allow_interruptions": self.allow_interruptions,
            "interruption_sensitivity": self.interruption_sensitivity,
            "silence_timeout_seconds": self.silence_timeout_seconds,
            "max_silence_prompts": self.max_silence_prompts,
            "thinking_acknowledgment": self.thinking_acknowledgment,
            "max_call_duration_seconds": self.max_call_duration_seconds,
            "call_recording_enabled": self.call_recording_enabled,
            "voicemail_detection": self.voicemail_detection,
            "voicemail_action": self.voicemail_action,
            "escalation_enabled": self.escalation_enabled,
            "escalation_triggers": [t.value for t in self.escalation_triggers],
            "max_attempts_before_escalation": self.max_attempts_before_escalation,
            "compliance_mode": self.compliance_mode.value,
            "required_disclosures": self.required_disclosures,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorSettings":
        """Create from dictionary."""
        return cls(
            response_style=data.get("response_style", "conversational"),
            response_length=data.get("response_length", "moderate"),
            use_filler_words=data.get("use_filler_words", True),
            allow_interruptions=data.get("allow_interruptions", True),
            interruption_sensitivity=data.get("interruption_sensitivity", 0.5),
            silence_timeout_seconds=data.get("silence_timeout_seconds", 10),
            max_silence_prompts=data.get("max_silence_prompts", 3),
            thinking_acknowledgment=data.get("thinking_acknowledgment", True),
            max_call_duration_seconds=data.get("max_call_duration_seconds", 1800),
            call_recording_enabled=data.get("call_recording_enabled", True),
            voicemail_detection=data.get("voicemail_detection", True),
            voicemail_action=data.get("voicemail_action", "leave_message"),
            escalation_enabled=data.get("escalation_enabled", True),
            escalation_triggers=[
                EscalationTrigger(t) for t in data.get("escalation_triggers", [])
            ],
            max_attempts_before_escalation=data.get("max_attempts_before_escalation", 3),
            compliance_mode=ComplianceMode(data.get("compliance_mode", "none")),
            required_disclosures=data.get("required_disclosures", []),
        )


@dataclass
class PersonalityProfile:
    """Complete personality profile for an agent."""

    id: str
    name: str
    description: str = ""

    # Organization
    organization_id: str = ""

    # Identity
    agent_name: str = ""
    agent_role: str = ""
    company_name: str = ""

    # Personality traits
    primary_traits: List[PersonalityTrait] = field(default_factory=list)
    secondary_traits: List[PersonalityTrait] = field(default_factory=list)

    # Industry
    industry: IndustryType = IndustryType.CUSTOM
    specialization: str = ""

    # Settings
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    behavior: BehaviorSettings = field(default_factory=BehaviorSettings)

    # Communication style
    greeting_style: str = "warm"  # warm, professional, casual, formal
    farewell_style: str = "friendly"  # friendly, formal, brief
    empathy_level: str = "high"  # low, medium, high
    assertiveness: str = "moderate"  # passive, moderate, assertive

    # Language preferences
    preferred_phrases: List[str] = field(default_factory=list)
    avoided_phrases: List[str] = field(default_factory=list)
    custom_vocabulary: Dict[str, str] = field(default_factory=dict)

    # Templates
    system_prompt_template_id: Optional[str] = None
    greeting_template_id: Optional[str] = None
    farewell_template_id: Optional[str] = None

    # Status
    is_active: bool = True
    is_default: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"pers_{uuid.uuid4().hex[:24]}"
        if not self.primary_traits:
            self.primary_traits = [
                PersonalityTrait.PROFESSIONAL,
                PersonalityTrait.FRIENDLY,
            ]

    def get_trait_description(self) -> str:
        """Generate natural language description of personality traits."""
        traits = []
        for trait in self.primary_traits[:3]:
            traits.append(trait.value)

        if len(traits) == 1:
            return traits[0]
        elif len(traits) == 2:
            return f"{traits[0]} and {traits[1]}"
        else:
            return f"{', '.join(traits[:-1])}, and {traits[-1]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "organization_id": self.organization_id,
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "company_name": self.company_name,
            "primary_traits": [t.value for t in self.primary_traits],
            "secondary_traits": [t.value for t in self.secondary_traits],
            "industry": self.industry.value,
            "specialization": self.specialization,
            "voice": self.voice.to_dict(),
            "behavior": self.behavior.to_dict(),
            "greeting_style": self.greeting_style,
            "farewell_style": self.farewell_style,
            "empathy_level": self.empathy_level,
            "assertiveness": self.assertiveness,
            "preferred_phrases": self.preferred_phrases,
            "avoided_phrases": self.avoided_phrases,
            "custom_vocabulary": self.custom_vocabulary,
            "system_prompt_template_id": self.system_prompt_template_id,
            "greeting_template_id": self.greeting_template_id,
            "farewell_template_id": self.farewell_template_id,
            "is_active": self.is_active,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityProfile":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            organization_id=data.get("organization_id", ""),
            agent_name=data.get("agent_name", ""),
            agent_role=data.get("agent_role", ""),
            company_name=data.get("company_name", ""),
            primary_traits=[
                PersonalityTrait(t) for t in data.get("primary_traits", [])
            ],
            secondary_traits=[
                PersonalityTrait(t) for t in data.get("secondary_traits", [])
            ],
            industry=IndustryType(data.get("industry", "custom")),
            specialization=data.get("specialization", ""),
            voice=VoiceSettings.from_dict(data.get("voice", {})),
            behavior=BehaviorSettings.from_dict(data.get("behavior", {})),
            greeting_style=data.get("greeting_style", "warm"),
            farewell_style=data.get("farewell_style", "friendly"),
            empathy_level=data.get("empathy_level", "high"),
            assertiveness=data.get("assertiveness", "moderate"),
            preferred_phrases=data.get("preferred_phrases", []),
            avoided_phrases=data.get("avoided_phrases", []),
            custom_vocabulary=data.get("custom_vocabulary", {}),
            system_prompt_template_id=data.get("system_prompt_template_id"),
            greeting_template_id=data.get("greeting_template_id"),
            farewell_template_id=data.get("farewell_template_id"),
            is_active=data.get("is_active", True),
            is_default=data.get("is_default", False),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
        )


# =============================================================================
# Agent Configuration
# =============================================================================


@dataclass
class LLMSettings:
    """LLM configuration settings."""

    provider: str = "openai"
    model: str = "gpt-4"

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Response configuration
    response_timeout_seconds: int = 30
    stream_response: bool = True

    # Fallback
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_timeout_seconds": self.response_timeout_seconds,
            "stream_response": self.stream_response,
            "fallback_provider": self.fallback_provider,
            "fallback_model": self.fallback_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMSettings":
        """Create from dictionary."""
        return cls(
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 150),
            top_p=data.get("top_p", 1.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            response_timeout_seconds=data.get("response_timeout_seconds", 30),
            stream_response=data.get("stream_response", True),
            fallback_provider=data.get("fallback_provider"),
            fallback_model=data.get("fallback_model"),
        )


@dataclass
class FunctionDefinition:
    """Definition of a callable function for the agent."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Execution
    timeout_seconds: int = 30
    retry_on_failure: bool = False
    max_retries: int = 2

    # Requirements
    required_permissions: List[str] = field(default_factory=list)
    required_integrations: List[str] = field(default_factory=list)

    # Status
    is_enabled: bool = True

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "timeout_seconds": self.timeout_seconds,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
            "required_permissions": self.required_permissions,
            "required_integrations": self.required_integrations,
            "is_enabled": self.is_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            timeout_seconds=data.get("timeout_seconds", 30),
            retry_on_failure=data.get("retry_on_failure", False),
            max_retries=data.get("max_retries", 2),
            required_permissions=data.get("required_permissions", []),
            required_integrations=data.get("required_integrations", []),
            is_enabled=data.get("is_enabled", True),
        )


@dataclass
class TranscriptionSettings:
    """Speech-to-text transcription settings."""

    provider: str = "deepgram"
    model: str = "nova-2"

    # Language
    language: str = "en-US"

    # Features
    punctuation: bool = True
    profanity_filter: bool = False
    diarization: bool = False
    smart_format: bool = True

    # Sensitivity
    endpointing_timeout_ms: int = 700
    interim_results: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "language": self.language,
            "punctuation": self.punctuation,
            "profanity_filter": self.profanity_filter,
            "diarization": self.diarization,
            "smart_format": self.smart_format,
            "endpointing_timeout_ms": self.endpointing_timeout_ms,
            "interim_results": self.interim_results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionSettings":
        """Create from dictionary."""
        return cls(
            provider=data.get("provider", "deepgram"),
            model=data.get("model", "nova-2"),
            language=data.get("language", "en-US"),
            punctuation=data.get("punctuation", True),
            profanity_filter=data.get("profanity_filter", False),
            diarization=data.get("diarization", False),
            smart_format=data.get("smart_format", True),
            endpointing_timeout_ms=data.get("endpointing_timeout_ms", 700),
            interim_results=data.get("interim_results", True),
        )


@dataclass
class AgentConfiguration:
    """Complete agent configuration."""

    id: str
    name: str
    organization_id: str

    # Description
    description: str = ""

    # Personality
    personality_id: Optional[str] = None
    personality: Optional[PersonalityProfile] = None

    # Prompts
    system_prompt: str = ""
    first_message: str = ""

    # Templates
    templates: Dict[str, str] = field(default_factory=dict)  # category -> template_id

    # Settings
    llm: LLMSettings = field(default_factory=LLMSettings)
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)

    # Functions
    functions: List[FunctionDefinition] = field(default_factory=list)

    # Knowledge
    knowledge_base_ids: List[str] = field(default_factory=list)

    # Transfer targets
    transfer_targets: Dict[str, str] = field(default_factory=dict)

    # Status
    status: ConfigStatus = ConfigStatus.DRAFT

    # Version
    version: int = 1
    parent_version_id: Optional[str] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    # Usage tracking
    total_calls: int = 0
    total_minutes: float = 0.0
    last_used_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"cfg_{uuid.uuid4().hex[:24]}"

    def get_config_hash(self) -> str:
        """Get hash of configuration for change detection."""
        config_str = f"{self.system_prompt}:{self.llm.model}:{self.llm.temperature}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def clone(self, new_name: Optional[str] = None) -> "AgentConfiguration":
        """Create a clone of this configuration."""
        return AgentConfiguration(
            id=f"cfg_{uuid.uuid4().hex[:24]}",
            name=new_name or f"{self.name} (Copy)",
            organization_id=self.organization_id,
            description=self.description,
            personality_id=self.personality_id,
            system_prompt=self.system_prompt,
            first_message=self.first_message,
            templates=self.templates.copy(),
            llm=LLMSettings.from_dict(self.llm.to_dict()),
            transcription=TranscriptionSettings.from_dict(self.transcription.to_dict()),
            functions=[
                FunctionDefinition.from_dict(f.to_dict())
                for f in self.functions
            ],
            knowledge_base_ids=self.knowledge_base_ids.copy(),
            transfer_targets=self.transfer_targets.copy(),
            status=ConfigStatus.DRAFT,
            version=1,
            tags=self.tags.copy(),
            metadata=self.metadata.copy(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "organization_id": self.organization_id,
            "description": self.description,
            "personality_id": self.personality_id,
            "personality": self.personality.to_dict() if self.personality else None,
            "system_prompt": self.system_prompt,
            "first_message": self.first_message,
            "templates": self.templates,
            "llm": self.llm.to_dict(),
            "transcription": self.transcription.to_dict(),
            "functions": [f.to_dict() for f in self.functions],
            "knowledge_base_ids": self.knowledge_base_ids,
            "transfer_targets": self.transfer_targets,
            "status": self.status.value,
            "version": self.version,
            "parent_version_id": self.parent_version_id,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "total_calls": self.total_calls,
            "total_minutes": self.total_minutes,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfiguration":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            organization_id=data["organization_id"],
            description=data.get("description", ""),
            personality_id=data.get("personality_id"),
            personality=PersonalityProfile.from_dict(data["personality"]) if data.get("personality") else None,
            system_prompt=data.get("system_prompt", ""),
            first_message=data.get("first_message", ""),
            templates=data.get("templates", {}),
            llm=LLMSettings.from_dict(data.get("llm", {})),
            transcription=TranscriptionSettings.from_dict(data.get("transcription", {})),
            functions=[
                FunctionDefinition.from_dict(f)
                for f in data.get("functions", [])
            ],
            knowledge_base_ids=data.get("knowledge_base_ids", []),
            transfer_targets=data.get("transfer_targets", {}),
            status=ConfigStatus(data.get("status", "draft")),
            version=data.get("version", 1),
            parent_version_id=data.get("parent_version_id"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            created_by=data.get("created_by"),
            updated_by=data.get("updated_by"),
            total_calls=data.get("total_calls", 0),
            total_minutes=data.get("total_minutes", 0.0),
            last_used_at=datetime.fromisoformat(data["last_used_at"]) if data.get("last_used_at") else None,
        )


# =============================================================================
# Version Management
# =============================================================================


@dataclass
class ConfigVersion:
    """Version record for agent configuration."""

    id: str
    config_id: str
    version_number: int

    # Snapshot of configuration
    config_snapshot: Dict[str, Any]

    # Change tracking
    change_summary: str = ""
    changes: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    # Rollback info
    is_rollback: bool = False
    rolled_back_from: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"ver_{uuid.uuid4().hex[:24]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "config_id": self.config_id,
            "version_number": self.version_number,
            "config_snapshot": self.config_snapshot,
            "change_summary": self.change_summary,
            "changes": self.changes,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "is_rollback": self.is_rollback,
            "rolled_back_from": self.rolled_back_from,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigVersion":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            config_id=data["config_id"],
            version_number=data["version_number"],
            config_snapshot=data["config_snapshot"],
            change_summary=data.get("change_summary", ""),
            changes=data.get("changes", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            created_by=data.get("created_by"),
            is_rollback=data.get("is_rollback", False),
            rolled_back_from=data.get("rolled_back_from"),
        )


# =============================================================================
# Exceptions
# =============================================================================


class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    pass


class TemplateError(ConfigurationError):
    """Error with template operations."""
    pass


class ValidationError(ConfigurationError):
    """Validation error."""
    pass


class VersionError(ConfigurationError):
    """Error with version operations."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "PersonalityTrait",
    "TemplateCategory",
    "VariableType",
    "ConfigStatus",
    "IndustryType",
    "ComplianceMode",
    "EscalationTrigger",
    # Template types
    "TemplateVariable",
    "PromptTemplate",
    # Personality types
    "VoiceSettings",
    "BehaviorSettings",
    "PersonalityProfile",
    # Configuration types
    "LLMSettings",
    "FunctionDefinition",
    "TranscriptionSettings",
    "AgentConfiguration",
    # Version types
    "ConfigVersion",
    # Exceptions
    "ConfigurationError",
    "TemplateError",
    "ValidationError",
    "VersionError",
]
