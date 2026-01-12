"""
Agent Builder System

Fluent builder pattern for agent construction:
- AgentBuilder for step-by-step configuration
- AgentFactory for predefined templates
- Validation and verification
- Agent lifecycle management
"""

from typing import Optional, Dict, Any, List, Callable, Set, Type, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import copy
import json
import logging

from .config import (
    AgentConfig, AgentType, AgentStatus, AgentCapability,
    VoiceSettings, TranscriptionSettings, LLMSettings,
    InterruptionConfig, SilenceConfig, ErrorConfig,
    AgentValidator,
)
from .persona import Persona, PersonaConfig, PersonaTrait, SpeakingStyle, EmotionalTone
from .behavior import ConversationBehavior, BehaviorManager
from .prompt import SystemPromptBuilder, SystemPromptConfig, PromptRegistry

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BuilderError(Exception):
    """Error during agent building."""
    pass


class ValidationError(BuilderError):
    """Validation error during building."""
    pass


@dataclass
class AgentBlueprint:
    """Blueprint for creating agents."""
    blueprint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Configuration templates
    config_template: Optional[AgentConfig] = None
    persona_template: Optional[PersonaConfig] = None
    behavior_template: Optional[ConversationBehavior] = None
    prompt_config: Optional[SystemPromptConfig] = None

    # Customization points
    customizable_fields: Set[str] = field(default_factory=set)
    required_fields: Set[str] = field(default_factory=set)

    # Metadata
    category: str = ""
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class AgentBuilder:
    """
    Fluent builder for constructing agents.

    Features:
    - Step-by-step configuration
    - Validation at each step
    - Immutable intermediate states
    - Rollback support
    """

    def __init__(self, agent_type: AgentType = AgentType.INBOUND):
        self._config = AgentConfig(agent_type=agent_type)
        self._persona: Optional[Persona] = None
        self._behavior: Optional[ConversationBehavior] = None
        self._prompt_builder: Optional[SystemPromptBuilder] = None
        self._validators: List[Callable[[AgentConfig], List[str]]] = []
        self._hooks: Dict[str, List[Callable]] = {}
        self._history: List[Dict[str, Any]] = []

    def _save_state(self, action: str) -> None:
        """Save current state to history."""
        self._history.append({
            "action": action,
            "timestamp": datetime.utcnow(),
            "config": copy.deepcopy(self._config),
        })

    def _emit_hook(self, event: str, data: Any = None) -> None:
        """Emit hook event."""
        for hook in self._hooks.get(event, []):
            try:
                hook(data)
            except Exception as e:
                logger.warning(f"Hook error: {e}")

    # Identity configuration
    def with_name(self, name: str) -> "AgentBuilder":
        """Set agent name."""
        self._save_state("set_name")
        self._config.name = name
        return self

    def with_description(self, description: str) -> "AgentBuilder":
        """Set agent description."""
        self._save_state("set_description")
        self._config.description = description
        return self

    def with_type(self, agent_type: AgentType) -> "AgentBuilder":
        """Set agent type."""
        self._save_state("set_type")
        self._config.agent_type = agent_type
        return self

    def with_status(self, status: AgentStatus) -> "AgentBuilder":
        """Set agent status."""
        self._save_state("set_status")
        self._config.status = status
        return self

    # Capabilities
    def with_capability(self, capability: AgentCapability) -> "AgentBuilder":
        """Add single capability."""
        self._save_state("add_capability")
        self._config.capabilities.add(capability)
        return self

    def with_capabilities(self, *capabilities: AgentCapability) -> "AgentBuilder":
        """Add multiple capabilities."""
        self._save_state("add_capabilities")
        for cap in capabilities:
            self._config.capabilities.add(cap)
        return self

    def without_capability(self, capability: AgentCapability) -> "AgentBuilder":
        """Remove capability."""
        self._save_state("remove_capability")
        self._config.capabilities.discard(capability)
        return self

    # Voice configuration
    def with_voice(
        self,
        provider: str = "elevenlabs",
        voice_id: str = "",
        model: str = "eleven_turbo_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        **kwargs,
    ) -> "AgentBuilder":
        """Configure voice settings."""
        self._save_state("set_voice")
        self._config.voice = VoiceSettings(
            provider=provider,
            voice_id=voice_id,
            model=model,
            stability=stability,
            similarity_boost=similarity_boost,
            **kwargs,
        )
        return self

    def with_voice_settings(self, settings: VoiceSettings) -> "AgentBuilder":
        """Set voice settings directly."""
        self._save_state("set_voice_settings")
        self._config.voice = settings
        return self

    # Transcription configuration
    def with_transcription(
        self,
        provider: str = "deepgram",
        model: str = "nova-2",
        language: str = "en",
        **kwargs,
    ) -> "AgentBuilder":
        """Configure transcription settings."""
        self._save_state("set_transcription")
        self._config.transcription = TranscriptionSettings(
            provider=provider,
            model=model,
            language=language,
            **kwargs,
        )
        return self

    def with_transcription_settings(self, settings: TranscriptionSettings) -> "AgentBuilder":
        """Set transcription settings directly."""
        self._save_state("set_transcription_settings")
        self._config.transcription = settings
        return self

    # LLM configuration
    def with_llm(
        self,
        provider: str = "openai",
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> "AgentBuilder":
        """Configure LLM settings."""
        self._save_state("set_llm")
        self._config.llm = LLMSettings(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return self

    def with_llm_settings(self, settings: LLMSettings) -> "AgentBuilder":
        """Set LLM settings directly."""
        self._save_state("set_llm_settings")
        self._config.llm = settings
        return self

    # Behavior configuration
    def with_interruption(
        self,
        enabled: bool = True,
        threshold: float = 0.5,
        min_words: int = 3,
        **kwargs,
    ) -> "AgentBuilder":
        """Configure interruption handling."""
        self._save_state("set_interruption")
        self._config.interruption = InterruptionConfig(
            enabled=enabled,
            threshold=threshold,
            min_words=min_words,
            **kwargs,
        )
        return self

    def with_silence_handling(
        self,
        timeout_ms: int = 2000,
        max_count: int = 3,
        **kwargs,
    ) -> "AgentBuilder":
        """Configure silence handling."""
        self._save_state("set_silence")
        self._config.silence = SilenceConfig(
            timeout_ms=timeout_ms,
            max_count=max_count,
            **kwargs,
        )
        return self

    def with_error_handling(
        self,
        max_retries: int = 3,
        fallback_message: str = "",
        **kwargs,
    ) -> "AgentBuilder":
        """Configure error handling."""
        self._save_state("set_error")
        self._config.error = ErrorConfig(
            max_retries=max_retries,
            fallback_message=fallback_message,
            **kwargs,
        )
        return self

    # Prompt configuration
    def with_system_prompt(self, prompt: str) -> "AgentBuilder":
        """Set system prompt."""
        self._save_state("set_system_prompt")
        self._config.system_prompt = prompt
        return self

    def with_first_message(self, message: str) -> "AgentBuilder":
        """Set first message."""
        self._save_state("set_first_message")
        self._config.first_message = message
        return self

    def with_end_call_message(self, message: str) -> "AgentBuilder":
        """Set end call message."""
        self._save_state("set_end_call_message")
        self._config.end_call_message = message
        return self

    # Persona configuration
    def with_persona(self, persona: Persona) -> "AgentBuilder":
        """Set agent persona."""
        self._save_state("set_persona")
        self._persona = persona
        return self

    def with_persona_config(
        self,
        name: str = "",
        role: str = "",
        traits: Optional[List[PersonaTrait]] = None,
        speaking_style: SpeakingStyle = SpeakingStyle.PROFESSIONAL,
        **kwargs,
    ) -> "AgentBuilder":
        """Configure persona."""
        self._save_state("configure_persona")
        config = PersonaConfig(
            display_name=name,
            role=role,
            primary_traits=traits or [],
            speaking_style=speaking_style,
            **kwargs,
        )
        self._persona = Persona(config=config)
        return self

    # Behavior configuration
    def with_behavior(self, behavior: ConversationBehavior) -> "AgentBuilder":
        """Set conversation behavior."""
        self._save_state("set_behavior")
        self._behavior = behavior
        return self

    # Tools and functions
    def with_tool(self, tool: Dict[str, Any]) -> "AgentBuilder":
        """Add tool/function."""
        self._save_state("add_tool")
        if not self._config.tools:
            self._config.tools = []
        self._config.tools.append(tool)
        return self

    def with_tools(self, tools: List[Dict[str, Any]]) -> "AgentBuilder":
        """Add multiple tools."""
        self._save_state("add_tools")
        if not self._config.tools:
            self._config.tools = []
        self._config.tools.extend(tools)
        return self

    # Webhooks
    def with_webhook(self, event: str, url: str, **kwargs) -> "AgentBuilder":
        """Add webhook."""
        self._save_state("add_webhook")
        if not self._config.webhooks:
            self._config.webhooks = {}
        self._config.webhooks[event] = {"url": url, **kwargs}
        return self

    # Metadata
    def with_metadata(self, key: str, value: Any) -> "AgentBuilder":
        """Add metadata."""
        self._save_state("add_metadata")
        self._config.metadata[key] = value
        return self

    def with_tag(self, tag: str) -> "AgentBuilder":
        """Add tag."""
        self._save_state("add_tag")
        if tag not in self._config.tags:
            self._config.tags.append(tag)
        return self

    # Validation
    def add_validator(
        self,
        validator: Callable[[AgentConfig], List[str]],
    ) -> "AgentBuilder":
        """Add custom validator."""
        self._validators.append(validator)
        return self

    def on_hook(self, event: str, handler: Callable) -> "AgentBuilder":
        """Register build hook."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(handler)
        return self

    def validate(self) -> List[str]:
        """Validate current configuration."""
        errors = []

        # Run built-in validator
        validator = AgentValidator()
        errors.extend(validator.validate(self._config))

        # Run custom validators
        for custom_validator in self._validators:
            try:
                errors.extend(custom_validator(self._config))
            except Exception as e:
                errors.append(f"Validator error: {str(e)}")

        return errors

    def rollback(self, steps: int = 1) -> "AgentBuilder":
        """Rollback to previous state."""
        if steps > len(self._history):
            steps = len(self._history)

        if steps > 0 and self._history:
            target_index = len(self._history) - steps
            if target_index >= 0:
                self._config = copy.deepcopy(self._history[target_index]["config"])
                self._history = self._history[:target_index]

        return self

    def reset(self) -> "AgentBuilder":
        """Reset builder to initial state."""
        self._config = AgentConfig()
        self._persona = None
        self._behavior = None
        self._prompt_builder = None
        self._history = []
        return self

    def build(self, validate: bool = True) -> "Agent":
        """Build the agent."""
        self._emit_hook("pre_build", self._config)

        if validate:
            errors = self.validate()
            if errors:
                raise ValidationError(f"Validation failed: {', '.join(errors)}")

        # Apply persona to system prompt if set
        if self._persona:
            persona_prompt = self._persona.get_system_prompt_section()
            if persona_prompt:
                self._config.system_prompt = f"{persona_prompt}\n\n{self._config.system_prompt}"

            # Set first message from persona if not set
            if not self._config.first_message:
                self._config.first_message = self._persona.get_greeting()

        # Create agent
        agent = Agent(
            config=copy.deepcopy(self._config),
            persona=self._persona,
            behavior=self._behavior,
        )

        self._emit_hook("post_build", agent)

        return agent

    def to_dict(self) -> Dict[str, Any]:
        """Export builder state as dictionary."""
        return {
            "config": self._config.to_dict() if hasattr(self._config, 'to_dict') else {},
            "persona": self._persona.config.to_dict() if self._persona else None,
            "behavior_id": self._behavior.behavior_id if self._behavior else None,
            "history_length": len(self._history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentBuilder":
        """Create builder from dictionary."""
        builder = cls()
        # Restore configuration from dict
        config_data = data.get("config", {})
        for key, value in config_data.items():
            if hasattr(builder._config, key):
                setattr(builder._config, key, value)
        return builder

    @classmethod
    def from_config(cls, config: AgentConfig) -> "AgentBuilder":
        """Create builder from existing config."""
        builder = cls(agent_type=config.agent_type)
        builder._config = copy.deepcopy(config)
        return builder


@dataclass
class Agent:
    """Complete agent instance."""
    config: AgentConfig
    persona: Optional[Persona] = None
    behavior: Optional[ConversationBehavior] = None

    # Runtime state
    is_active: bool = False
    current_calls: int = 0
    total_calls: int = 0

    # Statistics
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def name(self) -> str:
        return self.config.name

    def activate(self) -> None:
        """Activate agent."""
        self.is_active = True
        self.config.status = AgentStatus.ACTIVE

    def deactivate(self) -> None:
        """Deactivate agent."""
        self.is_active = False
        self.config.status = AgentStatus.INACTIVE

    def record_call(self) -> None:
        """Record a call."""
        self.current_calls += 1
        self.total_calls += 1
        self.last_used = datetime.utcnow()

    def end_call(self) -> None:
        """End a call."""
        self.current_calls = max(0, self.current_calls - 1)

    def get_system_prompt(self) -> str:
        """Get complete system prompt."""
        prompt = self.config.system_prompt

        if self.persona:
            persona_section = self.persona.get_system_prompt_section()
            if persona_section:
                prompt = f"{persona_section}\n\n{prompt}"

        return prompt

    def get_greeting(self) -> str:
        """Get greeting message."""
        if self.config.first_message:
            return self.config.first_message
        if self.persona:
            return self.persona.get_greeting()
        return "Hello, how can I help you today?"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
            "is_active": self.is_active,
            "current_calls": self.current_calls,
            "total_calls": self.total_calls,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


class AgentFactory:
    """
    Factory for creating agents from templates.

    Features:
    - Predefined agent templates
    - Blueprint-based creation
    - Batch agent creation
    """

    def __init__(self):
        self._blueprints: Dict[str, AgentBlueprint] = {}
        self._templates: Dict[str, Callable[[], AgentBuilder]] = {}
        self._register_default_templates()

    def _register_default_templates(self) -> None:
        """Register default agent templates."""
        self._templates["customer_service"] = self._create_customer_service_template
        self._templates["sales"] = self._create_sales_template
        self._templates["technical_support"] = self._create_technical_support_template
        self._templates["appointment_scheduler"] = self._create_appointment_scheduler_template
        self._templates["survey"] = self._create_survey_template
        self._templates["ivr"] = self._create_ivr_template
        self._templates["outbound_sales"] = self._create_outbound_sales_template
        self._templates["lead_qualifier"] = self._create_lead_qualifier_template

    def _create_customer_service_template(self) -> AgentBuilder:
        """Create customer service agent template."""
        return (
            AgentBuilder(AgentType.INBOUND)
            .with_name("Customer Service Agent")
            .with_description("Handles customer inquiries and support requests")
            .with_capabilities(
                AgentCapability.INBOUND_CALLS,
                AgentCapability.TRANSFER,
                AgentCapability.VOICEMAIL,
            )
            .with_voice(
                provider="elevenlabs",
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
                stability=0.6,
                similarity_boost=0.8,
            )
            .with_transcription(
                provider="deepgram",
                model="nova-2",
                language="en",
            )
            .with_llm(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.7,
                max_tokens=500,
            )
            .with_persona_config(
                name="Alex",
                role="Customer Service Representative",
                traits=[PersonaTrait.FRIENDLY, PersonaTrait.HELPFUL, PersonaTrait.PATIENT],
                speaking_style=SpeakingStyle.FRIENDLY,
            )
            .with_interruption(enabled=True, threshold=0.5)
            .with_silence_handling(timeout_ms=3000, max_count=3)
            .with_system_prompt("""You are a friendly and professional customer service representative.
Your goal is to help customers with their inquiries efficiently and empathetically.

Guidelines:
- Listen actively to the customer's concerns
- Provide clear and accurate information
- Offer solutions proactively
- Escalate to a human agent when necessary
- Always maintain a positive and helpful tone""")
        )

    def _create_sales_template(self) -> AgentBuilder:
        """Create sales agent template."""
        return (
            AgentBuilder(AgentType.INBOUND)
            .with_name("Sales Agent")
            .with_description("Handles sales inquiries and product information")
            .with_capabilities(
                AgentCapability.INBOUND_CALLS,
                AgentCapability.OUTBOUND_CALLS,
                AgentCapability.TRANSFER,
            )
            .with_voice(
                provider="elevenlabs",
                voice_id="EXAVITQu4vr4xnSDxMaL",  # Bella
                stability=0.5,
                similarity_boost=0.85,
            )
            .with_llm(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.8,
                max_tokens=600,
            )
            .with_persona_config(
                name="Morgan",
                role="Sales Representative",
                traits=[PersonaTrait.ENTHUSIASTIC, PersonaTrait.FRIENDLY, PersonaTrait.DIRECT],
                speaking_style=SpeakingStyle.CONVERSATIONAL,
            )
            .with_system_prompt("""You are an enthusiastic sales representative.
Your goal is to understand customer needs and present relevant solutions.

Guidelines:
- Build rapport with potential customers
- Ask discovery questions to understand needs
- Present product benefits clearly
- Handle objections gracefully
- Create urgency without being pushy
- Guide customers toward a decision""")
        )

    def _create_technical_support_template(self) -> AgentBuilder:
        """Create technical support agent template."""
        return (
            AgentBuilder(AgentType.INBOUND)
            .with_name("Technical Support Agent")
            .with_description("Provides technical assistance and troubleshooting")
            .with_capabilities(
                AgentCapability.INBOUND_CALLS,
                AgentCapability.TRANSFER,
                AgentCapability.RECORDING,
            )
            .with_voice(
                provider="elevenlabs",
                voice_id="yoZ06aMxZJJ28mfd3POQ",  # Sam
                stability=0.7,
                similarity_boost=0.75,
            )
            .with_llm(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.5,
                max_tokens=800,
            )
            .with_persona_config(
                name="Jordan",
                role="Technical Support Specialist",
                traits=[PersonaTrait.EXPERT, PersonaTrait.PATIENT, PersonaTrait.DETAILED],
                speaking_style=SpeakingStyle.TECHNICAL,
            )
            .with_silence_handling(timeout_ms=5000, max_count=5)
            .with_system_prompt("""You are a knowledgeable technical support specialist.
Your goal is to help users resolve technical issues efficiently.

Guidelines:
- Gather detailed information about the issue
- Walk users through troubleshooting steps clearly
- Explain technical concepts in simple terms
- Document issues for escalation if needed
- Verify the solution works before ending the call""")
        )

    def _create_appointment_scheduler_template(self) -> AgentBuilder:
        """Create appointment scheduler agent template."""
        return (
            AgentBuilder(AgentType.INBOUND)
            .with_name("Appointment Scheduler")
            .with_description("Schedules and manages appointments")
            .with_capabilities(
                AgentCapability.INBOUND_CALLS,
                AgentCapability.OUTBOUND_CALLS,
            )
            .with_voice(
                provider="elevenlabs",
                voice_id="21m00Tcm4TlvDq8ikWAM",
                stability=0.65,
                similarity_boost=0.8,
            )
            .with_llm(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.6,
                max_tokens=400,
            )
            .with_persona_config(
                name="Taylor",
                role="Scheduling Coordinator",
                traits=[PersonaTrait.PROFESSIONAL, PersonaTrait.CONCISE, PersonaTrait.HELPFUL],
                speaking_style=SpeakingStyle.PROFESSIONAL,
            )
            .with_system_prompt("""You are a professional scheduling coordinator.
Your goal is to efficiently schedule appointments and manage calendars.

Guidelines:
- Confirm caller identity when necessary
- Check availability before offering times
- Clearly confirm appointment details
- Send confirmation information
- Handle rescheduling and cancellations gracefully""")
        )

    def _create_survey_template(self) -> AgentBuilder:
        """Create survey agent template."""
        return (
            AgentBuilder(AgentType.OUTBOUND)
            .with_name("Survey Agent")
            .with_description("Conducts surveys and collects feedback")
            .with_capabilities(
                AgentCapability.OUTBOUND_CALLS,
                AgentCapability.RECORDING,
            )
            .with_voice(
                provider="elevenlabs",
                voice_id="MF3mGyEYCl7XYWbV9V6O",  # Emily
                stability=0.7,
                similarity_boost=0.75,
            )
            .with_llm(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.5,
                max_tokens=300,
            )
            .with_persona_config(
                name="Casey",
                role="Survey Researcher",
                traits=[PersonaTrait.FRIENDLY, PersonaTrait.PATIENT, PersonaTrait.ENCOURAGING],
                speaking_style=SpeakingStyle.CONVERSATIONAL,
            )
            .with_system_prompt("""You are a friendly survey researcher.
Your goal is to collect accurate feedback while respecting respondents' time.

Guidelines:
- Introduce yourself and the survey purpose clearly
- Ask questions neutrally without leading
- Accept all responses without judgment
- Keep the pace comfortable
- Thank respondents for their time""")
        )

    def _create_ivr_template(self) -> AgentBuilder:
        """Create IVR agent template."""
        return (
            AgentBuilder(AgentType.INBOUND)
            .with_name("IVR Agent")
            .with_description("Handles call routing and basic inquiries")
            .with_capabilities(
                AgentCapability.INBOUND_CALLS,
                AgentCapability.TRANSFER,
            )
            .with_voice(
                provider="elevenlabs",
                voice_id="21m00Tcm4TlvDq8ikWAM",
                stability=0.8,
                similarity_boost=0.7,
            )
            .with_llm(
                provider="openai",
                model="gpt-3.5-turbo",  # Faster for simple routing
                temperature=0.3,
                max_tokens=200,
            )
            .with_persona_config(
                name="Assistant",
                role="Automated Assistant",
                traits=[PersonaTrait.PROFESSIONAL, PersonaTrait.CONCISE],
                speaking_style=SpeakingStyle.FORMAL,
            )
            .with_system_prompt("""You are an automated phone assistant.
Your goal is to quickly route callers to the appropriate department or provide basic information.

Guidelines:
- Greet callers professionally
- Determine the purpose of the call quickly
- Provide clear menu options when needed
- Route calls accurately
- Handle common questions directly""")
        )

    def _create_outbound_sales_template(self) -> AgentBuilder:
        """Create outbound sales agent template."""
        return (
            AgentBuilder(AgentType.OUTBOUND)
            .with_name("Outbound Sales Agent")
            .with_description("Makes outbound sales calls")
            .with_capabilities(
                AgentCapability.OUTBOUND_CALLS,
                AgentCapability.RECORDING,
            )
            .with_voice(
                provider="elevenlabs",
                voice_id="EXAVITQu4vr4xnSDxMaL",
                stability=0.5,
                similarity_boost=0.9,
            )
            .with_llm(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.8,
                max_tokens=500,
            )
            .with_persona_config(
                name="Chris",
                role="Business Development Representative",
                traits=[PersonaTrait.ENERGETIC, PersonaTrait.FRIENDLY, PersonaTrait.DIRECT],
                speaking_style=SpeakingStyle.CONVERSATIONAL,
            )
            .with_system_prompt("""You are a professional outbound sales representative.
Your goal is to introduce products/services and generate interest.

Guidelines:
- Introduce yourself and your company clearly
- Respect the prospect's time
- Focus on value and benefits
- Handle objections professionally
- Aim to schedule follow-up or close
- Accept 'no' gracefully""")
        )

    def _create_lead_qualifier_template(self) -> AgentBuilder:
        """Create lead qualifier agent template."""
        return (
            AgentBuilder(AgentType.OUTBOUND)
            .with_name("Lead Qualifier")
            .with_description("Qualifies leads through discovery questions")
            .with_capabilities(
                AgentCapability.OUTBOUND_CALLS,
                AgentCapability.INBOUND_CALLS,
            )
            .with_voice(
                provider="elevenlabs",
                voice_id="21m00Tcm4TlvDq8ikWAM",
                stability=0.6,
                similarity_boost=0.8,
            )
            .with_llm(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.6,
                max_tokens=400,
            )
            .with_persona_config(
                name="Sam",
                role="Lead Qualification Specialist",
                traits=[PersonaTrait.FRIENDLY, PersonaTrait.DIRECT, PersonaTrait.EMPATHETIC],
                speaking_style=SpeakingStyle.PROFESSIONAL,
            )
            .with_system_prompt("""You are a lead qualification specialist.
Your goal is to determine if leads are a good fit for our solutions.

Guidelines:
- Ask qualifying questions naturally
- Listen for budget, authority, need, and timeline (BANT)
- Score leads based on responses
- Route qualified leads to sales
- Thank unqualified leads politely""")
        )

    def register_blueprint(self, blueprint: AgentBlueprint) -> None:
        """Register agent blueprint."""
        self._blueprints[blueprint.blueprint_id] = blueprint

    def register_template(
        self,
        name: str,
        template_func: Callable[[], AgentBuilder],
    ) -> None:
        """Register custom template."""
        self._templates[name] = template_func

    def create(self, template_name: str) -> Agent:
        """Create agent from template."""
        template_func = self._templates.get(template_name)
        if not template_func:
            raise ValueError(f"Unknown template: {template_name}")

        builder = template_func()
        return builder.build()

    def create_from_blueprint(
        self,
        blueprint_id: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """Create agent from blueprint with optional overrides."""
        blueprint = self._blueprints.get(blueprint_id)
        if not blueprint:
            raise ValueError(f"Unknown blueprint: {blueprint_id}")

        # Start with template config
        builder = AgentBuilder()

        if blueprint.config_template:
            builder = AgentBuilder.from_config(blueprint.config_template)

        if blueprint.persona_template:
            builder.with_persona(Persona(config=copy.deepcopy(blueprint.persona_template)))

        if blueprint.behavior_template:
            builder.with_behavior(copy.deepcopy(blueprint.behavior_template))

        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if key in blueprint.customizable_fields or not blueprint.customizable_fields:
                    if hasattr(builder._config, key):
                        setattr(builder._config, key, value)

        return builder.build()

    def create_batch(
        self,
        template_name: str,
        count: int,
        name_pattern: str = "{template}_{index}",
    ) -> List[Agent]:
        """Create multiple agents from template."""
        agents = []

        for i in range(count):
            builder = self._templates[template_name]()
            name = name_pattern.format(template=template_name, index=i + 1)
            builder.with_name(name)
            agents.append(builder.build())

        return agents

    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self._templates.keys())

    def list_blueprints(self) -> List[AgentBlueprint]:
        """List available blueprints."""
        return list(self._blueprints.values())

    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a template."""
        template_func = self._templates.get(template_name)
        if not template_func:
            return None

        builder = template_func()
        return builder.to_dict()


class AgentRegistry:
    """
    Registry for managing agent instances.

    Features:
    - Agent storage and retrieval
    - Lifecycle management
    - Concurrency control
    """

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._by_name: Dict[str, str] = {}
        self._by_status: Dict[AgentStatus, Set[str]] = {status: set() for status in AgentStatus}

    def register(self, agent: Agent) -> None:
        """Register agent."""
        self._agents[agent.agent_id] = agent
        self._by_name[agent.name] = agent.agent_id
        self._by_status[agent.config.status].add(agent.agent_id)

    def get(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def get_by_name(self, name: str) -> Optional[Agent]:
        """Get agent by name."""
        agent_id = self._by_name.get(name)
        return self._agents.get(agent_id) if agent_id else None

    def get_active(self) -> List[Agent]:
        """Get all active agents."""
        return [
            self._agents[aid]
            for aid in self._by_status.get(AgentStatus.ACTIVE, set())
            if aid in self._agents
        ]

    def update_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        # Remove from old status set
        old_status = agent.config.status
        self._by_status[old_status].discard(agent_id)

        # Update and add to new status set
        agent.config.status = status
        self._by_status[status].add(agent_id)

        return True

    def unregister(self, agent_id: str) -> bool:
        """Unregister agent."""
        agent = self._agents.pop(agent_id, None)
        if agent:
            self._by_name.pop(agent.name, None)
            for status_set in self._by_status.values():
                status_set.discard(agent_id)
            return True
        return False

    def list_all(self) -> List[Agent]:
        """List all agents."""
        return list(self._agents.values())

    def count(self) -> int:
        """Count total agents."""
        return len(self._agents)

    def count_by_status(self, status: AgentStatus) -> int:
        """Count agents by status."""
        return len(self._by_status.get(status, set()))

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total": len(self._agents),
            "by_status": {
                status.value: len(agent_ids)
                for status, agent_ids in self._by_status.items()
            },
            "total_calls": sum(a.total_calls for a in self._agents.values()),
            "active_calls": sum(a.current_calls for a in self._agents.values()),
        }


# Convenience functions
def create_agent(template: str = "customer_service", **kwargs) -> Agent:
    """Create agent from template with overrides."""
    factory = AgentFactory()
    builder = factory._templates[template]()

    for key, value in kwargs.items():
        method_name = f"with_{key}"
        if hasattr(builder, method_name):
            getattr(builder, method_name)(value)

    return builder.build()


def quick_agent(
    name: str,
    prompt: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    **kwargs,
) -> Agent:
    """Quickly create a basic agent."""
    return (
        AgentBuilder()
        .with_name(name)
        .with_system_prompt(prompt)
        .with_voice(voice_id=voice_id)
        .with_transcription()
        .with_llm()
        .build(validate=False)
    )
