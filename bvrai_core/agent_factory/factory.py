"""
Agent Factory Module

This is the main orchestrator that combines all components to
automatically generate complete AI voice agents from business information.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from .base import (
    BusinessInfo,
    BusinessCategory,
    AgentConfig,
    AgentPersona,
    VoiceConfig,
    BehaviorConfig,
    GreetingConfig,
    TransferConfig,
    EscalationConfig,
    ComplianceConfig,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
)
from .analyzer import BusinessAnalyzer, AnalysisResult
from .persona import PersonaGenerator
from .prompts import PromptGenerator, SystemPrompt
from .flows_generator import FlowGenerator, GeneratedFlow
from .knowledge_builder import KnowledgeBuilder, KnowledgeConfig, ProcessedKnowledge


logger = logging.getLogger(__name__)


@dataclass
class FactoryConfig:
    """Configuration for the agent factory."""

    # Component settings
    enable_llm_generation: bool = False
    llm_provider_name: str = "openai"

    # Knowledge base settings
    knowledge_config: KnowledgeConfig = field(default_factory=KnowledgeConfig)

    # Default behavior settings
    default_behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    default_compliance: ComplianceConfig = field(default_factory=ComplianceConfig)

    # Generation options
    generate_all_flows: bool = True
    include_industry_flows: bool = True
    generate_qa_pairs: bool = True

    # Voice defaults
    default_voice_provider: str = "elevenlabs"

    # Callbacks
    on_progress: Optional[Callable[[str, GenerationStatus, float], None]] = None

    # Validation
    validate_output: bool = True
    require_minimum_faqs: int = 0


@dataclass
class BuildResult:
    """Result of agent building."""

    # Status
    success: bool = False
    error: Optional[str] = None

    # Generated artifacts
    agent_config: Optional[AgentConfig] = None
    analysis: Optional[AnalysisResult] = None
    system_prompt: Optional[SystemPrompt] = None
    flows: List[GeneratedFlow] = field(default_factory=list)
    knowledge: Optional[ProcessedKnowledge] = None

    # Metrics
    processing_time_ms: float = 0.0
    tokens_used: int = 0

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "agent_id": self.agent_config.id if self.agent_config else None,
            "agent_name": self.agent_config.name if self.agent_config else None,
            "total_flows": len(self.flows),
            "processing_time_ms": self.processing_time_ms,
            "warnings": self.warnings,
        }


class AgentBuilder:
    """
    Builder pattern for creating agent configurations.

    Provides a fluent interface for customizing agent generation.
    """

    def __init__(self, factory: "AgentFactory"):
        self._factory = factory
        self._business_info: Optional[BusinessInfo] = None
        self._persona_prefs: Dict[str, Any] = {}
        self._voice_prefs: Dict[str, Any] = {}
        self._behavior_prefs: Dict[str, Any] = {}
        self._enabled_features: Dict[str, bool] = {}
        self._custom_flows: List[GeneratedFlow] = []
        self._custom_prompts: Dict[str, str] = {}

    def with_business_info(self, business_info: BusinessInfo) -> "AgentBuilder":
        """Set business information."""
        self._business_info = business_info
        return self

    def with_persona(
        self,
        name: Optional[str] = None,
        traits: Optional[List[str]] = None,
        communication_style: Optional[str] = None,
    ) -> "AgentBuilder":
        """Set persona preferences."""
        if name:
            self._persona_prefs["name"] = name
        if traits:
            self._persona_prefs["traits"] = traits
        if communication_style:
            self._persona_prefs["communication_style"] = communication_style
        return self

    def with_voice(
        self,
        provider: Optional[str] = None,
        voice_id: Optional[str] = None,
        gender: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> "AgentBuilder":
        """Set voice preferences."""
        if provider:
            self._voice_prefs["provider"] = provider
        if voice_id:
            self._voice_prefs["voice_id"] = voice_id
        if gender:
            self._voice_prefs["gender"] = gender
        if speed:
            self._voice_prefs["speed"] = speed
        return self

    def with_behavior(
        self,
        formality: Optional[str] = None,
        response_length: Optional[str] = None,
        show_empathy: Optional[bool] = None,
    ) -> "AgentBuilder":
        """Set behavior preferences."""
        if formality:
            self._behavior_prefs["formality_level"] = formality
        if response_length:
            self._behavior_prefs["response_length"] = response_length
        if show_empathy is not None:
            self._behavior_prefs["show_empathy"] = show_empathy
        return self

    def enable_scheduling(self, enabled: bool = True) -> "AgentBuilder":
        """Enable/disable scheduling capability."""
        self._enabled_features["scheduling"] = enabled
        return self

    def enable_lead_qualification(self, enabled: bool = True) -> "AgentBuilder":
        """Enable/disable lead qualification."""
        self._enabled_features["lead_qualification"] = enabled
        return self

    def enable_transfers(self, enabled: bool = True) -> "AgentBuilder":
        """Enable/disable call transfers."""
        self._enabled_features["transfers"] = enabled
        return self

    def add_custom_flow(self, flow: GeneratedFlow) -> "AgentBuilder":
        """Add a custom dialog flow."""
        self._custom_flows.append(flow)
        return self

    def with_custom_greeting(self, greeting: str) -> "AgentBuilder":
        """Set custom greeting."""
        self._custom_prompts["greeting"] = greeting
        return self

    async def build(self) -> BuildResult:
        """Build the agent."""
        if not self._business_info:
            return BuildResult(success=False, error="Business information is required")

        request = GenerationRequest(
            business_info=self._business_info,
            persona_preferences=self._persona_prefs,
            voice_preferences=self._voice_prefs,
            behavior_preferences=self._behavior_prefs,
        )

        return await self._factory.generate(
            request,
            custom_flows=self._custom_flows,
            custom_prompts=self._custom_prompts,
        )


class AgentFactory:
    """
    Main factory for generating AI voice agents.

    Orchestrates all components to automatically create complete,
    deployable agent configurations from business information.
    """

    def __init__(
        self,
        config: Optional[FactoryConfig] = None,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize the agent factory.

        Args:
            config: Factory configuration
            llm_provider: Optional LLM provider for enhanced generation
        """
        self.config = config or FactoryConfig()
        self._llm_provider = llm_provider

        # Initialize components
        self._analyzer = BusinessAnalyzer(
            llm_provider=llm_provider,
            enable_llm_analysis=self.config.enable_llm_generation,
        )
        self._persona_generator = PersonaGenerator(
            llm_provider=llm_provider,
            enable_llm_generation=self.config.enable_llm_generation,
        )
        self._prompt_generator = PromptGenerator(
            llm_provider=llm_provider,
            enable_llm_generation=self.config.enable_llm_generation,
        )
        self._flow_generator = FlowGenerator(
            llm_provider=llm_provider,
            enable_llm_generation=self.config.enable_llm_generation,
        )
        self._knowledge_builder = KnowledgeBuilder(
            config=self.config.knowledge_config,
        )

    def builder(self) -> AgentBuilder:
        """Get an agent builder instance."""
        return AgentBuilder(self)

    async def generate(
        self,
        request: GenerationRequest,
        custom_flows: Optional[List[GeneratedFlow]] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> BuildResult:
        """
        Generate a complete agent from a request.

        Args:
            request: Generation request with business info
            custom_flows: Optional custom dialog flows
            custom_prompts: Optional custom prompts

        Returns:
            Build result with agent configuration
        """
        start_time = time.time()
        result = BuildResult()

        try:
            business_info = request.business_info

            # Step 1: Analyze business information
            self._emit_progress(request.id, GenerationStatus.ANALYZING, 0.1)
            logger.info(f"Analyzing business: {business_info.name}")

            analysis = await self._analyzer.analyze(business_info)
            result.analysis = analysis

            # Check for critical gaps
            if analysis.missing_information:
                for gap in analysis.missing_information:
                    result.warnings.append(f"Missing info: {gap}")

            # Step 2: Generate persona
            self._emit_progress(request.id, GenerationStatus.GENERATING_PERSONA, 0.2)
            logger.info("Generating agent persona")

            persona = await self._persona_generator.generate(
                business_info,
                preferences=request.persona_preferences,
            )

            # Apply voice preferences
            if request.voice_preferences:
                for key, value in request.voice_preferences.items():
                    if hasattr(persona.voice_config, key):
                        setattr(persona.voice_config, key, value)

            # Step 3: Build knowledge base
            self._emit_progress(request.id, GenerationStatus.BUILDING_KNOWLEDGE, 0.4)
            logger.info("Building knowledge base")

            knowledge = await self._knowledge_builder.build(business_info)
            result.knowledge = knowledge

            if knowledge.errors:
                for error in knowledge.errors:
                    result.warnings.append(f"Knowledge error: {error}")

            # Step 4: Generate dialog flows
            self._emit_progress(request.id, GenerationStatus.CREATING_FLOWS, 0.6)
            logger.info("Generating dialog flows")

            flows = await self._flow_generator.generate(business_info)

            # Add custom flows
            if custom_flows:
                flows.extend(custom_flows)

            result.flows = flows

            # Step 5: Generate system prompt
            self._emit_progress(request.id, GenerationStatus.GENERATING_PROMPTS, 0.8)
            logger.info("Generating system prompt")

            # Build behavior config from preferences
            behavior = self._build_behavior_config(
                request.behavior_preferences,
                analysis,
            )

            # Build compliance config
            compliance = self._build_compliance_config(
                business_info,
                analysis,
            )

            system_prompt = await self._prompt_generator.generate(
                business_info,
                persona,
                behavior=behavior,
                compliance=compliance,
            )

            # Apply custom prompts
            if custom_prompts:
                if "greeting" in custom_prompts:
                    # Update greeting in prompt
                    pass

            result.system_prompt = system_prompt

            # Step 6: Assemble agent configuration
            agent_config = self._assemble_agent_config(
                business_info=business_info,
                persona=persona,
                behavior=behavior,
                compliance=compliance,
                system_prompt=system_prompt,
                flows=flows,
                knowledge=knowledge,
                request=request,
            )

            result.agent_config = agent_config

            # Step 7: Validate (if enabled)
            if self.config.validate_output:
                self._emit_progress(request.id, GenerationStatus.VALIDATING, 0.9)
                validation_warnings = self._validate_agent(agent_config, analysis)
                result.warnings.extend(validation_warnings)

            # Complete
            result.success = True
            result.processing_time_ms = (time.time() - start_time) * 1000

            self._emit_progress(request.id, GenerationStatus.COMPLETED, 1.0)
            logger.info(
                f"Agent generation complete: {agent_config.name} "
                f"({result.processing_time_ms:.0f}ms)"
            )

        except Exception as e:
            logger.error(f"Agent generation failed: {e}")
            result.success = False
            result.error = str(e)
            result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    def _build_behavior_config(
        self,
        preferences: Dict[str, Any],
        analysis: AnalysisResult,
    ) -> BehaviorConfig:
        """Build behavior configuration."""
        config = BehaviorConfig()

        # Apply analysis recommendations
        config.formality_level = analysis.insights.formality_level

        # Apply preferences
        if "formality_level" in preferences:
            config.formality_level = preferences["formality_level"]
        if "response_length" in preferences:
            config.response_length = preferences["response_length"]
        if "show_empathy" in preferences:
            config.show_empathy = preferences["show_empathy"]

        return config

    def _build_compliance_config(
        self,
        business_info: BusinessInfo,
        analysis: AnalysisResult,
    ) -> ComplianceConfig:
        """Build compliance configuration."""
        config = ComplianceConfig()

        # Apply industry requirements
        config.hipaa_compliant = analysis.insights.requires_hipaa
        config.pci_compliant = analysis.insights.requires_pci

        return config

    def _assemble_agent_config(
        self,
        business_info: BusinessInfo,
        persona: AgentPersona,
        behavior: BehaviorConfig,
        compliance: ComplianceConfig,
        system_prompt: SystemPrompt,
        flows: List[GeneratedFlow],
        knowledge: ProcessedKnowledge,
        request: GenerationRequest,
    ) -> AgentConfig:
        """Assemble the complete agent configuration."""
        config = AgentConfig(
            name=f"{business_info.name} Voice Agent",
            business_id=business_info.id,
            business_name=business_info.name,
            persona=persona,
            behavior=behavior,
            compliance=compliance,
            system_prompt=system_prompt.compile(),
            dialog_flows=[f.id for f in flows],
            knowledge_base_id=knowledge.id if knowledge else None,
        )

        # Build greeting config
        greeting_template = (
            persona.sample_greetings[0]
            if persona.sample_greetings
            else f"Hello, thank you for calling {business_info.name}. How can I help you today?"
        )

        config.greeting = GreetingConfig(
            greeting_template=greeting_template,
        )

        # Build transfer config
        config.transfer = TransferConfig(
            enabled=True,
            default_transfer_number=business_info.contact.phone,
        )

        # Add team transfer destinations
        for member in business_info.team_members:
            if member.can_receive_transfers and member.extension:
                config.transfer.department_numbers[member.name] = member.extension

        # Build escalation config
        config.escalation = EscalationConfig()

        # Enable tools based on request
        if request.enable_appointment_booking:
            config.enabled_tools.append("appointment_booking")
        if request.enable_lead_qualification:
            config.enabled_tools.append("lead_qualification")
        if request.enable_order_taking:
            config.enabled_tools.append("order_taking")

        return config

    def _validate_agent(
        self,
        config: AgentConfig,
        analysis: AnalysisResult,
    ) -> List[str]:
        """Validate agent configuration."""
        warnings = []

        # Check system prompt length
        if len(config.system_prompt) > 10000:
            warnings.append("System prompt is very long, consider condensing")

        # Check for required flows
        required_flow_types = ["greeting", "closing"]
        flow_names_lower = [f.lower() for f in config.dialog_flows]

        for flow_type in required_flow_types:
            if not any(flow_type in name for name in flow_names_lower):
                warnings.append(f"Missing recommended flow: {flow_type}")

        # Check confidence
        if analysis.analysis_confidence < 0.5:
            warnings.append(
                f"Low analysis confidence ({analysis.analysis_confidence:.1%}). "
                "Consider providing more business information."
            )

        return warnings

    def _emit_progress(
        self,
        request_id: str,
        status: GenerationStatus,
        progress: float,
    ) -> None:
        """Emit progress callback."""
        if self.config.on_progress:
            try:
                self.config.on_progress(request_id, status, progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    async def generate_from_dict(self, data: Dict[str, Any]) -> BuildResult:
        """
        Generate agent from a dictionary of business data.

        Convenience method for API integration.

        Args:
            data: Dictionary with business information

        Returns:
            Build result
        """
        # Build BusinessInfo from dict
        business_info = self._dict_to_business_info(data)

        request = GenerationRequest(business_info=business_info)

        return await self.generate(request)

    def _dict_to_business_info(self, data: Dict[str, Any]) -> BusinessInfo:
        """Convert dictionary to BusinessInfo."""
        from .base import (
            ContactInfo, ProductInfo, ServiceInfo, FAQEntry,
            PolicyInfo, TeamMember, HoursOfOperation, BusinessHours,
        )
        from datetime import time as dt_time

        business_info = BusinessInfo()

        # Basic info
        business_info.name = data.get("name", "")
        business_info.description = data.get("description", "")
        business_info.tagline = data.get("tagline")
        business_info.about_us = data.get("about_us")

        # Category
        category_str = data.get("category", "other")
        try:
            business_info.category = BusinessCategory(category_str)
        except ValueError:
            business_info.category = BusinessCategory.OTHER

        # Contact
        if "contact" in data:
            contact_data = data["contact"]
            business_info.contact = ContactInfo(
                phone=contact_data.get("phone"),
                email=contact_data.get("email"),
                website=contact_data.get("website"),
                address=contact_data.get("address"),
                city=contact_data.get("city"),
                state=contact_data.get("state"),
                zip_code=contact_data.get("zip_code"),
            )

        # Services
        for service_data in data.get("services", []):
            service = ServiceInfo(
                name=service_data.get("name", ""),
                description=service_data.get("description", ""),
                price=service_data.get("price"),
                price_range=service_data.get("price_range"),
                price_type=service_data.get("price_type", "fixed"),
                duration_minutes=service_data.get("duration_minutes"),
                booking_required=service_data.get("booking_required", False),
            )
            business_info.services.append(service)

        # Products
        for product_data in data.get("products", []):
            product = ProductInfo(
                name=product_data.get("name", ""),
                description=product_data.get("description", ""),
                price=product_data.get("price"),
                category=product_data.get("category", ""),
            )
            business_info.products.append(product)

        # FAQs
        for faq_data in data.get("faqs", []):
            faq = FAQEntry(
                question=faq_data.get("question", ""),
                answer=faq_data.get("answer", ""),
                category=faq_data.get("category", "general"),
            )
            business_info.faqs.append(faq)

        # Team members
        for member_data in data.get("team_members", []):
            member = TeamMember(
                name=member_data.get("name", ""),
                role=member_data.get("role", ""),
                title=member_data.get("title"),
                phone=member_data.get("phone"),
                email=member_data.get("email"),
            )
            business_info.team_members.append(member)

        # Hours
        if "hours" in data:
            hours_data = data["hours"]
            hours = HoursOfOperation(
                timezone=hours_data.get("timezone", "America/New_York")
            )

            day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            for day_name in day_names:
                if day_name in hours_data:
                    day_data = hours_data[day_name]
                    if day_data.get("closed"):
                        day_hours = BusinessHours(is_closed=True)
                    else:
                        open_time = None
                        close_time = None
                        if "open" in day_data:
                            parts = day_data["open"].split(":")
                            open_time = dt_time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
                        if "close" in day_data:
                            parts = day_data["close"].split(":")
                            close_time = dt_time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
                        day_hours = BusinessHours(open_time=open_time, close_time=close_time)

                    setattr(hours, day_name, day_hours)

            business_info.hours = hours

        # Brand settings
        business_info.brand_tone = data.get("brand_tone", "professional")
        business_info.brand_personality = data.get("brand_personality", [])
        business_info.common_call_reasons = data.get("common_call_reasons", [])
        business_info.differentiators = data.get("differentiators", [])

        return business_info


def create_agent_factory(
    llm_provider: Optional[Any] = None,
    config: Optional[FactoryConfig] = None,
) -> AgentFactory:
    """
    Create an agent factory instance.

    Args:
        llm_provider: Optional LLM provider for enhanced generation
        config: Optional factory configuration

    Returns:
        Configured agent factory
    """
    return AgentFactory(config=config, llm_provider=llm_provider)


__all__ = [
    "AgentFactory",
    "AgentBuilder",
    "FactoryConfig",
    "BuildResult",
    "create_agent_factory",
]
