"""
Agent Factory Module

This module provides automatic generation of AI voice agents from business information.
It analyzes business data and creates complete, deployable agent configurations.
"""

from .base import (
    # Business Information
    BusinessInfo,
    BusinessCategory,
    ContactInfo,
    BusinessHours,
    HoursOfOperation,
    ProductInfo,
    ServiceInfo,
    FAQEntry,
    PolicyInfo,
    TeamMember,
    LocationInfo,
    # Agent Configuration
    AgentConfig,
    AgentPersona,
    VoiceConfig,
    BehaviorConfig,
    GreetingConfig,
    TransferConfig,
    EscalationConfig,
    ComplianceConfig,
    # Generation
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
)

from .analyzer import (
    BusinessAnalyzer,
    AnalysisResult,
    BusinessInsights,
    ConversationTopics,
    KeyEntities,
)

from .persona import (
    PersonaGenerator,
    PersonaTemplate,
    PersonaTraits,
    CommunicationStyle,
    INDUSTRY_PERSONAS,
)

from .prompts import (
    PromptGenerator,
    SystemPrompt,
    PromptTemplate,
    PromptContext,
    PromptBuilder,
)

from .flows_generator import (
    FlowGenerator,
    FlowTemplate,
    GeneratedFlow,
    FlowLibrary,
    COMMON_FLOWS,
)

from .knowledge_builder import (
    KnowledgeBuilder,
    KnowledgeConfig,
    KnowledgeSource,
    ProcessedKnowledge,
)

from .factory import (
    AgentFactory,
    FactoryConfig,
    BuildResult,
    AgentBuilder,
    create_agent_factory,
)


__all__ = [
    # Business Information
    "BusinessInfo",
    "BusinessCategory",
    "ContactInfo",
    "BusinessHours",
    "HoursOfOperation",
    "ProductInfo",
    "ServiceInfo",
    "FAQEntry",
    "PolicyInfo",
    "TeamMember",
    "LocationInfo",
    # Agent Configuration
    "AgentConfig",
    "AgentPersona",
    "VoiceConfig",
    "BehaviorConfig",
    "GreetingConfig",
    "TransferConfig",
    "EscalationConfig",
    "ComplianceConfig",
    # Generation
    "GenerationRequest",
    "GenerationResult",
    "GenerationStatus",
    # Analyzer
    "BusinessAnalyzer",
    "AnalysisResult",
    "BusinessInsights",
    "ConversationTopics",
    "KeyEntities",
    # Persona
    "PersonaGenerator",
    "PersonaTemplate",
    "PersonaTraits",
    "CommunicationStyle",
    "INDUSTRY_PERSONAS",
    # Prompts
    "PromptGenerator",
    "SystemPrompt",
    "PromptTemplate",
    "PromptContext",
    "PromptBuilder",
    # Flows
    "FlowGenerator",
    "FlowTemplate",
    "GeneratedFlow",
    "FlowLibrary",
    "COMMON_FLOWS",
    # Knowledge
    "KnowledgeBuilder",
    "KnowledgeConfig",
    "KnowledgeSource",
    "ProcessedKnowledge",
    # Factory
    "AgentFactory",
    "FactoryConfig",
    "BuildResult",
    "AgentBuilder",
    "create_agent_factory",
]
