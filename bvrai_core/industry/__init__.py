"""
Industry Intelligence Module

This module provides comprehensive industry-specific intelligence
for voice agents, including profiles, compliance, terminology,
behaviors, and conversation patterns.
"""

from .base import (
    # Industry types
    IndustryType,
    # Compliance
    ComplianceLevel,
    RegulationType,
    ComplianceRequirement,
    # Conversation
    ConversationPhase,
    SentimentCategory,
    ConversationPattern,
    BehaviorGuideline,
    # Services and protocols
    ServiceOffering,
    EmergencyProtocol,
    # Metrics and patterns
    IndustryMetric,
    SeasonalPattern,
    # Core types
    TermDefinition,
    IndustryProfile,
    IndustryContext,
)

from .profiles import (
    INDUSTRY_PROFILES,
    get_industry_profile,
    get_all_industry_types,
    get_industry_by_name,
)

from .compliance import (
    ComplianceViolation,
    ComplianceCheckResult,
    ComplianceChecker,
    ComplianceManager,
    COMPLIANCE_REGISTRY,
    get_requirements_for_regulation,
    get_industry_regulations,
    create_compliance_checker,
)

from .terminology import (
    TerminologyManager,
    TERMINOLOGY_REGISTRY,
    get_industry_terminology,
    create_terminology_manager,
)

from .behaviors import (
    ResponseStyle,
    IntentHandler,
    SentimentResponse,
    TimeBasedBehavior,
    SENTIMENT_RESPONSES,
    ConversationBehavior,
    BehaviorEngine,
    create_behavior_handler,
    create_behavior_engine,
)

from .intelligence import (
    IntelligenceConfig,
    ProcessingResult,
    SessionIntelligence,
    IndustryIntelligence,
    create_industry_intelligence,
)


__all__ = [
    # Base types
    "IndustryType",
    "ComplianceLevel",
    "RegulationType",
    "ComplianceRequirement",
    "ConversationPhase",
    "SentimentCategory",
    "ConversationPattern",
    "BehaviorGuideline",
    "ServiceOffering",
    "EmergencyProtocol",
    "IndustryMetric",
    "SeasonalPattern",
    "TermDefinition",
    "IndustryProfile",
    "IndustryContext",
    # Profiles
    "INDUSTRY_PROFILES",
    "get_industry_profile",
    "get_all_industry_types",
    "get_industry_by_name",
    # Compliance
    "ComplianceViolation",
    "ComplianceCheckResult",
    "ComplianceChecker",
    "ComplianceManager",
    "COMPLIANCE_REGISTRY",
    "get_requirements_for_regulation",
    "get_industry_regulations",
    "create_compliance_checker",
    # Terminology
    "TerminologyManager",
    "TERMINOLOGY_REGISTRY",
    "get_industry_terminology",
    "create_terminology_manager",
    # Behaviors
    "ResponseStyle",
    "IntentHandler",
    "SentimentResponse",
    "TimeBasedBehavior",
    "SENTIMENT_RESPONSES",
    "ConversationBehavior",
    "BehaviorEngine",
    "create_behavior_handler",
    "create_behavior_engine",
    # Intelligence
    "IntelligenceConfig",
    "ProcessingResult",
    "SessionIntelligence",
    "IndustryIntelligence",
    "create_industry_intelligence",
]
