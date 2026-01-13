"""
Agent Configuration Module

This module provides comprehensive configuration management for voice agents,
including prompt templates, personality profiles, version control, and A/B testing.

Features:
- Prompt Templates: Create, manage, and render dynamic prompt templates
- Personality Profiles: Configure agent personality traits, voice, and behavior
- Version Control: Track changes and rollback configurations
- A/B Testing: Compare different configurations with statistical analysis
- Import/Export: Portable configuration format for sharing and backup

Example usage:

    from platform.agent_config import (
        AgentConfigurationService,
        AgentConfiguration,
        PersonalityProfile,
        PromptTemplate,
        IndustryType,
        PersonalityTrait,
        TemplateCategory,
    )

    # Initialize the service
    service = AgentConfigurationService()

    # Create a configuration from industry best practices
    config = await service.create_configuration_from_industry(
        name="Customer Support Agent",
        organization_id="org_123",
        company_name="Acme Corp",
        agent_name="Sarah",
        industry=IndustryType.RETAIL,
    )

    # Create a custom configuration
    config = await service.create_configuration(
        name="Sales Agent",
        organization_id="org_123",
        system_prompt="You are a helpful sales assistant...",
        first_message="Hi! Thanks for calling. How can I help?",
    )

    # Manage templates
    template = await service.templates.create_template(
        name="Custom Greeting",
        category=TemplateCategory.GREETING,
        content="Hello {{caller_name}}! Welcome to {{company_name}}.",
        organization_id="org_123",
    )

    # Render template
    greeting = service.templates.render_template(
        template,
        {"caller_name": "John", "company_name": "Acme Corp"},
    )

    # Create personality profile
    profile = await service.personalities.create_profile(
        name="Friendly Sales Rep",
        organization_id="org_123",
        agent_name="Alex",
        agent_role="Sales Representative",
        company_name="Acme Corp",
        industry=IndustryType.RETAIL,
        primary_traits=[PersonalityTrait.FRIENDLY, PersonalityTrait.ENTHUSIASTIC],
    )

    # Version control
    versions, total = await service.get_configuration_versions(config.id)

    # Rollback to previous version
    config = await service.rollback_configuration(
        config_id=config.id,
        target_version=1,
    )

    # A/B Testing
    test = await service.ab_tests.create_test(
        name="Greeting Test",
        organization_id="org_123",
    )
    await service.ab_tests.add_variant("Control", config_a.id, weight=0.5)
    await service.ab_tests.add_variant("Variant B", config_b.id, weight=0.5)
    await service.ab_tests.start_test(test.id)

    # Export/Import
    export_data = await service.export_configuration(config.id)
    imported_config = await service.import_configuration(
        export_data,
        organization_id="org_456",
    )
"""

# Base types and enums
from .base import (
    # Enums
    PersonalityTrait,
    TemplateCategory,
    VariableType,
    ConfigStatus,
    IndustryType,
    ComplianceMode,
    EscalationTrigger,
    # Template types
    TemplateVariable,
    PromptTemplate,
    # Personality types
    VoiceSettings,
    BehaviorSettings,
    PersonalityProfile,
    # Configuration types
    LLMSettings,
    FunctionDefinition,
    TranscriptionSettings,
    AgentConfiguration,
    # Version types
    ConfigVersion,
    # Exceptions
    ConfigurationError,
    TemplateError,
    ValidationError,
    VersionError,
)

# Template management
from .templates import (
    TemplateStorage,
    InMemoryTemplateStorage,
    TemplateRenderer,
    TemplateManager,
    TemplateLibrary,
)

# Personality management
from .personality import (
    PersonalityStorage,
    InMemoryPersonalityStorage,
    TraitAnalyzer,
    VoiceConfigurationHelper,
    PersonalityManager,
)

# Version management
from .versioning import (
    ChangeType,
    RollbackStrategy,
    ConfigDiff,
    VersionStorage,
    InMemoryVersionStorage,
    VersionManager,
    AuditAction,
    AuditEntry,
    AuditLogger,
)

# Service layer
from .service import (
    ConfigurationStorage,
    ABTestVariant,
    ABTest,
    ABTestManager,
    AgentConfigurationService,
)


__all__ = [
    # Enums
    "PersonalityTrait",
    "TemplateCategory",
    "VariableType",
    "ConfigStatus",
    "IndustryType",
    "ComplianceMode",
    "EscalationTrigger",
    "ChangeType",
    "RollbackStrategy",
    "AuditAction",
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
    "ConfigDiff",
    "AuditEntry",
    # Storage interfaces
    "TemplateStorage",
    "InMemoryTemplateStorage",
    "PersonalityStorage",
    "InMemoryPersonalityStorage",
    "VersionStorage",
    "InMemoryVersionStorage",
    "ConfigurationStorage",
    # Managers
    "TemplateRenderer",
    "TemplateManager",
    "TemplateLibrary",
    "TraitAnalyzer",
    "VoiceConfigurationHelper",
    "PersonalityManager",
    "VersionManager",
    "AuditLogger",
    # A/B Testing
    "ABTestVariant",
    "ABTest",
    "ABTestManager",
    # Service
    "AgentConfigurationService",
    # Exceptions
    "ConfigurationError",
    "TemplateError",
    "ValidationError",
    "VersionError",
]
