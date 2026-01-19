"""
Agent Configuration Service Module

This module provides a unified service layer for managing all aspects of agent
configuration including templates, personalities, versions, and A/B testing.
"""

import asyncio
import copy
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .base import (
    AgentConfiguration,
    ConfigStatus,
    ConfigVersion,
    PromptTemplate,
    TemplateCategory,
    TemplateVariable,
    PersonalityProfile,
    PersonalityTrait,
    IndustryType,
    LLMSettings,
    FunctionDefinition,
    VoiceSettings,
    BehaviorSettings,
    ConfigurationError,
    ValidationError,
)
from .templates import (
    TemplateManager,
    TemplateLibrary,
    InMemoryTemplateStorage,
)
from .personality import (
    PersonalityManager,
    TraitAnalyzer,
    VoiceConfigurationHelper,
    InMemoryPersonalityStorage,
)
from .versioning import (
    VersionManager,
    ConfigDiff,
    RollbackStrategy,
    AuditLogger,
    AuditAction,
    InMemoryVersionStorage,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Storage Interface
# =============================================================================


class ConfigurationStorage:
    """In-memory configuration storage for testing and development."""

    def __init__(self):
        self._configs: Dict[str, AgentConfiguration] = {}
        self._org_index: Dict[str, Set[str]] = {}

    async def save(self, config: AgentConfiguration) -> None:
        """Save a configuration."""
        config.updated_at = datetime.utcnow()
        self._configs[config.id] = config

        if config.organization_id not in self._org_index:
            self._org_index[config.organization_id] = set()
        self._org_index[config.organization_id].add(config.id)

    async def get(self, config_id: str) -> Optional[AgentConfiguration]:
        """Get a configuration by ID."""
        return self._configs.get(config_id)

    async def list(
        self,
        organization_id: str,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[AgentConfiguration], int]:
        """List configurations with pagination."""
        if organization_id not in self._org_index:
            return [], 0

        configs = []
        for config_id in self._org_index[organization_id]:
            config = self._configs.get(config_id)
            if not config:
                continue

            # Apply filters
            if filters:
                if "status" in filters:
                    if config.status != filters["status"]:
                        continue
                if "search" in filters:
                    search = filters["search"].lower()
                    if search not in config.name.lower() and search not in config.description.lower():
                        continue
                if "tags" in filters:
                    if not any(tag in config.tags for tag in filters["tags"]):
                        continue

            configs.append(config)

        # Sort by updated_at descending
        configs.sort(key=lambda c: c.updated_at, reverse=True)

        total = len(configs)
        configs = configs[offset:offset + limit]

        return configs, total

    async def delete(self, config_id: str) -> bool:
        """Delete a configuration."""
        config = self._configs.get(config_id)
        if not config:
            return False

        if config.organization_id in self._org_index:
            self._org_index[config.organization_id].discard(config_id)

        del self._configs[config_id]
        return True


# =============================================================================
# A/B Testing Support
# =============================================================================


class ABTestVariant:
    """A variant in an A/B test."""

    def __init__(
        self,
        id: str,
        name: str,
        config_id: str,
        weight: float = 0.5,
    ):
        self.id = id
        self.name = name
        self.config_id = config_id
        self.weight = weight
        self.impressions = 0
        self.conversions = 0
        self.total_duration_seconds = 0.0

    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions

    @property
    def avg_duration(self) -> float:
        """Calculate average call duration."""
        if self.impressions == 0:
            return 0.0
        return self.total_duration_seconds / self.impressions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "config_id": self.config_id,
            "weight": self.weight,
            "impressions": self.impressions,
            "conversions": self.conversions,
            "conversion_rate": self.conversion_rate,
            "avg_duration": self.avg_duration,
        }


class ABTest:
    """An A/B test for comparing agent configurations."""

    def __init__(
        self,
        id: str,
        name: str,
        organization_id: str,
        description: str = "",
    ):
        self.id = id
        self.name = name
        self.organization_id = organization_id
        self.description = description
        self.variants: List[ABTestVariant] = []
        self.is_active = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.created_at = datetime.utcnow()
        self.winning_variant_id: Optional[str] = None

    def add_variant(
        self,
        name: str,
        config_id: str,
        weight: float = 0.5,
    ) -> ABTestVariant:
        """Add a variant to the test."""
        variant = ABTestVariant(
            id=f"var_{uuid.uuid4().hex[:12]}",
            name=name,
            config_id=config_id,
            weight=weight,
        )
        self.variants.append(variant)
        return variant

    def normalize_weights(self) -> None:
        """Normalize variant weights to sum to 1.0."""
        total = sum(v.weight for v in self.variants)
        if total > 0:
            for variant in self.variants:
                variant.weight = variant.weight / total

    def select_variant(self) -> ABTestVariant:
        """Select a variant based on weights."""
        import random

        if not self.variants:
            raise ConfigurationError("No variants in test")

        self.normalize_weights()

        r = random.random()
        cumulative = 0.0

        for variant in self.variants:
            cumulative += variant.weight
            if r <= cumulative:
                variant.impressions += 1
                return variant

        # Fallback to last variant
        self.variants[-1].impressions += 1
        return self.variants[-1]

    def record_conversion(
        self,
        variant_id: str,
        duration_seconds: float = 0.0,
    ) -> None:
        """Record a conversion for a variant."""
        for variant in self.variants:
            if variant.id == variant_id:
                variant.conversions += 1
                variant.total_duration_seconds += duration_seconds
                return

    def get_results(self) -> Dict[str, Any]:
        """Get test results."""
        if not self.variants:
            return {
                "test_id": self.id,
                "status": "no_variants",
                "variants": [],
            }

        # Sort by conversion rate
        sorted_variants = sorted(
            self.variants,
            key=lambda v: v.conversion_rate,
            reverse=True,
        )

        return {
            "test_id": self.id,
            "name": self.name,
            "is_active": self.is_active,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "variants": [v.to_dict() for v in sorted_variants],
            "winner": sorted_variants[0].to_dict() if sorted_variants else None,
            "total_impressions": sum(v.impressions for v in self.variants),
            "total_conversions": sum(v.conversions for v in self.variants),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "organization_id": self.organization_id,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "is_active": self.is_active,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "created_at": self.created_at.isoformat(),
            "winning_variant_id": self.winning_variant_id,
        }


class ABTestManager:
    """Manages A/B tests for agent configurations."""

    def __init__(self):
        self._tests: Dict[str, ABTest] = {}
        self._org_index: Dict[str, Set[str]] = {}
        self._config_tests: Dict[str, Set[str]] = {}

    async def create_test(
        self,
        name: str,
        organization_id: str,
        description: str = "",
    ) -> ABTest:
        """Create a new A/B test."""
        test = ABTest(
            id=f"abtest_{uuid.uuid4().hex[:24]}",
            name=name,
            organization_id=organization_id,
            description=description,
        )

        self._tests[test.id] = test

        if organization_id not in self._org_index:
            self._org_index[organization_id] = set()
        self._org_index[organization_id].add(test.id)

        logger.info(f"Created A/B test: {test.id} ({name})")
        return test

    async def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get a test by ID."""
        return self._tests.get(test_id)

    async def add_variant(
        self,
        test_id: str,
        name: str,
        config_id: str,
        weight: float = 0.5,
    ) -> ABTestVariant:
        """Add a variant to a test."""
        test = self._tests.get(test_id)
        if not test:
            raise ConfigurationError(f"Test not found: {test_id}")

        variant = test.add_variant(name, config_id, weight)

        # Track config -> test mapping
        if config_id not in self._config_tests:
            self._config_tests[config_id] = set()
        self._config_tests[config_id].add(test_id)

        return variant

    async def start_test(self, test_id: str) -> ABTest:
        """Start an A/B test."""
        test = self._tests.get(test_id)
        if not test:
            raise ConfigurationError(f"Test not found: {test_id}")

        if len(test.variants) < 2:
            raise ConfigurationError("Test must have at least 2 variants")

        test.is_active = True
        test.start_time = datetime.utcnow()

        logger.info(f"Started A/B test: {test_id}")
        return test

    async def stop_test(self, test_id: str) -> ABTest:
        """Stop an A/B test."""
        test = self._tests.get(test_id)
        if not test:
            raise ConfigurationError(f"Test not found: {test_id}")

        test.is_active = False
        test.end_time = datetime.utcnow()

        # Determine winner
        if test.variants:
            winner = max(test.variants, key=lambda v: v.conversion_rate)
            test.winning_variant_id = winner.id

        logger.info(f"Stopped A/B test: {test_id}")
        return test

    async def select_config(
        self,
        test_id: str,
    ) -> Tuple[str, str]:
        """
        Select a configuration for a call based on test.

        Returns:
            Tuple of (config_id, variant_id)
        """
        test = self._tests.get(test_id)
        if not test or not test.is_active:
            raise ConfigurationError(f"Test not active: {test_id}")

        variant = test.select_variant()
        return variant.config_id, variant.id

    async def record_result(
        self,
        test_id: str,
        variant_id: str,
        converted: bool,
        duration_seconds: float = 0.0,
    ) -> None:
        """Record a call result for a variant."""
        test = self._tests.get(test_id)
        if not test:
            return

        if converted:
            test.record_conversion(variant_id, duration_seconds)

    async def list_tests(
        self,
        organization_id: str,
        is_active: Optional[bool] = None,
    ) -> List[ABTest]:
        """List tests for an organization."""
        if organization_id not in self._org_index:
            return []

        tests = []
        for test_id in self._org_index[organization_id]:
            test = self._tests.get(test_id)
            if test:
                if is_active is None or test.is_active == is_active:
                    tests.append(test)

        return tests

    async def get_config_tests(
        self,
        config_id: str,
    ) -> List[ABTest]:
        """Get tests that include a specific configuration."""
        if config_id not in self._config_tests:
            return []

        tests = []
        for test_id in self._config_tests[config_id]:
            test = self._tests.get(test_id)
            if test:
                tests.append(test)

        return tests


# =============================================================================
# Agent Configuration Service
# =============================================================================


class AgentConfigurationService:
    """
    Unified service for managing agent configurations.

    Provides:
    - Configuration CRUD operations
    - Template management
    - Personality profiles
    - Version control
    - A/B testing
    - Configuration validation
    - Import/export
    """

    def __init__(
        self,
        config_storage: Optional[ConfigurationStorage] = None,
        template_manager: Optional[TemplateManager] = None,
        personality_manager: Optional[PersonalityManager] = None,
        version_manager: Optional[VersionManager] = None,
        ab_test_manager: Optional[ABTestManager] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """
        Initialize the service.

        Args:
            config_storage: Configuration storage backend
            template_manager: Template manager
            personality_manager: Personality manager
            version_manager: Version manager
            ab_test_manager: A/B test manager
            audit_logger: Audit logger
        """
        self.config_storage = config_storage or ConfigurationStorage()
        self.templates = template_manager or TemplateManager()
        self.personalities = personality_manager or PersonalityManager()
        self.versions = version_manager or VersionManager()
        self.ab_tests = ab_test_manager or ABTestManager()
        self.audit = audit_logger or AuditLogger()

        # Template library
        self.template_library = TemplateLibrary(self.templates)

        # Cache
        self._cache: Dict[str, Tuple[AgentConfiguration, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    # =========================================================================
    # Configuration Management
    # =========================================================================

    async def create_configuration(
        self,
        name: str,
        organization_id: str,
        system_prompt: str,
        description: str = "",
        first_message: str = "",
        personality_id: Optional[str] = None,
        llm: Optional[LLMSettings] = None,
        functions: Optional[List[FunctionDefinition]] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        transfer_targets: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> AgentConfiguration:
        """
        Create a new agent configuration.

        Args:
            name: Configuration name
            organization_id: Organization ID
            system_prompt: System prompt for the agent
            description: Configuration description
            first_message: Initial greeting message
            personality_id: ID of personality profile to use
            llm: LLM settings
            functions: Function definitions
            knowledge_base_ids: Knowledge base IDs
            transfer_targets: Transfer target mappings
            tags: Tags for organization
            metadata: Additional metadata
            created_by: User creating the configuration

        Returns:
            Created configuration
        """
        # Load personality if specified
        personality = None
        if personality_id:
            personality = await self.personalities.get_profile(personality_id)

        config = AgentConfiguration(
            id=f"cfg_{uuid.uuid4().hex[:24]}",
            name=name,
            organization_id=organization_id,
            description=description,
            personality_id=personality_id,
            personality=personality,
            system_prompt=system_prompt,
            first_message=first_message,
            llm=llm or LLMSettings(),
            functions=functions or [],
            knowledge_base_ids=knowledge_base_ids or [],
            transfer_targets=transfer_targets or {},
            status=ConfigStatus.DRAFT,
            tags=tags or [],
            metadata=metadata or {},
            created_by=created_by,
            updated_by=created_by,
        )

        # Validate
        self._validate_configuration(config)

        # Save
        await self.config_storage.save(config)

        # Create initial version
        await self.versions.create_version(
            config=config,
            change_summary="Initial version",
            created_by=created_by,
        )

        # Audit log
        await self.audit.log(
            entity_type="configuration",
            entity_id=config.id,
            action=AuditAction.CREATE,
            organization_id=organization_id,
            actor_id=created_by,
            changes={"name": name},
        )

        logger.info(f"Created configuration: {config.id} ({name})")
        return config

    async def get_configuration(
        self,
        config_id: str,
    ) -> Optional[AgentConfiguration]:
        """
        Get a configuration by ID.

        Args:
            config_id: Configuration ID

        Returns:
            Configuration or None
        """
        # Check cache
        if config_id in self._cache:
            config, cached_at = self._cache[config_id]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return config

        # Load from storage
        config = await self.config_storage.get(config_id)

        # Load personality if referenced
        if config and config.personality_id and not config.personality:
            config.personality = await self.personalities.get_profile(
                config.personality_id
            )

        # Update cache
        if config:
            self._cache[config_id] = (config, datetime.utcnow())

        return config

    async def update_configuration(
        self,
        config_id: str,
        updates: Dict[str, Any],
        change_summary: Optional[str] = None,
        updated_by: Optional[str] = None,
    ) -> AgentConfiguration:
        """
        Update a configuration.

        Args:
            config_id: Configuration ID
            updates: Fields to update
            change_summary: Description of changes
            updated_by: User making the update

        Returns:
            Updated configuration
        """
        config = await self.config_storage.get(config_id)
        if not config:
            raise ConfigurationError(f"Configuration not found: {config_id}")

        # Track changes for audit
        original = copy.deepcopy(config.to_dict())

        # Apply updates
        if "name" in updates:
            config.name = updates["name"]
        if "description" in updates:
            config.description = updates["description"]
        if "system_prompt" in updates:
            config.system_prompt = updates["system_prompt"]
        if "first_message" in updates:
            config.first_message = updates["first_message"]
        if "personality_id" in updates:
            config.personality_id = updates["personality_id"]
            if updates["personality_id"]:
                config.personality = await self.personalities.get_profile(
                    updates["personality_id"]
                )
            else:
                config.personality = None
        if "llm" in updates:
            config.llm = LLMSettings.from_dict(updates["llm"]) if isinstance(updates["llm"], dict) else updates["llm"]
        if "functions" in updates:
            config.functions = [
                FunctionDefinition.from_dict(f) if isinstance(f, dict) else f
                for f in updates["functions"]
            ]
        if "knowledge_base_ids" in updates:
            config.knowledge_base_ids = updates["knowledge_base_ids"]
        if "transfer_targets" in updates:
            config.transfer_targets = updates["transfer_targets"]
        if "templates" in updates:
            config.templates = updates["templates"]
        if "status" in updates:
            config.status = ConfigStatus(updates["status"]) if isinstance(updates["status"], str) else updates["status"]
        if "tags" in updates:
            config.tags = updates["tags"]
        if "metadata" in updates:
            config.metadata = updates["metadata"]

        config.version += 1
        config.updated_at = datetime.utcnow()
        config.updated_by = updated_by

        # Validate
        self._validate_configuration(config)

        # Save
        await self.config_storage.save(config)

        # Create new version
        diff = ConfigDiff(original, config.to_dict())
        await self.versions.create_version(
            config=config,
            change_summary=change_summary or diff.get_summary(),
            created_by=updated_by,
        )

        # Invalidate cache
        if config_id in self._cache:
            del self._cache[config_id]

        # Audit log
        await self.audit.log(
            entity_type="configuration",
            entity_id=config_id,
            action=AuditAction.UPDATE,
            organization_id=config.organization_id,
            actor_id=updated_by,
            changes={"summary": diff.get_summary(), "changes": diff.changes},
        )

        logger.info(f"Updated configuration: {config_id}")
        return config

    async def delete_configuration(
        self,
        config_id: str,
        deleted_by: Optional[str] = None,
    ) -> bool:
        """
        Delete a configuration.

        Args:
            config_id: Configuration ID
            deleted_by: User deleting the configuration

        Returns:
            True if deleted
        """
        config = await self.config_storage.get(config_id)
        if not config:
            return False

        # Check if in active A/B tests
        tests = await self.ab_tests.get_config_tests(config_id)
        active_tests = [t for t in tests if t.is_active]
        if active_tests:
            raise ConfigurationError(
                f"Configuration is used in active A/B tests: {[t.name for t in active_tests]}"
            )

        # Delete
        success = await self.config_storage.delete(config_id)

        # Invalidate cache
        if config_id in self._cache:
            del self._cache[config_id]

        # Audit log
        if success:
            await self.audit.log(
                entity_type="configuration",
                entity_id=config_id,
                action=AuditAction.DELETE,
                organization_id=config.organization_id,
                actor_id=deleted_by,
            )
            logger.info(f"Deleted configuration: {config_id}")

        return success

    async def list_configurations(
        self,
        organization_id: str,
        status: Optional[ConfigStatus] = None,
        search: Optional[str] = None,
        tags: Optional[List[str]] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[AgentConfiguration], int]:
        """
        List configurations.

        Args:
            organization_id: Organization ID
            status: Filter by status
            search: Search in name/description
            tags: Filter by tags
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (configurations, total_count)
        """
        filters = {}
        if status:
            filters["status"] = status
        if search:
            filters["search"] = search
        if tags:
            filters["tags"] = tags

        return await self.config_storage.list(
            organization_id=organization_id,
            filters=filters if filters else None,
            offset=offset,
            limit=limit,
        )

    async def clone_configuration(
        self,
        config_id: str,
        new_name: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> AgentConfiguration:
        """
        Clone a configuration.

        Args:
            config_id: Configuration ID to clone
            new_name: Name for the clone
            created_by: User creating the clone

        Returns:
            Cloned configuration
        """
        original = await self.config_storage.get(config_id)
        if not original:
            raise ConfigurationError(f"Configuration not found: {config_id}")

        clone = original.clone(new_name)
        clone.created_by = created_by
        clone.updated_by = created_by

        await self.config_storage.save(clone)

        # Create initial version for clone
        await self.versions.create_version(
            config=clone,
            change_summary=f"Cloned from {original.name}",
            created_by=created_by,
        )

        # Audit log
        await self.audit.log(
            entity_type="configuration",
            entity_id=clone.id,
            action=AuditAction.CLONE,
            organization_id=clone.organization_id,
            actor_id=created_by,
            metadata={"cloned_from": config_id},
        )

        logger.info(f"Cloned configuration: {config_id} -> {clone.id}")
        return clone

    async def activate_configuration(
        self,
        config_id: str,
        activated_by: Optional[str] = None,
    ) -> AgentConfiguration:
        """
        Activate a configuration.

        Args:
            config_id: Configuration ID
            activated_by: User activating

        Returns:
            Activated configuration
        """
        return await self.update_configuration(
            config_id=config_id,
            updates={"status": ConfigStatus.ACTIVE},
            change_summary="Activated configuration",
            updated_by=activated_by,
        )

    async def deactivate_configuration(
        self,
        config_id: str,
        deactivated_by: Optional[str] = None,
    ) -> AgentConfiguration:
        """
        Deactivate a configuration.

        Args:
            config_id: Configuration ID
            deactivated_by: User deactivating

        Returns:
            Deactivated configuration
        """
        return await self.update_configuration(
            config_id=config_id,
            updates={"status": ConfigStatus.DRAFT},
            change_summary="Deactivated configuration",
            updated_by=deactivated_by,
        )

    # =========================================================================
    # Configuration Building
    # =========================================================================

    async def create_configuration_from_industry(
        self,
        name: str,
        organization_id: str,
        company_name: str,
        agent_name: str,
        industry: IndustryType,
        agent_role: Optional[str] = None,
        voice_gender: str = "female",
        created_by: Optional[str] = None,
    ) -> AgentConfiguration:
        """
        Create a fully configured agent based on industry best practices.

        Args:
            name: Configuration name
            organization_id: Organization ID
            company_name: Company name
            agent_name: Name for the agent persona
            industry: Industry type
            agent_role: Agent role (auto-generated if not provided)
            voice_gender: Preferred voice gender
            created_by: User creating the configuration

        Returns:
            Created configuration
        """
        # Create personality profile
        personality = await self.personalities.create_profile_from_industry(
            name=f"{name} Personality",
            organization_id=organization_id,
            agent_name=agent_name,
            company_name=company_name,
            industry=industry,
            agent_role=agent_role,
            voice_gender=voice_gender,
        )

        # Get system prompt template
        system_template = await self.templates.get_default_template(
            organization_id=organization_id,
            category=TemplateCategory.SYSTEM_PROMPT,
        )

        # Build system prompt
        if system_template:
            system_prompt = self.templates.render_template(
                system_template,
                {
                    "agent_name": agent_name,
                    "agent_role": agent_role or personality.agent_role,
                    "company_name": company_name,
                    "industry": industry.value,
                },
            )
        else:
            system_prompt = self._build_default_system_prompt(
                agent_name=agent_name,
                agent_role=agent_role or personality.agent_role,
                company_name=company_name,
                industry=industry,
            )

        # Get greeting template
        greeting_template = await self.templates.get_default_template(
            organization_id=organization_id,
            category=TemplateCategory.GREETING,
        )

        if greeting_template:
            first_message = self.templates.render_template(
                greeting_template,
                {
                    "agent_name": agent_name,
                    "company_name": company_name,
                    "time_of_day": "day",
                },
            )
        else:
            first_message = f"Hi there! This is {agent_name} from {company_name}. How can I help you today?"

        # Create configuration
        config = await self.create_configuration(
            name=name,
            organization_id=organization_id,
            system_prompt=system_prompt,
            first_message=first_message,
            personality_id=personality.id,
            tags=[industry.value, "auto-generated"],
            metadata={"industry": industry.value},
            created_by=created_by,
        )

        return config

    def _build_default_system_prompt(
        self,
        agent_name: str,
        agent_role: str,
        company_name: str,
        industry: IndustryType,
    ) -> str:
        """Build default system prompt for industry."""
        return f"""You are {agent_name}, a {agent_role} at {company_name}.

Your role is to assist callers professionally and efficiently with their inquiries.

Key behaviors:
- Be professional yet approachable
- Listen carefully and ask clarifying questions when needed
- Provide accurate information based on your knowledge
- If you cannot help with something, offer to connect the caller with someone who can

You specialize in {industry.value.replace('_', ' ')} and have expertise in this domain.

Keep your responses concise and natural for voice conversation."""

    # =========================================================================
    # Version Management
    # =========================================================================

    async def get_configuration_versions(
        self,
        config_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[ConfigVersion], int]:
        """Get version history for a configuration."""
        return await self.versions.list_versions(config_id, offset, limit)

    async def compare_versions(
        self,
        config_id: str,
        version_a: int,
        version_b: int,
    ) -> Dict[str, Any]:
        """Compare two versions of a configuration."""
        diff = await self.versions.compare_versions(config_id, version_a, version_b)
        return diff.to_dict()

    async def rollback_configuration(
        self,
        config_id: str,
        target_version: int,
        rolled_back_by: Optional[str] = None,
    ) -> AgentConfiguration:
        """
        Rollback configuration to a previous version.

        Args:
            config_id: Configuration ID
            target_version: Version number to rollback to
            rolled_back_by: User performing rollback

        Returns:
            Rolled back configuration
        """
        config = await self.versions.rollback(
            config_id=config_id,
            target_version=target_version,
            strategy=RollbackStrategy.FULL,
            created_by=rolled_back_by,
        )

        await self.config_storage.save(config)

        # Invalidate cache
        if config_id in self._cache:
            del self._cache[config_id]

        # Audit log
        await self.audit.log(
            entity_type="configuration",
            entity_id=config_id,
            action=AuditAction.ROLLBACK,
            organization_id=config.organization_id,
            actor_id=rolled_back_by,
            metadata={"target_version": target_version},
        )

        return config

    # =========================================================================
    # Import/Export
    # =========================================================================

    async def export_configuration(
        self,
        config_id: str,
        include_personality: bool = True,
        include_templates: bool = True,
        exported_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export a configuration as a portable format.

        Args:
            config_id: Configuration ID
            include_personality: Include personality profile
            include_templates: Include associated templates
            exported_by: User exporting

        Returns:
            Exportable configuration data
        """
        config = await self.config_storage.get(config_id)
        if not config:
            raise ConfigurationError(f"Configuration not found: {config_id}")

        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "configuration": config.to_dict(),
        }

        # Include personality
        if include_personality and config.personality_id:
            personality = await self.personalities.get_profile(config.personality_id)
            if personality:
                export_data["personality"] = personality.to_dict()

        # Include templates
        if include_templates and config.templates:
            templates_data = {}
            for category, template_id in config.templates.items():
                template = await self.templates.get_template(template_id)
                if template:
                    templates_data[category] = template.to_dict()
            export_data["templates"] = templates_data

        # Audit log
        await self.audit.log(
            entity_type="configuration",
            entity_id=config_id,
            action=AuditAction.EXPORT,
            organization_id=config.organization_id,
            actor_id=exported_by,
        )

        return export_data

    async def import_configuration(
        self,
        export_data: Dict[str, Any],
        organization_id: str,
        name_prefix: str = "",
        imported_by: Optional[str] = None,
    ) -> AgentConfiguration:
        """
        Import a configuration from exported data.

        Args:
            export_data: Exported configuration data
            organization_id: Target organization ID
            name_prefix: Prefix for imported names
            imported_by: User importing

        Returns:
            Imported configuration
        """
        config_data = export_data.get("configuration", {})
        if not config_data:
            raise ConfigurationError("No configuration data in export")

        # Import personality if included
        personality_id = None
        if "personality" in export_data:
            personality_data = export_data["personality"]
            personality_data["id"] = f"pers_{uuid.uuid4().hex[:24]}"
            personality_data["organization_id"] = organization_id
            personality_data["name"] = f"{name_prefix}{personality_data.get('name', 'Imported')}"

            personality = PersonalityProfile.from_dict(personality_data)
            await self.personalities.storage.save(personality)
            personality_id = personality.id

        # Import templates if included
        template_mapping = {}
        if "templates" in export_data:
            for category, template_data in export_data["templates"].items():
                template_data["id"] = f"tmpl_{uuid.uuid4().hex[:24]}"
                template_data["organization_id"] = organization_id
                template_data["name"] = f"{name_prefix}{template_data.get('name', 'Imported')}"

                template = PromptTemplate.from_dict(template_data)
                await self.templates.storage.save(template)
                template_mapping[category] = template.id

        # Create configuration
        config = await self.create_configuration(
            name=f"{name_prefix}{config_data.get('name', 'Imported Configuration')}",
            organization_id=organization_id,
            system_prompt=config_data.get("system_prompt", ""),
            description=config_data.get("description", ""),
            first_message=config_data.get("first_message", ""),
            personality_id=personality_id,
            llm=LLMSettings.from_dict(config_data.get("llm", {})) if config_data.get("llm") else None,
            functions=[
                FunctionDefinition.from_dict(f)
                for f in config_data.get("functions", [])
            ],
            knowledge_base_ids=config_data.get("knowledge_base_ids", []),
            transfer_targets=config_data.get("transfer_targets", {}),
            tags=config_data.get("tags", []) + ["imported"],
            metadata=config_data.get("metadata", {}),
            created_by=imported_by,
        )

        # Update template references
        if template_mapping:
            config.templates = template_mapping
            await self.config_storage.save(config)

        # Audit log
        await self.audit.log(
            entity_type="configuration",
            entity_id=config.id,
            action=AuditAction.IMPORT,
            organization_id=organization_id,
            actor_id=imported_by,
        )

        logger.info(f"Imported configuration: {config.id}")
        return config

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_configuration(self, config: AgentConfiguration) -> None:
        """Validate a configuration."""
        errors = []

        if not config.name:
            errors.append("Configuration name is required")

        if not config.organization_id:
            errors.append("Organization ID is required")

        if not config.system_prompt:
            errors.append("System prompt is required")

        if len(config.system_prompt) < 10:
            errors.append("System prompt is too short")

        # Validate LLM settings
        if config.llm:
            if config.llm.temperature < 0 or config.llm.temperature > 2:
                errors.append("Temperature must be between 0 and 2")
            if config.llm.max_tokens < 1:
                errors.append("Max tokens must be at least 1")

        # Validate functions
        for func in config.functions:
            if not func.name:
                errors.append("Function name is required")
            if not func.description:
                errors.append(f"Function '{func.name}' requires a description")

        if errors:
            raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize_organization(
        self,
        organization_id: str,
        install_templates: bool = True,
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Initialize configuration resources for a new organization.

        Args:
            organization_id: Organization ID
            install_templates: Whether to install built-in templates
            created_by: User performing initialization

        Returns:
            Summary of installed resources
        """
        result = {
            "organization_id": organization_id,
            "templates_installed": 0,
        }

        if install_templates:
            templates = await self.template_library.install_builtin_templates(
                organization_id=organization_id,
                created_by=created_by,
            )
            result["templates_installed"] = len(templates)

        logger.info(f"Initialized configuration for organization: {organization_id}")
        return result


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "ConfigurationStorage",
    "ABTestVariant",
    "ABTest",
    "ABTestManager",
    "AgentConfigurationService",
]
