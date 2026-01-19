"""
Tenant Model and Management

Core tenant functionality:
- Tenant lifecycle management
- Configuration management
- Feature flags per tenant
- Tenant metadata
"""

from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import secrets
import logging

logger = logging.getLogger(__name__)


class TenantStatus(str, Enum):
    """Tenant lifecycle status."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    TRIAL_EXPIRED = "trial_expired"
    DELETED = "deleted"
    ARCHIVED = "archived"


class TenantTier(str, Enum):
    """Tenant subscription tier."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class TenantLimits:
    """Resource limits for tenant tier."""
    max_users: int = 5
    max_agents: int = 1
    max_calls_per_month: int = 1000
    max_concurrent_calls: int = 5
    max_storage_gb: float = 1.0
    max_api_calls_per_day: int = 10000
    max_integrations: int = 2
    retention_days: int = 30

    @classmethod
    def for_tier(cls, tier: TenantTier) -> "TenantLimits":
        """Get limits for a specific tier."""
        tier_limits = {
            TenantTier.FREE: cls(
                max_users=2,
                max_agents=1,
                max_calls_per_month=100,
                max_concurrent_calls=1,
                max_storage_gb=0.5,
                max_api_calls_per_day=1000,
                max_integrations=1,
                retention_days=7,
            ),
            TenantTier.STARTER: cls(
                max_users=5,
                max_agents=3,
                max_calls_per_month=1000,
                max_concurrent_calls=5,
                max_storage_gb=5.0,
                max_api_calls_per_day=10000,
                max_integrations=3,
                retention_days=30,
            ),
            TenantTier.PROFESSIONAL: cls(
                max_users=25,
                max_agents=10,
                max_calls_per_month=10000,
                max_concurrent_calls=20,
                max_storage_gb=50.0,
                max_api_calls_per_day=100000,
                max_integrations=10,
                retention_days=90,
            ),
            TenantTier.BUSINESS: cls(
                max_users=100,
                max_agents=50,
                max_calls_per_month=100000,
                max_concurrent_calls=100,
                max_storage_gb=500.0,
                max_api_calls_per_day=1000000,
                max_integrations=25,
                retention_days=180,
            ),
            TenantTier.ENTERPRISE: cls(
                max_users=-1,  # Unlimited
                max_agents=-1,
                max_calls_per_month=-1,
                max_concurrent_calls=500,
                max_storage_gb=-1,
                max_api_calls_per_day=-1,
                max_integrations=-1,
                retention_days=365,
            ),
        }
        return tier_limits.get(tier, cls())


@dataclass
class TenantFeatures:
    """Feature flags for tenant."""
    # Core features
    voice_ai: bool = True
    custom_voices: bool = False
    voice_cloning: bool = False
    real_time_transcription: bool = True

    # Analytics
    basic_analytics: bool = True
    advanced_analytics: bool = False
    custom_reports: bool = False

    # Integrations
    webhooks: bool = True
    api_access: bool = True
    sip_integration: bool = False
    crm_integration: bool = False
    custom_integrations: bool = False

    # Security
    sso: bool = False
    mfa: bool = True
    ip_whitelisting: bool = False
    audit_logs: bool = False

    # Support
    email_support: bool = True
    chat_support: bool = False
    phone_support: bool = False
    dedicated_support: bool = False
    sla_guarantee: bool = False

    @classmethod
    def for_tier(cls, tier: TenantTier) -> "TenantFeatures":
        """Get features for a specific tier."""
        tier_features = {
            TenantTier.FREE: cls(
                custom_voices=False,
                advanced_analytics=False,
                sip_integration=False,
            ),
            TenantTier.STARTER: cls(
                custom_voices=True,
                advanced_analytics=False,
                chat_support=True,
            ),
            TenantTier.PROFESSIONAL: cls(
                custom_voices=True,
                advanced_analytics=True,
                custom_reports=True,
                sip_integration=True,
                crm_integration=True,
                mfa=True,
                audit_logs=True,
                chat_support=True,
            ),
            TenantTier.BUSINESS: cls(
                custom_voices=True,
                voice_cloning=True,
                advanced_analytics=True,
                custom_reports=True,
                sip_integration=True,
                crm_integration=True,
                custom_integrations=True,
                sso=True,
                mfa=True,
                ip_whitelisting=True,
                audit_logs=True,
                chat_support=True,
                phone_support=True,
            ),
            TenantTier.ENTERPRISE: cls(
                custom_voices=True,
                voice_cloning=True,
                advanced_analytics=True,
                custom_reports=True,
                sip_integration=True,
                crm_integration=True,
                custom_integrations=True,
                sso=True,
                mfa=True,
                ip_whitelisting=True,
                audit_logs=True,
                email_support=True,
                chat_support=True,
                phone_support=True,
                dedicated_support=True,
                sla_guarantee=True,
            ),
        }
        return tier_features.get(tier, cls())


@dataclass
class TenantConfig:
    """Tenant-specific configuration."""
    # Branding
    company_name: Optional[str] = None
    logo_url: Optional[str] = None
    primary_color: str = "#3B82F6"
    custom_domain: Optional[str] = None

    # Voice settings
    default_voice: str = "en-US-Neural2-F"
    default_language: str = "en-US"
    allowed_languages: List[str] = field(default_factory=lambda: ["en-US"])

    # Security settings
    mfa_required: bool = False
    password_policy: str = "standard"  # standard, strong, custom
    session_timeout_minutes: int = 60
    ip_whitelist: List[str] = field(default_factory=list)

    # Integration settings
    webhook_urls: Dict[str, str] = field(default_factory=dict)
    api_keys: List[str] = field(default_factory=list)

    # Notification settings
    email_notifications: bool = True
    slack_webhook: Optional[str] = None

    # Compliance
    data_region: str = "us-east-1"
    gdpr_compliant: bool = False
    hipaa_compliant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "company_name": self.company_name,
            "logo_url": self.logo_url,
            "primary_color": self.primary_color,
            "custom_domain": self.custom_domain,
            "default_voice": self.default_voice,
            "default_language": self.default_language,
            "allowed_languages": self.allowed_languages,
            "mfa_required": self.mfa_required,
            "password_policy": self.password_policy,
            "session_timeout_minutes": self.session_timeout_minutes,
            "email_notifications": self.email_notifications,
            "data_region": self.data_region,
            "gdpr_compliant": self.gdpr_compliant,
            "hipaa_compliant": self.hipaa_compliant,
        }


@dataclass
class Tenant:
    """Core tenant entity."""
    id: str
    name: str
    slug: str
    status: TenantStatus = TenantStatus.PENDING
    tier: TenantTier = TenantTier.FREE

    # Owner information
    owner_id: Optional[str] = None
    owner_email: Optional[str] = None

    # Configuration
    config: TenantConfig = field(default_factory=TenantConfig)
    limits: TenantLimits = field(default_factory=TenantLimits)
    features: TenantFeatures = field(default_factory=TenantFeatures)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = None
    suspended_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

    # Trial information
    trial_ends_at: Optional[datetime] = None

    # Billing
    billing_email: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    subscription_id: Optional[str] = None

    def __post_init__(self):
        """Initialize tier-based limits and features."""
        if self.limits == TenantLimits():
            self.limits = TenantLimits.for_tier(self.tier)
        if self.features == TenantFeatures():
            self.features = TenantFeatures.for_tier(self.tier)

    @classmethod
    def create(
        cls,
        name: str,
        owner_email: str,
        tier: TenantTier = TenantTier.FREE,
        trial_days: int = 14,
    ) -> "Tenant":
        """Create a new tenant."""
        slug = cls._generate_slug(name)
        tenant = cls(
            id=str(uuid.uuid4()),
            name=name,
            slug=slug,
            status=TenantStatus.TRIAL if trial_days > 0 else TenantStatus.ACTIVE,
            tier=tier,
            owner_email=owner_email,
            limits=TenantLimits.for_tier(tier),
            features=TenantFeatures.for_tier(tier),
            trial_ends_at=datetime.utcnow() + timedelta(days=trial_days) if trial_days > 0 else None,
        )
        return tenant

    @staticmethod
    def _generate_slug(name: str) -> str:
        """Generate URL-safe slug from name."""
        import re
        slug = name.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_-]+', '-', slug)
        slug = slug.strip('-')
        # Add random suffix for uniqueness
        suffix = secrets.token_hex(4)
        return f"{slug}-{suffix}"

    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE

    def is_trial(self) -> bool:
        """Check if tenant is in trial."""
        return self.status == TenantStatus.TRIAL

    def is_trial_expired(self) -> bool:
        """Check if trial has expired."""
        if not self.trial_ends_at:
            return False
        return datetime.utcnow() > self.trial_ends_at

    def can_access(self) -> bool:
        """Check if tenant can access the system."""
        return self.status in (TenantStatus.ACTIVE, TenantStatus.TRIAL)

    def has_feature(self, feature: str) -> bool:
        """Check if tenant has a specific feature."""
        return getattr(self.features, feature, False)

    def check_limit(self, limit_name: str, current_value: int) -> bool:
        """Check if a limit is exceeded."""
        limit_value = getattr(self.limits, limit_name, -1)
        if limit_value == -1:  # Unlimited
            return True
        return current_value < limit_value

    def activate(self) -> None:
        """Activate the tenant."""
        self.status = TenantStatus.ACTIVE
        self.activated_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def suspend(self, reason: Optional[str] = None) -> None:
        """Suspend the tenant."""
        self.status = TenantStatus.SUSPENDED
        self.suspended_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        if reason:
            self.metadata["suspension_reason"] = reason

    def delete(self) -> None:
        """Mark tenant as deleted."""
        self.status = TenantStatus.DELETED
        self.deleted_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def upgrade_tier(self, new_tier: TenantTier) -> None:
        """Upgrade tenant tier."""
        self.tier = new_tier
        self.limits = TenantLimits.for_tier(new_tier)
        self.features = TenantFeatures.for_tier(new_tier)
        self.updated_at = datetime.utcnow()

        # If upgrading from trial, activate
        if self.status == TenantStatus.TRIAL:
            self.activate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "status": self.status.value,
            "tier": self.tier.value,
            "owner_id": self.owner_id,
            "owner_email": self.owner_email,
            "config": self.config.to_dict(),
            "metadata": self.metadata,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "trial_ends_at": self.trial_ends_at.isoformat() if self.trial_ends_at else None,
        }


class TenantRepository(ABC):
    """Abstract tenant repository."""

    @abstractmethod
    async def create(self, tenant: Tenant) -> Tenant:
        """Create tenant."""
        pass

    @abstractmethod
    async def get(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        pass

    @abstractmethod
    async def get_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        pass

    @abstractmethod
    async def update(self, tenant: Tenant) -> Tenant:
        """Update tenant."""
        pass

    @abstractmethod
    async def delete(self, tenant_id: str) -> bool:
        """Delete tenant."""
        pass

    @abstractmethod
    async def list(
        self,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Tenant]:
        """List tenants."""
        pass


class InMemoryTenantRepository(TenantRepository):
    """In-memory tenant repository for development."""

    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._slug_index: Dict[str, str] = {}

    async def create(self, tenant: Tenant) -> Tenant:
        """Create tenant."""
        if tenant.id in self._tenants:
            raise ValueError(f"Tenant {tenant.id} already exists")
        if tenant.slug in self._slug_index:
            raise ValueError(f"Tenant slug {tenant.slug} already exists")

        self._tenants[tenant.id] = tenant
        self._slug_index[tenant.slug] = tenant.id
        return tenant

    async def get(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)

    async def get_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        tenant_id = self._slug_index.get(slug)
        if tenant_id:
            return self._tenants.get(tenant_id)
        return None

    async def update(self, tenant: Tenant) -> Tenant:
        """Update tenant."""
        if tenant.id not in self._tenants:
            raise ValueError(f"Tenant {tenant.id} not found")

        old_tenant = self._tenants[tenant.id]
        if old_tenant.slug != tenant.slug:
            del self._slug_index[old_tenant.slug]
            self._slug_index[tenant.slug] = tenant.id

        tenant.updated_at = datetime.utcnow()
        self._tenants[tenant.id] = tenant
        return tenant

    async def delete(self, tenant_id: str) -> bool:
        """Delete tenant."""
        if tenant_id not in self._tenants:
            return False

        tenant = self._tenants[tenant_id]
        del self._slug_index[tenant.slug]
        del self._tenants[tenant_id]
        return True

    async def list(
        self,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Tenant]:
        """List tenants."""
        tenants = list(self._tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status]
        if tier:
            tenants = [t for t in tenants if t.tier == tier]

        return tenants[offset:offset + limit]


class TenantManager:
    """
    Tenant lifecycle manager.

    Handles tenant creation, updates, and lifecycle events.
    """

    def __init__(
        self,
        repository: Optional[TenantRepository] = None,
    ):
        self.repository = repository or InMemoryTenantRepository()
        self._event_handlers: Dict[str, List[callable]] = {}

    async def create_tenant(
        self,
        name: str,
        owner_email: str,
        tier: TenantTier = TenantTier.FREE,
        trial_days: int = 14,
        config: Optional[TenantConfig] = None,
    ) -> Tenant:
        """Create a new tenant."""
        tenant = Tenant.create(
            name=name,
            owner_email=owner_email,
            tier=tier,
            trial_days=trial_days,
        )

        if config:
            tenant.config = config

        tenant = await self.repository.create(tenant)
        await self._emit_event("tenant.created", tenant)

        logger.info(f"Created tenant: {tenant.id} ({tenant.name})")
        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return await self.repository.get(tenant_id)

    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        return await self.repository.get_by_slug(slug)

    async def update_tenant(self, tenant: Tenant) -> Tenant:
        """Update tenant."""
        tenant = await self.repository.update(tenant)
        await self._emit_event("tenant.updated", tenant)
        return tenant

    async def activate_tenant(self, tenant_id: str) -> Tenant:
        """Activate a tenant."""
        tenant = await self.repository.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        tenant.activate()
        tenant = await self.repository.update(tenant)
        await self._emit_event("tenant.activated", tenant)

        logger.info(f"Activated tenant: {tenant.id}")
        return tenant

    async def suspend_tenant(
        self,
        tenant_id: str,
        reason: Optional[str] = None,
    ) -> Tenant:
        """Suspend a tenant."""
        tenant = await self.repository.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        tenant.suspend(reason)
        tenant = await self.repository.update(tenant)
        await self._emit_event("tenant.suspended", tenant)

        logger.info(f"Suspended tenant: {tenant.id} (reason: {reason})")
        return tenant

    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant (soft delete)."""
        tenant = await self.repository.get(tenant_id)
        if not tenant:
            return False

        tenant.delete()
        await self.repository.update(tenant)
        await self._emit_event("tenant.deleted", tenant)

        logger.info(f"Deleted tenant: {tenant.id}")
        return True

    async def upgrade_tenant(
        self,
        tenant_id: str,
        new_tier: TenantTier,
    ) -> Tenant:
        """Upgrade tenant tier."""
        tenant = await self.repository.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        old_tier = tenant.tier
        tenant.upgrade_tier(new_tier)
        tenant = await self.repository.update(tenant)

        await self._emit_event("tenant.upgraded", {
            "tenant": tenant,
            "old_tier": old_tier,
            "new_tier": new_tier,
        })

        logger.info(f"Upgraded tenant {tenant.id}: {old_tier} -> {new_tier}")
        return tenant

    async def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Tenant]:
        """List tenants with filtering."""
        return await self.repository.list(status, tier, limit, offset)

    async def check_expired_trials(self) -> List[Tenant]:
        """Check and update expired trials."""
        trials = await self.repository.list(status=TenantStatus.TRIAL)
        expired = []

        for tenant in trials:
            if tenant.is_trial_expired():
                tenant.status = TenantStatus.TRIAL_EXPIRED
                await self.repository.update(tenant)
                await self._emit_event("tenant.trial_expired", tenant)
                expired.append(tenant)
                logger.info(f"Trial expired for tenant: {tenant.id}")

        return expired

    def on_event(self, event_name: str, handler: callable) -> None:
        """Register event handler."""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    async def _emit_event(self, event_name: str, data: Any) -> None:
        """Emit event to handlers."""
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error for {event_name}: {e}")


import asyncio
