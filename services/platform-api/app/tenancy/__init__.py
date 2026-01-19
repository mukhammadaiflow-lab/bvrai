"""
Multi-Tenant Isolation System

Enterprise multi-tenancy with:
- Tenant management
- Data isolation (row-level, schema, database)
- Resource quotas
- Tenant context propagation
- Cross-tenant security
"""

from app.tenancy.tenant import (
    Tenant,
    TenantConfig,
    TenantStatus,
    TenantTier,
    TenantManager,
    TenantRepository,
)

from app.tenancy.isolation import (
    IsolationStrategy,
    IsolationLevel,
    RowLevelIsolation,
    SchemaIsolation,
    DatabaseIsolation,
    IsolationManager,
)

from app.tenancy.context import (
    TenantContext,
    get_current_tenant,
    set_current_tenant,
    clear_tenant_context,
    tenant_context,
    TenantContextVar,
)

from app.tenancy.middleware import (
    TenantMiddleware,
    TenantResolver,
    HeaderTenantResolver,
    SubdomainTenantResolver,
    PathTenantResolver,
    JWTTenantResolver,
)

from app.tenancy.quotas import (
    TenantQuota,
    QuotaType,
    QuotaLimit,
    QuotaUsage,
    TenantQuotaManager,
    QuotaEnforcer,
)

__all__ = [
    # Tenant
    "Tenant",
    "TenantConfig",
    "TenantStatus",
    "TenantTier",
    "TenantManager",
    "TenantRepository",
    # Isolation
    "IsolationStrategy",
    "IsolationLevel",
    "RowLevelIsolation",
    "SchemaIsolation",
    "DatabaseIsolation",
    "IsolationManager",
    # Context
    "TenantContext",
    "get_current_tenant",
    "set_current_tenant",
    "clear_tenant_context",
    "tenant_context",
    "TenantContextVar",
    # Middleware
    "TenantMiddleware",
    "TenantResolver",
    "HeaderTenantResolver",
    "SubdomainTenantResolver",
    "PathTenantResolver",
    "JWTTenantResolver",
    # Quotas
    "TenantQuota",
    "QuotaType",
    "QuotaLimit",
    "QuotaUsage",
    "TenantQuotaManager",
    "QuotaEnforcer",
]
