"""Add additional performance indexes for common queries

Revision ID: 005_performance_indexes
Revises: 004_row_level_security
Create Date: 2026-01-21 00:00:03.000000

This migration adds indexes to optimize common query patterns:
1. Composite indexes for frequently used WHERE clauses
2. Partial indexes for status-based queries
3. Indexes for time-based filtering
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '005_performance_indexes'
down_revision: Union[str, None] = '004_row_level_security'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add performance indexes."""

    # =========================================================================
    # 1. Composite indexes for common query patterns
    # =========================================================================

    # Calls: org + status + started_at (for dashboard queries)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_calls_org_status_started
        ON calls (organization_id, status, started_at DESC)
    """)

    # Calls: agent + started_at (for agent-specific analytics)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_calls_agent_started
        ON calls (agent_id, started_at DESC)
    """)

    # Conversations: org + status + created_at
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_conversations_org_status_created
        ON conversations (organization_id, status, created_at DESC)
    """)

    # Agents: org + is_active (for listing active agents)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_agents_org_active
        ON agents (organization_id, is_active)
        WHERE is_active = true
    """)

    # =========================================================================
    # 2. Partial indexes for status-based queries
    # These are smaller and faster for common status filters
    # =========================================================================

    # Active calls index (smaller than full index)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_calls_active
        ON calls (organization_id, started_at DESC)
        WHERE status = 'in_progress'
    """)

    # Pending calls (waiting to be processed)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_calls_pending
        ON calls (organization_id, created_at)
        WHERE status = 'initiated'
    """)

    # Failed calls (for error tracking)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_calls_failed
        ON calls (organization_id, ended_at DESC)
        WHERE status = 'failed'
    """)

    # Active agents
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_agents_active_only
        ON agents (organization_id, name)
        WHERE is_active = true
    """)

    # =========================================================================
    # 3. Time-based indexes for analytics and reporting
    # =========================================================================

    # Calls by date (for daily aggregates)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_calls_date
        ON calls (DATE(started_at), organization_id)
    """)

    # Usage records by date and type
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_usage_records_date_type
        ON usage_records (DATE(recorded_at), organization_id, usage_type)
    """)

    # =========================================================================
    # 4. Text search indexes for name/description searches
    # =========================================================================

    # Agents name search
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_agents_name_trgm
        ON agents USING gin (name gin_trgm_ops)
    """)

    # Enable trigram extension if not already enabled
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # =========================================================================
    # 5. Covering indexes for common SELECT queries
    # Includes frequently selected columns to avoid table lookups
    # =========================================================================

    # API keys lookup (include scopes to avoid table lookup)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_api_keys_covering
        ON api_keys (key_prefix, organization_id)
        INCLUDE (scopes, expires_at, is_active)
    """)


def downgrade() -> None:
    """Remove performance indexes."""

    # Drop all created indexes
    op.execute("DROP INDEX IF EXISTS ix_api_keys_covering")
    op.execute("DROP INDEX IF EXISTS ix_agents_name_trgm")
    op.execute("DROP INDEX IF EXISTS ix_usage_records_date_type")
    op.execute("DROP INDEX IF EXISTS ix_calls_date")
    op.execute("DROP INDEX IF EXISTS ix_agents_active_only")
    op.execute("DROP INDEX IF EXISTS ix_calls_failed")
    op.execute("DROP INDEX IF EXISTS ix_calls_pending")
    op.execute("DROP INDEX IF EXISTS ix_calls_active")
    op.execute("DROP INDEX IF EXISTS ix_agents_org_active")
    op.execute("DROP INDEX IF EXISTS ix_conversations_org_status_created")
    op.execute("DROP INDEX IF EXISTS ix_calls_agent_started")
    op.execute("DROP INDEX IF EXISTS ix_calls_org_status_started")
