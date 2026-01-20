"""Production readiness schema changes

Revision ID: 002_production_readiness
Revises: 001_initial
Create Date: 2026-01-20 00:00:01.000000

This migration applies critical production-readiness improvements:
1. Float → Numeric for precise monetary calculations
2. JSON → JSONB for PostgreSQL performance and indexing
3. Composite indexes for common query patterns
4. Version columns for optimistic locking
5. Phone number blind index columns for encrypted search
6. Webhook secret encryption column width increase

Note: This migration is designed to be run on PostgreSQL.
For other databases, some operations may need adjustment.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '002_production_readiness'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply production readiness schema changes."""

    # =========================================================================
    # 1. Convert Float columns to Numeric for monetary precision
    # =========================================================================

    # Usage records: metric_value may be used for costs
    op.alter_column(
        'usage_records',
        'metric_value',
        type_=sa.Numeric(18, 6),
        existing_type=sa.Float(),
        existing_nullable=False,
    )

    # Voice configurations: speed and pitch
    # These are not money but benefit from fixed precision for consistency
    op.alter_column(
        'voice_configurations',
        'speed',
        type_=sa.Numeric(4, 2),  # e.g., 1.50
        existing_type=sa.Float(),
        existing_nullable=True,
    )
    op.alter_column(
        'voice_configurations',
        'pitch',
        type_=sa.Numeric(4, 2),  # e.g., 1.25
        existing_type=sa.Float(),
        existing_nullable=True,
    )

    # =========================================================================
    # 2. Convert JSON columns to JSONB for PostgreSQL
    # JSONB provides: binary storage, indexing, faster queries, deduplication
    # =========================================================================

    # API Keys
    op.alter_column(
        'api_keys',
        'scopes',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='scopes::jsonb',
    )

    # Agents
    op.alter_column(
        'agents',
        'llm_config',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='llm_config::jsonb',
    )
    op.alter_column(
        'agents',
        'tools',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='tools::jsonb',
    )

    # Agent versions
    op.alter_column(
        'agent_versions',
        'config_snapshot',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=False,
        postgresql_using='config_snapshot::jsonb',
    )

    # Voice configurations
    op.alter_column(
        'voice_configurations',
        'stt_config',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='stt_config::jsonb',
    )
    op.alter_column(
        'voice_configurations',
        'tts_config',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='tts_config::jsonb',
    )

    # Conversations
    op.alter_column(
        'conversations',
        'metadata',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='metadata::jsonb',
    )

    # Messages
    op.alter_column(
        'messages',
        'metadata',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='metadata::jsonb',
    )

    # Calls
    op.alter_column(
        'calls',
        'metadata',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='metadata::jsonb',
    )

    # Call events
    op.alter_column(
        'call_events',
        'event_data',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='event_data::jsonb',
    )

    # Analytics events
    op.alter_column(
        'analytics_events',
        'event_data',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='event_data::jsonb',
    )

    # Usage records
    op.alter_column(
        'usage_records',
        'metadata',
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using='metadata::jsonb',
    )

    # =========================================================================
    # 3. Add composite indexes for common query patterns
    # These improve performance for multi-column WHERE clauses
    # =========================================================================

    # Calls: Organization + Status + Created (for dashboards)
    op.create_index(
        'ix_calls_org_status_created',
        'calls',
        ['organization_id', 'status', 'created_at'],
    )

    # Calls: Organization + Agent + Created (for agent-specific queries)
    op.create_index(
        'ix_calls_org_agent_created',
        'calls',
        ['organization_id', 'agent_id', 'created_at'],
    )

    # Conversations: Organization + Started (for timeline views)
    op.create_index(
        'ix_conversations_org_started',
        'conversations',
        ['organization_id', 'started_at'],
    )

    # Conversations: Organization + Status + Started
    op.create_index(
        'ix_conversations_org_status_started',
        'conversations',
        ['organization_id', 'status', 'started_at'],
    )

    # Messages: Conversation + Created (for message timeline)
    op.create_index(
        'ix_messages_conv_created',
        'messages',
        ['conversation_id', 'created_at'],
    )

    # Analytics: Organization + Type + Created (for filtered analytics)
    op.create_index(
        'ix_analytics_org_type_created',
        'analytics_events',
        ['organization_id', 'event_type', 'created_at'],
    )

    # Users: Organization + Active (for active user queries)
    op.create_index(
        'ix_users_org_active',
        'users',
        ['organization_id', 'is_active'],
    )

    # Agents: Organization + Status (for filtered agent lists)
    op.create_index(
        'ix_agents_org_status',
        'agents',
        ['organization_id', 'status'],
    )

    # =========================================================================
    # 4. Add version columns for optimistic locking
    # Prevents lost updates in concurrent edit scenarios
    # =========================================================================

    # Organizations - critical for billing/plan changes
    op.add_column(
        'organizations',
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
    )

    # Agents - frequently edited, version conflicts possible
    op.add_column(
        'agents',
        sa.Column('opt_lock_version', sa.Integer(), nullable=False, server_default='1'),
    )
    # Note: 'agents' already has a 'version' column for semantic versioning
    # We add 'opt_lock_version' to avoid confusion

    # Voice configurations - edited by multiple users
    op.add_column(
        'voice_configurations',
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
    )

    # Users - profile updates, permission changes
    op.add_column(
        'users',
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
    )

    # API Keys - security-critical, concurrent updates must be serialized
    op.add_column(
        'api_keys',
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
    )

    # =========================================================================
    # 5. Add blind index columns for encrypted phone search
    # Allows searching encrypted phone numbers without decryption
    # =========================================================================

    # Conversations: blind index for customer_phone
    op.add_column(
        'conversations',
        sa.Column('customer_phone_index', sa.String(32), nullable=True),
    )
    op.create_index(
        'ix_conversations_phone_index',
        'conversations',
        ['customer_phone_index'],
    )

    # Calls: blind index for from_number and to_number
    op.add_column(
        'calls',
        sa.Column('from_number_index', sa.String(32), nullable=True),
    )
    op.add_column(
        'calls',
        sa.Column('to_number_index', sa.String(32), nullable=True),
    )
    op.create_index(
        'ix_calls_from_number_index',
        'calls',
        ['from_number_index'],
    )
    op.create_index(
        'ix_calls_to_number_index',
        'calls',
        ['to_number_index'],
    )

    # =========================================================================
    # 6. Increase column widths for encrypted data
    # Encrypted data is larger than plaintext (nonce + ciphertext + tag + base64)
    # =========================================================================

    # Phone numbers: 50 chars → 100 chars (encrypted)
    op.alter_column(
        'conversations',
        'customer_phone',
        type_=sa.String(100),
        existing_type=sa.String(50),
        existing_nullable=True,
    )
    op.alter_column(
        'calls',
        'from_number',
        type_=sa.String(100),
        existing_type=sa.String(50),
        existing_nullable=True,
    )
    op.alter_column(
        'calls',
        'to_number',
        type_=sa.String(100),
        existing_type=sa.String(50),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Revert production readiness schema changes."""

    # =========================================================================
    # 6. Revert column widths
    # =========================================================================
    op.alter_column(
        'calls',
        'to_number',
        type_=sa.String(50),
        existing_type=sa.String(100),
        existing_nullable=True,
    )
    op.alter_column(
        'calls',
        'from_number',
        type_=sa.String(50),
        existing_type=sa.String(100),
        existing_nullable=True,
    )
    op.alter_column(
        'conversations',
        'customer_phone',
        type_=sa.String(50),
        existing_type=sa.String(100),
        existing_nullable=True,
    )

    # =========================================================================
    # 5. Remove blind index columns
    # =========================================================================
    op.drop_index('ix_calls_to_number_index', table_name='calls')
    op.drop_index('ix_calls_from_number_index', table_name='calls')
    op.drop_column('calls', 'to_number_index')
    op.drop_column('calls', 'from_number_index')

    op.drop_index('ix_conversations_phone_index', table_name='conversations')
    op.drop_column('conversations', 'customer_phone_index')

    # =========================================================================
    # 4. Remove version columns
    # =========================================================================
    op.drop_column('api_keys', 'version')
    op.drop_column('users', 'version')
    op.drop_column('voice_configurations', 'version')
    op.drop_column('agents', 'opt_lock_version')
    op.drop_column('organizations', 'version')

    # =========================================================================
    # 3. Remove composite indexes
    # =========================================================================
    op.drop_index('ix_agents_org_status', table_name='agents')
    op.drop_index('ix_users_org_active', table_name='users')
    op.drop_index('ix_analytics_org_type_created', table_name='analytics_events')
    op.drop_index('ix_messages_conv_created', table_name='messages')
    op.drop_index('ix_conversations_org_status_started', table_name='conversations')
    op.drop_index('ix_conversations_org_started', table_name='conversations')
    op.drop_index('ix_calls_org_agent_created', table_name='calls')
    op.drop_index('ix_calls_org_status_created', table_name='calls')

    # =========================================================================
    # 2. Revert JSONB to JSON
    # =========================================================================
    op.alter_column(
        'usage_records',
        'metadata',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )
    op.alter_column(
        'analytics_events',
        'event_data',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )
    op.alter_column(
        'call_events',
        'event_data',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )
    op.alter_column(
        'calls',
        'metadata',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )
    op.alter_column(
        'messages',
        'metadata',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )
    op.alter_column(
        'conversations',
        'metadata',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )
    op.alter_column(
        'voice_configurations',
        'tts_config',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )
    op.alter_column(
        'voice_configurations',
        'stt_config',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )
    op.alter_column(
        'agent_versions',
        'config_snapshot',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=False,
    )
    op.alter_column(
        'agents',
        'tools',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )
    op.alter_column(
        'agents',
        'llm_config',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )
    op.alter_column(
        'api_keys',
        'scopes',
        type_=sa.JSON(),
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
    )

    # =========================================================================
    # 1. Revert Numeric to Float
    # =========================================================================
    op.alter_column(
        'voice_configurations',
        'pitch',
        type_=sa.Float(),
        existing_type=sa.Numeric(4, 2),
        existing_nullable=True,
    )
    op.alter_column(
        'voice_configurations',
        'speed',
        type_=sa.Float(),
        existing_type=sa.Numeric(4, 2),
        existing_nullable=True,
    )
    op.alter_column(
        'usage_records',
        'metric_value',
        type_=sa.Float(),
        existing_type=sa.Numeric(18, 6),
        existing_nullable=False,
    )
