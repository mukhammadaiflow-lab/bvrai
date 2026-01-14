"""Initial database schema

Revision ID: 001_initial
Revises:
Create Date: 2024-01-14 00:00:00.000000

This migration creates all initial tables for the Builder Voice AI Platform.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all initial tables."""

    # =========================================================================
    # Organizations
    # =========================================================================
    op.create_table(
        'organizations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(100), unique=True, nullable=False),
        sa.Column('plan', sa.String(50), default='free'),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), default=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_organizations_slug', 'organizations', ['slug'])

    op.create_table(
        'organization_settings',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('key', sa.String(100), nullable=False),
        sa.Column('value', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_org_settings_org_key', 'organization_settings', ['organization_id', 'key'], unique=True)

    # =========================================================================
    # Users
    # =========================================================================
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('role', sa.String(50), default='member'),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), default=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_org', 'users', ['organization_id'])

    # =========================================================================
    # API Keys
    # =========================================================================
    op.create_table(
        'api_keys',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False),
        sa.Column('key_prefix', sa.String(20), nullable=False),
        sa.Column('scopes', sa.JSON(), nullable=True),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_api_keys_org', 'api_keys', ['organization_id'])
    op.create_index('ix_api_keys_prefix', 'api_keys', ['key_prefix'])

    # =========================================================================
    # Agents
    # =========================================================================
    op.create_table(
        'agents',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('system_prompt', sa.Text(), nullable=True),
        sa.Column('voice_config_id', sa.String(36), nullable=True),
        sa.Column('llm_config', sa.JSON(), nullable=True),
        sa.Column('tools', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(50), default='draft'),
        sa.Column('version', sa.Integer(), default=1),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), default=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('created_by', sa.String(36), nullable=True),
        sa.Column('updated_by', sa.String(36), nullable=True),
    )
    op.create_index('ix_agents_org', 'agents', ['organization_id'])
    op.create_index('ix_agents_status', 'agents', ['status'])

    op.create_table(
        'agent_versions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('agent_id', sa.String(36), sa.ForeignKey('agents.id'), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('config_snapshot', sa.JSON(), nullable=False),
        sa.Column('changelog', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('created_by', sa.String(36), nullable=True),
    )
    op.create_index('ix_agent_versions_agent', 'agent_versions', ['agent_id', 'version'], unique=True)

    # =========================================================================
    # Voice Configurations
    # =========================================================================
    op.create_table(
        'voice_configurations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('stt_provider', sa.String(50), nullable=False),
        sa.Column('stt_config', sa.JSON(), nullable=True),
        sa.Column('tts_provider', sa.String(50), nullable=False),
        sa.Column('tts_config', sa.JSON(), nullable=True),
        sa.Column('voice_id', sa.String(100), nullable=True),
        sa.Column('language', sa.String(10), default='en'),
        sa.Column('speed', sa.Float(), default=1.0),
        sa.Column('pitch', sa.Float(), default=1.0),
        sa.Column('is_default', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_voice_configs_org', 'voice_configurations', ['organization_id'])

    # =========================================================================
    # Conversations & Messages
    # =========================================================================
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('agent_id', sa.String(36), sa.ForeignKey('agents.id'), nullable=False),
        sa.Column('call_id', sa.String(100), nullable=True),
        sa.Column('customer_phone', sa.String(50), nullable=True),
        sa.Column('customer_id', sa.String(100), nullable=True),
        sa.Column('channel', sa.String(50), default='phone'),
        sa.Column('status', sa.String(50), default='active'),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_conversations_org', 'conversations', ['organization_id'])
    op.create_index('ix_conversations_agent', 'conversations', ['agent_id'])
    op.create_index('ix_conversations_call', 'conversations', ['call_id'])
    op.create_index('ix_conversations_customer', 'conversations', ['customer_phone'])

    op.create_table(
        'messages',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id'), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('audio_url', sa.String(500), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_messages_conversation', 'messages', ['conversation_id'])

    # =========================================================================
    # Calls
    # =========================================================================
    op.create_table(
        'calls',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('agent_id', sa.String(36), sa.ForeignKey('agents.id'), nullable=False),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id'), nullable=True),
        sa.Column('external_call_id', sa.String(100), nullable=True),
        sa.Column('direction', sa.String(20), default='inbound'),
        sa.Column('from_number', sa.String(50), nullable=True),
        sa.Column('to_number', sa.String(50), nullable=True),
        sa.Column('status', sa.String(50), default='initiated'),
        sa.Column('hangup_reason', sa.String(100), nullable=True),
        sa.Column('recording_url', sa.String(500), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('answered_at', sa.DateTime(), nullable=True),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('cost_cents', sa.Integer(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_calls_org', 'calls', ['organization_id'])
    op.create_index('ix_calls_agent', 'calls', ['agent_id'])
    op.create_index('ix_calls_external', 'calls', ['external_call_id'])
    op.create_index('ix_calls_status', 'calls', ['status'])

    op.create_table(
        'call_events',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('call_id', sa.String(36), sa.ForeignKey('calls.id'), nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('event_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_call_events_call', 'call_events', ['call_id'])
    op.create_index('ix_call_events_type', 'call_events', ['event_type'])

    # =========================================================================
    # Analytics
    # =========================================================================
    op.create_table(
        'analytics_events',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('agent_id', sa.String(36), nullable=True),
        sa.Column('call_id', sa.String(36), nullable=True),
        sa.Column('conversation_id', sa.String(36), nullable=True),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('event_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_analytics_org', 'analytics_events', ['organization_id'])
    op.create_index('ix_analytics_type', 'analytics_events', ['event_type'])
    op.create_index('ix_analytics_created', 'analytics_events', ['created_at'])

    op.create_table(
        'usage_records',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('period_start', sa.DateTime(), nullable=False),
        sa.Column('period_end', sa.DateTime(), nullable=False),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('unit', sa.String(20), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_usage_org_period', 'usage_records', ['organization_id', 'period_start'])
    op.create_index('ix_usage_metric', 'usage_records', ['metric_type'])


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.drop_table('usage_records')
    op.drop_table('analytics_events')
    op.drop_table('call_events')
    op.drop_table('calls')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('voice_configurations')
    op.drop_table('agent_versions')
    op.drop_table('agents')
    op.drop_table('api_keys')
    op.drop_table('users')
    op.drop_table('organization_settings')
    op.drop_table('organizations')
