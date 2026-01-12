"""Initial database schema.

Revision ID: 001
Revises:
Create Date: 2026-01-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('company', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
    )
    op.create_index('ix_users_email', 'users', ['email'])

    # Agents table
    op.create_table(
        'agents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('system_prompt', sa.Text(), nullable=True),
        sa.Column('first_message', sa.Text(), nullable=True),
        sa.Column('voice_id', sa.String(100), nullable=True),
        sa.Column('voice_provider', sa.String(50), server_default='elevenlabs', nullable=False),
        sa.Column('llm_provider', sa.String(50), server_default='openai', nullable=False),
        sa.Column('llm_model', sa.String(100), server_default='gpt-4o-mini', nullable=False),
        sa.Column('temperature', sa.Float(), server_default='0.7', nullable=False),
        sa.Column('max_tokens', sa.Integer(), server_default='1000', nullable=False),
        sa.Column('tools', postgresql.JSON(astext_type=sa.Text()), server_default='[]', nullable=False),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_agents_user_id', 'agents', ['user_id'])
    op.create_index('ix_agents_name', 'agents', ['name'])

    # Calls table
    op.create_table(
        'calls',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('external_id', sa.String(100), nullable=True),
        sa.Column('direction', sa.String(20), nullable=False),
        sa.Column('status', sa.String(50), server_default='pending', nullable=False),
        sa.Column('caller_number', sa.String(50), nullable=True),
        sa.Column('callee_number', sa.String(50), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('answered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('end_reason', sa.String(100), nullable=True),
        sa.Column('recording_url', sa.String(500), nullable=True),
        sa.Column('transcript', sa.Text(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('sentiment', sa.String(50), nullable=True),
        sa.Column('cost_cents', sa.Integer(), server_default='0', nullable=False),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_calls_agent_id', 'calls', ['agent_id'])
    op.create_index('ix_calls_user_id', 'calls', ['user_id'])
    op.create_index('ix_calls_external_id', 'calls', ['external_id'])
    op.create_index('ix_calls_status', 'calls', ['status'])
    op.create_index('ix_calls_started_at', 'calls', ['started_at'])

    # Call logs table (for detailed turn-by-turn logs)
    op.create_table(
        'call_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('call_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('speaker', sa.String(20), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['call_id'], ['calls.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_call_logs_call_id', 'call_logs', ['call_id'])
    op.create_index('ix_call_logs_timestamp', 'call_logs', ['timestamp'])
    op.create_index('ix_call_logs_event_type', 'call_logs', ['event_type'])

    # Knowledge bases table
    op.create_table(
        'knowledge_bases',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('source_type', sa.String(50), nullable=False),
        sa.Column('source_url', sa.String(500), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('chunk_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('vector_ids', postgresql.ARRAY(sa.String()), server_default='{}', nullable=False),
        sa.Column('status', sa.String(50), server_default='pending', nullable=False),
        sa.Column('last_synced_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_knowledge_bases_agent_id', 'knowledge_bases', ['agent_id'])

    # API keys table
    op.create_table(
        'api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False),
        sa.Column('key_prefix', sa.String(20), nullable=False),
        sa.Column('scopes', postgresql.ARRAY(sa.String()), server_default='{}', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_api_keys_user_id', 'api_keys', ['user_id'])
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'])
    op.create_index('ix_api_keys_key_prefix', 'api_keys', ['key_prefix'])

    # Phone numbers table
    op.create_table(
        'phone_numbers',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('number', sa.String(50), nullable=False),
        sa.Column('provider', sa.String(50), nullable=False),
        sa.Column('provider_id', sa.String(100), nullable=True),
        sa.Column('capabilities', postgresql.ARRAY(sa.String()), server_default='{}', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('monthly_cost_cents', sa.Integer(), server_default='0', nullable=False),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='SET NULL'),
        sa.UniqueConstraint('number'),
    )
    op.create_index('ix_phone_numbers_user_id', 'phone_numbers', ['user_id'])
    op.create_index('ix_phone_numbers_number', 'phone_numbers', ['number'])

    # Webhooks table
    op.create_table(
        'webhooks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('url', sa.String(500), nullable=False),
        sa.Column('events', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('secret', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('failure_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('last_triggered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_webhooks_user_id', 'webhooks', ['user_id'])


def downgrade() -> None:
    op.drop_table('webhooks')
    op.drop_table('phone_numbers')
    op.drop_table('api_keys')
    op.drop_table('knowledge_bases')
    op.drop_table('call_logs')
    op.drop_table('calls')
    op.drop_table('agents')
    op.drop_table('users')
