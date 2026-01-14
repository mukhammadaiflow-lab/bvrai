"""Add telephony integration fields.

Revision ID: 002
Revises: 001
Create Date: 2026-01-14

Adds fields for Twilio telephony integration:
- twilio_call_sid to calls table
- session_id to calls table
- Updated call status enum values
- Additional call_logs fields
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add twilio_call_sid to calls table
    op.add_column(
        'calls',
        sa.Column('twilio_call_sid', sa.String(100), nullable=True)
    )
    op.create_index('ix_calls_twilio_call_sid', 'calls', ['twilio_call_sid'])

    # Add session_id to calls table if not exists
    op.add_column(
        'calls',
        sa.Column('session_id', sa.String(100), nullable=True)
    )
    op.create_index('ix_calls_session_id', 'calls', ['session_id'], unique=True)

    # Add from_number and to_number columns (renamed from caller/callee)
    op.add_column(
        'calls',
        sa.Column('from_number', sa.String(50), nullable=True)
    )
    op.add_column(
        'calls',
        sa.Column('to_number', sa.String(50), nullable=True)
    )

    # Add turn_number and role to call_logs
    op.add_column(
        'call_logs',
        sa.Column('turn_number', sa.Integer(), nullable=True)
    )
    op.add_column(
        'call_logs',
        sa.Column('role', sa.String(20), nullable=True)
    )
    op.add_column(
        'call_logs',
        sa.Column('intent', sa.String(100), nullable=True)
    )
    op.add_column(
        'call_logs',
        sa.Column('entities', postgresql.JSON(astext_type=sa.Text()), nullable=True)
    )
    op.add_column(
        'call_logs',
        sa.Column('function_call', postgresql.JSON(astext_type=sa.Text()), nullable=True)
    )
    op.add_column(
        'call_logs',
        sa.Column('function_result', postgresql.JSON(astext_type=sa.Text()), nullable=True)
    )

    # Create usage_records table
    op.create_table(
        'usage_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('usage_type', sa.String(50), nullable=False),
        sa.Column('amount', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('unit', sa.String(20), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=True),
        sa.Column('resource_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('details', postgresql.JSON(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_usage_user_type_date', 'usage_records', ['user_id', 'usage_type', 'recorded_at'])
    op.create_index('ix_usage_recorded_at', 'usage_records', ['recorded_at'])

    # Create audit_events table
    op.create_table(
        'audit_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_id', sa.String(36), nullable=False),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('actor_id', sa.String(255), nullable=True),
        sa.Column('action', sa.Text(), nullable=False),
        sa.Column('outcome', sa.String(20), nullable=False),
        sa.Column('resource_type', sa.String(100), nullable=True),
        sa.Column('resource_id', sa.String(255), nullable=True),
        sa.Column('request_id', sa.String(100), nullable=True),
        sa.Column('session_id', sa.String(100), nullable=True),
        sa.Column('tenant_id', sa.String(100), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('correlation_id', sa.String(100), nullable=True),
        sa.Column('details', postgresql.JSON(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('previous_hash', sa.String(64), nullable=True),
        sa.Column('event_hash', sa.String(64), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_audit_events_event_id', 'audit_events', ['event_id'], unique=True)
    op.create_index('ix_audit_events_event_type', 'audit_events', ['event_type'])
    op.create_index('ix_audit_events_actor_id', 'audit_events', ['actor_id'])
    op.create_index('ix_audit_events_tenant_id', 'audit_events', ['tenant_id'])
    op.create_index('ix_audit_events_timestamp', 'audit_events', ['timestamp'])
    op.create_index('ix_audit_type_timestamp', 'audit_events', ['event_type', 'timestamp'])
    op.create_index('ix_audit_actor_timestamp', 'audit_events', ['actor_id', 'timestamp'])
    op.create_index('ix_audit_resource', 'audit_events', ['resource_type', 'resource_id'])

    # Create response_time_logs table
    op.create_table(
        'response_time_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('call_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('asr_latency_ms', sa.Integer(), nullable=True),
        sa.Column('llm_latency_ms', sa.Integer(), nullable=True),
        sa.Column('tts_latency_ms', sa.Integer(), nullable=True),
        sa.Column('total_latency_ms', sa.Integer(), nullable=True),
        sa.Column('turn_number', sa.Integer(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['call_id'], ['calls.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_response_call_turn', 'response_time_logs', ['call_id', 'turn_number'])
    op.create_index('ix_response_recorded', 'response_time_logs', ['recorded_at'])

    # Create queue_wait_times table
    op.create_table(
        'queue_wait_times',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('call_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('wait_time_seconds', sa.Integer(), nullable=False),
        sa.Column('initial_position', sa.Integer(), nullable=True),
        sa.Column('entered_queue_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('exited_queue_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('outcome', sa.String(50), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['call_id'], ['calls.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_queue_agent_entered', 'queue_wait_times', ['agent_id', 'entered_queue_at'])

    # Add revoked_at to api_keys table
    op.add_column(
        'api_keys',
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    # Drop new tables
    op.drop_table('queue_wait_times')
    op.drop_table('response_time_logs')
    op.drop_table('audit_events')
    op.drop_table('usage_records')

    # Remove added columns from api_keys
    op.drop_column('api_keys', 'revoked_at')

    # Remove added columns from call_logs
    op.drop_column('call_logs', 'function_result')
    op.drop_column('call_logs', 'function_call')
    op.drop_column('call_logs', 'entities')
    op.drop_column('call_logs', 'intent')
    op.drop_column('call_logs', 'role')
    op.drop_column('call_logs', 'turn_number')

    # Remove added columns from calls
    op.drop_column('calls', 'to_number')
    op.drop_column('calls', 'from_number')
    op.drop_index('ix_calls_session_id', 'calls')
    op.drop_column('calls', 'session_id')
    op.drop_index('ix_calls_twilio_call_sid', 'calls')
    op.drop_column('calls', 'twilio_call_sid')
