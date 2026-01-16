"""Add all feature tables - phone numbers, webhooks, knowledge bases, campaigns

Revision ID: 002_all_features
Revises: 001_initial
Create Date: 2026-01-16 00:00:00.000000

This migration adds all remaining tables for the complete MVP.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002_all_features'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add all new feature tables."""

    # =========================================================================
    # Update Agents table with new columns
    # =========================================================================
    op.add_column('agents', sa.Column('industry', sa.String(100), nullable=True))
    op.add_column('agents', sa.Column('voice_config_json', sa.JSON(), nullable=True))
    op.add_column('agents', sa.Column('behavior_config', sa.JSON(), nullable=True))
    op.add_column('agents', sa.Column('transcription_config', sa.JSON(), nullable=True))
    op.add_column('agents', sa.Column('knowledge_base_ids', sa.JSON(), nullable=True))
    op.add_column('agents', sa.Column('functions', sa.JSON(), nullable=True))
    op.add_column('agents', sa.Column('extra_data', sa.JSON(), nullable=True))
    op.add_column('agents', sa.Column('metadata_json', sa.JSON(), nullable=True))
    op.add_column('agents', sa.Column('tags', sa.JSON(), nullable=True))
    op.add_column('agents', sa.Column('phone_number', sa.String(50), nullable=True))
    op.add_column('agents', sa.Column('is_active', sa.Boolean(), server_default='true'))
    op.add_column('agents', sa.Column('total_calls', sa.Integer(), server_default='0'))
    op.add_column('agents', sa.Column('total_minutes', sa.Float(), server_default='0.0'))

    # =========================================================================
    # Update Calls table with new columns
    # =========================================================================
    op.add_column('calls', sa.Column('initiated_at', sa.DateTime(), nullable=True))
    op.add_column('calls', sa.Column('end_reason', sa.String(100), nullable=True))
    op.add_column('calls', sa.Column('transferred', sa.Boolean(), server_default='false'))
    op.add_column('calls', sa.Column('transfer_target', sa.String(100), nullable=True))
    op.add_column('calls', sa.Column('cost_amount', sa.Float(), server_default='0.0'))
    op.add_column('calls', sa.Column('cost_currency', sa.String(10), server_default="'USD'"))
    op.add_column('calls', sa.Column('is_deleted', sa.Boolean(), server_default='false'))
    op.add_column('calls', sa.Column('deleted_at', sa.DateTime(), nullable=True))

    # =========================================================================
    # Phone Numbers
    # =========================================================================
    op.create_table(
        'phone_numbers',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('number', sa.String(50), unique=True, nullable=False),
        sa.Column('friendly_name', sa.String(255), nullable=True),
        sa.Column('provider', sa.String(50), server_default='twilio'),
        sa.Column('provider_id', sa.String(100), nullable=True),
        sa.Column('number_type', sa.String(20), server_default='local'),
        sa.Column('country_code', sa.String(5), server_default='US'),
        sa.Column('voice_enabled', sa.Boolean(), server_default='true'),
        sa.Column('sms_enabled', sa.Boolean(), server_default='false'),
        sa.Column('mms_enabled', sa.Boolean(), server_default='false'),
        sa.Column('agent_id', sa.String(36), sa.ForeignKey('agents.id', ondelete='SET NULL'), nullable=True),
        sa.Column('webhook_url', sa.String(500), nullable=True),
        sa.Column('fallback_url', sa.String(500), nullable=True),
        sa.Column('status_callback_url', sa.String(500), nullable=True),
        sa.Column('status', sa.String(20), server_default='active'),
        sa.Column('monthly_cost', sa.Float(), server_default='0.0'),
        sa.Column('currency', sa.String(10), server_default='USD'),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), server_default='false'),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_phone_numbers_organization_id', 'phone_numbers', ['organization_id'])
    op.create_index('ix_phone_numbers_number', 'phone_numbers', ['number'])
    op.create_index('ix_phone_numbers_agent_id', 'phone_numbers', ['agent_id'])
    op.create_index('ix_phone_numbers_status', 'phone_numbers', ['status'])

    # =========================================================================
    # Webhooks
    # =========================================================================
    op.create_table(
        'webhooks',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('url', sa.String(500), nullable=False),
        sa.Column('secret', sa.String(255), nullable=True),
        sa.Column('auth_type', sa.String(20), server_default='none'),
        sa.Column('auth_value', sa.String(500), nullable=True),
        sa.Column('events', sa.JSON(), nullable=True),
        sa.Column('agent_ids', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('max_retries', sa.Integer(), server_default='3'),
        sa.Column('retry_delay_seconds', sa.Integer(), server_default='60'),
        sa.Column('timeout_seconds', sa.Integer(), server_default='30'),
        sa.Column('total_deliveries', sa.Integer(), server_default='0'),
        sa.Column('successful_deliveries', sa.Integer(), server_default='0'),
        sa.Column('failed_deliveries', sa.Integer(), server_default='0'),
        sa.Column('last_triggered_at', sa.DateTime(), nullable=True),
        sa.Column('last_success_at', sa.DateTime(), nullable=True),
        sa.Column('last_failure_at', sa.DateTime(), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), server_default='false'),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_webhooks_organization_id', 'webhooks', ['organization_id'])
    op.create_index('ix_webhooks_is_active', 'webhooks', ['is_active'])

    op.create_table(
        'webhook_deliveries',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('webhook_id', sa.String(36), sa.ForeignKey('webhooks.id', ondelete='CASCADE'), nullable=False),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('event_id', sa.String(36), nullable=False),
        sa.Column('request_url', sa.String(500), nullable=False),
        sa.Column('request_headers', sa.JSON(), nullable=True),
        sa.Column('request_body', sa.JSON(), nullable=True),
        sa.Column('response_status', sa.Integer(), nullable=True),
        sa.Column('response_headers', sa.JSON(), nullable=True),
        sa.Column('response_body', sa.Text(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('attempt_number', sa.Integer(), server_default='1'),
        sa.Column('next_retry_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_webhook_deliveries_webhook_id', 'webhook_deliveries', ['webhook_id'])
    op.create_index('ix_webhook_deliveries_event_type', 'webhook_deliveries', ['event_type'])
    op.create_index('ix_webhook_deliveries_status', 'webhook_deliveries', ['status'])
    op.create_index('ix_webhook_deliveries_created_at', 'webhook_deliveries', ['created_at'])

    # =========================================================================
    # Knowledge Bases
    # =========================================================================
    op.create_table(
        'knowledge_bases',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('embedding_model', sa.String(100), server_default='text-embedding-3-small'),
        sa.Column('embedding_provider', sa.String(50), server_default='openai'),
        sa.Column('chunk_size', sa.Integer(), server_default='1000'),
        sa.Column('chunk_overlap', sa.Integer(), server_default='200'),
        sa.Column('vector_store', sa.String(50), server_default='qdrant'),
        sa.Column('vector_collection', sa.String(100), nullable=True),
        sa.Column('status', sa.String(20), server_default='active'),
        sa.Column('document_count', sa.Integer(), server_default='0'),
        sa.Column('chunk_count', sa.Integer(), server_default='0'),
        sa.Column('total_tokens', sa.Integer(), server_default='0'),
        sa.Column('last_synced_at', sa.DateTime(), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), server_default='false'),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_knowledge_bases_organization_id', 'knowledge_bases', ['organization_id'])
    op.create_index('ix_knowledge_bases_status', 'knowledge_bases', ['status'])

    op.create_table(
        'documents',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('knowledge_base_id', sa.String(36), sa.ForeignKey('knowledge_bases.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('doc_type', sa.String(50), server_default='text'),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('source_url', sa.String(500), nullable=True),
        sa.Column('file_path', sa.String(500), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('mime_type', sa.String(100), nullable=True),
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('chunk_count', sa.Integer(), server_default='0'),
        sa.Column('token_count', sa.Integer(), server_default='0'),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), server_default='false'),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_documents_knowledge_base_id', 'documents', ['knowledge_base_id'])
    op.create_index('ix_documents_organization_id', 'documents', ['organization_id'])
    op.create_index('ix_documents_status', 'documents', ['status'])
    op.create_index('ix_documents_doc_type', 'documents', ['doc_type'])

    op.create_table(
        'document_chunks',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('document_id', sa.String(36), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('knowledge_base_id', sa.String(36), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('start_char', sa.Integer(), nullable=True),
        sa.Column('end_char', sa.Integer(), nullable=True),
        sa.Column('token_count', sa.Integer(), server_default='0'),
        sa.Column('vector_id', sa.String(100), nullable=True),
        sa.Column('embedding_model', sa.String(100), nullable=True),
        sa.Column('chunk_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_document_chunks_document_id', 'document_chunks', ['document_id'])
    op.create_index('ix_document_chunks_knowledge_base_id', 'document_chunks', ['knowledge_base_id'])
    op.create_index('ix_document_chunks_vector_id', 'document_chunks', ['vector_id'])

    # =========================================================================
    # Campaigns
    # =========================================================================
    op.create_table(
        'campaigns',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('organization_id', sa.String(36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('agent_id', sa.String(36), sa.ForeignKey('agents.id', ondelete='SET NULL'), nullable=True),
        sa.Column('phone_number_id', sa.String(36), sa.ForeignKey('phone_numbers.id', ondelete='SET NULL'), nullable=True),
        sa.Column('schedule_config', sa.JSON(), nullable=True),
        sa.Column('retry_config', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(20), server_default='draft'),
        sa.Column('total_contacts', sa.Integer(), server_default='0'),
        sa.Column('calls_completed', sa.Integer(), server_default='0'),
        sa.Column('calls_successful', sa.Integer(), server_default='0'),
        sa.Column('calls_failed', sa.Integer(), server_default='0'),
        sa.Column('calls_pending', sa.Integer(), server_default='0'),
        sa.Column('calls_in_progress', sa.Integer(), server_default='0'),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('paused_at', sa.DateTime(), nullable=True),
        sa.Column('total_cost', sa.Float(), server_default='0.0'),
        sa.Column('total_minutes', sa.Float(), server_default='0.0'),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), server_default='false'),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_campaigns_organization_id', 'campaigns', ['organization_id'])
    op.create_index('ix_campaigns_agent_id', 'campaigns', ['agent_id'])
    op.create_index('ix_campaigns_status', 'campaigns', ['status'])
    op.create_index('ix_campaigns_created_at', 'campaigns', ['created_at'])

    op.create_table(
        'campaign_contacts',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('campaign_id', sa.String(36), sa.ForeignKey('campaigns.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.String(36), nullable=False),
        sa.Column('phone_number', sa.String(50), nullable=False),
        sa.Column('first_name', sa.String(100), nullable=True),
        sa.Column('last_name', sa.String(100), nullable=True),
        sa.Column('email', sa.String(255), nullable=True),
        sa.Column('context', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('call_id', sa.String(36), nullable=True),
        sa.Column('call_outcome', sa.String(50), nullable=True),
        sa.Column('attempt_count', sa.Integer(), server_default='0'),
        sa.Column('last_attempt_at', sa.DateTime(), nullable=True),
        sa.Column('next_attempt_at', sa.DateTime(), nullable=True),
        sa.Column('call_duration_seconds', sa.Float(), server_default='0.0'),
        sa.Column('call_cost', sa.Float(), server_default='0.0'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_campaign_contacts_campaign_id', 'campaign_contacts', ['campaign_id'])
    op.create_index('ix_campaign_contacts_organization_id', 'campaign_contacts', ['organization_id'])
    op.create_index('ix_campaign_contacts_status', 'campaign_contacts', ['status'])
    op.create_index('ix_campaign_contacts_phone_number', 'campaign_contacts', ['phone_number'])
    op.create_unique_constraint('uq_campaign_contact_phone', 'campaign_contacts', ['campaign_id', 'phone_number'])


def downgrade() -> None:
    """Drop all new tables and columns."""
    # Drop campaign tables
    op.drop_table('campaign_contacts')
    op.drop_table('campaigns')

    # Drop knowledge base tables
    op.drop_table('document_chunks')
    op.drop_table('documents')
    op.drop_table('knowledge_bases')

    # Drop webhook tables
    op.drop_table('webhook_deliveries')
    op.drop_table('webhooks')

    # Drop phone numbers
    op.drop_table('phone_numbers')

    # Remove columns from calls
    op.drop_column('calls', 'deleted_at')
    op.drop_column('calls', 'is_deleted')
    op.drop_column('calls', 'cost_currency')
    op.drop_column('calls', 'cost_amount')
    op.drop_column('calls', 'transfer_target')
    op.drop_column('calls', 'transferred')
    op.drop_column('calls', 'end_reason')
    op.drop_column('calls', 'initiated_at')

    # Remove columns from agents
    op.drop_column('agents', 'total_minutes')
    op.drop_column('agents', 'total_calls')
    op.drop_column('agents', 'is_active')
    op.drop_column('agents', 'phone_number')
    op.drop_column('agents', 'tags')
    op.drop_column('agents', 'metadata_json')
    op.drop_column('agents', 'extra_data')
    op.drop_column('agents', 'functions')
    op.drop_column('agents', 'knowledge_base_ids')
    op.drop_column('agents', 'transcription_config')
    op.drop_column('agents', 'behavior_config')
    op.drop_column('agents', 'voice_config_json')
    op.drop_column('agents', 'industry')
