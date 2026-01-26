"""Add security constraints and fix foreign keys

Revision ID: 003_security_constraints
Revises: 002_production_readiness
Create Date: 2026-01-21 00:00:01.000000

This migration adds critical security constraints:
1. Missing foreign keys for multi-tenancy data isolation
2. Fix CASCADE behavior for proper data cleanup
3. Add CHECK constraints for status fields
4. Add JSONB GIN indexes for better query performance
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '003_security_constraints'
down_revision: Union[str, None] = '002_production_readiness'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply security constraints."""

    # =========================================================================
    # 1. Add missing foreign key on voice_configurations.organization_id
    # This is CRITICAL for multi-tenant data isolation
    # =========================================================================

    op.create_foreign_key(
        'fk_voice_configurations_organization',
        'voice_configurations',
        'organizations',
        ['organization_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # =========================================================================
    # 2. Add missing foreign key on conversations.organization_id
    # =========================================================================

    op.create_foreign_key(
        'fk_conversations_organization',
        'conversations',
        'organizations',
        ['organization_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # =========================================================================
    # 3. Fix messages.conversation_id cascade behavior
    # When a conversation is deleted, messages should also be deleted
    # SET NULL leaves orphan messages which is incorrect
    # =========================================================================

    # First drop the existing constraint
    op.drop_constraint(
        'messages_conversation_id_fkey',
        'messages',
        type_='foreignkey'
    )

    # Recreate with CASCADE
    op.create_foreign_key(
        'fk_messages_conversation',
        'messages',
        'conversations',
        ['conversation_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # =========================================================================
    # 4. Add CHECK constraints for status fields
    # Prevents invalid status values from being inserted
    # =========================================================================

    # Call status constraint
    op.execute("""
        ALTER TABLE calls ADD CONSTRAINT check_call_status
        CHECK (status IN (
            'initiated', 'ringing', 'in_progress', 'completed',
            'failed', 'no_answer', 'busy', 'cancelled', 'transferred'
        ))
    """)

    # Conversation status constraint
    op.execute("""
        ALTER TABLE conversations ADD CONSTRAINT check_conversation_status
        CHECK (status IN ('active', 'completed', 'abandoned', 'transferred'))
    """)

    # Message role constraint
    op.execute("""
        ALTER TABLE messages ADD CONSTRAINT check_message_role
        CHECK (role IN ('user', 'assistant', 'system', 'tool'))
    """)

    # =========================================================================
    # 5. Add GIN indexes for JSONB columns for better query performance
    # These enable fast JSON path queries and containment checks
    # =========================================================================

    # Agents metadata GIN index
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_agents_metadata_gin
        ON agents USING GIN (metadata_json)
    """)

    # Conversations metadata GIN index
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_conversations_metadata_gin
        ON conversations USING GIN (metadata_json)
    """)

    # Calls metadata GIN index
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_calls_metadata_gin
        ON calls USING GIN (metadata)
    """)

    # =========================================================================
    # 6. Add index on organization_id for voice_configurations
    # Enables fast lookups by organization
    # =========================================================================

    op.create_index(
        'ix_voice_configurations_organization_id',
        'voice_configurations',
        ['organization_id']
    )

    # =========================================================================
    # 7. Add NOT NULL constraints where missing
    # =========================================================================

    # Ensure conversation always has an organization
    op.alter_column(
        'conversations',
        'organization_id',
        existing_type=sa.String(36),
        nullable=False
    )


def downgrade() -> None:
    """Revert security constraints."""

    # 7. Revert NOT NULL
    op.alter_column(
        'conversations',
        'organization_id',
        existing_type=sa.String(36),
        nullable=True
    )

    # 6. Drop organization index
    op.drop_index('ix_voice_configurations_organization_id', table_name='voice_configurations')

    # 5. Drop GIN indexes
    op.execute("DROP INDEX IF EXISTS ix_calls_metadata_gin")
    op.execute("DROP INDEX IF EXISTS ix_conversations_metadata_gin")
    op.execute("DROP INDEX IF EXISTS ix_agents_metadata_gin")

    # 4. Drop CHECK constraints
    op.execute("ALTER TABLE messages DROP CONSTRAINT IF EXISTS check_message_role")
    op.execute("ALTER TABLE conversations DROP CONSTRAINT IF EXISTS check_conversation_status")
    op.execute("ALTER TABLE calls DROP CONSTRAINT IF EXISTS check_call_status")

    # 3. Revert message FK to SET NULL
    op.drop_constraint('fk_messages_conversation', 'messages', type_='foreignkey')
    op.create_foreign_key(
        'messages_conversation_id_fkey',
        'messages',
        'conversations',
        ['conversation_id'],
        ['id'],
        ondelete='SET NULL'
    )

    # 2. Drop conversations FK
    op.drop_constraint('fk_conversations_organization', 'conversations', type_='foreignkey')

    # 1. Drop voice_configurations FK
    op.drop_constraint('fk_voice_configurations_organization', 'voice_configurations', type_='foreignkey')
