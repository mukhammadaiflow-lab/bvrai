"""Enable Row-Level Security for multi-tenancy

Revision ID: 004_row_level_security
Revises: 003_security_constraints
Create Date: 2026-01-21 00:00:02.000000

This migration enables PostgreSQL Row-Level Security (RLS) for
critical multi-tenant tables to ensure data isolation between
organizations.

IMPORTANT: This requires the application to set the current organization
context for each request using:
    SET app.current_org_id = '<organization_id>';

The policies ensure that:
1. Users can only see data belonging to their organization
2. Insert/update operations automatically use the current org context
3. Superusers bypass RLS (for admin operations)
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '004_row_level_security'
down_revision: Union[str, None] = '003_security_constraints'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Tables requiring RLS for multi-tenant isolation
RLS_TABLES = [
    'agents',
    'agent_versions',
    'voice_configurations',
    'conversations',
    'messages',
    'calls',
    'call_events',
    'api_keys',
    'webhooks',
    'webhook_events',
    'analytics_events',
    'usage_records',
    'knowledge_bases',
    'documents',
]


def upgrade() -> None:
    """Enable Row-Level Security on multi-tenant tables."""

    # =========================================================================
    # 1. Create helper function to get current org
    # =========================================================================

    op.execute("""
        CREATE OR REPLACE FUNCTION current_org_id()
        RETURNS text AS $$
        BEGIN
            RETURN COALESCE(
                current_setting('app.current_org_id', true),
                ''
            );
        END;
        $$ LANGUAGE plpgsql STABLE;
    """)

    # =========================================================================
    # 2. Enable RLS and create policies for each table
    # =========================================================================

    for table in RLS_TABLES:
        # Enable RLS on table
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")

        # Create policy for SELECT (users see only their org's data)
        op.execute(f"""
            CREATE POLICY {table}_org_isolation_select ON {table}
                FOR SELECT
                USING (
                    organization_id = current_org_id()
                    OR current_org_id() = ''
                );
        """)

        # Create policy for INSERT (users can only insert to their org)
        op.execute(f"""
            CREATE POLICY {table}_org_isolation_insert ON {table}
                FOR INSERT
                WITH CHECK (
                    organization_id = current_org_id()
                    OR current_org_id() = ''
                );
        """)

        # Create policy for UPDATE (users can only update their org's data)
        op.execute(f"""
            CREATE POLICY {table}_org_isolation_update ON {table}
                FOR UPDATE
                USING (
                    organization_id = current_org_id()
                    OR current_org_id() = ''
                )
                WITH CHECK (
                    organization_id = current_org_id()
                    OR current_org_id() = ''
                );
        """)

        # Create policy for DELETE (users can only delete their org's data)
        op.execute(f"""
            CREATE POLICY {table}_org_isolation_delete ON {table}
                FOR DELETE
                USING (
                    organization_id = current_org_id()
                    OR current_org_id() = ''
                );
        """)

    # =========================================================================
    # 3. Handle tables with indirect organization reference (messages)
    # Messages reference conversations which have organization_id
    # =========================================================================

    # Drop the simple policy for messages and create a join-based one
    op.execute("DROP POLICY IF EXISTS messages_org_isolation_select ON messages;")
    op.execute("DROP POLICY IF EXISTS messages_org_isolation_insert ON messages;")
    op.execute("DROP POLICY IF EXISTS messages_org_isolation_update ON messages;")
    op.execute("DROP POLICY IF EXISTS messages_org_isolation_delete ON messages;")

    # Messages: Check via conversation's organization_id
    op.execute("""
        CREATE POLICY messages_org_isolation_select ON messages
            FOR SELECT
            USING (
                EXISTS (
                    SELECT 1 FROM conversations c
                    WHERE c.id = messages.conversation_id
                    AND (c.organization_id = current_org_id() OR current_org_id() = '')
                )
            );
    """)

    op.execute("""
        CREATE POLICY messages_org_isolation_insert ON messages
            FOR INSERT
            WITH CHECK (
                EXISTS (
                    SELECT 1 FROM conversations c
                    WHERE c.id = messages.conversation_id
                    AND (c.organization_id = current_org_id() OR current_org_id() = '')
                )
            );
    """)

    op.execute("""
        CREATE POLICY messages_org_isolation_update ON messages
            FOR UPDATE
            USING (
                EXISTS (
                    SELECT 1 FROM conversations c
                    WHERE c.id = messages.conversation_id
                    AND (c.organization_id = current_org_id() OR current_org_id() = '')
                )
            );
    """)

    op.execute("""
        CREATE POLICY messages_org_isolation_delete ON messages
            FOR DELETE
            USING (
                EXISTS (
                    SELECT 1 FROM conversations c
                    WHERE c.id = messages.conversation_id
                    AND (c.organization_id = current_org_id() OR current_org_id() = '')
                )
            );
    """)

    # =========================================================================
    # 4. Similarly for agent_versions (via agent's organization_id)
    # =========================================================================

    op.execute("DROP POLICY IF EXISTS agent_versions_org_isolation_select ON agent_versions;")
    op.execute("DROP POLICY IF EXISTS agent_versions_org_isolation_insert ON agent_versions;")
    op.execute("DROP POLICY IF EXISTS agent_versions_org_isolation_update ON agent_versions;")
    op.execute("DROP POLICY IF EXISTS agent_versions_org_isolation_delete ON agent_versions;")

    op.execute("""
        CREATE POLICY agent_versions_org_isolation_select ON agent_versions
            FOR SELECT
            USING (
                EXISTS (
                    SELECT 1 FROM agents a
                    WHERE a.id = agent_versions.agent_id
                    AND (a.organization_id = current_org_id() OR current_org_id() = '')
                )
            );
    """)

    op.execute("""
        CREATE POLICY agent_versions_org_isolation_insert ON agent_versions
            FOR INSERT
            WITH CHECK (
                EXISTS (
                    SELECT 1 FROM agents a
                    WHERE a.id = agent_versions.agent_id
                    AND (a.organization_id = current_org_id() OR current_org_id() = '')
                )
            );
    """)

    op.execute("""
        CREATE POLICY agent_versions_org_isolation_update ON agent_versions
            FOR UPDATE
            USING (
                EXISTS (
                    SELECT 1 FROM agents a
                    WHERE a.id = agent_versions.agent_id
                    AND (a.organization_id = current_org_id() OR current_org_id() = '')
                )
            );
    """)

    op.execute("""
        CREATE POLICY agent_versions_org_isolation_delete ON agent_versions
            FOR DELETE
            USING (
                EXISTS (
                    SELECT 1 FROM agents a
                    WHERE a.id = agent_versions.agent_id
                    AND (a.organization_id = current_org_id() OR current_org_id() = '')
                )
            );
    """)


def downgrade() -> None:
    """Disable Row-Level Security."""

    # Drop all policies and disable RLS for each table
    for table in RLS_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_org_isolation_select ON {table};")
        op.execute(f"DROP POLICY IF EXISTS {table}_org_isolation_insert ON {table};")
        op.execute(f"DROP POLICY IF EXISTS {table}_org_isolation_update ON {table};")
        op.execute(f"DROP POLICY IF EXISTS {table}_org_isolation_delete ON {table};")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")

    # Drop helper function
    op.execute("DROP FUNCTION IF EXISTS current_org_id();")
