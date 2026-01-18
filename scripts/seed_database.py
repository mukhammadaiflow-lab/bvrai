#!/usr/bin/env python3
"""
Database Seeding Script

Creates demo data for development and testing purposes.
"""

import asyncio
import hashlib
import secrets
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select

from bvrai_core.database.models import (
    Base,
    Organization,
    User,
    APIKey,
    Agent,
    Call,
    PhoneNumber,
    Webhook,
    KnowledgeBase,
    Document,
    Campaign,
    CampaignContact,
)


# =============================================================================
# Configuration
# =============================================================================

DATABASE_URL = "sqlite+aiosqlite:///./bvrai.db"

# Demo organization
DEMO_ORG = {
    "id": "demo-org-001",
    "name": "Demo Organization",
    "slug": "demo-org",
    "plan": "professional",
}

# Demo user
DEMO_USER = {
    "id": "demo-user-001",
    "email": "demo@example.com",
    "password": "demo1234",  # Will be hashed
    "first_name": "Demo",
    "last_name": "User",
    "role": "owner",
}

# Demo API key (for development)
DEMO_API_KEY = "bvr_demo_key_for_development_only_do_not_use_in_production"


# =============================================================================
# Seed Functions
# =============================================================================

async def seed_organization(session: AsyncSession) -> Organization:
    """Create demo organization."""
    # Check if exists
    result = await session.execute(
        select(Organization).where(Organization.id == DEMO_ORG["id"])
    )
    org = result.scalar_one_or_none()

    if org:
        print(f"  Organization '{org.name}' already exists")
        return org

    org = Organization(
        id=DEMO_ORG["id"],
        name=DEMO_ORG["name"],
        slug=DEMO_ORG["slug"],
        plan=DEMO_ORG["plan"],
        is_active=True,
    )
    session.add(org)
    await session.flush()
    print(f"  Created organization: {org.name}")
    return org


async def seed_user(session: AsyncSession, org: Organization) -> User:
    """Create demo user."""
    result = await session.execute(
        select(User).where(User.email == DEMO_USER["email"])
    )
    user = result.scalar_one_or_none()

    if user:
        print(f"  User '{user.email}' already exists")
        return user

    user = User(
        id=DEMO_USER["id"],
        organization_id=org.id,
        email=DEMO_USER["email"],
        first_name=DEMO_USER["first_name"],
        last_name=DEMO_USER["last_name"],
        role=DEMO_USER["role"],
        is_active=True,
        is_verified=True,
    )
    user.set_password(DEMO_USER["password"])
    session.add(user)
    await session.flush()
    print(f"  Created user: {user.email}")
    return user


async def seed_api_key(session: AsyncSession, org: Organization, user: User) -> APIKey:
    """Create demo API key."""
    key_hash = hashlib.sha256(DEMO_API_KEY.encode()).hexdigest()

    result = await session.execute(
        select(APIKey).where(APIKey.key_hash == key_hash)
    )
    api_key = result.scalar_one_or_none()

    if api_key:
        print(f"  API key already exists")
        return api_key

    api_key = APIKey(
        organization_id=org.id,
        created_by_user_id=user.id,
        name="Demo API Key",
        key_hash=key_hash,
        key_prefix=DEMO_API_KEY[:12],
        scopes=["*"],  # Full access
        is_active=True,
    )
    session.add(api_key)
    await session.flush()
    print(f"  Created API key: {DEMO_API_KEY[:20]}...")
    return api_key


async def seed_agents(session: AsyncSession, org: Organization) -> list[Agent]:
    """Create demo agents."""
    agents_data = [
        {
            "name": "Sales Agent",
            "description": "Handles inbound sales calls and qualifies leads",
            "system_prompt": """You are a friendly sales assistant for a software company.
Your goal is to understand the customer's needs, qualify them as a potential lead,
and schedule a demo with a sales representative if they're interested.

Be conversational, professional, and helpful. Ask about their business needs,
team size, and timeline. If they're a good fit, offer to schedule a demo.""",
            "first_message": "Hi! Thanks for calling. I'm here to help you learn more about our software. What brings you to us today?",
            "industry": "technology",
        },
        {
            "name": "Customer Support Agent",
            "description": "Handles customer support inquiries and troubleshooting",
            "system_prompt": """You are a helpful customer support agent.
Your goal is to resolve customer issues efficiently and professionally.

Listen carefully to the customer's problem, ask clarifying questions if needed,
and provide clear step-by-step solutions. If you can't resolve the issue,
offer to escalate to a human agent.""",
            "first_message": "Hello! I'm here to help with any questions or issues you may have. How can I assist you today?",
            "industry": "customer_service",
        },
        {
            "name": "Appointment Scheduler",
            "description": "Books appointments and manages calendar",
            "system_prompt": """You are an appointment scheduling assistant.
Your goal is to help customers book, reschedule, or cancel appointments.

Be efficient and friendly. Confirm all appointment details including date, time,
and any special requirements. Send a confirmation at the end of the call.""",
            "first_message": "Hi there! I can help you schedule an appointment. What date and time works best for you?",
            "industry": "healthcare",
        },
    ]

    agents = []
    for data in agents_data:
        result = await session.execute(
            select(Agent).where(
                Agent.organization_id == org.id,
                Agent.name == data["name"],
            )
        )
        agent = result.scalar_one_or_none()

        if agent:
            print(f"  Agent '{agent.name}' already exists")
            agents.append(agent)
            continue

        agent = Agent(
            organization_id=org.id,
            name=data["name"],
            description=data["description"],
            system_prompt=data["system_prompt"],
            first_message=data["first_message"],
            industry=data.get("industry"),
            is_active=True,
            is_published=True,
            voice_config_json={
                "provider": "elevenlabs",
                "voice_id": "21m00Tcm4TlvDq8ikWAM",
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 500,
            },
            behavior_config={
                "interruption_handling": "graceful",
                "silence_timeout_seconds": 5,
                "max_call_duration_minutes": 30,
            },
        )
        session.add(agent)
        await session.flush()
        print(f"  Created agent: {agent.name}")
        agents.append(agent)

    return agents


async def seed_phone_numbers(
    session: AsyncSession, org: Organization, agents: list[Agent]
) -> list[PhoneNumber]:
    """Create demo phone numbers."""
    numbers_data = [
        {
            "number": "+14155551234",
            "friendly_name": "Main Sales Line",
            "country_code": "US",
            "number_type": "local",
            "provider": "twilio",
        },
        {
            "number": "+14155555678",
            "friendly_name": "Support Line",
            "country_code": "US",
            "number_type": "local",
            "provider": "twilio",
        },
        {
            "number": "+18005551234",
            "friendly_name": "Toll-Free Line",
            "country_code": "US",
            "number_type": "toll_free",
            "provider": "twilio",
        },
    ]

    phone_numbers = []
    for i, data in enumerate(numbers_data):
        result = await session.execute(
            select(PhoneNumber).where(PhoneNumber.number == data["number"])
        )
        phone = result.scalar_one_or_none()

        if phone:
            print(f"  Phone number '{phone.number}' already exists")
            phone_numbers.append(phone)
            continue

        phone = PhoneNumber(
            organization_id=org.id,
            number=data["number"],
            friendly_name=data["friendly_name"],
            country_code=data["country_code"],
            number_type=data["number_type"],
            provider=data["provider"],
            voice_enabled=True,
            sms_enabled=False,
            status="active",
            monthly_cost=1.50,
            # Assign to agents if available
            agent_id=agents[i].id if i < len(agents) else None,
        )
        session.add(phone)
        await session.flush()
        print(f"  Created phone number: {phone.number}")
        phone_numbers.append(phone)

    return phone_numbers


async def seed_webhooks(session: AsyncSession, org: Organization) -> list[Webhook]:
    """Create demo webhooks."""
    webhooks_data = [
        {
            "name": "Call Notifications",
            "url": "https://webhook.site/demo-endpoint",
            "events": ["call.started", "call.ended", "call.failed"],
        },
        {
            "name": "Campaign Updates",
            "url": "https://webhook.site/campaign-endpoint",
            "events": ["campaign.started", "campaign.completed"],
        },
    ]

    webhooks = []
    for data in webhooks_data:
        result = await session.execute(
            select(Webhook).where(
                Webhook.organization_id == org.id,
                Webhook.name == data["name"],
            )
        )
        webhook = result.scalar_one_or_none()

        if webhook:
            print(f"  Webhook '{webhook.name}' already exists")
            webhooks.append(webhook)
            continue

        webhook = Webhook(
            organization_id=org.id,
            name=data["name"],
            url=data["url"],
            events=data["events"],
            secret=secrets.token_hex(16),
            is_active=True,
            max_retries=3,
            timeout_seconds=30,
        )
        session.add(webhook)
        await session.flush()
        print(f"  Created webhook: {webhook.name}")
        webhooks.append(webhook)

    return webhooks


async def seed_knowledge_bases(
    session: AsyncSession, org: Organization
) -> list[KnowledgeBase]:
    """Create demo knowledge bases."""
    kb_data = [
        {
            "name": "Product FAQ",
            "description": "Frequently asked questions about our products",
        },
        {
            "name": "Pricing Information",
            "description": "Pricing plans and feature comparison",
        },
    ]

    knowledge_bases = []
    for data in kb_data:
        result = await session.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.organization_id == org.id,
                KnowledgeBase.name == data["name"],
            )
        )
        kb = result.scalar_one_or_none()

        if kb:
            print(f"  Knowledge base '{kb.name}' already exists")
            knowledge_bases.append(kb)
            continue

        kb = KnowledgeBase(
            organization_id=org.id,
            name=data["name"],
            description=data["description"],
            embedding_model="text-embedding-3-small",
            embedding_provider="openai",
            chunk_size=1000,
            chunk_overlap=200,
            status="active",
        )
        session.add(kb)
        await session.flush()

        # Add some sample documents
        documents = [
            {
                "name": "FAQ Document",
                "doc_type": "text",
                "content": """Q: What is your return policy?
A: We offer a 30-day money-back guarantee on all products.

Q: How do I contact support?
A: You can reach us at support@example.com or call 1-800-555-0123.

Q: Do you offer bulk discounts?
A: Yes, we offer discounts for orders of 100+ units. Contact sales for details.""",
            },
        ]

        for doc_data in documents:
            doc = Document(
                knowledge_base_id=kb.id,
                organization_id=org.id,
                name=doc_data["name"],
                doc_type=doc_data["doc_type"],
                content=doc_data["content"],
                status="completed",
                chunk_count=3,
                token_count=100,
            )
            session.add(doc)

        await session.flush()
        print(f"  Created knowledge base: {kb.name}")
        knowledge_bases.append(kb)

    return knowledge_bases


async def seed_campaigns(
    session: AsyncSession,
    org: Organization,
    agents: list[Agent],
    phone_numbers: list[PhoneNumber],
) -> list[Campaign]:
    """Create demo campaigns."""
    if not agents or not phone_numbers:
        print("  Skipping campaigns - no agents or phone numbers")
        return []

    campaigns_data = [
        {
            "name": "Q1 Outreach Campaign",
            "description": "Quarterly customer outreach campaign",
            "status": "completed",
        },
        {
            "name": "Product Launch Campaign",
            "description": "New product announcement calls",
            "status": "draft",
        },
    ]

    campaigns = []
    for data in campaigns_data:
        result = await session.execute(
            select(Campaign).where(
                Campaign.organization_id == org.id,
                Campaign.name == data["name"],
            )
        )
        campaign = result.scalar_one_or_none()

        if campaign:
            print(f"  Campaign '{campaign.name}' already exists")
            campaigns.append(campaign)
            continue

        campaign = Campaign(
            organization_id=org.id,
            name=data["name"],
            description=data["description"],
            agent_id=agents[0].id,
            phone_number_id=phone_numbers[0].id,
            status=data["status"],
            schedule_config={
                "timezone": "America/New_York",
                "daily_start_hour": 9,
                "daily_end_hour": 17,
                "days_of_week": [1, 2, 3, 4, 5],
            },
            total_contacts=10,
            calls_completed=5 if data["status"] == "completed" else 0,
            calls_successful=3 if data["status"] == "completed" else 0,
        )
        session.add(campaign)
        await session.flush()

        # Add sample contacts
        for i in range(5):
            contact = CampaignContact(
                campaign_id=campaign.id,
                organization_id=org.id,
                phone_number=f"+1415555{1000 + i}",
                first_name=f"Contact{i + 1}",
                last_name="Demo",
                email=f"contact{i + 1}@example.com",
                status="completed" if data["status"] == "completed" else "pending",
            )
            session.add(contact)

        await session.flush()
        print(f"  Created campaign: {campaign.name}")
        campaigns.append(campaign)

    return campaigns


async def seed_calls(
    session: AsyncSession,
    org: Organization,
    agents: list[Agent],
    phone_numbers: list[PhoneNumber],
) -> list[Call]:
    """Create demo call history."""
    if not agents:
        print("  Skipping calls - no agents")
        return []

    # Create some sample calls
    calls = []
    statuses = ["completed", "completed", "completed", "failed", "completed"]
    directions = ["inbound", "inbound", "outbound", "inbound", "outbound"]

    for i in range(5):
        initiated = datetime.utcnow() - timedelta(hours=i * 2)
        call = Call(
            organization_id=org.id,
            agent_id=agents[0].id,
            direction=directions[i],
            status=statuses[i],
            from_number=f"+1415555{2000 + i}",
            to_number=phone_numbers[0].number if phone_numbers else "+14155551234",
            initiated_at=initiated,
            answered_at=initiated + timedelta(seconds=5),
            ended_at=initiated + timedelta(minutes=5),
            duration_seconds=300.0,
            cost_amount=0.15,
        )
        session.add(call)
        calls.append(call)

    await session.flush()
    print(f"  Created {len(calls)} demo calls")
    return calls


# =============================================================================
# Main Seeding Function
# =============================================================================

async def seed_database():
    """Main database seeding function."""
    print("\n=== BVRAI Database Seeding ===\n")

    # Create engine and session
    engine = create_async_engine(DATABASE_URL, echo=False)

    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created/verified\n")

    # Create session
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        try:
            # Seed data
            print("Seeding organization...")
            org = await seed_organization(session)

            print("\nSeeding user...")
            user = await seed_user(session, org)

            print("\nSeeding API key...")
            api_key = await seed_api_key(session, org, user)

            print("\nSeeding agents...")
            agents = await seed_agents(session, org)

            print("\nSeeding phone numbers...")
            phone_numbers = await seed_phone_numbers(session, org, agents)

            print("\nSeeding webhooks...")
            webhooks = await seed_webhooks(session, org)

            print("\nSeeding knowledge bases...")
            knowledge_bases = await seed_knowledge_bases(session, org)

            print("\nSeeding campaigns...")
            campaigns = await seed_campaigns(session, org, agents, phone_numbers)

            print("\nSeeding call history...")
            calls = await seed_calls(session, org, agents, phone_numbers)

            # Commit all changes
            await session.commit()

            print("\n" + "=" * 50)
            print("Database seeding completed successfully!")
            print("=" * 50)
            print(f"\nDemo credentials:")
            print(f"  Email: {DEMO_USER['email']}")
            print(f"  Password: {DEMO_USER['password']}")
            print(f"  API Key: {DEMO_API_KEY}")
            print(f"\nTo use dev mode, set environment variable:")
            print(f"  export BVRAI_DEV_MODE=true")
            print(f"  export BVRAI_DEV_ORG_ID={DEMO_ORG['id']}")

        except Exception as e:
            await session.rollback()
            print(f"\nError during seeding: {e}")
            raise

    await engine.dispose()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    asyncio.run(seed_database())
