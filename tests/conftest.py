"""Shared pytest fixtures for testing."""

import asyncio
import os
from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# Set test environment
os.environ["ENVIRONMENT"] = "development"
os.environ["DATABASE_URL"] = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/bvrai_test"
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def db_engine():
    """Create test database engine."""
    engine = create_async_engine(
        os.environ["DATABASE_URL"],
        echo=False,
        pool_pre_ping=True,
    )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session with transaction rollback."""
    async_session = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session
        # Rollback any changes made during the test
        await session.rollback()


# =============================================================================
# API Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def app() -> FastAPI:
    """Create test FastAPI application."""
    from bvrai_core.api.app import create_app, AppConfig

    config = AppConfig(
        debug=True,
        docs_enabled=False,
    )
    return create_app(config)


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def authenticated_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create authenticated test HTTP client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Set auth headers for testing
        ac.headers["Authorization"] = "Bearer test_token"
        ac.headers["X-Organization-ID"] = "org_test123"
        yield ac


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def agent_id() -> str:
    """Generate a test agent ID."""
    return f"agt_{uuid4().hex}"


@pytest.fixture
def organization_id() -> str:
    """Generate a test organization ID."""
    return f"org_{uuid4().hex}"


@pytest.fixture
def user_id() -> str:
    """Generate a test user ID."""
    return f"usr_{uuid4().hex}"


@pytest.fixture
def session_id() -> str:
    """Generate a test session ID."""
    return str(uuid4())


@pytest.fixture
def test_agent_data(organization_id: str):
    """Sample agent data for testing."""
    return {
        "name": "Test Agent",
        "description": "A test voice agent",
        "system_prompt": "You are a helpful test assistant.",
        "voice": {
            "provider": "elevenlabs",
            "voice_id": "test_voice",
            "speed": 1.0,
            "pitch": 1.0,
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "behavior": {
            "greeting_message": "Hello! How can I help you today?",
            "silence_timeout_seconds": 10,
            "max_call_duration_seconds": 1800,
        },
    }


@pytest.fixture
def test_call_data(agent_id: str):
    """Sample call data for testing."""
    return {
        "agent_id": agent_id,
        "to_number": "+15555551234",
        "from_number": "+15555555678",
        "metadata": {"test": True},
    }


@pytest.fixture
def test_document_data():
    """Sample document data for testing."""
    return {
        "title": "Test Document",
        "content": "This is a test document with some sample content for RAG testing.",
        "source": "test",
        "metadata": {"type": "test"},
    }


@pytest.fixture
def sample_audio_data() -> bytes:
    """Generate sample audio data (silence)."""
    # 1 second of silence at 16kHz, 16-bit mono
    samples = 16000
    return bytes(samples * 2)


@pytest.fixture
def sample_transcript():
    """Sample transcript for testing."""
    return {
        "speaker": "user",
        "text": "Hello, I need help with my order.",
        "is_final": True,
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def test_organization(db_session: AsyncSession, organization_id: str):
    """Create a test organization in the database."""
    from bvrai_core.database.models import Organization

    org = Organization(
        id=organization_id,
        name="Test Organization",
        slug="test-org",
        email="test@example.com",
        is_active=True,
    )
    db_session.add(org)
    await db_session.commit()
    await db_session.refresh(org)
    return org


@pytest_asyncio.fixture
async def test_agent(
    db_session: AsyncSession,
    test_organization,
    test_agent_data: dict,
):
    """Create a test agent in the database."""
    from bvrai_core.database.models import Agent

    agent = Agent(
        id=f"agt_{uuid4().hex}",
        organization_id=test_organization.id,
        name=test_agent_data["name"],
        description=test_agent_data["description"],
        slug="test-agent",
        system_prompt=test_agent_data["system_prompt"],
        first_message=test_agent_data["behavior"]["greeting_message"],
        llm_provider=test_agent_data["llm"]["provider"],
        llm_model=test_agent_data["llm"]["model"],
        llm_temperature=test_agent_data["llm"]["temperature"],
        llm_max_tokens=test_agent_data["llm"]["max_tokens"],
        is_active=True,
        metadata_json={
            "voice": test_agent_data["voice"],
            "behavior": test_agent_data["behavior"],
        },
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest_asyncio.fixture
async def test_agents(
    db_session: AsyncSession,
    test_organization,
):
    """Create multiple test agents in the database."""
    from bvrai_core.database.models import Agent

    agents = []
    for i in range(5):
        agent = Agent(
            id=f"agt_{uuid4().hex}",
            organization_id=test_organization.id,
            name=f"Test Agent {i + 1}",
            description=f"Test agent number {i + 1}",
            slug=f"test-agent-{i + 1}",
            system_prompt="You are a helpful assistant.",
            llm_provider="openai",
            llm_model="gpt-4",
            llm_temperature=0.7,
            llm_max_tokens=1000,
            is_active=i % 2 == 0,  # Alternate active/inactive
        )
        agents.append(agent)
        db_session.add(agent)

    await db_session.commit()
    for agent in agents:
        await db_session.refresh(agent)

    return agents


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession, test_organization, user_id: str):
    """Create a test user in the database."""
    from bvrai_core.database.models import User

    user = User(
        id=user_id,
        organization_id=test_organization.id,
        email="test@example.com",
        first_name="Test",
        last_name="User",
        role="admin",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user
