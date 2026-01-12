"""Tests for Platform API agent endpoints."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from uuid import uuid4

# Mock database session for testing
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_db():
    """Create mock database session."""
    db = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.close = AsyncMock()
    db.execute = AsyncMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    return db


@pytest_asyncio.fixture
async def client():
    """Create test client for Platform API."""
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, "/home/user/bvrai/services/platform-api")

    from app.main import app
    from app.database.session import get_db

    # Override database dependency
    async def override_get_db():
        db = AsyncMock()
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        db.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None),
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
            scalar=MagicMock(return_value=0),
        ))
        yield db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    app.dependency_overrides.clear()


class TestHealthCheck:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "platform-api"

    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data


class TestAgentEndpoints:
    """Tests for agent CRUD endpoints."""

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, client):
        """Test listing agents when none exist."""
        response = await client.get("/api/v1/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_create_agent_validation(self, client, test_agent_data):
        """Test agent creation with validation."""
        # Missing required fields should fail
        response = await client.post(
            "/api/v1/agents",
            json={"name": ""},  # Empty name
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, client, agent_id):
        """Test getting non-existent agent returns 404."""
        response = await client.get(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_agent_not_found(self, client, agent_id):
        """Test deleting non-existent agent returns 404."""
        response = await client.delete(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 404


class TestCallEndpoints:
    """Tests for call endpoints."""

    @pytest.mark.asyncio
    async def test_list_calls_empty(self, client):
        """Test listing calls when none exist."""
        response = await client.get("/api/v1/calls")
        assert response.status_code == 200
        data = response.json()
        assert "calls" in data

    @pytest.mark.asyncio
    async def test_get_active_calls(self, client):
        """Test getting active calls."""
        response = await client.get("/api/v1/calls/active")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_call_not_found(self, client):
        """Test getting non-existent call returns 404."""
        call_id = str(uuid4())
        response = await client.get(f"/api/v1/calls/{call_id}")
        assert response.status_code == 404


class TestAnalyticsEndpoints:
    """Tests for analytics endpoints."""

    @pytest.mark.asyncio
    async def test_get_overview(self, client):
        """Test getting dashboard overview."""
        response = await client.get("/api/v1/analytics/overview")
        assert response.status_code == 200
        data = response.json()
        assert "total_calls" in data

    @pytest.mark.asyncio
    async def test_get_realtime_metrics(self, client):
        """Test getting realtime metrics."""
        response = await client.get("/api/v1/analytics/realtime")
        assert response.status_code == 200
        data = response.json()
        assert "active_calls" in data
        assert "timestamp" in data


class TestWebhookEndpoints:
    """Tests for webhook endpoints."""

    @pytest.mark.asyncio
    async def test_webhook_health(self, client):
        """Test webhook health endpoint."""
        response = await client.get("/api/v1/webhooks/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
