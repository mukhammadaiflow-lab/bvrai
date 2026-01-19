"""Tests for WebRTC Gateway signaling."""

import json
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch


@pytest_asyncio.fixture
async def client():
    """Create test client for WebRTC Gateway."""
    import sys
    sys.path.insert(0, "/home/user/bvrai/services/webrtc-gateway")

    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


class TestHealthCheck:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "webrtc-gateway"
        assert "active_sessions" in data
        assert "max_sessions" in data

    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "websocket" in data


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, client):
        """Test listing sessions when none exist."""
        response = await client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client, session_id):
        """Test getting non-existent session returns 404."""
        response = await client.get(f"/sessions/{session_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_end_session_not_found(self, client, session_id):
        """Test ending non-existent session returns 404."""
        response = await client.delete(f"/sessions/{session_id}")
        assert response.status_code == 404


class TestIceServerEndpoint:
    """Tests for ICE server configuration."""

    @pytest.mark.asyncio
    async def test_get_ice_servers(self, client):
        """Test getting ICE server configuration."""
        response = await client.get("/ice-servers")
        assert response.status_code == 200
        data = response.json()
        assert "ice_servers" in data
        assert isinstance(data["ice_servers"], list)


class TestStatsEndpoint:
    """Tests for statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats(self, client):
        """Test getting gateway statistics."""
        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_sessions" in data
        assert "active_sessions" in data
        assert "max_sessions" in data


class TestSignalingModels:
    """Tests for signaling message models."""

    def test_signaling_message_creation(self):
        """Test creating signaling messages."""
        import sys
        sys.path.insert(0, "/home/user/bvrai/services/webrtc-gateway")

        from app.signaling.models import (
            SignalingMessage,
            SignalingMessageType,
            ConnectPayload,
        )

        # Test basic message
        msg = SignalingMessage(type=SignalingMessageType.PING)
        assert msg.type == SignalingMessageType.PING
        assert msg.id is not None
        assert msg.timestamp is not None

        # Test message with payload
        connect_payload = ConnectPayload(
            agent_id="00000000-0000-0000-0000-000000000001",
            api_key="test_key",
        )
        msg = SignalingMessage(
            type=SignalingMessageType.CONNECT,
            payload=connect_payload.model_dump(),
        )
        assert msg.type == SignalingMessageType.CONNECT
        assert msg.payload["agent_id"] == "00000000-0000-0000-0000-000000000001"

    def test_offer_payload(self):
        """Test SDP offer payload."""
        import sys
        sys.path.insert(0, "/home/user/bvrai/services/webrtc-gateway")

        from app.signaling.models import OfferPayload

        offer = OfferPayload(sdp="v=0\r\no=- 123 2 IN IP4 127.0.0.1\r\n")
        assert offer.type == "offer"
        assert "v=0" in offer.sdp

    def test_ice_candidate_payload(self):
        """Test ICE candidate payload."""
        import sys
        sys.path.insert(0, "/home/user/bvrai/services/webrtc-gateway")

        from app.signaling.models import IceCandidatePayload

        candidate = IceCandidatePayload(
            candidate="candidate:1 1 UDP 2130706431 192.168.1.1 5000 typ host",
            sdp_mid="audio",
            sdp_m_line_index=0,
        )
        assert "UDP" in candidate.candidate
        assert candidate.sdp_mid == "audio"
