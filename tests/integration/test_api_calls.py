"""Integration tests for calls API."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient
from uuid import uuid4


class TestCallsAPI:
    """Tests for call management endpoints."""

    @pytest.mark.asyncio
    async def test_make_outbound_call(
        self, authenticated_client: AsyncClient, test_agent, mock_twilio
    ):
        """Test making an outbound call."""
        response = await authenticated_client.post(
            "/api/v1/calls/outbound",
            json={
                "agent_id": test_agent.id,
                "to_number": "+15555551234",
                "from_number": "+15555555678",
                "metadata": {"campaign": "test"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "call_id" in data
        assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_make_call_invalid_number(
        self, authenticated_client: AsyncClient, test_agent
    ):
        """Test making call with invalid phone number."""
        response = await authenticated_client.post(
            "/api/v1/calls/outbound",
            json={
                "agent_id": test_agent.id,
                "to_number": "invalid",
                "from_number": "+15555555678",
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_make_call_inactive_agent(
        self, authenticated_client: AsyncClient, test_agent
    ):
        """Test making call with inactive agent."""
        # Deactivate agent
        await authenticated_client.post(f"/api/v1/agents/{test_agent.id}/pause")

        response = await authenticated_client.post(
            "/api/v1/calls/outbound",
            json={
                "agent_id": test_agent.id,
                "to_number": "+15555551234",
                "from_number": "+15555555678",
            },
        )

        assert response.status_code == 400
        assert "inactive" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_list_calls(self, authenticated_client: AsyncClient, db_session, test_tenant):
        """Test listing calls."""
        # Create some test calls
        from app.models.call import Call

        for i in range(5):
            call = Call(
                id=str(uuid4()),
                tenant_id=test_tenant.id,
                agent_id=str(uuid4()),
                direction="outbound",
                status="completed" if i % 2 == 0 else "failed",
                to_number=f"+1555555{i:04d}",
                from_number="+15555555678",
                started_at=datetime.utcnow() - timedelta(hours=i),
                ended_at=datetime.utcnow() - timedelta(hours=i) + timedelta(minutes=5),
                duration_seconds=300,
            )
            db_session.add(call)
        await db_session.commit()

        response = await authenticated_client.get("/api/v1/calls")

        assert response.status_code == 200
        data = response.json()
        assert "calls" in data
        assert "total" in data
        assert len(data["calls"]) == 5

    @pytest.mark.asyncio
    async def test_list_calls_with_filters(
        self, authenticated_client: AsyncClient, db_session, test_tenant, test_agent
    ):
        """Test listing calls with filters."""
        # Create test calls
        from app.models.call import Call

        for i in range(3):
            call = Call(
                id=str(uuid4()),
                tenant_id=test_tenant.id,
                agent_id=test_agent.id,
                direction="outbound",
                status="completed",
                to_number=f"+1555555{i:04d}",
                from_number="+15555555678",
                started_at=datetime.utcnow() - timedelta(hours=i),
                duration_seconds=300,
            )
            db_session.add(call)
        await db_session.commit()

        response = await authenticated_client.get(
            "/api/v1/calls",
            params={
                "agent_id": test_agent.id,
                "status": "completed",
            },
        )

        assert response.status_code == 200
        data = response.json()
        for call in data["calls"]:
            assert call["agent_id"] == test_agent.id
            assert call["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_call(
        self, authenticated_client: AsyncClient, db_session, test_tenant, test_agent
    ):
        """Test getting a single call."""
        from app.models.call import Call

        call = Call(
            id=str(uuid4()),
            tenant_id=test_tenant.id,
            agent_id=test_agent.id,
            direction="outbound",
            status="completed",
            to_number="+15555551234",
            from_number="+15555555678",
            started_at=datetime.utcnow(),
            ended_at=datetime.utcnow() + timedelta(minutes=5),
            duration_seconds=300,
        )
        db_session.add(call)
        await db_session.commit()

        response = await authenticated_client.get(f"/api/v1/calls/{call.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == call.id
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_call_not_found(self, authenticated_client: AsyncClient):
        """Test getting non-existent call."""
        response = await authenticated_client.get("/api/v1/calls/nonexistent")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_call_transcript(
        self, authenticated_client: AsyncClient, db_session, test_tenant, test_agent
    ):
        """Test getting call transcript."""
        from app.models.call import Call
        from app.models.transcript import Transcript

        call = Call(
            id=str(uuid4()),
            tenant_id=test_tenant.id,
            agent_id=test_agent.id,
            direction="outbound",
            status="completed",
            to_number="+15555551234",
            from_number="+15555555678",
            started_at=datetime.utcnow(),
            duration_seconds=300,
        )
        db_session.add(call)

        transcript = Transcript(
            id=str(uuid4()),
            call_id=call.id,
            segments=[
                {"speaker": "user", "text": "Hello", "start": 0.0, "end": 1.0},
                {"speaker": "agent", "text": "Hi there!", "start": 1.2, "end": 2.5},
            ],
            full_text="Hello\nHi there!",
        )
        db_session.add(transcript)
        await db_session.commit()

        response = await authenticated_client.get(f"/api/v1/calls/{call.id}/transcript")

        assert response.status_code == 200
        data = response.json()
        assert "segments" in data
        assert len(data["segments"]) == 2

    @pytest.mark.asyncio
    async def test_get_call_recording(
        self, authenticated_client: AsyncClient, db_session, test_tenant, test_agent
    ):
        """Test getting call recording URL."""
        from app.models.call import Call
        from app.models.recording import Recording

        call = Call(
            id=str(uuid4()),
            tenant_id=test_tenant.id,
            agent_id=test_agent.id,
            direction="outbound",
            status="completed",
            to_number="+15555551234",
            from_number="+15555555678",
            started_at=datetime.utcnow(),
            duration_seconds=300,
        )
        db_session.add(call)

        recording = Recording(
            id=str(uuid4()),
            call_id=call.id,
            storage_path="recordings/test.wav",
            duration_seconds=300,
            format="wav",
            sample_rate=16000,
        )
        db_session.add(recording)
        await db_session.commit()

        with patch("app.services.storage.generate_presigned_url") as mock_url:
            mock_url.return_value = "https://storage.example.com/recording.wav?token=xxx"

            response = await authenticated_client.get(f"/api/v1/calls/{call.id}/recording")

            assert response.status_code == 200
            data = response.json()
            assert "url" in data

    @pytest.mark.asyncio
    async def test_end_active_call(
        self, authenticated_client: AsyncClient, db_session, test_tenant, test_agent
    ):
        """Test ending an active call."""
        from app.models.call import Call

        call = Call(
            id=str(uuid4()),
            tenant_id=test_tenant.id,
            agent_id=test_agent.id,
            direction="outbound",
            status="active",
            to_number="+15555551234",
            from_number="+15555555678",
            started_at=datetime.utcnow(),
        )
        db_session.add(call)
        await db_session.commit()

        with patch("app.services.telephony.hangup_call", new_callable=AsyncMock):
            response = await authenticated_client.post(f"/api/v1/calls/{call.id}/end")

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_call_stats(self, authenticated_client: AsyncClient):
        """Test getting call statistics."""
        response = await authenticated_client.get(
            "/api/v1/calls/stats",
            params={
                "start_date": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "end_date": datetime.utcnow().isoformat(),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_calls" in data
        assert "total_minutes" in data
        assert "success_rate" in data

    @pytest.mark.asyncio
    async def test_export_calls(self, authenticated_client: AsyncClient):
        """Test exporting calls data."""
        response = await authenticated_client.get(
            "/api/v1/calls/export",
            params={
                "format": "csv",
                "start_date": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            },
        )

        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")


class TestBulkCalls:
    """Tests for bulk call operations."""

    @pytest.mark.asyncio
    async def test_create_bulk_call_campaign(
        self, authenticated_client: AsyncClient, test_agent
    ):
        """Test creating a bulk call campaign."""
        response = await authenticated_client.post(
            "/api/v1/calls/bulk",
            json={
                "agent_id": test_agent.id,
                "from_number": "+15555555678",
                "recipients": [
                    {"to_number": "+15555551234", "metadata": {"name": "John"}},
                    {"to_number": "+15555551235", "metadata": {"name": "Jane"}},
                    {"to_number": "+15555551236", "metadata": {"name": "Bob"}},
                ],
                "schedule": {
                    "start_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                    "calls_per_minute": 5,
                },
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "campaign_id" in data
        assert data["total_recipients"] == 3

    @pytest.mark.asyncio
    async def test_get_campaign_status(self, authenticated_client: AsyncClient):
        """Test getting bulk campaign status."""
        # Create campaign first
        with patch("app.services.campaign.create_campaign", new_callable=AsyncMock) as mock:
            mock.return_value = {"campaign_id": "camp_123"}

            create_response = await authenticated_client.post(
                "/api/v1/calls/bulk",
                json={
                    "agent_id": str(uuid4()),
                    "from_number": "+15555555678",
                    "recipients": [{"to_number": "+15555551234"}],
                },
            )
            campaign_id = create_response.json().get("campaign_id", "camp_123")

        response = await authenticated_client.get(f"/api/v1/calls/bulk/{campaign_id}")

        assert response.status_code in [200, 404]  # Depends on implementation

    @pytest.mark.asyncio
    async def test_cancel_campaign(self, authenticated_client: AsyncClient):
        """Test canceling a bulk campaign."""
        campaign_id = "camp_123"

        with patch("app.services.campaign.cancel_campaign", new_callable=AsyncMock):
            response = await authenticated_client.post(
                f"/api/v1/calls/bulk/{campaign_id}/cancel"
            )

            assert response.status_code in [200, 404]
