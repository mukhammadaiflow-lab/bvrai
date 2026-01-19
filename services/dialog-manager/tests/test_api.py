"""API tests for Dialog Manager Service."""
import pytest


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """Test health check endpoint."""
    response = await async_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "dialog-manager"


@pytest.mark.asyncio
async def test_ready_endpoint(async_client):
    """Test readiness check endpoint."""
    response = await async_client.get("/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_dialog_turn_endpoint(async_client):
    """Test dialog turn endpoint."""
    response = await async_client.post(
        "/dialog/turn",
        json={
            "tenant_id": "test-tenant",
            "session_id": "api-test-session",
            "transcript": "Hello, I need help",
            "is_final": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "speak_text" in data
    assert "action_object" in data
    assert "confidence" in data
    assert "session_id" in data
    assert data["session_id"] == "api-test-session"


@pytest.mark.asyncio
async def test_dialog_turn_validation_error(async_client):
    """Test dialog turn endpoint with invalid request."""
    response = await async_client.post(
        "/dialog/turn",
        json={
            "tenant_id": "",  # Invalid: empty
            "session_id": "test",
            "transcript": "Hello",
        },
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_dialog_turn_missing_field(async_client):
    """Test dialog turn endpoint with missing required field."""
    response = await async_client.post(
        "/dialog/turn",
        json={
            "tenant_id": "test",
            # Missing session_id
            "transcript": "Hello",
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_dialog_turn_booking_intent(async_client):
    """Test dialog turn that triggers booking action."""
    response = await async_client.post(
        "/dialog/turn",
        json={
            "tenant_id": "test-tenant",
            "session_id": "booking-session",
            "transcript": "I want to schedule an appointment",
            "is_final": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["action_object"] is not None
    assert data["action_object"]["action_type"] == "initiate_booking"


@pytest.mark.asyncio
async def test_get_session_endpoint(async_client):
    """Test get session endpoint."""
    # First create a session by making a dialog turn
    await async_client.post(
        "/dialog/turn",
        json={
            "tenant_id": "test-tenant",
            "session_id": "get-session-test",
            "transcript": "Hello",
            "is_final": True,
        },
    )

    # Then get the session
    response = await async_client.get("/sessions/get-session-test")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "get-session-test"
    assert data["tenant_id"] == "test-tenant"
    assert "history" in data


@pytest.mark.asyncio
async def test_get_session_not_found(async_client):
    """Test get session endpoint with non-existent session."""
    response = await async_client.get("/sessions/non-existent-session")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_endpoint(async_client):
    """Test delete session endpoint."""
    # First create a session
    await async_client.post(
        "/dialog/turn",
        json={
            "tenant_id": "test-tenant",
            "session_id": "delete-session-test",
            "transcript": "Hello",
            "is_final": True,
        },
    )

    # Delete the session
    response = await async_client.delete("/sessions/delete-session-test")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"

    # Verify session is gone
    get_response = await async_client.get("/sessions/delete-session-test")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_not_found(async_client):
    """Test delete session endpoint with non-existent session."""
    response = await async_client.delete("/sessions/non-existent-session")

    assert response.status_code == 404
