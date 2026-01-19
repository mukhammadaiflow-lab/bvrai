"""Tests for Dialog Service."""
import pytest
from app.adapters.vector_adapter import VectorDocument


@pytest.mark.asyncio
async def test_process_turn_basic(dialog_service):
    """Test basic dialog turn processing."""
    response = await dialog_service.process_turn(
        tenant_id="test-tenant",
        session_id="test-session",
        transcript="Hello, how are you?",
        is_final=True,
    )

    assert response.speak_text
    assert response.session_id == "test-session"
    assert 0 <= response.confidence <= 1


@pytest.mark.asyncio
async def test_process_turn_with_booking_intent(dialog_service):
    """Test dialog turn that triggers booking action."""
    response = await dialog_service.process_turn(
        tenant_id="test-tenant",
        session_id="test-session-2",
        transcript="I would like to book an appointment",
        is_final=True,
    )

    assert response.speak_text
    assert response.action_object is not None
    assert response.action_object.action_type == "initiate_booking"


@pytest.mark.asyncio
async def test_process_turn_interim_transcript(dialog_service):
    """Test interim (non-final) transcript handling."""
    response = await dialog_service.process_turn(
        tenant_id="test-tenant",
        session_id="test-session-3",
        transcript="I need...",
        is_final=False,
    )

    assert response.speak_text == ""
    assert response.action_object is None
    assert response.confidence == 0.0


@pytest.mark.asyncio
async def test_session_history_maintained(dialog_service):
    """Test that session history is maintained across turns."""
    session_id = "history-test-session"

    # First turn
    await dialog_service.process_turn(
        tenant_id="test-tenant",
        session_id=session_id,
        transcript="Hello",
        is_final=True,
    )

    # Second turn
    await dialog_service.process_turn(
        tenant_id="test-tenant",
        session_id=session_id,
        transcript="What are your hours?",
        is_final=True,
    )

    # Check history
    history = dialog_service.sessions.get_history(session_id)
    assert len(history) == 4  # 2 user + 2 assistant turns


@pytest.mark.asyncio
async def test_context_retrieval(dialog_service, local_vector_db):
    """Test that context is retrieved from vector DB."""
    # Add test document
    await local_vector_db.upsert([
        VectorDocument(
            id="doc-1",
            content="Our business hours are 9 AM to 5 PM, Monday through Friday.",
            metadata={"tenant_id": "test-tenant", "type": "faq"},
        )
    ])

    response = await dialog_service.process_turn(
        tenant_id="test-tenant",
        session_id="context-test-session",
        transcript="What are your business hours?",
        is_final=True,
    )

    assert response.speak_text
    # Note: With mock LLM, context doesn't affect response
    # In production, the LLM would use the retrieved context


@pytest.mark.asyncio
async def test_action_extraction_hours(dialog_service):
    """Test action extraction for hours inquiry."""
    response = await dialog_service.process_turn(
        tenant_id="test-tenant",
        session_id="hours-session",
        transcript="When do you open?",
        is_final=True,
    )

    assert response.action_object is not None
    assert response.action_object.action_type == "lookup_faq"


@pytest.mark.asyncio
async def test_multiple_sessions_isolated(dialog_service):
    """Test that multiple sessions are properly isolated."""
    # Session 1
    await dialog_service.process_turn(
        tenant_id="tenant-1",
        session_id="session-1",
        transcript="Hello",
        is_final=True,
    )

    # Session 2
    await dialog_service.process_turn(
        tenant_id="tenant-2",
        session_id="session-2",
        transcript="Hi there",
        is_final=True,
    )

    # Check sessions are separate
    session1 = dialog_service.sessions.get_session("session-1")
    session2 = dialog_service.sessions.get_session("session-2")

    assert session1.tenant_id == "tenant-1"
    assert session2.tenant_id == "tenant-2"
    assert len(session1.history) == 2
    assert len(session2.history) == 2
