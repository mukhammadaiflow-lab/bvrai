"""Unit tests for conversation engine."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.conversation.engine import ConversationEngine
from app.conversation.context import ConversationContext
from app.conversation.history import ConversationHistory, Message
from app.conversation.turn_taking import TurnTakingManager


class TestConversationContext:
    """Tests for ConversationContext."""

    def test_create_context(self):
        """Test creating conversation context."""
        context = ConversationContext(
            session_id=str(uuid4()),
            agent_id=str(uuid4()),
            user_id=str(uuid4()),
        )

        assert context.session_id is not None
        assert context.turn_count == 0
        assert context.metadata == {}

    def test_add_variable(self):
        """Test adding context variables."""
        context = ConversationContext(
            session_id=str(uuid4()),
            agent_id=str(uuid4()),
        )

        context.set_variable("customer_name", "John Doe")
        context.set_variable("order_id", "12345")

        assert context.get_variable("customer_name") == "John Doe"
        assert context.get_variable("order_id") == "12345"

    def test_get_missing_variable(self):
        """Test getting missing variable returns default."""
        context = ConversationContext(
            session_id=str(uuid4()),
            agent_id=str(uuid4()),
        )

        assert context.get_variable("missing") is None
        assert context.get_variable("missing", default="default") == "default"

    def test_increment_turn(self):
        """Test incrementing turn count."""
        context = ConversationContext(
            session_id=str(uuid4()),
            agent_id=str(uuid4()),
        )

        assert context.turn_count == 0

        context.increment_turn()
        assert context.turn_count == 1

        context.increment_turn()
        assert context.turn_count == 2

    def test_context_to_dict(self):
        """Test converting context to dictionary."""
        context = ConversationContext(
            session_id="sess_123",
            agent_id="agent_456",
            user_id="user_789",
        )
        context.set_variable("key", "value")

        data = context.to_dict()

        assert data["session_id"] == "sess_123"
        assert data["agent_id"] == "agent_456"
        assert data["variables"]["key"] == "value"


class TestMessage:
    """Tests for Message class."""

    def test_create_user_message(self):
        """Test creating a user message."""
        message = Message(
            role="user",
            content="Hello, I need help.",
        )

        assert message.role == "user"
        assert message.content == "Hello, I need help."
        assert message.timestamp is not None

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        message = Message(
            role="assistant",
            content="Hi! How can I assist you today?",
        )

        assert message.role == "assistant"
        assert message.content == "Hi! How can I assist you today?"

    def test_message_with_metadata(self):
        """Test creating message with metadata."""
        message = Message(
            role="user",
            content="Test message",
            metadata={
                "confidence": 0.95,
                "language": "en",
            },
        )

        assert message.metadata["confidence"] == 0.95
        assert message.metadata["language"] == "en"

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        message = Message(
            role="assistant",
            content="Hello!",
            metadata={"key": "value"},
        )

        data = message.to_dict()

        assert data["role"] == "assistant"
        assert data["content"] == "Hello!"
        assert data["metadata"]["key"] == "value"


class TestConversationHistory:
    """Tests for ConversationHistory."""

    def test_create_history(self):
        """Test creating conversation history."""
        history = ConversationHistory(max_messages=100)

        assert len(history) == 0
        assert history.max_messages == 100

    def test_add_message(self):
        """Test adding messages to history."""
        history = ConversationHistory()

        history.add_message(Message(role="user", content="Hello"))
        history.add_message(Message(role="assistant", content="Hi there!"))

        assert len(history) == 2
        assert history.messages[0].role == "user"
        assert history.messages[1].role == "assistant"

    def test_history_max_messages(self):
        """Test history respects max messages limit."""
        history = ConversationHistory(max_messages=3)

        for i in range(5):
            history.add_message(Message(role="user", content=f"Message {i}"))

        assert len(history) == 3
        assert "Message 2" in history.messages[0].content

    def test_get_recent_messages(self):
        """Test getting recent messages."""
        history = ConversationHistory()

        for i in range(10):
            history.add_message(Message(role="user", content=f"Message {i}"))

        recent = history.get_recent(5)

        assert len(recent) == 5
        assert "Message 9" in recent[-1].content

    def test_clear_history(self):
        """Test clearing history."""
        history = ConversationHistory()
        history.add_message(Message(role="user", content="Test"))

        history.clear()

        assert len(history) == 0

    def test_to_messages_format(self):
        """Test converting to LLM messages format."""
        history = ConversationHistory()
        history.add_message(Message(role="user", content="Hello"))
        history.add_message(Message(role="assistant", content="Hi!"))

        messages = history.to_messages_format()

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi!"}


class TestTurnTakingManager:
    """Tests for TurnTakingManager."""

    def test_create_manager(self):
        """Test creating turn taking manager."""
        manager = TurnTakingManager(
            min_silence_for_turn_end=0.5,
            max_turn_duration=30.0,
        )

        assert manager.min_silence_for_turn_end == 0.5
        assert manager.max_turn_duration == 30.0

    def test_start_turn(self):
        """Test starting a turn."""
        manager = TurnTakingManager()

        manager.start_turn("user")

        assert manager.current_speaker == "user"
        assert manager.turn_start_time is not None

    def test_end_turn(self):
        """Test ending a turn."""
        manager = TurnTakingManager()
        manager.start_turn("user")

        manager.end_turn()

        assert manager.current_speaker is None
        assert manager.turn_start_time is None

    def test_detect_turn_end_by_silence(self):
        """Test detecting turn end by silence."""
        manager = TurnTakingManager(min_silence_for_turn_end=0.5)
        manager.start_turn("user")

        # Short silence - turn not over
        assert manager.should_end_turn(silence_duration=0.3) is False

        # Long silence - turn should end
        assert manager.should_end_turn(silence_duration=0.7) is True

    def test_detect_turn_end_by_duration(self):
        """Test detecting turn end by max duration."""
        manager = TurnTakingManager(max_turn_duration=5.0)
        manager.start_turn("user")

        # Within max duration
        assert manager.should_end_turn(turn_duration=3.0) is False

        # Exceeded max duration
        assert manager.should_end_turn(turn_duration=6.0) is True

    def test_barge_in_detection(self):
        """Test barge-in detection."""
        manager = TurnTakingManager(allow_barge_in=True)
        manager.start_turn("agent")

        result = manager.detect_barge_in(
            speech_detected=True,
            confidence=0.8,
        )

        assert result["barge_in_detected"] is True
        assert result["confidence"] >= 0.8


class TestConversationEngine:
    """Tests for ConversationEngine."""

    @pytest.mark.asyncio
    async def test_create_engine(self):
        """Test creating conversation engine."""
        engine = ConversationEngine(
            agent_id=str(uuid4()),
            session_id=str(uuid4()),
        )

        assert engine.agent_id is not None
        assert engine.session_id is not None
        assert engine.is_active is False

    @pytest.mark.asyncio
    async def test_start_conversation(self):
        """Test starting a conversation."""
        engine = ConversationEngine(
            agent_id=str(uuid4()),
            session_id=str(uuid4()),
        )

        with patch.object(engine, '_load_agent', new_callable=AsyncMock):
            await engine.start()

        assert engine.is_active is True

    @pytest.mark.asyncio
    async def test_end_conversation(self):
        """Test ending a conversation."""
        engine = ConversationEngine(
            agent_id=str(uuid4()),
            session_id=str(uuid4()),
        )
        engine.is_active = True

        await engine.end()

        assert engine.is_active is False

    @pytest.mark.asyncio
    async def test_process_user_input(self):
        """Test processing user input."""
        engine = ConversationEngine(
            agent_id=str(uuid4()),
            session_id=str(uuid4()),
        )
        engine.is_active = True

        with patch.object(engine, '_generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "I understand, let me help you with that."

            response = await engine.process_input("I need help with my order")

            assert response is not None
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_interruption(self):
        """Test handling user interruption."""
        engine = ConversationEngine(
            agent_id=str(uuid4()),
            session_id=str(uuid4()),
        )
        engine.is_active = True
        engine._current_response = "I was saying..."

        await engine.handle_interruption()

        assert engine._current_response is None
        assert engine.context.get_variable("last_interrupted") is True

    @pytest.mark.asyncio
    async def test_context_persistence(self):
        """Test context persists across turns."""
        engine = ConversationEngine(
            agent_id=str(uuid4()),
            session_id=str(uuid4()),
        )
        engine.is_active = True

        engine.context.set_variable("customer_name", "John")

        with patch.object(engine, '_generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "Hello John!"

            await engine.process_input("My name is John")

            assert engine.context.get_variable("customer_name") == "John"


class TestConversationAnalytics:
    """Tests for conversation analytics."""

    def test_track_message_metrics(self):
        """Test tracking message metrics."""
        from app.conversation.analytics import ConversationAnalytics

        analytics = ConversationAnalytics(session_id=str(uuid4()))

        analytics.track_message(
            role="user",
            content="Hello",
            latency=0.5,
        )

        assert analytics.message_count == 1
        assert analytics.user_message_count == 1
        assert analytics.avg_latency == 0.5

    def test_track_turn_metrics(self):
        """Test tracking turn metrics."""
        from app.conversation.analytics import ConversationAnalytics

        analytics = ConversationAnalytics(session_id=str(uuid4()))

        analytics.start_turn()
        analytics.end_turn(duration=5.0)

        assert analytics.turn_count == 1
        assert analytics.avg_turn_duration == 5.0

    def test_calculate_engagement_score(self):
        """Test calculating engagement score."""
        from app.conversation.analytics import ConversationAnalytics

        analytics = ConversationAnalytics(session_id=str(uuid4()))

        # Add some interactions
        for i in range(10):
            analytics.track_message(role="user", content=f"Message {i}", latency=0.3)
            analytics.track_message(role="assistant", content=f"Response {i}", latency=0.5)

        score = analytics.calculate_engagement_score()

        assert 0 <= score <= 1.0

    def test_get_summary(self):
        """Test getting analytics summary."""
        from app.conversation.analytics import ConversationAnalytics

        analytics = ConversationAnalytics(session_id=str(uuid4()))

        analytics.track_message(role="user", content="Test", latency=0.4)
        analytics.track_message(role="assistant", content="Response", latency=0.6)

        summary = analytics.get_summary()

        assert "message_count" in summary
        assert "avg_latency" in summary
        assert summary["message_count"] == 2
