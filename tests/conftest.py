"""Shared pytest fixtures for testing."""

import asyncio
from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def agent_id() -> str:
    """Generate a test agent ID."""
    return str(uuid4())


@pytest.fixture
def session_id() -> str:
    """Generate a test session ID."""
    return str(uuid4())


@pytest.fixture
def user_id() -> str:
    """Generate a test user ID."""
    return str(uuid4())


@pytest.fixture
def test_agent_data():
    """Sample agent data for testing."""
    return {
        "name": "Test Agent",
        "description": "A test voice agent",
        "voice_id": "test_voice",
        "voice_name": "Test Voice",
        "language": "en-US",
        "system_prompt": "You are a helpful test assistant.",
        "greeting_message": "Hello! How can I help you today?",
        "llm_provider": "mock",
        "llm_model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "tools": [],
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
