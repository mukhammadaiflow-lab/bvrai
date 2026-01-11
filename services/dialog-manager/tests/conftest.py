"""Pytest configuration and fixtures."""
import os
import pytest
from httpx import AsyncClient

# Set test environment before importing app
os.environ["ENVIRONMENT"] = "test"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["VECTOR_DB_PROVIDER"] = "local"
os.environ["VECTOR_DB_PATH"] = "./test_data/vectors.db"

from app.main import app
from app.adapters import MockLLMAdapter, LocalVectorAdapter
from app.services import SessionService, DialogService


@pytest.fixture
def mock_llm():
    """Create a mock LLM adapter."""
    return MockLLMAdapter()


@pytest.fixture
def local_vector_db(tmp_path):
    """Create a local vector DB adapter with temp path."""
    db_path = str(tmp_path / "test_vectors.db")
    return LocalVectorAdapter(db_path)


@pytest.fixture
def session_service():
    """Create a session service."""
    return SessionService()


@pytest.fixture
def dialog_service(mock_llm, local_vector_db, session_service):
    """Create a dialog service with mock adapters."""
    return DialogService(
        llm_adapter=mock_llm,
        vector_adapter=local_vector_db,
        session_service=session_service,
    )


@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
