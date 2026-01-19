"""Integration tests for agents API."""

import pytest
from datetime import datetime
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient


class TestAgentsAPI:
    """Tests for agent management endpoints."""

    @pytest.mark.asyncio
    async def test_create_agent(self, authenticated_client: AsyncClient):
        """Test creating an agent."""
        response = await authenticated_client.post(
            "/api/v1/agents",
            json={
                "name": "Customer Support Bot",
                "description": "Handles customer support inquiries",
                "voice_id": "alloy",
                "voice_provider": "openai",
                "language": "en-US",
                "system_prompt": "You are a helpful customer support assistant.",
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Customer Support Bot"
        assert data["status"] == "inactive"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_create_agent_validation(self, authenticated_client: AsyncClient):
        """Test agent creation validation."""
        # Missing required field
        response = await authenticated_client.post(
            "/api/v1/agents",
            json={
                "description": "Missing name",
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_agents(self, authenticated_client: AsyncClient, test_agents):
        """Test listing agents."""
        response = await authenticated_client.get("/api/v1/agents")

        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "total" in data
        assert len(data["agents"]) == len(test_agents)

    @pytest.mark.asyncio
    async def test_list_agents_with_pagination(
        self, authenticated_client: AsyncClient, test_agents
    ):
        """Test listing agents with pagination."""
        response = await authenticated_client.get(
            "/api/v1/agents",
            params={"page": 1, "page_size": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 2
        assert data["total"] == len(test_agents)
        assert data["page"] == 1
        assert data["page_size"] == 2

    @pytest.mark.asyncio
    async def test_list_agents_filter_by_status(
        self, authenticated_client: AsyncClient, test_agents
    ):
        """Test filtering agents by status."""
        response = await authenticated_client.get(
            "/api/v1/agents",
            params={"status": "active"},
        )

        assert response.status_code == 200
        data = response.json()
        for agent in data["agents"]:
            assert agent["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_agent(self, authenticated_client: AsyncClient, test_agent):
        """Test getting a single agent."""
        response = await authenticated_client.get(f"/api/v1/agents/{test_agent.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_agent.id
        assert data["name"] == test_agent.name

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, authenticated_client: AsyncClient):
        """Test getting non-existent agent."""
        response = await authenticated_client.get("/api/v1/agents/nonexistent")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_agent(self, authenticated_client: AsyncClient, test_agent):
        """Test updating an agent."""
        response = await authenticated_client.patch(
            f"/api/v1/agents/{test_agent.id}",
            json={
                "name": "Updated Agent Name",
                "description": "Updated description",
                "settings": {
                    "temperature": 0.5,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Agent Name"
        assert data["description"] == "Updated description"

    @pytest.mark.asyncio
    async def test_delete_agent(self, authenticated_client: AsyncClient, test_agent):
        """Test deleting an agent."""
        response = await authenticated_client.delete(f"/api/v1/agents/{test_agent.id}")

        assert response.status_code == 204

        # Verify deletion
        get_response = await authenticated_client.get(f"/api/v1/agents/{test_agent.id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_activate_agent(self, authenticated_client: AsyncClient, test_agent):
        """Test activating an agent."""
        # First deactivate
        await authenticated_client.post(f"/api/v1/agents/{test_agent.id}/pause")

        response = await authenticated_client.post(f"/api/v1/agents/{test_agent.id}/activate")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_pause_agent(self, authenticated_client: AsyncClient, test_agent):
        """Test pausing an agent."""
        response = await authenticated_client.post(f"/api/v1/agents/{test_agent.id}/pause")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"

    @pytest.mark.asyncio
    async def test_get_agent_stats(self, authenticated_client: AsyncClient, test_agent):
        """Test getting agent statistics."""
        response = await authenticated_client.get(f"/api/v1/agents/{test_agent.id}/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_calls" in data
        assert "total_minutes" in data
        assert "success_rate" in data

    @pytest.mark.asyncio
    async def test_duplicate_agent(self, authenticated_client: AsyncClient, test_agent):
        """Test duplicating an agent."""
        response = await authenticated_client.post(
            f"/api/v1/agents/{test_agent.id}/duplicate",
            json={"name": "Duplicated Agent"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Duplicated Agent"
        assert data["id"] != test_agent.id
        # Should have same config
        assert data["voice_id"] == test_agent.voice_id

    @pytest.mark.asyncio
    async def test_test_agent(self, authenticated_client: AsyncClient, test_agent):
        """Test agent testing endpoint."""
        with patch("app.services.agent.run_test", new_callable=AsyncMock) as mock_test:
            mock_test.return_value = {
                "success": True,
                "response": "Hello, how can I help you?",
                "latency_ms": 250,
            }

            response = await authenticated_client.post(
                f"/api/v1/agents/{test_agent.id}/test",
                json={"message": "Hello"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "response" in data


class TestAgentVersioning:
    """Tests for agent version management."""

    @pytest.mark.asyncio
    async def test_create_version(self, authenticated_client: AsyncClient, test_agent):
        """Test creating an agent version."""
        response = await authenticated_client.post(
            f"/api/v1/agents/{test_agent.id}/versions",
            json={
                "description": "Initial production version",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "version" in data
        assert data["version"] == 1

    @pytest.mark.asyncio
    async def test_list_versions(self, authenticated_client: AsyncClient, test_agent):
        """Test listing agent versions."""
        # Create a version first
        await authenticated_client.post(
            f"/api/v1/agents/{test_agent.id}/versions",
            json={"description": "Version 1"},
        )

        response = await authenticated_client.get(f"/api/v1/agents/{test_agent.id}/versions")

        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
        assert len(data["versions"]) >= 1

    @pytest.mark.asyncio
    async def test_rollback_version(self, authenticated_client: AsyncClient, test_agent):
        """Test rolling back to a previous version."""
        # Create versions
        await authenticated_client.post(
            f"/api/v1/agents/{test_agent.id}/versions",
            json={"description": "Version 1"},
        )

        # Update agent
        await authenticated_client.patch(
            f"/api/v1/agents/{test_agent.id}",
            json={"name": "Updated Name"},
        )

        await authenticated_client.post(
            f"/api/v1/agents/{test_agent.id}/versions",
            json={"description": "Version 2"},
        )

        # Rollback to version 1
        response = await authenticated_client.post(
            f"/api/v1/agents/{test_agent.id}/versions/1/rollback"
        )

        assert response.status_code == 200


class TestAgentTools:
    """Tests for agent tool configuration."""

    @pytest.mark.asyncio
    async def test_add_tool(self, authenticated_client: AsyncClient, test_agent):
        """Test adding a tool to an agent."""
        response = await authenticated_client.post(
            f"/api/v1/agents/{test_agent.id}/tools",
            json={
                "name": "calendar_book",
                "type": "function",
                "description": "Book a calendar appointment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "format": "date"},
                        "time": {"type": "string"},
                        "duration": {"type": "integer"},
                    },
                    "required": ["date", "time"],
                },
            },
        )

        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_list_tools(self, authenticated_client: AsyncClient, test_agent):
        """Test listing agent tools."""
        response = await authenticated_client.get(f"/api/v1/agents/{test_agent.id}/tools")

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data

    @pytest.mark.asyncio
    async def test_remove_tool(self, authenticated_client: AsyncClient, test_agent):
        """Test removing a tool from an agent."""
        # Add a tool first
        await authenticated_client.post(
            f"/api/v1/agents/{test_agent.id}/tools",
            json={
                "name": "test_tool",
                "type": "function",
                "description": "Test",
            },
        )

        response = await authenticated_client.delete(
            f"/api/v1/agents/{test_agent.id}/tools/test_tool"
        )

        assert response.status_code == 204
