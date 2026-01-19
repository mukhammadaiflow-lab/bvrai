"""
Builder Engine Python SDK - Agents Resource

This module provides methods for managing AI voice agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List, Union

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import (
    Agent,
    AgentConfig,
    AgentStatus,
    VoiceConfig,
    STTConfig,
    LLMConfig,
    FunctionDefinition,
    Call,
    Analytics,
)
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class AgentsResource(BaseResource):
    """
    Resource for managing AI voice agents.

    Agents are the core building blocks of Builder Engine. Each agent
    represents a configured AI assistant that can handle voice calls.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> # Create an agent
        >>> agent = client.agents.create(
        ...     name="Sales Agent",
        ...     voice_id="voice_abc123",
        ...     llm_config={"model": "gpt-4-turbo", "temperature": 0.7}
        ... )
        >>> # List all agents
        >>> agents = client.agents.list()
        >>> for agent in agents:
        ...     print(agent.name)
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[AgentStatus] = None,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
    ) -> PaginatedResponse[Agent]:
        """
        List all agents.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page (max 100)
            status: Filter by agent status
            search: Search by name or description
            sort_by: Field to sort by (created_at, name, total_calls)
            sort_order: Sort order (asc, desc)

        Returns:
            PaginatedResponse containing Agent objects

        Example:
            >>> agents = client.agents.list(status=AgentStatus.ACTIVE)
            >>> print(f"Found {agents.total} active agents")
        """
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            status=status.value if status else None,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        response = self._get(Endpoints.AGENTS, params=params)
        return self._parse_paginated_response(response, Agent)

    def get(self, agent_id: str) -> Agent:
        """
        Get an agent by ID.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            Agent object

        Raises:
            NotFoundError: If the agent doesn't exist

        Example:
            >>> agent = client.agents.get("agent_abc123")
            >>> print(agent.name)
        """
        path = Endpoints.AGENT.format(agent_id=agent_id)
        response = self._get(path)
        return Agent.from_dict(response)

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        voice_id: Optional[str] = None,
        phone_number_id: Optional[str] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        workflow_ids: Optional[List[str]] = None,
        config: Optional[Union[AgentConfig, Dict[str, Any]]] = None,
        voice_config: Optional[Union[VoiceConfig, Dict[str, Any]]] = None,
        stt_config: Optional[Union[STTConfig, Dict[str, Any]]] = None,
        llm_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
        first_message: Optional[str] = None,
        system_prompt: Optional[str] = None,
        functions: Optional[List[Union[FunctionDefinition, Dict[str, Any]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """
        Create a new agent.

        Args:
            name: Name of the agent
            description: Description of the agent's purpose
            voice_id: ID of the voice to use
            phone_number_id: ID of the phone number to assign
            knowledge_base_ids: List of knowledge base IDs to attach
            workflow_ids: List of workflow IDs to attach
            config: Complete agent configuration
            voice_config: Voice synthesis configuration
            stt_config: Speech-to-text configuration
            llm_config: LLM configuration
            first_message: Initial greeting message
            system_prompt: System prompt for the LLM
            functions: List of function definitions
            metadata: Custom metadata

        Returns:
            Created Agent object

        Example:
            >>> agent = client.agents.create(
            ...     name="Customer Support Agent",
            ...     system_prompt="You are a helpful customer support agent...",
            ...     voice_id="voice_abc123",
            ...     first_message="Hello! How can I help you today?",
            ...     llm_config={
            ...         "model": "gpt-4-turbo",
            ...         "temperature": 0.7,
            ...         "max_tokens": 500
            ...     }
            ... )
        """
        data: Dict[str, Any] = {"name": name}

        if description is not None:
            data["description"] = description
        if voice_id is not None:
            data["voice_id"] = voice_id
        if phone_number_id is not None:
            data["phone_number_id"] = phone_number_id
        if knowledge_base_ids is not None:
            data["knowledge_base_ids"] = knowledge_base_ids
        if workflow_ids is not None:
            data["workflow_ids"] = workflow_ids
        if metadata is not None:
            data["metadata"] = metadata

        # Build config
        agent_config: Dict[str, Any] = {}

        if config is not None:
            if isinstance(config, AgentConfig):
                agent_config = config.to_dict()
            else:
                agent_config = config

        if voice_config is not None:
            if isinstance(voice_config, VoiceConfig):
                agent_config["voice"] = voice_config.to_dict()
            else:
                agent_config["voice"] = voice_config

        if stt_config is not None:
            if isinstance(stt_config, STTConfig):
                agent_config["stt"] = stt_config.to_dict()
            else:
                agent_config["stt"] = stt_config

        if llm_config is not None:
            if isinstance(llm_config, LLMConfig):
                agent_config["llm"] = llm_config.to_dict()
            else:
                agent_config["llm"] = llm_config

        if first_message is not None:
            agent_config["first_message"] = first_message

        if system_prompt is not None:
            if "llm" not in agent_config:
                agent_config["llm"] = {}
            agent_config["llm"]["system_prompt"] = system_prompt

        if functions is not None:
            agent_config["functions"] = [
                f.to_dict() if isinstance(f, FunctionDefinition) else f
                for f in functions
            ]

        if agent_config:
            data["config"] = agent_config

        response = self._post(Endpoints.AGENTS, json=data)
        return Agent.from_dict(response)

    def update(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[AgentStatus] = None,
        voice_id: Optional[str] = None,
        phone_number_id: Optional[str] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        workflow_ids: Optional[List[str]] = None,
        config: Optional[Union[AgentConfig, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """
        Update an existing agent.

        Args:
            agent_id: The agent's unique identifier
            name: New name for the agent
            description: New description
            status: New status
            voice_id: New voice ID
            phone_number_id: New phone number ID
            knowledge_base_ids: New list of knowledge base IDs
            workflow_ids: New list of workflow IDs
            config: New agent configuration
            metadata: New metadata (merged with existing)

        Returns:
            Updated Agent object

        Example:
            >>> agent = client.agents.update(
            ...     agent_id="agent_abc123",
            ...     name="Updated Agent Name",
            ...     status=AgentStatus.INACTIVE
            ... )
        """
        data: Dict[str, Any] = {}

        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if status is not None:
            data["status"] = status.value
        if voice_id is not None:
            data["voice_id"] = voice_id
        if phone_number_id is not None:
            data["phone_number_id"] = phone_number_id
        if knowledge_base_ids is not None:
            data["knowledge_base_ids"] = knowledge_base_ids
        if workflow_ids is not None:
            data["workflow_ids"] = workflow_ids
        if config is not None:
            if isinstance(config, AgentConfig):
                data["config"] = config.to_dict()
            else:
                data["config"] = config
        if metadata is not None:
            data["metadata"] = metadata

        path = Endpoints.AGENT.format(agent_id=agent_id)
        response = self._patch(path, json=data)
        return Agent.from_dict(response)

    def delete(self, agent_id: str) -> None:
        """
        Delete an agent.

        Args:
            agent_id: The agent's unique identifier

        Raises:
            NotFoundError: If the agent doesn't exist

        Example:
            >>> client.agents.delete("agent_abc123")
        """
        path = Endpoints.AGENT.format(agent_id=agent_id)
        self._delete(path)

    def duplicate(
        self,
        agent_id: str,
        name: Optional[str] = None,
    ) -> Agent:
        """
        Duplicate an existing agent.

        Creates a copy of the agent with all its configuration.

        Args:
            agent_id: The agent's unique identifier
            name: Name for the duplicated agent (defaults to "Copy of {original}")

        Returns:
            New Agent object

        Example:
            >>> new_agent = client.agents.duplicate(
            ...     agent_id="agent_abc123",
            ...     name="Sales Agent v2"
            ... )
        """
        path = Endpoints.AGENT_DUPLICATE.format(agent_id=agent_id)
        data = {}
        if name is not None:
            data["name"] = name
        response = self._post(path, json=data)
        return Agent.from_dict(response)

    def get_calls(
        self,
        agent_id: str,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> PaginatedResponse[Call]:
        """
        Get calls for an agent.

        Args:
            agent_id: The agent's unique identifier
            page: Page number
            page_size: Items per page
            status: Filter by call status
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)

        Returns:
            PaginatedResponse containing Call objects

        Example:
            >>> calls = client.agents.get_calls(
            ...     agent_id="agent_abc123",
            ...     status="completed",
            ...     start_date="2024-01-01"
            ... )
        """
        path = Endpoints.AGENT_CALLS.format(agent_id=agent_id)
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )
        response = self._get(path, params=params)
        return self._parse_paginated_response(response, Call)

    def get_analytics(
        self,
        agent_id: str,
        period: str = "week",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Analytics:
        """
        Get analytics for an agent.

        Args:
            agent_id: The agent's unique identifier
            period: Time period (day, week, month, custom)
            start_date: Start date for custom period
            end_date: End date for custom period

        Returns:
            Analytics object

        Example:
            >>> analytics = client.agents.get_analytics(
            ...     agent_id="agent_abc123",
            ...     period="month"
            ... )
            >>> print(f"Total calls: {analytics.call_metrics.total_calls}")
        """
        path = Endpoints.AGENT_ANALYTICS.format(agent_id=agent_id)
        params = {
            "period": period,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        response = self._get(path, params=params)
        return Analytics.from_dict(response)

    def test(
        self,
        agent_id: str,
        phone_number: Optional[str] = None,
        simulate_user_messages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Test an agent with a simulated call.

        Args:
            agent_id: The agent's unique identifier
            phone_number: Phone number for test call (optional)
            simulate_user_messages: List of simulated user messages

        Returns:
            Test results including conversation transcript

        Example:
            >>> result = client.agents.test(
            ...     agent_id="agent_abc123",
            ...     simulate_user_messages=[
            ...         "Hi, I need help with my order",
            ...         "Order number is 12345"
            ...     ]
            ... )
            >>> print(result["transcript"])
        """
        path = Endpoints.AGENT_TEST.format(agent_id=agent_id)
        data: Dict[str, Any] = {}
        if phone_number:
            data["phone_number"] = phone_number
        if simulate_user_messages:
            data["simulate_user_messages"] = simulate_user_messages

        return self._post(path, json=data)

    def update_functions(
        self,
        agent_id: str,
        functions: List[Union[FunctionDefinition, Dict[str, Any]]],
    ) -> Agent:
        """
        Update the functions/tools for an agent.

        Args:
            agent_id: The agent's unique identifier
            functions: List of function definitions

        Returns:
            Updated Agent object

        Example:
            >>> agent = client.agents.update_functions(
            ...     agent_id="agent_abc123",
            ...     functions=[
            ...         {
            ...             "name": "get_order_status",
            ...             "description": "Get the status of an order",
            ...             "parameters": {
            ...                 "type": "object",
            ...                 "properties": {
            ...                     "order_id": {"type": "string"}
            ...                 }
            ...             },
            ...             "webhook_url": "https://api.example.com/orders"
            ...         }
            ...     ]
            ... )
        """
        config = {
            "functions": [
                f.to_dict() if isinstance(f, FunctionDefinition) else f
                for f in functions
            ]
        }
        return self.update(agent_id, config=config)

    def archive(self, agent_id: str) -> Agent:
        """
        Archive an agent.

        Archived agents are not deleted but cannot receive calls.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            Updated Agent object

        Example:
            >>> agent = client.agents.archive("agent_abc123")
        """
        return self.update(agent_id, status=AgentStatus.ARCHIVED)

    def activate(self, agent_id: str) -> Agent:
        """
        Activate an agent.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            Updated Agent object

        Example:
            >>> agent = client.agents.activate("agent_abc123")
        """
        return self.update(agent_id, status=AgentStatus.ACTIVE)

    def deactivate(self, agent_id: str) -> Agent:
        """
        Deactivate an agent.

        Deactivated agents cannot receive new calls.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            Updated Agent object

        Example:
            >>> agent = client.agents.deactivate("agent_abc123")
        """
        return self.update(agent_id, status=AgentStatus.INACTIVE)
