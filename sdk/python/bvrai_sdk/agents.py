"""Agents API for Builder Engine."""

from typing import Optional, Dict, List, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from bvrai_sdk.client import BvraiClient


class VoiceProvider(str, Enum):
    """Voice provider options."""
    ELEVENLABS = "elevenlabs"
    AZURE = "azure"
    GOOGLE = "google"
    AWS = "aws"


class LLMProvider(str, Enum):
    """LLM provider options."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str
    system_prompt: str
    first_message: Optional[str] = None
    voice_id: Optional[str] = None
    voice_provider: VoiceProvider = VoiceProvider.ELEVENLABS
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    tools: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """An AI voice agent."""
    id: str
    name: str
    system_prompt: str
    first_message: Optional[str] = None
    voice_id: Optional[str] = None
    voice_provider: str = "elevenlabs"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    tools: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """Create Agent from API response."""
        return cls(
            id=data["id"],
            name=data["name"],
            system_prompt=data.get("system_prompt", ""),
            first_message=data.get("first_message"),
            voice_id=data.get("voice_id"),
            voice_provider=data.get("voice_provider", "elevenlabs"),
            llm_provider=data.get("llm_provider", "openai"),
            llm_model=data.get("llm_model", "gpt-4o-mini"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 1000),
            tools=data.get("tools", []),
            is_active=data.get("is_active", True),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "first_message": self.first_message,
            "voice_id": self.voice_id,
            "voice_provider": self.voice_provider,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": self.tools,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }


class AgentsAPI:
    """
    Agents API client.

    Create and manage voice AI agents.
    """

    def __init__(self, client: "BvraiClient"):
        self._client = client

    async def create(
        self,
        name: str,
        system_prompt: str,
        first_message: Optional[str] = None,
        voice_id: Optional[str] = None,
        voice_provider: str = "elevenlabs",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """
        Create a new agent.

        Args:
            name: Agent name
            system_prompt: System prompt for the agent
            first_message: Initial greeting message
            voice_id: Voice ID for TTS
            voice_provider: TTS provider
            llm_provider: LLM provider
            llm_model: LLM model name
            temperature: LLM temperature
            max_tokens: Max response tokens
            tools: List of tools/functions
            metadata: Custom metadata

        Returns:
            Created Agent

        Example:
            agent = await client.agents.create(
                name="Sales Assistant",
                system_prompt="You are a friendly sales assistant...",
                first_message="Hello! How can I help you today?",
                voice_id="21m00Tcm4TlvDq8ikWAM",
            )
        """
        data = {
            "name": name,
            "system_prompt": system_prompt,
            "first_message": first_message,
            "voice_id": voice_id,
            "voice_provider": voice_provider,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools or [],
            "metadata": metadata or {},
        }

        response = await self._client.post("/v1/agents", data=data)
        return Agent.from_dict(response)

    async def create_from_config(self, config: AgentConfig) -> Agent:
        """Create agent from AgentConfig."""
        return await self.create(
            name=config.name,
            system_prompt=config.system_prompt,
            first_message=config.first_message,
            voice_id=config.voice_id,
            voice_provider=config.voice_provider.value,
            llm_provider=config.llm_provider.value,
            llm_model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            tools=config.tools,
            metadata=config.metadata,
        )

    async def get(self, agent_id: str) -> Agent:
        """
        Get an agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent

        Raises:
            NotFoundError: If agent not found
        """
        response = await self._client.get(f"/v1/agents/{agent_id}")
        return Agent.from_dict(response)

    async def list(
        self,
        limit: int = 20,
        offset: int = 0,
        is_active: Optional[bool] = None,
    ) -> List[Agent]:
        """
        List all agents.

        Args:
            limit: Maximum agents to return
            offset: Pagination offset
            is_active: Filter by active status

        Returns:
            List of Agents
        """
        params = {"limit": limit, "offset": offset}
        if is_active is not None:
            params["is_active"] = is_active

        response = await self._client.get("/v1/agents", params=params)
        return [Agent.from_dict(a) for a in response.get("agents", [])]

    async def update(
        self,
        agent_id: str,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        first_message: Optional[str] = None,
        voice_id: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        is_active: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """
        Update an agent.

        Args:
            agent_id: Agent ID to update
            **kwargs: Fields to update

        Returns:
            Updated Agent
        """
        data = {}
        if name is not None:
            data["name"] = name
        if system_prompt is not None:
            data["system_prompt"] = system_prompt
        if first_message is not None:
            data["first_message"] = first_message
        if voice_id is not None:
            data["voice_id"] = voice_id
        if llm_model is not None:
            data["llm_model"] = llm_model
        if temperature is not None:
            data["temperature"] = temperature
        if is_active is not None:
            data["is_active"] = is_active
        if tools is not None:
            data["tools"] = tools
        if metadata is not None:
            data["metadata"] = metadata

        response = await self._client.patch(f"/v1/agents/{agent_id}", data=data)
        return Agent.from_dict(response)

    async def delete(self, agent_id: str) -> bool:
        """
        Delete an agent.

        Args:
            agent_id: Agent ID to delete

        Returns:
            True if deleted

        Raises:
            NotFoundError: If agent not found
        """
        await self._client.delete(f"/v1/agents/{agent_id}")
        return True

    async def add_tool(
        self,
        agent_id: str,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ) -> Agent:
        """
        Add a tool to an agent.

        Args:
            agent_id: Agent ID
            name: Tool name
            description: Tool description
            parameters: JSON Schema for parameters

        Returns:
            Updated Agent
        """
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

        agent = await self.get(agent_id)
        tools = agent.tools + [tool]

        return await self.update(agent_id, tools=tools)

    async def remove_tool(self, agent_id: str, tool_name: str) -> Agent:
        """
        Remove a tool from an agent.

        Args:
            agent_id: Agent ID
            tool_name: Tool name to remove

        Returns:
            Updated Agent
        """
        agent = await self.get(agent_id)
        tools = [
            t for t in agent.tools
            if t.get("function", {}).get("name") != tool_name
        ]

        return await self.update(agent_id, tools=tools)

    async def clone(
        self,
        agent_id: str,
        new_name: Optional[str] = None,
    ) -> Agent:
        """
        Clone an existing agent.

        Args:
            agent_id: Agent to clone
            new_name: Name for the clone

        Returns:
            New Agent (clone)
        """
        original = await self.get(agent_id)

        return await self.create(
            name=new_name or f"{original.name} (Copy)",
            system_prompt=original.system_prompt,
            first_message=original.first_message,
            voice_id=original.voice_id,
            voice_provider=original.voice_provider,
            llm_provider=original.llm_provider,
            llm_model=original.llm_model,
            temperature=original.temperature,
            max_tokens=original.max_tokens,
            tools=original.tools,
            metadata=original.metadata,
        )
