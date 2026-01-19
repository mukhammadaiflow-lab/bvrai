"""Prompt manager for runtime prompt handling."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import json

from app.prompts.template import PromptTemplate
from app.prompts.builder import SystemPromptBuilder, AgentPersona, ConversationRules
from app.prompts.library import PromptLibrary


logger = structlog.get_logger()


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    agent_id: str
    name: str
    system_prompt: Optional[str] = None
    template_name: Optional[str] = None
    template_variables: Dict[str, Any] = None
    persona: Optional[AgentPersona] = None
    rules: Optional[ConversationRules] = None
    tools: List[str] = None
    knowledge: List[str] = None
    first_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.template_variables = self.template_variables or {}
        self.tools = self.tools or []
        self.knowledge = self.knowledge or []
        self.metadata = self.metadata or {}


class PromptManager:
    """
    Manages prompts for voice AI agents.

    Handles:
    - Agent configuration loading
    - Dynamic prompt generation
    - Context injection
    - Template management
    - Caching
    """

    def __init__(
        self,
        library: Optional[PromptLibrary] = None,
        redis_client=None,
        cache_ttl_seconds: int = 300,
    ):
        self._library = library or PromptLibrary()
        self._redis = redis_client
        self._cache_ttl = cache_ttl_seconds

        # In-memory caches
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._rendered_prompts: Dict[str, str] = {}

    async def register_agent(self, config: AgentConfig) -> None:
        """
        Register an agent configuration.

        Args:
            config: Agent configuration
        """
        self._agent_configs[config.agent_id] = config

        # Persist to Redis if available
        if self._redis:
            await self._redis.setex(
                f"agent_config:{config.agent_id}",
                self._cache_ttl,
                json.dumps(self._serialize_config(config)),
            )

        logger.info(
            "agent_registered",
            agent_id=config.agent_id,
            name=config.name,
        )

    async def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent configuration."""
        # Check memory cache
        if agent_id in self._agent_configs:
            return self._agent_configs[agent_id]

        # Check Redis
        if self._redis:
            data = await self._redis.get(f"agent_config:{agent_id}")
            if data:
                config = self._deserialize_config(json.loads(data))
                self._agent_configs[agent_id] = config
                return config

        return None

    async def build_system_prompt(
        self,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None,
        force_rebuild: bool = False,
    ) -> str:
        """
        Build system prompt for an agent.

        Args:
            agent_id: Agent ID
            context: Runtime context to inject
            force_rebuild: Bypass cache

        Returns:
            Complete system prompt
        """
        # Check cache
        cache_key = f"{agent_id}:{hash(str(context or {}))}"
        if not force_rebuild and cache_key in self._rendered_prompts:
            return self._rendered_prompts[cache_key]

        config = await self.get_agent_config(agent_id)
        if not config:
            raise ValueError(f"Agent not found: {agent_id}")

        # Build prompt
        if config.system_prompt:
            # Direct system prompt
            prompt = config.system_prompt
        elif config.template_name:
            # Template-based
            template = self._library.get(config.template_name)
            if not template:
                raise ValueError(f"Template not found: {config.template_name}")

            variables = {**config.template_variables}
            if context:
                variables.update(context)

            prompt = template.render(variables)
        else:
            # Build using SystemPromptBuilder
            prompt = self._build_from_config(config, context)

        # Cache result
        self._rendered_prompts[cache_key] = prompt

        return prompt

    def _build_from_config(
        self,
        config: AgentConfig,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build prompt from agent config."""
        builder = SystemPromptBuilder()

        # Set persona
        if config.persona:
            builder.with_persona(config.persona)
        else:
            builder.with_simple_persona(config.name, "AI assistant")

        # Set rules
        if config.rules:
            builder.with_rules(config.rules)
        else:
            builder.with_rules(ConversationRules())

        # Add knowledge
        for knowledge in config.knowledge:
            builder.add_knowledge(knowledge)

        # Add tools
        for tool in config.tools:
            builder.add_tool(tool, f"Execute {tool} function")

        # Add context
        if context:
            for key, value in context.items():
                builder.with_context(key, value)

        return builder.build()

    async def get_first_message(
        self,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Get the first message for an agent.

        Args:
            agent_id: Agent ID
            context: Runtime context

        Returns:
            First message or None
        """
        config = await self.get_agent_config(agent_id)
        if not config or not config.first_message:
            return None

        message = config.first_message

        # Substitute context variables
        if context:
            for key, value in context.items():
                message = message.replace(f"{{{key}}}", str(value))

        return message

    async def update_agent(
        self,
        agent_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """
        Update agent configuration.

        Args:
            agent_id: Agent ID
            updates: Fields to update

        Returns:
            True if updated
        """
        config = await self.get_agent_config(agent_id)
        if not config:
            return False

        # Apply updates
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Clear rendered cache
        keys_to_remove = [
            k for k in self._rendered_prompts
            if k.startswith(f"{agent_id}:")
        ]
        for key in keys_to_remove:
            del self._rendered_prompts[key]

        # Re-register
        await self.register_agent(config)

        return True

    def add_template(self, template: PromptTemplate) -> None:
        """Add a template to the library."""
        self._library.add(template)

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template from the library."""
        return self._library.get(name)

    def list_templates(self) -> List[str]:
        """List available templates."""
        return self._library.list_all()

    def _serialize_config(self, config: AgentConfig) -> Dict[str, Any]:
        """Serialize agent config for storage."""
        data = {
            "agent_id": config.agent_id,
            "name": config.name,
            "system_prompt": config.system_prompt,
            "template_name": config.template_name,
            "template_variables": config.template_variables,
            "tools": config.tools,
            "knowledge": config.knowledge,
            "first_message": config.first_message,
            "metadata": config.metadata,
        }

        if config.persona:
            data["persona"] = {
                "name": config.persona.name,
                "role": config.persona.role,
                "company": config.persona.company,
                "personality_traits": config.persona.personality_traits,
                "speaking_style": config.persona.speaking_style,
            }

        if config.rules:
            data["rules"] = {
                "allow_interruptions": config.rules.allow_interruptions,
                "confirm_important_info": config.rules.confirm_important_info,
                "max_response_length": config.rules.max_response_length,
            }

        return data

    def _deserialize_config(self, data: Dict[str, Any]) -> AgentConfig:
        """Deserialize agent config from storage."""
        config = AgentConfig(
            agent_id=data["agent_id"],
            name=data["name"],
            system_prompt=data.get("system_prompt"),
            template_name=data.get("template_name"),
            template_variables=data.get("template_variables", {}),
            tools=data.get("tools", []),
            knowledge=data.get("knowledge", []),
            first_message=data.get("first_message"),
            metadata=data.get("metadata", {}),
        )

        if data.get("persona"):
            config.persona = AgentPersona(**data["persona"])

        if data.get("rules"):
            config.rules = ConversationRules(**data["rules"])

        return config

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "registered_agents": len(self._agent_configs),
            "cached_prompts": len(self._rendered_prompts),
            "templates": len(self._library.list_all()),
            "categories": self._library.get_categories(),
        }
