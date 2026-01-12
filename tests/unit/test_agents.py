"""Unit tests for agent functionality."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.agent.persona import PersonaManager, AgentPersona
from app.agent.prompt import PromptManager, PromptTemplate
from app.agent.behavior import BehaviorController, InterruptionHandler
from app.agent.builder import AgentBuilder


class TestPersonaManager:
    """Tests for PersonaManager."""

    def test_create_persona(self):
        """Test creating a new persona."""
        persona = AgentPersona(
            name="Support Agent",
            role="Customer Support Representative",
            personality_traits=["friendly", "helpful", "patient"],
            communication_style="professional",
            background="Experienced customer service agent with 5 years experience.",
            goals=["resolve customer issues", "provide accurate information"],
        )

        assert persona.name == "Support Agent"
        assert persona.role == "Customer Support Representative"
        assert "friendly" in persona.personality_traits
        assert persona.communication_style == "professional"

    def test_persona_to_system_prompt(self):
        """Test converting persona to system prompt."""
        persona = AgentPersona(
            name="Sales Agent",
            role="Sales Representative",
            personality_traits=["confident", "persuasive"],
            communication_style="casual",
            background="Top sales performer",
            goals=["qualify leads", "book appointments"],
        )

        prompt = persona.to_system_prompt()

        assert "Sales Agent" in prompt
        assert "Sales Representative" in prompt
        assert "confident" in prompt
        assert "qualify leads" in prompt

    def test_persona_merge_traits(self):
        """Test merging personality traits."""
        persona = AgentPersona(
            name="Test Agent",
            role="Test",
            personality_traits=["friendly", "helpful"],
        )

        persona.add_traits(["patient", "helpful"])  # duplicate should not be added twice

        assert len(set(persona.personality_traits)) == 3
        assert "patient" in persona.personality_traits

    @pytest.mark.asyncio
    async def test_persona_manager_load(self):
        """Test loading personas from storage."""
        manager = PersonaManager()

        with patch.object(manager, '_load_from_storage', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = [
                {"name": "Agent 1", "role": "Support"},
                {"name": "Agent 2", "role": "Sales"},
            ]

            personas = await manager.list_personas()

            mock_load.assert_called_once()
            assert len(personas) == 2


class TestPromptManager:
    """Tests for PromptManager."""

    def test_create_template(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            name="greeting",
            template="Hello {customer_name}, how can I help you today?",
            variables=["customer_name"],
        )

        assert template.name == "greeting"
        assert "customer_name" in template.variables

    def test_render_template(self):
        """Test rendering a template with variables."""
        template = PromptTemplate(
            name="greeting",
            template="Hello {customer_name}, I'm {agent_name}. How can I help?",
            variables=["customer_name", "agent_name"],
        )

        rendered = template.render(customer_name="John", agent_name="Sarah")

        assert "Hello John" in rendered
        assert "Sarah" in rendered

    def test_render_missing_variable(self):
        """Test rendering with missing variable."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!",
            variables=["name"],
        )

        with pytest.raises(KeyError):
            template.render()

    def test_prompt_manager_register(self):
        """Test registering templates."""
        manager = PromptManager()

        template = PromptTemplate(
            name="test",
            template="Hello {name}",
            variables=["name"],
        )

        manager.register_template(template)

        assert "test" in manager.templates
        assert manager.get_template("test") == template

    def test_prompt_manager_unregister(self):
        """Test unregistering templates."""
        manager = PromptManager()
        template = PromptTemplate(name="test", template="test", variables=[])

        manager.register_template(template)
        manager.unregister_template("test")

        assert "test" not in manager.templates


class TestBehaviorController:
    """Tests for BehaviorController."""

    def test_create_controller(self):
        """Test creating a behavior controller."""
        controller = BehaviorController(
            agent_id=str(uuid4()),
            interruption_threshold=0.5,
            response_delay=0.2,
            max_silence_duration=5.0,
        )

        assert controller.interruption_threshold == 0.5
        assert controller.response_delay == 0.2
        assert controller.max_silence_duration == 5.0

    def test_should_interrupt(self):
        """Test interruption detection."""
        controller = BehaviorController(
            agent_id=str(uuid4()),
            interruption_threshold=0.5,
        )

        # Should interrupt - confidence above threshold
        assert controller.should_interrupt(confidence=0.7) is True

        # Should not interrupt - confidence below threshold
        assert controller.should_interrupt(confidence=0.3) is False

    def test_detect_silence(self):
        """Test silence detection."""
        controller = BehaviorController(
            agent_id=str(uuid4()),
            max_silence_duration=3.0,
        )

        # Simulate silence
        controller.start_silence_timer()

        # Not silent yet
        assert controller.is_prolonged_silence(elapsed=2.0) is False

        # Prolonged silence
        assert controller.is_prolonged_silence(elapsed=4.0) is True

    def test_behavior_modes(self):
        """Test different behavior modes."""
        controller = BehaviorController(agent_id=str(uuid4()))

        controller.set_mode("listening")
        assert controller.current_mode == "listening"

        controller.set_mode("speaking")
        assert controller.current_mode == "speaking"

        controller.set_mode("thinking")
        assert controller.current_mode == "thinking"


class TestInterruptionHandler:
    """Tests for InterruptionHandler."""

    def test_create_handler(self):
        """Test creating interruption handler."""
        handler = InterruptionHandler(
            sensitivity=0.6,
            cooldown_period=1.0,
        )

        assert handler.sensitivity == 0.6
        assert handler.cooldown_period == 1.0

    def test_handle_interruption(self):
        """Test handling an interruption."""
        handler = InterruptionHandler(sensitivity=0.5)

        result = handler.handle_interruption(
            audio_level=0.8,
            is_speech=True,
            current_state="speaking",
        )

        assert result["should_stop_speaking"] is True
        assert result["reason"] == "user_interruption"

    def test_cooldown_period(self):
        """Test cooldown period after interruption."""
        handler = InterruptionHandler(cooldown_period=2.0)

        # First interruption
        handler.record_interruption()

        # During cooldown
        assert handler.in_cooldown(elapsed=1.0) is True

        # After cooldown
        assert handler.in_cooldown(elapsed=3.0) is False


class TestAgentBuilder:
    """Tests for AgentBuilder."""

    def test_build_basic_agent(self):
        """Test building a basic agent."""
        builder = AgentBuilder()

        agent = (
            builder
            .set_name("Test Agent")
            .set_role("Support")
            .set_voice("alloy")
            .set_llm("gpt-4")
            .build()
        )

        assert agent.name == "Test Agent"
        assert agent.role == "Support"
        assert agent.voice_id == "alloy"
        assert agent.llm_model == "gpt-4"

    def test_build_with_persona(self):
        """Test building agent with persona."""
        builder = AgentBuilder()

        persona = AgentPersona(
            name="Sales Bot",
            role="Sales",
            personality_traits=["confident"],
        )

        agent = (
            builder
            .set_persona(persona)
            .set_voice("nova")
            .build()
        )

        assert agent.name == "Sales Bot"
        assert agent.persona == persona

    def test_build_with_tools(self):
        """Test building agent with tools."""
        builder = AgentBuilder()

        tools = [
            {"name": "calendar", "type": "function"},
            {"name": "crm", "type": "function"},
        ]

        agent = (
            builder
            .set_name("Tool Agent")
            .add_tools(tools)
            .build()
        )

        assert len(agent.tools) == 2
        assert any(t["name"] == "calendar" for t in agent.tools)

    def test_build_with_knowledge_base(self):
        """Test building agent with knowledge base."""
        builder = AgentBuilder()

        agent = (
            builder
            .set_name("KB Agent")
            .set_knowledge_base("kb_123")
            .build()
        )

        assert agent.knowledge_base_id == "kb_123"

    def test_builder_validation(self):
        """Test builder validates required fields."""
        builder = AgentBuilder()

        with pytest.raises(ValueError):
            # Missing required name
            builder.build()

    def test_builder_chaining(self):
        """Test builder method chaining."""
        builder = AgentBuilder()

        # All methods should return the builder for chaining
        result = (
            builder
            .set_name("Test")
            .set_role("Test Role")
            .set_voice("alloy")
            .set_llm("gpt-4")
            .set_temperature(0.7)
            .set_max_tokens(1000)
        )

        assert result is builder


class TestAgentConfig:
    """Tests for agent configuration."""

    def test_default_config(self):
        """Test default agent configuration."""
        from app.agent.config import AgentConfig

        config = AgentConfig()

        assert config.language == "en-US"
        assert 0 <= config.temperature <= 1
        assert config.max_tokens > 0

    def test_config_validation(self):
        """Test configuration validation."""
        from app.agent.config import AgentConfig

        # Invalid temperature
        with pytest.raises(ValueError):
            AgentConfig(temperature=2.0)

        # Invalid max_tokens
        with pytest.raises(ValueError):
            AgentConfig(max_tokens=-100)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from app.agent.config import AgentConfig

        config = AgentConfig(
            language="en-GB",
            temperature=0.5,
            max_tokens=500,
        )

        data = config.to_dict()

        assert data["language"] == "en-GB"
        assert data["temperature"] == 0.5
        assert data["max_tokens"] == 500

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from app.agent.config import AgentConfig

        data = {
            "language": "es-ES",
            "temperature": 0.8,
            "max_tokens": 2000,
        }

        config = AgentConfig.from_dict(data)

        assert config.language == "es-ES"
        assert config.temperature == 0.8
        assert config.max_tokens == 2000
