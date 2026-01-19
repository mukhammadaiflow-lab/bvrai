"""
Agent Management Module

Comprehensive agent system:
- Agent configuration and settings
- Persona management
- Behavior engine
- Prompt building
- Agent builder and factory
"""

from app.agents.service import AgentService
from app.agents.schemas import (
    AgentCreate,
    AgentUpdate,
    AgentResponse,
    AgentListResponse,
)

# Configuration
from app.agents.config import (
    AgentConfig,
    AgentType,
    AgentStatus,
    AgentCapability,
    VoiceSettings,
    TranscriptionSettings,
    LLMSettings,
    InterruptionConfig,
    SilenceConfig,
    ErrorConfig,
    AgentValidator,
)

# Persona
from app.agents.persona import (
    Persona,
    PersonaConfig,
    PersonaTrait,
    SpeakingStyle,
    EmotionalTone,
    PersonaManager,
)

# Behavior
from app.agents.behavior import (
    ConversationBehavior,
    InterruptionBehavior,
    SilenceBehavior,
    ErrorBehavior,
    TurnTakingBehavior,
    BehaviorEngine,
    BehaviorManager,
    InterruptionMode,
    SilenceAction,
    ErrorAction,
    TurnTakingMode,
    ConversationPhase,
)

# Prompts
from app.agents.prompt import (
    PromptTemplate,
    PromptVariable,
    PromptSection,
    SystemPromptConfig,
    SystemPromptBuilder,
    ConversationPromptBuilder,
    FunctionPromptBuilder,
    PromptChain,
    PromptRegistry,
    PromptOptimizer,
    PromptType,
    PromptCategory,
)

# Builder
from app.agents.builder import (
    Agent,
    AgentBuilder,
    AgentFactory,
    AgentBlueprint,
    AgentRegistry,
    create_agent,
    quick_agent,
)

__all__ = [
    # Service
    "AgentService",
    "AgentCreate",
    "AgentUpdate",
    "AgentResponse",
    "AgentListResponse",
    # Configuration
    "AgentConfig",
    "AgentType",
    "AgentStatus",
    "AgentCapability",
    "VoiceSettings",
    "TranscriptionSettings",
    "LLMSettings",
    "InterruptionConfig",
    "SilenceConfig",
    "ErrorConfig",
    "AgentValidator",
    # Persona
    "Persona",
    "PersonaConfig",
    "PersonaTrait",
    "SpeakingStyle",
    "EmotionalTone",
    "PersonaManager",
    # Behavior
    "ConversationBehavior",
    "InterruptionBehavior",
    "SilenceBehavior",
    "ErrorBehavior",
    "TurnTakingBehavior",
    "BehaviorEngine",
    "BehaviorManager",
    "InterruptionMode",
    "SilenceAction",
    "ErrorAction",
    "TurnTakingMode",
    "ConversationPhase",
    # Prompts
    "PromptTemplate",
    "PromptVariable",
    "PromptSection",
    "SystemPromptConfig",
    "SystemPromptBuilder",
    "ConversationPromptBuilder",
    "FunctionPromptBuilder",
    "PromptChain",
    "PromptRegistry",
    "PromptOptimizer",
    "PromptType",
    "PromptCategory",
    # Builder
    "Agent",
    "AgentBuilder",
    "AgentFactory",
    "AgentBlueprint",
    "AgentRegistry",
    "create_agent",
    "quick_agent",
]
