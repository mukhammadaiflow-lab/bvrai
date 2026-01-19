"""Prompt builder for constructing system prompts."""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class AgentPersona:
    """Agent persona configuration."""
    name: str
    role: str
    company: Optional[str] = None
    personality_traits: List[str] = field(default_factory=list)
    speaking_style: Optional[str] = None
    background: Optional[str] = None


@dataclass
class ConversationRules:
    """Rules for conversation behavior."""
    allow_interruptions: bool = True
    confirm_important_info: bool = True
    ask_clarifying_questions: bool = True
    stay_on_topic: bool = True
    max_response_length: int = 150  # words
    use_filler_words: bool = True
    acknowledge_emotions: bool = True


class PromptBuilder:
    """
    Builder for constructing prompts.

    Provides a fluent API for building prompts.
    """

    def __init__(self):
        self._parts: List[str] = []
        self._variables: Dict[str, Any] = {}

    def add(self, text: str) -> "PromptBuilder":
        """Add text to prompt."""
        self._parts.append(text)
        return self

    def add_section(self, title: str, content: str) -> "PromptBuilder":
        """Add a titled section."""
        self._parts.append(f"\n## {title}\n{content}")
        return self

    def add_list(self, title: str, items: List[str]) -> "PromptBuilder":
        """Add a bullet list section."""
        formatted = "\n".join(f"- {item}" for item in items)
        self._parts.append(f"\n## {title}\n{formatted}")
        return self

    def add_numbered_list(self, title: str, items: List[str]) -> "PromptBuilder":
        """Add a numbered list section."""
        formatted = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        self._parts.append(f"\n## {title}\n{formatted}")
        return self

    def add_if(self, condition: bool, text: str) -> "PromptBuilder":
        """Add text only if condition is true."""
        if condition:
            self._parts.append(text)
        return self

    def set_variable(self, name: str, value: Any) -> "PromptBuilder":
        """Set a variable for substitution."""
        self._variables[name] = value
        return self

    def build(self) -> str:
        """Build the final prompt."""
        result = "\n".join(self._parts)

        # Substitute variables
        for name, value in self._variables.items():
            result = result.replace(f"{{{name}}}", str(value))

        return result.strip()

    def clear(self) -> "PromptBuilder":
        """Clear all parts."""
        self._parts.clear()
        self._variables.clear()
        return self


class SystemPromptBuilder:
    """
    Specialized builder for voice AI system prompts.

    Creates comprehensive prompts with:
    - Agent persona
    - Conversation rules
    - Domain knowledge
    - Tool descriptions
    - Behavioral guidelines
    """

    def __init__(self):
        self._persona: Optional[AgentPersona] = None
        self._rules: Optional[ConversationRules] = None
        self._objective: Optional[str] = None
        self._knowledge: List[str] = []
        self._tools: List[Dict[str, str]] = []
        self._guidelines: List[str] = []
        self._examples: List[Dict[str, str]] = []
        self._constraints: List[str] = []
        self._context: Dict[str, Any] = {}

    def with_persona(self, persona: AgentPersona) -> "SystemPromptBuilder":
        """Set agent persona."""
        self._persona = persona
        return self

    def with_simple_persona(
        self,
        name: str,
        role: str,
        company: Optional[str] = None,
    ) -> "SystemPromptBuilder":
        """Set simple persona."""
        self._persona = AgentPersona(name=name, role=role, company=company)
        return self

    def with_rules(self, rules: ConversationRules) -> "SystemPromptBuilder":
        """Set conversation rules."""
        self._rules = rules
        return self

    def with_objective(self, objective: str) -> "SystemPromptBuilder":
        """Set conversation objective."""
        self._objective = objective
        return self

    def add_knowledge(self, knowledge: str) -> "SystemPromptBuilder":
        """Add domain knowledge."""
        self._knowledge.append(knowledge)
        return self

    def add_tool(
        self,
        name: str,
        description: str,
        when_to_use: Optional[str] = None,
    ) -> "SystemPromptBuilder":
        """Add tool description."""
        self._tools.append({
            "name": name,
            "description": description,
            "when_to_use": when_to_use or f"Use when you need to {description.lower()}",
        })
        return self

    def add_guideline(self, guideline: str) -> "SystemPromptBuilder":
        """Add behavioral guideline."""
        self._guidelines.append(guideline)
        return self

    def add_example(self, user: str, assistant: str) -> "SystemPromptBuilder":
        """Add example exchange."""
        self._examples.append({"user": user, "assistant": assistant})
        return self

    def add_constraint(self, constraint: str) -> "SystemPromptBuilder":
        """Add constraint/limitation."""
        self._constraints.append(constraint)
        return self

    def with_context(self, key: str, value: Any) -> "SystemPromptBuilder":
        """Add context information."""
        self._context[key] = value
        return self

    def build(self) -> str:
        """Build the complete system prompt."""
        parts = []

        # Identity section
        parts.append(self._build_identity())

        # Objective
        if self._objective:
            parts.append(f"\n## Objective\n{self._objective}")

        # Knowledge
        if self._knowledge:
            knowledge_text = "\n".join(f"- {k}" for k in self._knowledge)
            parts.append(f"\n## Knowledge\n{knowledge_text}")

        # Tools
        if self._tools:
            parts.append(self._build_tools_section())

        # Guidelines
        if self._guidelines:
            guidelines_text = "\n".join(f"- {g}" for g in self._guidelines)
            parts.append(f"\n## Guidelines\n{guidelines_text}")

        # Conversation rules
        if self._rules:
            parts.append(self._build_rules_section())

        # Examples
        if self._examples:
            parts.append(self._build_examples_section())

        # Constraints
        if self._constraints:
            constraints_text = "\n".join(f"- {c}" for c in self._constraints)
            parts.append(f"\n## Constraints\n{constraints_text}")

        # Context
        if self._context:
            parts.append(self._build_context_section())

        return "\n".join(parts).strip()

    def _build_identity(self) -> str:
        """Build identity section."""
        if not self._persona:
            return "You are a helpful voice AI assistant."

        parts = [f"You are {self._persona.name}, a {self._persona.role}"]

        if self._persona.company:
            parts[0] += f" at {self._persona.company}"
        parts[0] += "."

        if self._persona.personality_traits:
            traits = ", ".join(self._persona.personality_traits)
            parts.append(f"Your personality: {traits}.")

        if self._persona.speaking_style:
            parts.append(f"Speaking style: {self._persona.speaking_style}.")

        if self._persona.background:
            parts.append(f"Background: {self._persona.background}")

        return "## Identity\n" + " ".join(parts)

    def _build_tools_section(self) -> str:
        """Build tools section."""
        lines = ["## Available Tools"]
        for tool in self._tools:
            lines.append(f"\n### {tool['name']}")
            lines.append(tool['description'])
            lines.append(f"When to use: {tool['when_to_use']}")
        return "\n".join(lines)

    def _build_rules_section(self) -> str:
        """Build rules section."""
        if not self._rules:
            return ""

        rules = []
        if self._rules.allow_interruptions:
            rules.append("Allow the user to interrupt you mid-sentence")
        if self._rules.confirm_important_info:
            rules.append("Always confirm important information before taking action")
        if self._rules.ask_clarifying_questions:
            rules.append("Ask clarifying questions when unsure")
        if self._rules.stay_on_topic:
            rules.append("Keep the conversation focused on the objective")
        if self._rules.use_filler_words:
            rules.append("Use natural filler words (um, well, let me see)")
        if self._rules.acknowledge_emotions:
            rules.append("Acknowledge and respond appropriately to emotions")

        rules.append(f"Keep responses under {self._rules.max_response_length} words")

        return "## Conversation Rules\n" + "\n".join(f"- {r}" for r in rules)

    def _build_examples_section(self) -> str:
        """Build examples section."""
        lines = ["## Example Exchanges"]
        for ex in self._examples:
            lines.append(f"\nUser: {ex['user']}")
            lines.append(f"Assistant: {ex['assistant']}")
        return "\n".join(lines)

    def _build_context_section(self) -> str:
        """Build context section."""
        lines = ["## Current Context"]
        for key, value in self._context.items():
            if isinstance(value, dict):
                import json
                value = json.dumps(value, indent=2)
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def clear(self) -> "SystemPromptBuilder":
        """Reset builder."""
        self._persona = None
        self._rules = None
        self._objective = None
        self._knowledge.clear()
        self._tools.clear()
        self._guidelines.clear()
        self._examples.clear()
        self._constraints.clear()
        self._context.clear()
        return self
