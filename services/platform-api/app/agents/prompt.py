"""
Prompt Management System

Advanced prompt engineering:
- System prompt building
- Prompt templates with variables
- Context injection
- Prompt versioning
- A/B testing support
"""

from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import re
import json
import logging

logger = logging.getLogger(__name__)


class PromptType(str, Enum):
    """Types of prompts."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class PromptCategory(str, Enum):
    """Prompt categories."""
    GREETING = "greeting"
    FAREWELL = "farewell"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    ERROR = "error"
    TRANSFER = "transfer"
    HOLD = "hold"
    FALLBACK = "fallback"
    CUSTOM = "custom"


class VariableType(str, Enum):
    """Variable types for prompts."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    LIST = "list"
    OBJECT = "object"


@dataclass
class PromptVariable:
    """Variable definition for prompts."""
    name: str
    var_type: VariableType = VariableType.STRING
    description: str = ""
    default_value: Any = None
    required: bool = False
    validators: List[str] = field(default_factory=list)

    def validate(self, value: Any) -> bool:
        """Validate variable value."""
        if value is None:
            return not self.required

        # Type validation
        if self.var_type == VariableType.STRING:
            return isinstance(value, str)
        elif self.var_type == VariableType.NUMBER:
            return isinstance(value, (int, float))
        elif self.var_type == VariableType.BOOLEAN:
            return isinstance(value, bool)
        elif self.var_type == VariableType.LIST:
            return isinstance(value, list)
        elif self.var_type == VariableType.OBJECT:
            return isinstance(value, dict)

        return True

    def format_value(self, value: Any) -> str:
        """Format value for prompt insertion."""
        if value is None:
            return str(self.default_value) if self.default_value else ""

        if self.var_type == VariableType.LIST:
            return ", ".join(str(v) for v in value)
        elif self.var_type == VariableType.OBJECT:
            return json.dumps(value)
        elif self.var_type == VariableType.BOOLEAN:
            return "yes" if value else "no"

        return str(value)


@dataclass
class PromptTemplate:
    """Reusable prompt template."""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: PromptCategory = PromptCategory.CUSTOM

    # Template content
    template: str = ""
    variables: List[PromptVariable] = field(default_factory=list)

    # Versioning
    version: str = "1.0.0"
    is_active: bool = True

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def get_variable_names(self) -> Set[str]:
        """Extract variable names from template."""
        pattern = r'\{\{(\w+)\}\}'
        return set(re.findall(pattern, self.template))

    def render(self, variables: Dict[str, Any]) -> str:
        """Render template with variables."""
        result = self.template

        # Get all variable definitions
        var_defs = {v.name: v for v in self.variables}

        # Replace variables
        for var_name in self.get_variable_names():
            var_def = var_defs.get(var_name)
            value = variables.get(var_name)

            if var_def:
                formatted = var_def.format_value(value)
            else:
                formatted = str(value) if value else ""

            result = result.replace(f"{{{{{var_name}}}}}", formatted)

        return result

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate provided variables against definitions."""
        errors = []
        var_defs = {v.name: v for v in self.variables}

        for var_name, var_def in var_defs.items():
            value = variables.get(var_name)

            if var_def.required and value is None:
                errors.append(f"Required variable '{var_name}' is missing")
            elif value is not None and not var_def.validate(value):
                errors.append(f"Invalid type for variable '{var_name}'")

        return errors


@dataclass
class PromptSection:
    """Section of a system prompt."""
    section_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    content: str = ""
    order: int = 0
    is_enabled: bool = True
    conditions: List[Dict[str, Any]] = field(default_factory=list)

    def should_include(self, context: Dict[str, Any]) -> bool:
        """Check if section should be included based on conditions."""
        if not self.is_enabled:
            return False

        if not self.conditions:
            return True

        for condition in self.conditions:
            field = condition.get("field")
            operator = condition.get("operator", "eq")
            value = condition.get("value")

            ctx_value = context.get(field)

            if operator == "eq" and ctx_value != value:
                return False
            elif operator == "neq" and ctx_value == value:
                return False
            elif operator == "contains" and value not in str(ctx_value):
                return False
            elif operator == "exists" and ctx_value is None:
                return False

        return True


@dataclass
class SystemPromptConfig:
    """Configuration for system prompt generation."""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Core identity
    agent_name: str = ""
    agent_role: str = ""
    company_name: str = ""

    # Sections
    sections: List[PromptSection] = field(default_factory=list)

    # Templates
    intro_template: str = ""
    capabilities_template: str = ""
    rules_template: str = ""
    personality_template: str = ""

    # Behavioral settings
    response_style: str = "professional"
    tone: str = "friendly"
    verbosity: str = "concise"  # concise, moderate, detailed

    # Constraints
    max_length: int = 4000
    include_timestamps: bool = False
    include_context: bool = True

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class PromptBuilder(ABC):
    """Abstract prompt builder."""

    @abstractmethod
    def build(self, context: Dict[str, Any]) -> str:
        """Build prompt from context."""
        pass


class SystemPromptBuilder(PromptBuilder):
    """
    Builds comprehensive system prompts.

    Features:
    - Section-based composition
    - Context-aware generation
    - Variable interpolation
    - Conditional sections
    """

    def __init__(self, config: SystemPromptConfig):
        self.config = config
        self._custom_sections: List[PromptSection] = []
        self._transformers: List[Callable[[str], str]] = []

    def add_section(self, section: PromptSection) -> "SystemPromptBuilder":
        """Add custom section."""
        self._custom_sections.append(section)
        return self

    def add_transformer(self, transformer: Callable[[str], str]) -> "SystemPromptBuilder":
        """Add prompt transformer."""
        self._transformers.append(transformer)
        return self

    def build(self, context: Dict[str, Any]) -> str:
        """Build complete system prompt."""
        parts = []

        # Build identity section
        identity = self._build_identity(context)
        if identity:
            parts.append(identity)

        # Build capabilities section
        capabilities = self._build_capabilities(context)
        if capabilities:
            parts.append(capabilities)

        # Build rules section
        rules = self._build_rules(context)
        if rules:
            parts.append(rules)

        # Build personality section
        personality = self._build_personality(context)
        if personality:
            parts.append(personality)

        # Add configured sections
        for section in sorted(self.config.sections, key=lambda s: s.order):
            if section.should_include(context):
                parts.append(section.content)

        # Add custom sections
        for section in sorted(self._custom_sections, key=lambda s: s.order):
            if section.should_include(context):
                parts.append(section.content)

        # Build context section
        if self.config.include_context:
            ctx_section = self._build_context(context)
            if ctx_section:
                parts.append(ctx_section)

        # Combine parts
        prompt = "\n\n".join(filter(None, parts))

        # Apply transformers
        for transformer in self._transformers:
            prompt = transformer(prompt)

        # Enforce max length
        if len(prompt) > self.config.max_length:
            prompt = prompt[:self.config.max_length - 3] + "..."

        return prompt

    def _build_identity(self, context: Dict[str, Any]) -> str:
        """Build identity section."""
        if self.config.intro_template:
            return self._interpolate(self.config.intro_template, context)

        parts = []

        if self.config.agent_name:
            parts.append(f"You are {self.config.agent_name}")
            if self.config.agent_role:
                parts.append(f", a {self.config.agent_role}")
            if self.config.company_name:
                parts.append(f" at {self.config.company_name}")
            parts.append(".")

        return "".join(parts) if parts else ""

    def _build_capabilities(self, context: Dict[str, Any]) -> str:
        """Build capabilities section."""
        if self.config.capabilities_template:
            return self._interpolate(self.config.capabilities_template, context)

        capabilities = context.get("capabilities", [])
        if not capabilities:
            return ""

        lines = ["Your capabilities include:"]
        for cap in capabilities:
            lines.append(f"- {cap}")

        return "\n".join(lines)

    def _build_rules(self, context: Dict[str, Any]) -> str:
        """Build rules section."""
        if self.config.rules_template:
            return self._interpolate(self.config.rules_template, context)

        rules = context.get("rules", [])
        if not rules:
            return ""

        lines = ["Important rules to follow:"]
        for i, rule in enumerate(rules, 1):
            lines.append(f"{i}. {rule}")

        return "\n".join(lines)

    def _build_personality(self, context: Dict[str, Any]) -> str:
        """Build personality section."""
        if self.config.personality_template:
            return self._interpolate(self.config.personality_template, context)

        parts = []

        if self.config.response_style:
            parts.append(f"Maintain a {self.config.response_style} response style.")

        if self.config.tone:
            parts.append(f"Keep a {self.config.tone} tone in all interactions.")

        if self.config.verbosity == "concise":
            parts.append("Be concise and to the point in your responses.")
        elif self.config.verbosity == "detailed":
            parts.append("Provide detailed and thorough responses.")

        return " ".join(parts) if parts else ""

    def _build_context(self, context: Dict[str, Any]) -> str:
        """Build context section."""
        ctx_items = context.get("context_items", {})
        if not ctx_items:
            return ""

        lines = ["Current context:"]
        for key, value in ctx_items.items():
            if value is not None:
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    def _interpolate(self, template: str, context: Dict[str, Any]) -> str:
        """Interpolate variables in template."""
        result = template

        # Simple variable replacement
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                result = result.replace(f"{{{{{key}}}}}", str(value))

        # Also replace from config
        result = result.replace("{{agent_name}}", self.config.agent_name)
        result = result.replace("{{agent_role}}", self.config.agent_role)
        result = result.replace("{{company_name}}", self.config.company_name)

        return result


class ConversationPromptBuilder(PromptBuilder):
    """
    Builds conversation-specific prompts.

    Features:
    - Message history formatting
    - Context injection
    - Turn-based prompts
    """

    def __init__(
        self,
        max_history_turns: int = 10,
        include_system: bool = True,
        include_timestamps: bool = False,
    ):
        self.max_history_turns = max_history_turns
        self.include_system = include_system
        self.include_timestamps = include_timestamps

    def build(self, context: Dict[str, Any]) -> str:
        """Build conversation prompt."""
        parts = []

        # System message
        if self.include_system:
            system = context.get("system_prompt", "")
            if system:
                parts.append(f"[SYSTEM]\n{system}")

        # Conversation history
        history = context.get("history", [])
        recent_history = history[-self.max_history_turns:] if history else []

        for turn in recent_history:
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")

            if self.include_timestamps:
                timestamp = turn.get("timestamp", "")
                parts.append(f"[{role} - {timestamp}]\n{content}")
            else:
                parts.append(f"[{role}]\n{content}")

        # Current input
        current_input = context.get("current_input", "")
        if current_input:
            parts.append(f"[USER]\n{current_input}")

        return "\n\n".join(parts)

    def build_messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build message list for chat models."""
        messages = []

        # System message
        if self.include_system:
            system = context.get("system_prompt", "")
            if system:
                messages.append({"role": "system", "content": system})

        # History
        history = context.get("history", [])
        recent_history = history[-self.max_history_turns:] if history else []

        for turn in recent_history:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", ""),
            })

        # Current input
        current_input = context.get("current_input", "")
        if current_input:
            messages.append({"role": "user", "content": current_input})

        return messages


class FunctionPromptBuilder(PromptBuilder):
    """
    Builds function/tool calling prompts.

    Features:
    - Function schema formatting
    - Tool descriptions
    - Parameter documentation
    """

    def __init__(self, format_style: str = "json"):
        self.format_style = format_style

    def build(self, context: Dict[str, Any]) -> str:
        """Build function prompt."""
        functions = context.get("functions", [])
        if not functions:
            return ""

        parts = ["Available functions:"]

        for func in functions:
            name = func.get("name", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            parts.append(f"\n{name}:")
            parts.append(f"  Description: {description}")

            if parameters:
                parts.append("  Parameters:")
                for param_name, param_info in parameters.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    required = param_info.get("required", False)
                    req_str = " (required)" if required else ""
                    parts.append(f"    - {param_name} ({param_type}){req_str}: {param_desc}")

        return "\n".join(parts)

    def build_schema(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build function schema for API calls."""
        functions = context.get("functions", [])
        schemas = []

        for func in functions:
            schema = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }

            for param_name, param_info in func.get("parameters", {}).items():
                schema["parameters"]["properties"][param_name] = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", ""),
                }
                if param_info.get("required"):
                    schema["parameters"]["required"].append(param_name)

            schemas.append(schema)

        return schemas


class PromptChain:
    """
    Chain multiple prompts together.

    Features:
    - Sequential prompt execution
    - Context passing between prompts
    - Conditional branching
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._steps: List[Dict[str, Any]] = []

    def add_step(
        self,
        builder: PromptBuilder,
        name: str = "",
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        transform_context: Optional[Callable[[Dict[str, Any], str], Dict[str, Any]]] = None,
    ) -> "PromptChain":
        """Add step to chain."""
        self._steps.append({
            "builder": builder,
            "name": name,
            "condition": condition,
            "transform_context": transform_context,
        })
        return self

    def build(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build all prompts in chain."""
        results = []
        current_context = context.copy()

        for step in self._steps:
            # Check condition
            condition = step.get("condition")
            if condition and not condition(current_context):
                continue

            # Build prompt
            builder = step["builder"]
            prompt = builder.build(current_context)

            results.append({
                "name": step.get("name", ""),
                "prompt": prompt,
            })

            # Transform context for next step
            transform = step.get("transform_context")
            if transform:
                current_context = transform(current_context, prompt)

        return results


class PromptRegistry:
    """
    Registry for prompt templates.

    Features:
    - Template storage and retrieval
    - Version management
    - A/B testing support
    """

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._versions: Dict[str, Dict[str, PromptTemplate]] = {}
        self._ab_tests: Dict[str, Dict[str, Any]] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register template."""
        self._templates[template.template_id] = template

        # Track versions
        if template.name not in self._versions:
            self._versions[template.name] = {}
        self._versions[template.name][template.version] = template

    def get(self, template_id: str) -> Optional[PromptTemplate]:
        """Get template by ID."""
        return self._templates.get(template_id)

    def get_by_name(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[PromptTemplate]:
        """Get template by name and optional version."""
        versions = self._versions.get(name, {})

        if version:
            return versions.get(version)

        # Get latest active version
        for ver in sorted(versions.keys(), reverse=True):
            template = versions[ver]
            if template.is_active:
                return template

        return None

    def get_by_category(self, category: PromptCategory) -> List[PromptTemplate]:
        """Get templates by category."""
        return [
            t for t in self._templates.values()
            if t.category == category and t.is_active
        ]

    def create_ab_test(
        self,
        test_name: str,
        template_a_id: str,
        template_b_id: str,
        split_ratio: float = 0.5,
    ) -> None:
        """Create A/B test between templates."""
        self._ab_tests[test_name] = {
            "template_a": template_a_id,
            "template_b": template_b_id,
            "split_ratio": split_ratio,
            "results_a": {"count": 0, "success": 0},
            "results_b": {"count": 0, "success": 0},
        }

    def get_ab_template(
        self,
        test_name: str,
        user_id: str,
    ) -> Optional[PromptTemplate]:
        """Get template for A/B test."""
        test = self._ab_tests.get(test_name)
        if not test:
            return None

        # Deterministic split based on user_id
        hash_val = hash(user_id + test_name) % 100
        split_threshold = int(test["split_ratio"] * 100)

        if hash_val < split_threshold:
            return self.get(test["template_a"])
        else:
            return self.get(test["template_b"])

    def record_ab_result(
        self,
        test_name: str,
        user_id: str,
        success: bool,
    ) -> None:
        """Record A/B test result."""
        test = self._ab_tests.get(test_name)
        if not test:
            return

        # Determine which variant
        hash_val = hash(user_id + test_name) % 100
        split_threshold = int(test["split_ratio"] * 100)

        if hash_val < split_threshold:
            test["results_a"]["count"] += 1
            if success:
                test["results_a"]["success"] += 1
        else:
            test["results_b"]["count"] += 1
            if success:
                test["results_b"]["success"] += 1

    def get_ab_stats(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get A/B test statistics."""
        test = self._ab_tests.get(test_name)
        if not test:
            return None

        results_a = test["results_a"]
        results_b = test["results_b"]

        rate_a = results_a["success"] / results_a["count"] if results_a["count"] > 0 else 0
        rate_b = results_b["success"] / results_b["count"] if results_b["count"] > 0 else 0

        return {
            "test_name": test_name,
            "variant_a": {
                "template_id": test["template_a"],
                "count": results_a["count"],
                "success": results_a["success"],
                "success_rate": rate_a,
            },
            "variant_b": {
                "template_id": test["template_b"],
                "count": results_b["count"],
                "success": results_b["success"],
                "success_rate": rate_b,
            },
            "winner": "a" if rate_a > rate_b else "b" if rate_b > rate_a else "tie",
        }

    def list_templates(self) -> List[PromptTemplate]:
        """List all templates."""
        return list(self._templates.values())

    def delete(self, template_id: str) -> bool:
        """Delete template."""
        template = self._templates.pop(template_id, None)
        if template:
            versions = self._versions.get(template.name, {})
            versions.pop(template.version, None)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_templates": len(self._templates),
            "active_templates": sum(1 for t in self._templates.values() if t.is_active),
            "categories": {
                cat.value: sum(1 for t in self._templates.values() if t.category == cat)
                for cat in PromptCategory
            },
            "ab_tests": len(self._ab_tests),
        }


class PromptOptimizer:
    """
    Optimizes prompts for performance.

    Features:
    - Token estimation
    - Prompt compression
    - Context prioritization
    """

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimation: ~4 characters per token for English
        return len(text) // 4

    def compress(self, prompt: str, target_tokens: Optional[int] = None) -> str:
        """Compress prompt to target token count."""
        target = target_tokens or self.max_tokens
        current_tokens = self.estimate_tokens(prompt)

        if current_tokens <= target:
            return prompt

        # Apply compression strategies
        result = prompt

        # Remove extra whitespace
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\n\s*\n', '\n', result)

        # Abbreviate common phrases
        abbreviations = {
            "for example": "e.g.",
            "that is": "i.e.",
            "and so on": "etc.",
            "in other words": "i.e.",
        }
        for phrase, abbrev in abbreviations.items():
            result = result.replace(phrase, abbrev)

        # Truncate if still too long
        if self.estimate_tokens(result) > target:
            char_limit = target * 4
            result = result[:char_limit - 3] + "..."

        return result

    def prioritize_context(
        self,
        items: List[Dict[str, Any]],
        max_items: int = 10,
    ) -> List[Dict[str, Any]]:
        """Prioritize context items by relevance."""
        # Sort by priority (higher = more important)
        sorted_items = sorted(
            items,
            key=lambda x: x.get("priority", 0),
            reverse=True,
        )
        return sorted_items[:max_items]

    def split_prompt(
        self,
        prompt: str,
        chunk_size: int = 2000,
    ) -> List[str]:
        """Split prompt into chunks."""
        chunks = []
        current_chunk = ""

        # Split by paragraphs
        paragraphs = prompt.split("\n\n")

        for para in paragraphs:
            if self.estimate_tokens(current_chunk + para) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


# Factory functions
def create_system_prompt_builder(
    agent_name: str = "",
    agent_role: str = "",
    company_name: str = "",
    **kwargs,
) -> SystemPromptBuilder:
    """Create system prompt builder with configuration."""
    config = SystemPromptConfig(
        agent_name=agent_name,
        agent_role=agent_role,
        company_name=company_name,
        **kwargs,
    )
    return SystemPromptBuilder(config)


def create_greeting_template(
    agent_name: str,
    style: str = "professional",
) -> PromptTemplate:
    """Create greeting prompt template."""
    templates = {
        "professional": f"Hello, my name is {agent_name}. How may I assist you today?",
        "friendly": f"Hi there! I'm {agent_name}. What can I help you with?",
        "formal": f"Good day. I am {agent_name}, and I will be assisting you today.",
        "casual": f"Hey! {agent_name} here. What's up?",
    }

    return PromptTemplate(
        name="greeting",
        category=PromptCategory.GREETING,
        template=templates.get(style, templates["professional"]),
    )


def create_farewell_template(style: str = "professional") -> PromptTemplate:
    """Create farewell prompt template."""
    templates = {
        "professional": "Thank you for contacting us. Have a great day!",
        "friendly": "Thanks so much! Take care and have an awesome day!",
        "formal": "Thank you for your time. Please do not hesitate to contact us again.",
        "casual": "Alright, thanks! Catch you later!",
    }

    return PromptTemplate(
        name="farewell",
        category=PromptCategory.FAREWELL,
        template=templates.get(style, templates["professional"]),
    )
