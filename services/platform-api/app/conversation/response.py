"""
Response Generation

Response generation and templating:
- Template-based responses
- Dynamic content generation
- Multi-language support
- SSML generation
"""

from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import re
import random

logger = logging.getLogger(__name__)


class ResponseType(str, Enum):
    """Response types."""
    TEXT = "text"
    SSML = "ssml"
    AUDIO = "audio"
    ACTION = "action"
    CARD = "card"
    SUGGESTION = "suggestion"


class SSMLElement(str, Enum):
    """SSML elements."""
    SPEAK = "speak"
    BREAK = "break"
    EMPHASIS = "emphasis"
    PROSODY = "prosody"
    SAY_AS = "say-as"
    PHONEME = "phoneme"
    SUB = "sub"
    AUDIO = "audio"


@dataclass
class GeneratedResponse:
    """Generated response."""
    response_id: str
    response_type: ResponseType = ResponseType.TEXT

    # Content
    text: str = ""
    ssml: Optional[str] = None
    audio_url: Optional[str] = None

    # Actions
    actions: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Metadata
    language: str = "en"
    voice: Optional[str] = None
    emotion: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)
    template_id: Optional[str] = None

    # Flags
    should_end_conversation: bool = False
    expects_response: bool = True
    is_interruptible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response_id": self.response_id,
            "type": self.response_type.value,
            "text": self.text,
            "ssml": self.ssml,
            "audio_url": self.audio_url,
            "actions": self.actions,
            "suggestions": self.suggestions,
            "language": self.language,
            "should_end": self.should_end_conversation,
        }


@dataclass
class ResponseTemplate:
    """Response template."""
    template_id: str
    name: str

    # Content variants
    texts: List[str] = field(default_factory=list)
    ssml_template: Optional[str] = None

    # Conditions
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Options
    language: str = "en"
    priority: int = 0
    weight: float = 1.0

    # Actions
    actions: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Flags
    ends_conversation: bool = False
    expects_response: bool = True

    def get_text(self) -> str:
        """Get random text variant."""
        if self.texts:
            return random.choice(self.texts)
        return ""

    def matches_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if conditions match context."""
        for key, expected in self.conditions.items():
            actual = context.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True


class TemplateEngine:
    """
    Template rendering engine.

    Features:
    - Variable substitution
    - Conditional blocks
    - Loops
    - SSML generation
    """

    def __init__(self):
        self._filters: Dict[str, Callable[[Any], str]] = {}
        self._functions: Dict[str, Callable[..., str]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default filters and functions."""
        # Filters
        self._filters["upper"] = lambda x: str(x).upper()
        self._filters["lower"] = lambda x: str(x).lower()
        self._filters["title"] = lambda x: str(x).title()
        self._filters["currency"] = lambda x: f"${float(x):.2f}"
        self._filters["phone"] = self._format_phone
        self._filters["date"] = self._format_date

        # Functions
        self._functions["random"] = lambda *args: random.choice(args)
        self._functions["pluralize"] = self._pluralize

    def _format_phone(self, value: Any) -> str:
        """Format phone number."""
        digits = re.sub(r"[^\d]", "", str(value))
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        if len(digits) == 11 and digits[0] == "1":
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        return str(value)

    def _format_date(self, value: Any) -> str:
        """Format date."""
        if isinstance(value, datetime):
            return value.strftime("%B %d, %Y")
        return str(value)

    def _pluralize(self, count: int, singular: str, plural: str = "") -> str:
        """Pluralize word based on count."""
        if count == 1:
            return singular
        return plural or f"{singular}s"

    def add_filter(self, name: str, func: Callable[[Any], str]) -> None:
        """Add custom filter."""
        self._filters[name] = func

    def add_function(self, name: str, func: Callable[..., str]) -> None:
        """Add custom function."""
        self._functions[name] = func

    def render(
        self,
        template: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Render template with context.

        Supports:
        - ${variable} - Variable substitution
        - ${variable|filter} - Filter application
        - {{#if condition}}...{{/if}} - Conditionals
        - {{#each items}}...{{/each}} - Loops
        - {{function(args)}} - Function calls
        """
        result = template

        # Process conditionals
        result = self._process_conditionals(result, context)

        # Process loops
        result = self._process_loops(result, context)

        # Process function calls
        result = self._process_functions(result, context)

        # Process variables
        result = self._process_variables(result, context)

        return result.strip()

    def _process_variables(
        self,
        template: str,
        context: Dict[str, Any],
    ) -> str:
        """Process variable substitutions."""
        def replace_var(match):
            expression = match.group(1)

            # Check for filter
            if "|" in expression:
                var_name, filter_name = expression.split("|", 1)
                value = self._resolve_path(var_name.strip(), context)

                filter_func = self._filters.get(filter_name.strip())
                if filter_func:
                    return filter_func(value)
                return str(value) if value is not None else ""
            else:
                value = self._resolve_path(expression.strip(), context)
                return str(value) if value is not None else ""

        return re.sub(r"\$\{([^}]+)\}", replace_var, template)

    def _process_conditionals(
        self,
        template: str,
        context: Dict[str, Any],
    ) -> str:
        """Process conditional blocks."""
        pattern = r"\{\{#if\s+(.+?)\}\}(.*?)(?:\{\{#else\}\}(.*?))?\{\{/if\}\}"

        def replace_cond(match):
            condition = match.group(1)
            if_content = match.group(2)
            else_content = match.group(3) or ""

            # Evaluate condition
            result = self._evaluate_condition(condition, context)

            return if_content if result else else_content

        return re.sub(pattern, replace_cond, template, flags=re.DOTALL)

    def _process_loops(
        self,
        template: str,
        context: Dict[str, Any],
    ) -> str:
        """Process loop blocks."""
        pattern = r"\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}"

        def replace_loop(match):
            var_name = match.group(1)
            loop_content = match.group(2)

            items = context.get(var_name, [])
            if not isinstance(items, (list, tuple)):
                return ""

            result_parts = []
            for i, item in enumerate(items):
                item_context = {
                    **context,
                    "item": item,
                    "index": i,
                    "first": i == 0,
                    "last": i == len(items) - 1,
                }
                result_parts.append(self.render(loop_content, item_context))

            return "".join(result_parts)

        return re.sub(pattern, replace_loop, template, flags=re.DOTALL)

    def _process_functions(
        self,
        template: str,
        context: Dict[str, Any],
    ) -> str:
        """Process function calls."""
        pattern = r"\{\{(\w+)\((.*?)\)\}\}"

        def replace_func(match):
            func_name = match.group(1)
            args_str = match.group(2)

            func = self._functions.get(func_name)
            if not func:
                return ""

            # Parse arguments
            args = []
            if args_str:
                for arg in args_str.split(","):
                    arg = arg.strip().strip("'\"")
                    # Check if it's a variable reference
                    if arg.startswith("$"):
                        arg = self._resolve_path(arg[1:], context)
                    args.append(arg)

            try:
                return str(func(*args))
            except Exception as e:
                logger.error(f"Function error: {e}")
                return ""

        return re.sub(pattern, replace_func, template)

    def _resolve_path(self, path: str, context: Dict[str, Any]) -> Any:
        """Resolve dot-notation path in context."""
        parts = path.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

            if value is None:
                return None

        return value

    def _evaluate_condition(
        self,
        condition: str,
        context: Dict[str, Any],
    ) -> bool:
        """Evaluate condition expression."""
        condition = condition.strip()

        # Simple variable truthiness
        if not any(op in condition for op in ["==", "!=", "<", ">", " and ", " or "]):
            value = self._resolve_path(condition, context)
            return bool(value)

        # Comparison operators
        for op in ["==", "!=", ">=", "<=", ">", "<"]:
            if op in condition:
                left, right = condition.split(op, 1)
                left_val = self._resolve_path(left.strip(), context)
                right_val = right.strip().strip("'\"")

                # Try numeric comparison
                try:
                    left_val = float(left_val)
                    right_val = float(right_val)
                except (ValueError, TypeError):
                    pass

                if op == "==":
                    return left_val == right_val
                elif op == "!=":
                    return left_val != right_val
                elif op == ">":
                    return left_val > right_val
                elif op == "<":
                    return left_val < right_val
                elif op == ">=":
                    return left_val >= right_val
                elif op == "<=":
                    return left_val <= right_val

        return False


class SSMLBuilder:
    """
    SSML markup builder.

    Creates Speech Synthesis Markup Language for TTS.
    """

    def __init__(self, text: str = ""):
        self._parts: List[str] = []
        if text:
            self._parts.append(text)

    def text(self, content: str) -> "SSMLBuilder":
        """Add plain text."""
        self._parts.append(self._escape(content))
        return self

    def pause(self, duration_ms: int = 500) -> "SSMLBuilder":
        """Add pause/break."""
        self._parts.append(f'<break time="{duration_ms}ms"/>')
        return self

    def emphasis(
        self,
        text: str,
        level: str = "moderate",
    ) -> "SSMLBuilder":
        """Add emphasized text."""
        self._parts.append(
            f'<emphasis level="{level}">{self._escape(text)}</emphasis>'
        )
        return self

    def prosody(
        self,
        text: str,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None,
    ) -> "SSMLBuilder":
        """Add prosody-modified text."""
        attrs = []
        if rate:
            attrs.append(f'rate="{rate}"')
        if pitch:
            attrs.append(f'pitch="{pitch}"')
        if volume:
            attrs.append(f'volume="{volume}"')

        attr_str = " ".join(attrs)
        self._parts.append(
            f'<prosody {attr_str}>{self._escape(text)}</prosody>'
        )
        return self

    def say_as(
        self,
        text: str,
        interpret_as: str,
        format: Optional[str] = None,
    ) -> "SSMLBuilder":
        """Add interpreted text."""
        attrs = f'interpret-as="{interpret_as}"'
        if format:
            attrs += f' format="{format}"'

        self._parts.append(
            f'<say-as {attrs}>{self._escape(text)}</say-as>'
        )
        return self

    def phoneme(
        self,
        text: str,
        pronunciation: str,
        alphabet: str = "ipa",
    ) -> "SSMLBuilder":
        """Add phonetic pronunciation."""
        self._parts.append(
            f'<phoneme alphabet="{alphabet}" ph="{pronunciation}">'
            f'{self._escape(text)}</phoneme>'
        )
        return self

    def sub(self, text: str, alias: str) -> "SSMLBuilder":
        """Add substitution."""
        self._parts.append(
            f'<sub alias="{alias}">{self._escape(text)}</sub>'
        )
        return self

    def audio(self, url: str, alt_text: str = "") -> "SSMLBuilder":
        """Add audio."""
        self._parts.append(
            f'<audio src="{url}">{self._escape(alt_text)}</audio>'
        )
        return self

    def _escape(self, text: str) -> str:
        """Escape special characters."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def build(self) -> str:
        """Build SSML string."""
        content = "".join(self._parts)
        return f'<speak>{content}</speak>'


class ResponseGenerator:
    """
    Generates responses from templates.

    Features:
    - Template selection
    - Variable substitution
    - SSML generation
    - Multi-language support
    """

    def __init__(self):
        self._templates: Dict[str, List[ResponseTemplate]] = {}
        self._engine = TemplateEngine()
        self._default_language = "en"

    def add_template(self, template: ResponseTemplate) -> None:
        """Add response template."""
        key = f"{template.name}:{template.language}"
        if key not in self._templates:
            self._templates[key] = []
        self._templates[key].append(template)
        # Sort by priority
        self._templates[key].sort(key=lambda t: t.priority, reverse=True)

    def add_templates_from_dict(self, data: Dict[str, Any]) -> int:
        """Add templates from dictionary."""
        count = 0
        for name, config in data.items():
            if isinstance(config, list):
                # Simple list of text variants
                template = ResponseTemplate(
                    template_id=name,
                    name=name,
                    texts=config,
                )
            elif isinstance(config, dict):
                template = ResponseTemplate(
                    template_id=config.get("id", name),
                    name=name,
                    texts=config.get("texts", [config.get("text", "")]),
                    ssml_template=config.get("ssml"),
                    conditions=config.get("conditions", {}),
                    language=config.get("language", "en"),
                    priority=config.get("priority", 0),
                    actions=config.get("actions", []),
                    suggestions=config.get("suggestions", []),
                    ends_conversation=config.get("ends_conversation", False),
                )
            else:
                continue

            self.add_template(template)
            count += 1

        return count

    async def generate(
        self,
        template_name: str,
        context: Dict[str, Any],
        language: Optional[str] = None,
    ) -> Optional[GeneratedResponse]:
        """Generate response from template."""
        import uuid

        language = language or self._default_language
        key = f"{template_name}:{language}"

        # Fall back to default language
        if key not in self._templates:
            key = f"{template_name}:{self._default_language}"

        templates = self._templates.get(key, [])
        if not templates:
            logger.warning(f"No template found: {template_name}")
            return None

        # Find matching template
        template = None
        for t in templates:
            if t.matches_conditions(context):
                template = t
                break

        if not template:
            # Use first template as default
            template = templates[0]

        # Get text
        text = template.get_text()

        # Render with context
        text = self._engine.render(text, context)

        # Generate SSML if template defined
        ssml = None
        if template.ssml_template:
            ssml = self._engine.render(template.ssml_template, context)

        return GeneratedResponse(
            response_id=str(uuid.uuid4()),
            response_type=ResponseType.SSML if ssml else ResponseType.TEXT,
            text=text,
            ssml=ssml,
            actions=template.actions,
            suggestions=template.suggestions,
            language=language,
            template_id=template.template_id,
            should_end_conversation=template.ends_conversation,
            expects_response=template.expects_response,
        )

    async def generate_ssml(
        self,
        text: str,
        context: Dict[str, Any],
        include_pauses: bool = True,
    ) -> str:
        """Generate SSML from plain text."""
        # Render variables
        text = self._engine.render(text, context)

        builder = SSMLBuilder()

        # Split into sentences and add pauses
        sentences = re.split(r"([.!?]+)", text)

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""

            if sentence:
                builder.text(sentence + punctuation)

                if include_pauses and punctuation:
                    pause_duration = 500 if "." in punctuation else 300
                    builder.pause(pause_duration)

        # Handle remaining text
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            builder.text(sentences[-1].strip())

        return builder.build()

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "template_count": sum(len(t) for t in self._templates.values()),
            "languages": list(set(k.split(":")[1] for k in self._templates.keys())),
        }
