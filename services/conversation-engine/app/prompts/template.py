"""Prompt template system."""

import re
from typing import Optional, Dict, List, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class VariableType(str, Enum):
    """Types of template variables."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"
    DATE = "date"
    CONDITIONAL = "conditional"


@dataclass
class PromptVariable:
    """A variable in a prompt template."""
    name: str
    type: VariableType = VariableType.STRING
    description: Optional[str] = None
    required: bool = True
    default: Optional[Any] = None
    validation_regex: Optional[str] = None
    examples: List[str] = field(default_factory=list)

    def validate(self, value: Any) -> bool:
        """Validate a value for this variable."""
        if value is None:
            return not self.required

        if self.type == VariableType.STRING:
            if not isinstance(value, str):
                return False
            if self.validation_regex:
                return bool(re.match(self.validation_regex, value))
            return True

        elif self.type == VariableType.NUMBER:
            return isinstance(value, (int, float))

        elif self.type == VariableType.BOOLEAN:
            return isinstance(value, bool)

        elif self.type == VariableType.LIST:
            return isinstance(value, list)

        elif self.type == VariableType.OBJECT:
            return isinstance(value, dict)

        return True


@dataclass
class PromptTemplate:
    """
    A reusable prompt template.

    Supports:
    - Variable substitution: {variable_name}
    - Conditional sections: {?condition}...{/condition}
    - Loops: {#items}...{/items}
    - Filters: {variable|uppercase}
    """

    name: str
    template: str
    description: Optional[str] = None
    variables: List[PromptVariable] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Compiled patterns
    _var_pattern = re.compile(r'\{(\w+)(?:\|(\w+))?\}')
    _conditional_pattern = re.compile(r'\{\?(\w+)\}(.*?)\{/\1\}', re.DOTALL)
    _loop_pattern = re.compile(r'\{#(\w+)\}(.*?)\{/\1\}', re.DOTALL)

    def render(
        self,
        variables: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> str:
        """
        Render the template with variables.

        Args:
            variables: Variable values
            strict: Raise error on missing required variables

        Returns:
            Rendered prompt string
        """
        variables = variables or {}
        result = self.template

        # Apply defaults for missing variables
        for var in self.variables:
            if var.name not in variables and var.default is not None:
                variables[var.name] = var.default

        # Validate required variables
        if strict:
            for var in self.variables:
                if var.required and var.name not in variables:
                    raise ValueError(f"Missing required variable: {var.name}")

        # Process conditionals first
        result = self._process_conditionals(result, variables)

        # Process loops
        result = self._process_loops(result, variables)

        # Substitute variables
        result = self._substitute_variables(result, variables)

        # Clean up whitespace
        result = self._clean_whitespace(result)

        return result

    def _substitute_variables(
        self,
        text: str,
        variables: Dict[str, Any],
    ) -> str:
        """Substitute variables in text."""
        def replace_var(match):
            var_name = match.group(1)
            filter_name = match.group(2)

            value = variables.get(var_name, f"{{{var_name}}}")

            if value is None:
                return ""

            # Convert to string
            if isinstance(value, (list, dict)):
                import json
                value = json.dumps(value)
            else:
                value = str(value)

            # Apply filter if present
            if filter_name:
                value = self._apply_filter(value, filter_name)

            return value

        return self._var_pattern.sub(replace_var, text)

    def _process_conditionals(
        self,
        text: str,
        variables: Dict[str, Any],
    ) -> str:
        """Process conditional sections."""
        def replace_conditional(match):
            condition_var = match.group(1)
            content = match.group(2)

            value = variables.get(condition_var)

            # Check if condition is true
            if value and (not isinstance(value, str) or value.lower() not in ('false', 'no', '0', '')):
                return content.strip()
            return ""

        return self._conditional_pattern.sub(replace_conditional, text)

    def _process_loops(
        self,
        text: str,
        variables: Dict[str, Any],
    ) -> str:
        """Process loop sections."""
        def replace_loop(match):
            list_var = match.group(1)
            item_template = match.group(2)

            items = variables.get(list_var, [])
            if not isinstance(items, list):
                return ""

            rendered_items = []
            for i, item in enumerate(items):
                item_vars = {
                    "item": item,
                    "index": i,
                    "first": i == 0,
                    "last": i == len(items) - 1,
                }

                # If item is a dict, merge its keys
                if isinstance(item, dict):
                    item_vars.update(item)

                rendered = self._substitute_variables(item_template, item_vars)
                rendered_items.append(rendered.strip())

            return "\n".join(rendered_items)

        return self._loop_pattern.sub(replace_loop, text)

    def _apply_filter(self, value: str, filter_name: str) -> str:
        """Apply a filter to a value."""
        filters = {
            "uppercase": str.upper,
            "lowercase": str.lower,
            "capitalize": str.capitalize,
            "title": str.title,
            "strip": str.strip,
            "truncate50": lambda s: s[:50] + "..." if len(s) > 50 else s,
            "truncate100": lambda s: s[:100] + "..." if len(s) > 100 else s,
        }

        filter_fn = filters.get(filter_name)
        if filter_fn:
            return filter_fn(value)
        return value

    def _clean_whitespace(self, text: str) -> str:
        """Clean up excessive whitespace."""
        # Remove multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        return '\n'.join(lines).strip()

    def get_required_variables(self) -> List[str]:
        """Get list of required variable names."""
        return [v.name for v in self.variables if v.required]

    def get_all_variable_names(self) -> Set[str]:
        """Extract all variable names from template."""
        # Find all {variable} patterns
        var_names = set()

        for match in self._var_pattern.finditer(self.template):
            var_names.add(match.group(1))

        for match in self._conditional_pattern.finditer(self.template):
            var_names.add(match.group(1))

        for match in self._loop_pattern.finditer(self.template):
            var_names.add(match.group(1))

        return var_names

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "variables": [
                {
                    "name": v.name,
                    "type": v.type.value,
                    "description": v.description,
                    "required": v.required,
                    "default": v.default,
                }
                for v in self.variables
            ],
            "category": self.category,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        variables = []
        for v in data.get("variables", []):
            variables.append(PromptVariable(
                name=v["name"],
                type=VariableType(v.get("type", "string")),
                description=v.get("description"),
                required=v.get("required", True),
                default=v.get("default"),
            ))

        return cls(
            name=data["name"],
            template=data["template"],
            description=data.get("description"),
            variables=variables,
            category=data.get("category", "general"),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )
