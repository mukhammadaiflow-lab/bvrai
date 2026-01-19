"""
Prompt Template Management Module

This module provides comprehensive template management capabilities including
creation, storage, retrieval, versioning, and rendering of prompt templates.
"""

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .base import (
    PromptTemplate,
    TemplateCategory,
    TemplateVariable,
    VariableType,
    IndustryType,
    TemplateError,
    ValidationError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Template Storage Interface
# =============================================================================


class TemplateStorage(ABC):
    """Abstract base class for template storage backends."""

    @abstractmethod
    async def save(self, template: PromptTemplate) -> None:
        """Save a template."""
        pass

    @abstractmethod
    async def get(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        pass

    @abstractmethod
    async def get_by_category(
        self,
        organization_id: str,
        category: TemplateCategory,
        include_inactive: bool = False,
    ) -> List[PromptTemplate]:
        """Get templates by category."""
        pass

    @abstractmethod
    async def list(
        self,
        organization_id: str,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[PromptTemplate], int]:
        """List templates with pagination."""
        pass

    @abstractmethod
    async def delete(self, template_id: str) -> bool:
        """Delete a template."""
        pass

    @abstractmethod
    async def get_default(
        self,
        organization_id: str,
        category: TemplateCategory,
    ) -> Optional[PromptTemplate]:
        """Get default template for category."""
        pass


class InMemoryTemplateStorage(TemplateStorage):
    """In-memory template storage for testing and development."""

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._org_index: Dict[str, Set[str]] = {}
        self._category_index: Dict[str, Dict[TemplateCategory, Set[str]]] = {}

    async def save(self, template: PromptTemplate) -> None:
        """Save a template to memory."""
        template.updated_at = datetime.utcnow()
        self._templates[template.id] = template

        # Update indices
        if template.organization_id not in self._org_index:
            self._org_index[template.organization_id] = set()
        self._org_index[template.organization_id].add(template.id)

        if template.organization_id not in self._category_index:
            self._category_index[template.organization_id] = {}
        if template.category not in self._category_index[template.organization_id]:
            self._category_index[template.organization_id][template.category] = set()
        self._category_index[template.organization_id][template.category].add(template.id)

    async def get(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    async def get_by_category(
        self,
        organization_id: str,
        category: TemplateCategory,
        include_inactive: bool = False,
    ) -> List[PromptTemplate]:
        """Get templates by category."""
        if organization_id not in self._category_index:
            return []

        if category not in self._category_index[organization_id]:
            return []

        template_ids = self._category_index[organization_id][category]
        templates = []
        for tid in template_ids:
            template = self._templates.get(tid)
            if template and (include_inactive or template.is_active):
                templates.append(template)

        return sorted(templates, key=lambda t: t.updated_at, reverse=True)

    async def list(
        self,
        organization_id: str,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[PromptTemplate], int]:
        """List templates with pagination."""
        if organization_id not in self._org_index:
            return [], 0

        template_ids = self._org_index[organization_id]
        templates = []

        for tid in template_ids:
            template = self._templates.get(tid)
            if not template:
                continue

            # Apply filters
            if filters:
                if "category" in filters:
                    if template.category != filters["category"]:
                        continue
                if "is_active" in filters:
                    if template.is_active != filters["is_active"]:
                        continue
                if "industry" in filters:
                    if filters["industry"] not in template.industries:
                        continue
                if "search" in filters:
                    search = filters["search"].lower()
                    if search not in template.name.lower() and search not in template.description.lower():
                        continue
                if "tags" in filters:
                    if not any(tag in template.tags for tag in filters["tags"]):
                        continue

            templates.append(template)

        # Sort by updated_at descending
        templates.sort(key=lambda t: t.updated_at, reverse=True)

        total = len(templates)
        templates = templates[offset:offset + limit]

        return templates, total

    async def delete(self, template_id: str) -> bool:
        """Delete a template."""
        template = self._templates.get(template_id)
        if not template:
            return False

        # Remove from indices
        if template.organization_id in self._org_index:
            self._org_index[template.organization_id].discard(template_id)

        if template.organization_id in self._category_index:
            if template.category in self._category_index[template.organization_id]:
                self._category_index[template.organization_id][template.category].discard(template_id)

        del self._templates[template_id]
        return True

    async def get_default(
        self,
        organization_id: str,
        category: TemplateCategory,
    ) -> Optional[PromptTemplate]:
        """Get default template for category."""
        templates = await self.get_by_category(organization_id, category)
        for template in templates:
            if template.is_default:
                return template
        return templates[0] if templates else None


# =============================================================================
# Template Renderer
# =============================================================================


class TemplateRenderer:
    """Renders templates with variable substitution and conditional logic."""

    # Patterns for template syntax
    VARIABLE_PATTERN = re.compile(r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}')
    CONDITIONAL_PATTERN = re.compile(
        r'\{\%\s*if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\%\}(.*?)\{\%\s*endif\s*\%\}',
        re.DOTALL
    )
    LOOP_PATTERN = re.compile(
        r'\{\%\s*for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\%\}(.*?)\{\%\s*endfor\s*\%\}',
        re.DOTALL
    )
    ELSE_PATTERN = re.compile(
        r'\{\%\s*if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\%\}(.*?)\{\%\s*else\s*\%\}(.*?)\{\%\s*endif\s*\%\}',
        re.DOTALL
    )

    def __init__(
        self,
        custom_filters: Optional[Dict[str, Callable[[Any], str]]] = None,
    ):
        """
        Initialize renderer.

        Args:
            custom_filters: Custom filter functions for variable transformation
        """
        self.filters = {
            "upper": lambda x: str(x).upper(),
            "lower": lambda x: str(x).lower(),
            "title": lambda x: str(x).title(),
            "capitalize": lambda x: str(x).capitalize(),
            "strip": lambda x: str(x).strip(),
            "default": lambda x, d="": str(x) if x else d,
            "join": lambda x, sep=", ": sep.join(str(i) for i in x) if isinstance(x, list) else str(x),
            "truncate": lambda x, l=50: str(x)[:l] + "..." if len(str(x)) > l else str(x),
            "length": lambda x: str(len(x)) if hasattr(x, "__len__") else "0",
            "first": lambda x: str(x[0]) if isinstance(x, list) and x else "",
            "last": lambda x: str(x[-1]) if isinstance(x, list) and x else "",
        }
        if custom_filters:
            self.filters.update(custom_filters)

    def render(
        self,
        template: Union[str, PromptTemplate],
        values: Dict[str, Any],
        strict: bool = False,
    ) -> str:
        """
        Render a template with provided values.

        Args:
            template: Template string or PromptTemplate object
            values: Variable values for substitution
            strict: If True, raise error for undefined variables

        Returns:
            Rendered string
        """
        content = template.content if isinstance(template, PromptTemplate) else template

        # Process conditionals with else first
        content = self._process_conditionals_with_else(content, values)

        # Process simple conditionals
        content = self._process_conditionals(content, values)

        # Process loops
        content = self._process_loops(content, values)

        # Process variables with filters
        content = self._process_variables(content, values, strict)

        # Clean up extra whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()

        return content

    def _process_variables(
        self,
        content: str,
        values: Dict[str, Any],
        strict: bool,
    ) -> str:
        """Process variable substitutions."""
        def replace_variable(match: re.Match) -> str:
            var_expr = match.group(1)

            # Check for filter syntax: variable|filter
            if "|" in var_expr:
                parts = var_expr.split("|")
                var_name = parts[0].strip()
                filter_name = parts[1].strip()

                value = values.get(var_name)
                if value is None and strict:
                    raise TemplateError(f"Undefined variable: {var_name}")

                # Apply filter
                if filter_name in self.filters:
                    return self.filters[filter_name](value)
                else:
                    return str(value) if value is not None else ""
            else:
                var_name = var_expr.strip()
                value = values.get(var_name)

                if value is None:
                    if strict:
                        raise TemplateError(f"Undefined variable: {var_name}")
                    return ""

                # Convert value to string
                if isinstance(value, list):
                    return ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    return json.dumps(value)
                elif isinstance(value, datetime):
                    return value.strftime("%Y-%m-%d %H:%M")
                else:
                    return str(value)

        return self.VARIABLE_PATTERN.sub(replace_variable, content)

    def _process_conditionals(
        self,
        content: str,
        values: Dict[str, Any],
    ) -> str:
        """Process conditional blocks."""
        def replace_conditional(match: re.Match) -> str:
            var_name = match.group(1)
            inner_content = match.group(2)

            value = values.get(var_name)
            if value:
                return inner_content.strip()
            return ""

        return self.CONDITIONAL_PATTERN.sub(replace_conditional, content)

    def _process_conditionals_with_else(
        self,
        content: str,
        values: Dict[str, Any],
    ) -> str:
        """Process conditional blocks with else clauses."""
        def replace_conditional(match: re.Match) -> str:
            var_name = match.group(1)
            if_content = match.group(2)
            else_content = match.group(3)

            value = values.get(var_name)
            if value:
                return if_content.strip()
            return else_content.strip()

        return self.ELSE_PATTERN.sub(replace_conditional, content)

    def _process_loops(
        self,
        content: str,
        values: Dict[str, Any],
    ) -> str:
        """Process loop blocks."""
        def replace_loop(match: re.Match) -> str:
            item_name = match.group(1)
            list_name = match.group(2)
            inner_content = match.group(3)

            items = values.get(list_name, [])
            if not isinstance(items, list):
                return ""

            results = []
            for item in items:
                # Create local context with loop item
                local_values = values.copy()
                local_values[item_name] = item

                # Render inner content with item
                rendered = self._process_variables(inner_content, local_values, strict=False)
                results.append(rendered.strip())

            return "\n".join(results)

        return self.LOOP_PATTERN.sub(replace_loop, content)

    def extract_variables(self, content: str) -> Set[str]:
        """Extract all variable names from template content."""
        variables = set()

        # Extract from variable pattern
        for match in self.VARIABLE_PATTERN.finditer(content):
            var_expr = match.group(1)
            if "|" in var_expr:
                var_name = var_expr.split("|")[0].strip()
            else:
                var_name = var_expr.strip()
            variables.add(var_name)

        # Extract from conditionals
        for match in self.CONDITIONAL_PATTERN.finditer(content):
            variables.add(match.group(1))

        for match in self.ELSE_PATTERN.finditer(content):
            variables.add(match.group(1))

        # Extract from loops
        for match in self.LOOP_PATTERN.finditer(content):
            variables.add(match.group(2))

        return variables

    def validate_syntax(self, content: str) -> List[str]:
        """
        Validate template syntax.

        Returns:
            List of syntax errors
        """
        errors = []

        # Check for unmatched braces
        open_braces = content.count("{{")
        close_braces = content.count("}}")
        if open_braces != close_braces:
            errors.append(f"Unmatched braces: {open_braces} opening, {close_braces} closing")

        # Check for unmatched conditionals
        if_count = len(re.findall(r'\{\%\s*if\s+', content))
        endif_count = len(re.findall(r'\{\%\s*endif\s*\%\}', content))
        if if_count != endif_count:
            errors.append(f"Unmatched conditionals: {if_count} if, {endif_count} endif")

        # Check for unmatched loops
        for_count = len(re.findall(r'\{\%\s*for\s+', content))
        endfor_count = len(re.findall(r'\{\%\s*endfor\s*\%\}', content))
        if for_count != endfor_count:
            errors.append(f"Unmatched loops: {for_count} for, {endfor_count} endfor")

        # Check variable name validity
        for match in self.VARIABLE_PATTERN.finditer(content):
            var_expr = match.group(1)
            var_name = var_expr.split("|")[0].strip()
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
                errors.append(f"Invalid variable name: {var_name}")

        return errors


# =============================================================================
# Template Manager
# =============================================================================


class TemplateManager:
    """
    Manages prompt templates with CRUD operations, versioning, and rendering.
    """

    def __init__(
        self,
        storage: Optional[TemplateStorage] = None,
        renderer: Optional[TemplateRenderer] = None,
    ):
        """
        Initialize template manager.

        Args:
            storage: Storage backend
            renderer: Template renderer
        """
        self.storage = storage or InMemoryTemplateStorage()
        self.renderer = renderer or TemplateRenderer()

        # Cache for frequently used templates
        self._cache: Dict[str, Tuple[PromptTemplate, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def create_template(
        self,
        name: str,
        category: TemplateCategory,
        content: str,
        organization_id: str,
        description: str = "",
        variables: Optional[List[TemplateVariable]] = None,
        industries: Optional[List[IndustryType]] = None,
        tags: Optional[List[str]] = None,
        is_default: bool = False,
        created_by: Optional[str] = None,
    ) -> PromptTemplate:
        """
        Create a new template.

        Args:
            name: Template name
            category: Template category
            content: Template content with variables
            organization_id: Organization ID
            description: Template description
            variables: Variable definitions
            industries: Target industries
            tags: Tags for organization
            is_default: Whether this is the default for category
            created_by: User who created the template

        Returns:
            Created template
        """
        # Validate syntax
        errors = self.renderer.validate_syntax(content)
        if errors:
            raise ValidationError(f"Template syntax errors: {'; '.join(errors)}")

        # Create template
        template = PromptTemplate(
            id=f"tmpl_{uuid.uuid4().hex[:24]}",
            name=name,
            category=category,
            content=content,
            organization_id=organization_id,
            description=description,
            variables=variables or [],
            industries=industries or [],
            tags=tags or [],
            is_default=is_default,
            created_by=created_by,
            updated_by=created_by,
        )

        # If setting as default, unset other defaults
        if is_default:
            await self._unset_default(organization_id, category)

        # Save
        await self.storage.save(template)

        logger.info(f"Created template: {template.id} ({name})")
        return template

    async def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get a template by ID.

        Args:
            template_id: Template ID

        Returns:
            Template or None
        """
        # Check cache
        if template_id in self._cache:
            template, cached_at = self._cache[template_id]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return template

        # Get from storage
        template = await self.storage.get(template_id)

        # Update cache
        if template:
            self._cache[template_id] = (template, datetime.utcnow())

        return template

    async def update_template(
        self,
        template_id: str,
        updates: Dict[str, Any],
        updated_by: Optional[str] = None,
    ) -> PromptTemplate:
        """
        Update a template.

        Args:
            template_id: Template ID
            updates: Fields to update
            updated_by: User making the update

        Returns:
            Updated template
        """
        template = await self.storage.get(template_id)
        if not template:
            raise TemplateError(f"Template not found: {template_id}")

        # Validate content if being updated
        if "content" in updates:
            errors = self.renderer.validate_syntax(updates["content"])
            if errors:
                raise ValidationError(f"Template syntax errors: {'; '.join(errors)}")

        # Apply updates
        if "name" in updates:
            template.name = updates["name"]
        if "content" in updates:
            template.content = updates["content"]
            template._extract_variables_from_content()
        if "description" in updates:
            template.description = updates["description"]
        if "variables" in updates:
            template.variables = [
                TemplateVariable.from_dict(v) if isinstance(v, dict) else v
                for v in updates["variables"]
            ]
        if "industries" in updates:
            template.industries = [
                IndustryType(i) if isinstance(i, str) else i
                for i in updates["industries"]
            ]
        if "tags" in updates:
            template.tags = updates["tags"]
        if "is_active" in updates:
            template.is_active = updates["is_active"]
        if "is_default" in updates:
            if updates["is_default"]:
                await self._unset_default(template.organization_id, template.category)
            template.is_default = updates["is_default"]

        template.version += 1
        template.updated_at = datetime.utcnow()
        template.updated_by = updated_by

        # Save
        await self.storage.save(template)

        # Invalidate cache
        if template_id in self._cache:
            del self._cache[template_id]

        logger.info(f"Updated template: {template_id}")
        return template

    async def delete_template(self, template_id: str) -> bool:
        """
        Delete a template.

        Args:
            template_id: Template ID

        Returns:
            True if deleted
        """
        success = await self.storage.delete(template_id)

        # Invalidate cache
        if template_id in self._cache:
            del self._cache[template_id]

        if success:
            logger.info(f"Deleted template: {template_id}")

        return success

    async def list_templates(
        self,
        organization_id: str,
        category: Optional[TemplateCategory] = None,
        industry: Optional[IndustryType] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[PromptTemplate], int]:
        """
        List templates with filtering.

        Args:
            organization_id: Organization ID
            category: Filter by category
            industry: Filter by industry
            tags: Filter by tags
            search: Search in name/description
            is_active: Filter by active status
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (templates, total_count)
        """
        filters = {}
        if category:
            filters["category"] = category
        if industry:
            filters["industry"] = industry
        if tags:
            filters["tags"] = tags
        if search:
            filters["search"] = search
        if is_active is not None:
            filters["is_active"] = is_active

        return await self.storage.list(
            organization_id=organization_id,
            filters=filters if filters else None,
            offset=offset,
            limit=limit,
        )

    async def get_templates_by_category(
        self,
        organization_id: str,
        category: TemplateCategory,
        include_inactive: bool = False,
    ) -> List[PromptTemplate]:
        """
        Get all templates for a category.

        Args:
            organization_id: Organization ID
            category: Template category
            include_inactive: Include inactive templates

        Returns:
            List of templates
        """
        return await self.storage.get_by_category(
            organization_id=organization_id,
            category=category,
            include_inactive=include_inactive,
        )

    async def get_default_template(
        self,
        organization_id: str,
        category: TemplateCategory,
    ) -> Optional[PromptTemplate]:
        """
        Get default template for a category.

        Args:
            organization_id: Organization ID
            category: Template category

        Returns:
            Default template or None
        """
        return await self.storage.get_default(organization_id, category)

    async def clone_template(
        self,
        template_id: str,
        new_name: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> PromptTemplate:
        """
        Clone a template.

        Args:
            template_id: Template ID to clone
            new_name: Name for the clone
            created_by: User creating the clone

        Returns:
            Cloned template
        """
        original = await self.storage.get(template_id)
        if not original:
            raise TemplateError(f"Template not found: {template_id}")

        clone = original.clone(new_name)
        clone.created_by = created_by
        clone.updated_by = created_by

        await self.storage.save(clone)

        logger.info(f"Cloned template: {template_id} -> {clone.id}")
        return clone

    def render_template(
        self,
        template: Union[str, PromptTemplate],
        values: Dict[str, Any],
        strict: bool = False,
    ) -> str:
        """
        Render a template with values.

        Args:
            template: Template or template content
            values: Variable values
            strict: Raise error for undefined variables

        Returns:
            Rendered string
        """
        return self.renderer.render(template, values, strict)

    async def render_by_id(
        self,
        template_id: str,
        values: Dict[str, Any],
        strict: bool = False,
    ) -> str:
        """
        Render a template by ID.

        Args:
            template_id: Template ID
            values: Variable values
            strict: Raise error for undefined variables

        Returns:
            Rendered string
        """
        template = await self.get_template(template_id)
        if not template:
            raise TemplateError(f"Template not found: {template_id}")

        return self.render_template(template, values, strict)

    async def render_default(
        self,
        organization_id: str,
        category: TemplateCategory,
        values: Dict[str, Any],
        strict: bool = False,
    ) -> str:
        """
        Render the default template for a category.

        Args:
            organization_id: Organization ID
            category: Template category
            values: Variable values
            strict: Raise error for undefined variables

        Returns:
            Rendered string
        """
        template = await self.get_default_template(organization_id, category)
        if not template:
            raise TemplateError(f"No default template for category: {category.value}")

        return self.render_template(template, values, strict)

    def validate_template(self, content: str) -> List[str]:
        """
        Validate template syntax.

        Args:
            content: Template content

        Returns:
            List of validation errors
        """
        return self.renderer.validate_syntax(content)

    def extract_variables(self, content: str) -> Set[str]:
        """
        Extract variable names from template.

        Args:
            content: Template content

        Returns:
            Set of variable names
        """
        return self.renderer.extract_variables(content)

    async def _unset_default(
        self,
        organization_id: str,
        category: TemplateCategory,
    ) -> None:
        """Unset default flag for all templates in category."""
        templates = await self.storage.get_by_category(
            organization_id=organization_id,
            category=category,
            include_inactive=True,
        )

        for template in templates:
            if template.is_default:
                template.is_default = False
                await self.storage.save(template)
                if template.id in self._cache:
                    del self._cache[template.id]


# =============================================================================
# Template Library
# =============================================================================


class TemplateLibrary:
    """
    Library of built-in and shared templates.

    Provides default templates for common use cases and allows
    organizations to share templates.
    """

    def __init__(self, manager: TemplateManager):
        """
        Initialize template library.

        Args:
            manager: Template manager instance
        """
        self.manager = manager
        self._builtin_templates: Dict[TemplateCategory, List[Dict[str, Any]]] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """Load built-in template definitions."""
        self._builtin_templates = {
            TemplateCategory.SYSTEM_PROMPT: [
                {
                    "name": "Professional Assistant",
                    "description": "A professional, helpful assistant for general business use",
                    "content": """You are {{agent_name}}, a {{agent_role}} at {{company_name}}.

Your primary responsibility is to assist callers with their inquiries professionally and efficiently.

Key behaviors:
- Be professional yet approachable
- Listen carefully and ask clarifying questions when needed
- Provide accurate information based on your knowledge
- If you cannot help with something, offer to connect the caller with someone who can

{% if industry %}
You specialize in {{industry}} and have expertise in this domain.
{% endif %}

{% if special_instructions %}
Additional instructions: {{special_instructions}}
{% endif %}

Keep your responses concise and natural for voice conversation.""",
                    "industries": [],
                },
                {
                    "name": "Healthcare Receptionist",
                    "description": "HIPAA-compliant healthcare scheduling assistant",
                    "content": """You are {{agent_name}}, a {{agent_role}} at {{company_name}}.

You assist patients with scheduling appointments, answering general questions about services, and directing calls to the appropriate department.

IMPORTANT COMPLIANCE REQUIREMENTS:
- Never share specific patient health information over the phone without proper verification
- Direct any clinical questions to appropriate medical staff
- Maintain patient privacy at all times

Your capabilities:
- Schedule, reschedule, or cancel appointments
- Provide office hours and location information
- Answer questions about services offered
- Transfer calls to clinical staff when appropriate

{% if accepted_insurance %}
We accept the following insurance plans: {{accepted_insurance|join}}
{% endif %}

Be warm, empathetic, and professional. Patients may be anxious or unwell.""",
                    "industries": [IndustryType.HEALTHCARE],
                },
                {
                    "name": "Real Estate Agent",
                    "description": "Real estate inquiry and scheduling assistant",
                    "content": """You are {{agent_name}}, a {{agent_role}} at {{company_name}}.

You help potential buyers and sellers with property inquiries, schedule showings, and provide information about listings.

Your responsibilities:
- Answer questions about available properties
- Schedule property viewings
- Collect lead information from interested parties
- Provide basic information about the buying/selling process

{% if featured_listings %}
Current featured listings to mention when appropriate:
{% for listing in featured_listings %}
- {{listing}}
{% endfor %}
{% endif %}

Be enthusiastic but not pushy. Focus on understanding the caller's needs and preferences.""",
                    "industries": [IndustryType.REAL_ESTATE],
                },
            ],
            TemplateCategory.GREETING: [
                {
                    "name": "Warm Greeting",
                    "description": "A warm, friendly greeting",
                    "content": "Hi there! This is {{agent_name}} from {{company_name}}. How can I help you today?",
                    "industries": [],
                },
                {
                    "name": "Professional Greeting",
                    "description": "A formal, professional greeting",
                    "content": "Good {{time_of_day}}. You've reached {{company_name}}. My name is {{agent_name}}. How may I assist you?",
                    "industries": [],
                },
                {
                    "name": "Casual Greeting",
                    "description": "A casual, relaxed greeting",
                    "content": "Hey! {{agent_name}} here from {{company_name}}. What can I do for you?",
                    "industries": [],
                },
            ],
            TemplateCategory.FAREWELL: [
                {
                    "name": "Standard Farewell",
                    "description": "A standard goodbye message",
                    "content": "Thank you for calling {{company_name}}. Have a great {{time_of_day}}!",
                    "industries": [],
                },
                {
                    "name": "Appointment Confirmation Farewell",
                    "description": "Farewell after scheduling an appointment",
                    "content": "Great! Your appointment is confirmed for {{appointment_time}}. We'll see you then! Thank you for calling {{company_name}}.",
                    "industries": [],
                },
            ],
            TemplateCategory.TRANSFER: [
                {
                    "name": "Standard Transfer",
                    "description": "Standard transfer announcement",
                    "content": "I'll connect you with {{transfer_target}} now. Please hold for just a moment.",
                    "industries": [],
                },
                {
                    "name": "Warm Transfer",
                    "description": "Transfer with context",
                    "content": "Let me connect you with {{transfer_target}} who can better assist you with {{reason}}. I'll briefly explain your situation to them. Please hold.",
                    "industries": [],
                },
            ],
            TemplateCategory.HOLD: [
                {
                    "name": "Brief Hold",
                    "description": "Request for brief hold",
                    "content": "Would you mind holding for just a moment while I look into that for you?",
                    "industries": [],
                },
                {
                    "name": "Extended Hold",
                    "description": "Request for extended hold with estimate",
                    "content": "I need to check on a few things for you. Would you be able to hold for about {{estimated_wait}} minutes?",
                    "industries": [],
                },
            ],
            TemplateCategory.CLARIFICATION: [
                {
                    "name": "General Clarification",
                    "description": "Ask for clarification",
                    "content": "I want to make sure I understand correctly. You're asking about {{topic}}, is that right?",
                    "industries": [],
                },
                {
                    "name": "Detailed Clarification",
                    "description": "Request more details",
                    "content": "Could you tell me a bit more about {{topic}}? That will help me assist you better.",
                    "industries": [],
                },
            ],
            TemplateCategory.APOLOGY: [
                {
                    "name": "General Apology",
                    "description": "Standard apology",
                    "content": "I apologize for the inconvenience. Let me see what I can do to help resolve this for you.",
                    "industries": [],
                },
                {
                    "name": "Wait Apology",
                    "description": "Apology for wait time",
                    "content": "Thank you so much for your patience. I apologize for the wait. How can I assist you?",
                    "industries": [],
                },
            ],
            TemplateCategory.ERROR_RECOVERY: [
                {
                    "name": "Misunderstanding",
                    "description": "Recovery from misunderstanding",
                    "content": "I'm sorry, I didn't quite catch that. Could you please repeat that for me?",
                    "industries": [],
                },
                {
                    "name": "Technical Issue",
                    "description": "Recovery from technical issue",
                    "content": "I apologize, I'm having a bit of trouble. Let me try that again. You were asking about {{topic}}?",
                    "industries": [],
                },
            ],
            TemplateCategory.OBJECTION_HANDLING: [
                {
                    "name": "Price Objection",
                    "description": "Handling price concerns",
                    "content": "I understand that price is an important consideration. Let me share some options that might work better for your budget, and explain the value you'd be getting.",
                    "industries": [],
                },
                {
                    "name": "Timing Objection",
                    "description": "Handling timing concerns",
                    "content": "I completely understand. There's no pressure to decide today. What if we schedule a follow-up call for a time that works better for you?",
                    "industries": [],
                },
            ],
            TemplateCategory.SCHEDULING: [
                {
                    "name": "Appointment Offer",
                    "description": "Offering appointment times",
                    "content": "I have availability on {{available_dates}}. Would any of those work for you?",
                    "industries": [],
                },
                {
                    "name": "Appointment Confirmation",
                    "description": "Confirming appointment details",
                    "content": "Perfect! I've got you scheduled for {{appointment_time}} with {{provider_name}}. We're located at {{address}}. Is there anything else you need?",
                    "industries": [],
                },
            ],
            TemplateCategory.INFORMATION: [
                {
                    "name": "Hours of Operation",
                    "description": "Business hours information",
                    "content": "Our hours are {{hours}}. {% if holiday_hours %}Please note, our holiday hours may vary.{% endif %}",
                    "industries": [],
                },
                {
                    "name": "Location Information",
                    "description": "Location and directions",
                    "content": "We're located at {{address}}. {% if parking_info %}{{parking_info}}{% endif %} {% if landmarks %}Look for {{landmarks}}.{% endif %}",
                    "industries": [],
                },
            ],
        }

    async def install_builtin_templates(
        self,
        organization_id: str,
        categories: Optional[List[TemplateCategory]] = None,
        industries: Optional[List[IndustryType]] = None,
        created_by: Optional[str] = None,
    ) -> List[PromptTemplate]:
        """
        Install built-in templates for an organization.

        Args:
            organization_id: Organization ID
            categories: Categories to install (all if None)
            industries: Industries to filter by (all if None)
            created_by: User installing templates

        Returns:
            List of installed templates
        """
        installed = []
        categories_to_install = categories or list(self._builtin_templates.keys())

        for category in categories_to_install:
            if category not in self._builtin_templates:
                continue

            for template_def in self._builtin_templates[category]:
                # Filter by industry if specified
                template_industries = template_def.get("industries", [])
                if industries and template_industries:
                    if not any(i in template_industries for i in industries):
                        continue

                # Create template
                template = await self.manager.create_template(
                    name=template_def["name"],
                    category=category,
                    content=template_def["content"],
                    organization_id=organization_id,
                    description=template_def.get("description", ""),
                    industries=template_industries,
                    tags=["builtin"],
                    is_default=len(installed) == 0,  # First template in category is default
                    created_by=created_by,
                )
                installed.append(template)

        logger.info(f"Installed {len(installed)} built-in templates for org {organization_id}")
        return installed

    def get_builtin_template_names(
        self,
        category: Optional[TemplateCategory] = None,
    ) -> Dict[TemplateCategory, List[str]]:
        """
        Get names of available built-in templates.

        Args:
            category: Filter by category

        Returns:
            Dictionary of category -> template names
        """
        result = {}
        categories = [category] if category else list(self._builtin_templates.keys())

        for cat in categories:
            if cat in self._builtin_templates:
                result[cat] = [t["name"] for t in self._builtin_templates[cat]]

        return result


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "TemplateStorage",
    "InMemoryTemplateStorage",
    "TemplateRenderer",
    "TemplateManager",
    "TemplateLibrary",
]
