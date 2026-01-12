"""
Workflow Variables System

Variable management for workflows:
- Variable scoping
- Type validation
- Expression evaluation
- Variable transformation
"""

from typing import Optional, Dict, Any, List, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, date, time
from enum import Enum
from abc import ABC, abstractmethod
import re
import json
import logging

logger = logging.getLogger(__name__)


class VariableType(str, Enum):
    """Variable data types."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


class VariableScope(str, Enum):
    """Variable scopes."""
    LOCAL = "local"
    WORKFLOW = "workflow"
    GLOBAL = "global"
    SESSION = "session"
    SYSTEM = "system"


@dataclass
class VariableDefinition:
    """Definition for a workflow variable."""
    name: str
    var_type: VariableType = VariableType.ANY
    scope: VariableScope = VariableScope.WORKFLOW
    default_value: Any = None
    description: str = ""
    required: bool = False
    readonly: bool = False
    sensitive: bool = False  # For masking in logs
    validators: List[str] = field(default_factory=list)
    transformers: List[str] = field(default_factory=list)

    def validate(self, value: Any) -> bool:
        """Validate value against definition."""
        if value is None:
            return not self.required

        return self._type_check(value)

    def _type_check(self, value: Any) -> bool:
        """Check value type."""
        type_checks = {
            VariableType.STRING: lambda v: isinstance(v, str),
            VariableType.NUMBER: lambda v: isinstance(v, (int, float)),
            VariableType.INTEGER: lambda v: isinstance(v, int),
            VariableType.FLOAT: lambda v: isinstance(v, float),
            VariableType.BOOLEAN: lambda v: isinstance(v, bool),
            VariableType.ARRAY: lambda v: isinstance(v, list),
            VariableType.OBJECT: lambda v: isinstance(v, dict),
            VariableType.ANY: lambda v: True,
        }

        checker = type_checks.get(self.var_type, lambda v: True)
        return checker(value)

    def coerce(self, value: Any) -> Any:
        """Coerce value to expected type."""
        if value is None:
            return self.default_value

        try:
            if self.var_type == VariableType.STRING:
                return str(value)
            elif self.var_type == VariableType.INTEGER:
                return int(value)
            elif self.var_type == VariableType.FLOAT:
                return float(value)
            elif self.var_type == VariableType.NUMBER:
                return float(value) if '.' in str(value) else int(value)
            elif self.var_type == VariableType.BOOLEAN:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            elif self.var_type == VariableType.ARRAY:
                if isinstance(value, str):
                    return json.loads(value)
                return list(value)
            elif self.var_type == VariableType.OBJECT:
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

        return value


@dataclass
class Variable:
    """Runtime variable instance."""
    definition: VariableDefinition
    value: Any = None
    is_set: bool = False
    last_modified: Optional[datetime] = None
    modified_by: str = ""  # Node ID that modified it

    def get(self) -> Any:
        """Get variable value."""
        if not self.is_set:
            return self.definition.default_value
        return self.value

    def set(self, value: Any, modified_by: str = "") -> bool:
        """Set variable value."""
        if self.definition.readonly and self.is_set:
            logger.warning(f"Attempted to modify readonly variable: {self.definition.name}")
            return False

        # Validate
        if not self.definition.validate(value):
            logger.warning(f"Invalid value for variable {self.definition.name}: {value}")
            return False

        # Coerce and set
        self.value = self.definition.coerce(value)
        self.is_set = True
        self.last_modified = datetime.utcnow()
        self.modified_by = modified_by
        return True

    def reset(self) -> None:
        """Reset to default value."""
        self.value = self.definition.default_value
        self.is_set = False
        self.last_modified = None
        self.modified_by = ""


class VariableStore:
    """
    Store for managing workflow variables.

    Features:
    - Scoped variable access
    - Type validation
    - Change tracking
    """

    def __init__(self):
        self._variables: Dict[str, Variable] = {}
        self._scopes: Dict[VariableScope, Dict[str, Variable]] = {
            scope: {} for scope in VariableScope
        }
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100

    def define(self, definition: VariableDefinition) -> None:
        """Define a variable."""
        var = Variable(definition=definition)
        self._variables[definition.name] = var
        self._scopes[definition.scope][definition.name] = var

    def define_many(self, definitions: List[VariableDefinition]) -> None:
        """Define multiple variables."""
        for defn in definitions:
            self.define(defn)

    def get(self, name: str, default: Any = None) -> Any:
        """Get variable value."""
        var = self._variables.get(name)
        if var:
            return var.get()
        return default

    def set(self, name: str, value: Any, modified_by: str = "") -> bool:
        """Set variable value."""
        var = self._variables.get(name)
        if not var:
            # Auto-create variable
            defn = VariableDefinition(name=name)
            var = Variable(definition=defn)
            self._variables[name] = var
            self._scopes[VariableScope.WORKFLOW][name] = var

        old_value = var.value
        success = var.set(value, modified_by)

        if success:
            self._record_change(name, old_value, value, modified_by)

        return success

    def delete(self, name: str) -> bool:
        """Delete variable."""
        var = self._variables.pop(name, None)
        if var:
            self._scopes[var.definition.scope].pop(name, None)
            return True
        return False

    def exists(self, name: str) -> bool:
        """Check if variable exists."""
        return name in self._variables

    def get_scope(self, scope: VariableScope) -> Dict[str, Any]:
        """Get all variables in scope."""
        return {
            name: var.get()
            for name, var in self._scopes[scope].items()
        }

    def get_all(self) -> Dict[str, Any]:
        """Get all variable values."""
        return {name: var.get() for name, var in self._variables.items()}

    def _record_change(
        self,
        name: str,
        old_value: Any,
        new_value: Any,
        modified_by: str,
    ) -> None:
        """Record variable change in history."""
        self._history.append({
            "name": name,
            "old_value": old_value,
            "new_value": new_value,
            "modified_by": modified_by,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Limit history size
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get variable change history."""
        if name:
            return [h for h in self._history if h["name"] == name]
        return self._history.copy()

    def clear(self, scope: Optional[VariableScope] = None) -> None:
        """Clear variables, optionally by scope."""
        if scope:
            for name in list(self._scopes[scope].keys()):
                self.delete(name)
        else:
            self._variables.clear()
            self._scopes = {s: {} for s in VariableScope}
            self._history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export variables to dictionary."""
        return {
            name: {
                "value": var.get(),
                "type": var.definition.var_type.value,
                "scope": var.definition.scope.value,
                "is_set": var.is_set,
            }
            for name, var in self._variables.items()
            if not var.definition.sensitive
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Import variables from dictionary."""
        for name, value in data.items():
            if isinstance(value, dict) and "value" in value:
                self.set(name, value["value"])
            else:
                self.set(name, value)


class ExpressionEvaluator:
    """
    Evaluates expressions with variable substitution.

    Features:
    - Variable interpolation
    - Function calls
    - Safe evaluation
    """

    def __init__(self, variable_store: VariableStore):
        self.store = variable_store
        self._functions: Dict[str, Callable] = self._register_default_functions()

    def _register_default_functions(self) -> Dict[str, Callable]:
        """Register default expression functions."""
        return {
            # String functions
            "upper": lambda s: str(s).upper(),
            "lower": lambda s: str(s).lower(),
            "trim": lambda s: str(s).strip(),
            "length": lambda s: len(s),
            "substring": lambda s, start, end=None: s[start:end],
            "replace": lambda s, old, new: str(s).replace(old, new),
            "split": lambda s, sep: str(s).split(sep),
            "join": lambda arr, sep: sep.join(str(x) for x in arr),
            "contains": lambda s, sub: sub in str(s),
            "startswith": lambda s, prefix: str(s).startswith(prefix),
            "endswith": lambda s, suffix: str(s).endswith(suffix),

            # Number functions
            "abs": abs,
            "round": round,
            "floor": lambda x: int(x),
            "ceil": lambda x: int(x) + (1 if x > int(x) else 0),
            "min": min,
            "max": max,
            "sum": sum,

            # Array functions
            "first": lambda arr: arr[0] if arr else None,
            "last": lambda arr: arr[-1] if arr else None,
            "count": len,
            "reverse": lambda arr: list(reversed(arr)),
            "sort": lambda arr: sorted(arr),
            "unique": lambda arr: list(set(arr)),
            "filter": lambda arr, pred: [x for x in arr if pred(x)],

            # Date/Time functions
            "now": lambda: datetime.utcnow().isoformat(),
            "today": lambda: date.today().isoformat(),
            "format_date": lambda d, fmt: d.strftime(fmt) if hasattr(d, 'strftime') else str(d),

            # Type conversions
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "json": json.dumps,
            "parse_json": json.loads,

            # Conditional
            "if": lambda cond, true_val, false_val: true_val if cond else false_val,
            "default": lambda val, default: val if val is not None else default,
            "coalesce": lambda *args: next((a for a in args if a is not None), None),
        }

    def register_function(self, name: str, func: Callable) -> None:
        """Register custom function."""
        self._functions[name] = func

    def evaluate(self, expression: str) -> Any:
        """Evaluate expression."""
        if not expression:
            return expression

        # Check if entire expression is a variable reference
        var_match = re.match(r'^\{\{(\w+(?:\.\w+)*)\}\}$', expression)
        if var_match:
            return self._resolve_variable_path(var_match.group(1))

        # Interpolate variables in string
        result = self._interpolate_variables(expression)

        # Check for function calls
        func_match = re.match(r'^(\w+)\((.*)\)$', result)
        if func_match:
            func_name = func_match.group(1)
            if func_name in self._functions:
                args = self._parse_function_args(func_match.group(2))
                return self._functions[func_name](*args)

        return result

    def _interpolate_variables(self, expression: str) -> str:
        """Replace {{variable}} with values."""
        pattern = r'\{\{(\w+(?:\.\w+)*)\}\}'

        def replacer(match):
            var_path = match.group(1)
            value = self._resolve_variable_path(var_path)
            return str(value) if value is not None else ""

        return re.sub(pattern, replacer, expression)

    def _resolve_variable_path(self, path: str) -> Any:
        """Resolve variable path like 'object.property.subproperty'."""
        parts = path.split('.')
        value = self.store.get(parts[0])

        for part in parts[1:]:
            if value is None:
                return None

            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list):
                try:
                    index = int(part)
                    value = value[index] if 0 <= index < len(value) else None
                except (ValueError, IndexError):
                    value = None
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                value = None

        return value

    def _parse_function_args(self, args_str: str) -> List[Any]:
        """Parse function arguments."""
        if not args_str.strip():
            return []

        args = []
        current = ""
        depth = 0

        for char in args_str:
            if char == '(' or char == '[' or char == '{':
                depth += 1
                current += char
            elif char == ')' or char == ']' or char == '}':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                args.append(self._parse_arg(current.strip()))
                current = ""
            else:
                current += char

        if current.strip():
            args.append(self._parse_arg(current.strip()))

        return args

    def _parse_arg(self, arg: str) -> Any:
        """Parse single argument."""
        # Variable reference
        if arg.startswith('{{') and arg.endswith('}}'):
            return self._resolve_variable_path(arg[2:-2])

        # String literal
        if (arg.startswith('"') and arg.endswith('"')) or \
           (arg.startswith("'") and arg.endswith("'")):
            return arg[1:-1]

        # Boolean
        if arg.lower() == 'true':
            return True
        if arg.lower() == 'false':
            return False

        # Null
        if arg.lower() == 'null' or arg.lower() == 'none':
            return None

        # Number
        try:
            if '.' in arg:
                return float(arg)
            return int(arg)
        except ValueError:
            pass

        # JSON
        try:
            return json.loads(arg)
        except json.JSONDecodeError:
            pass

        # Variable name (without brackets)
        return self.store.get(arg, arg)

    def evaluate_condition(self, condition: str) -> bool:
        """Evaluate boolean condition."""
        # Interpolate variables first
        resolved = self._interpolate_variables(condition)

        # Parse comparison operators
        operators = [
            ('==', lambda a, b: a == b),
            ('!=', lambda a, b: a != b),
            ('>=', lambda a, b: float(a) >= float(b)),
            ('<=', lambda a, b: float(a) <= float(b)),
            ('>', lambda a, b: float(a) > float(b)),
            ('<', lambda a, b: float(a) < float(b)),
            (' contains ', lambda a, b: b in str(a)),
            (' in ', lambda a, b: a in b),
            (' startswith ', lambda a, b: str(a).startswith(str(b))),
            (' endswith ', lambda a, b: str(a).endswith(str(b))),
        ]

        for op, func in operators:
            if op in resolved:
                parts = resolved.split(op, 1)
                if len(parts) == 2:
                    left = self._parse_arg(parts[0].strip())
                    right = self._parse_arg(parts[1].strip())
                    try:
                        return func(left, right)
                    except (ValueError, TypeError):
                        return False

        # Boolean value
        return resolved.lower() in ('true', '1', 'yes')


class VariableTransformer:
    """
    Transform variable values.

    Features:
    - Built-in transformations
    - Custom transformers
    - Chained transformations
    """

    def __init__(self):
        self._transformers: Dict[str, Callable[[Any], Any]] = {
            # String transformations
            "uppercase": lambda v: str(v).upper(),
            "lowercase": lambda v: str(v).lower(),
            "capitalize": lambda v: str(v).capitalize(),
            "trim": lambda v: str(v).strip(),
            "slug": lambda v: re.sub(r'[^a-z0-9]+', '-', str(v).lower()).strip('-'),

            # Number transformations
            "round": lambda v: round(float(v)),
            "floor": lambda v: int(float(v)),
            "abs": lambda v: abs(float(v)),
            "currency": lambda v: f"${float(v):.2f}",
            "percent": lambda v: f"{float(v) * 100:.1f}%",

            # Date transformations
            "date_iso": lambda v: v.isoformat() if hasattr(v, 'isoformat') else str(v),
            "date_short": lambda v: v.strftime('%m/%d/%Y') if hasattr(v, 'strftime') else str(v),
            "date_long": lambda v: v.strftime('%B %d, %Y') if hasattr(v, 'strftime') else str(v),

            # Phone formatting
            "phone_format": lambda v: self._format_phone(v),
            "phone_e164": lambda v: self._format_phone_e164(v),

            # Masking
            "mask": lambda v: '*' * len(str(v)),
            "mask_partial": lambda v: str(v)[:2] + '*' * (len(str(v)) - 4) + str(v)[-2:] if len(str(v)) > 4 else '*' * len(str(v)),
            "mask_email": lambda v: self._mask_email(v),

            # JSON
            "json_encode": lambda v: json.dumps(v),
            "json_decode": lambda v: json.loads(v) if isinstance(v, str) else v,

            # Arrays
            "first": lambda v: v[0] if isinstance(v, list) and v else None,
            "last": lambda v: v[-1] if isinstance(v, list) and v else None,
            "join": lambda v: ', '.join(str(x) for x in v) if isinstance(v, list) else str(v),
            "sort": lambda v: sorted(v) if isinstance(v, list) else v,
            "unique": lambda v: list(set(v)) if isinstance(v, list) else v,
            "reverse": lambda v: list(reversed(v)) if isinstance(v, list) else v,
        }

    def _format_phone(self, value: Any) -> str:
        """Format phone number."""
        digits = re.sub(r'\D', '', str(value))
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        return str(value)

    def _format_phone_e164(self, value: Any) -> str:
        """Format phone to E.164."""
        digits = re.sub(r'\D', '', str(value))
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+{digits}"
        return f"+{digits}"

    def _mask_email(self, value: Any) -> str:
        """Mask email address."""
        email = str(value)
        if '@' in email:
            local, domain = email.split('@')
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1] if len(local) > 2 else '*' * len(local)
            return f"{masked_local}@{domain}"
        return '*' * len(email)

    def register(self, name: str, transformer: Callable[[Any], Any]) -> None:
        """Register custom transformer."""
        self._transformers[name] = transformer

    def transform(self, value: Any, transformer_name: str) -> Any:
        """Apply single transformation."""
        transformer = self._transformers.get(transformer_name)
        if transformer:
            try:
                return transformer(value)
            except Exception as e:
                logger.warning(f"Transformation error ({transformer_name}): {e}")
                return value
        return value

    def transform_chain(self, value: Any, transformers: List[str]) -> Any:
        """Apply chain of transformations."""
        result = value
        for transformer_name in transformers:
            result = self.transform(result, transformer_name)
        return result

    def list_transformers(self) -> List[str]:
        """List available transformers."""
        return list(self._transformers.keys())


# Singleton instances
_variable_store: Optional[VariableStore] = None
_expression_evaluator: Optional[ExpressionEvaluator] = None
_variable_transformer: Optional[VariableTransformer] = None


def get_variable_store() -> VariableStore:
    """Get singleton variable store."""
    global _variable_store
    if _variable_store is None:
        _variable_store = VariableStore()
    return _variable_store


def get_expression_evaluator() -> ExpressionEvaluator:
    """Get singleton expression evaluator."""
    global _expression_evaluator
    if _expression_evaluator is None:
        _expression_evaluator = ExpressionEvaluator(get_variable_store())
    return _expression_evaluator


def get_variable_transformer() -> VariableTransformer:
    """Get singleton variable transformer."""
    global _variable_transformer
    if _variable_transformer is None:
        _variable_transformer = VariableTransformer()
    return _variable_transformer
