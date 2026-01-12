"""
Input Validation and Sanitization

Security validation with:
- Input sanitization
- Injection detection (SQL, XSS, Command)
- Schema validation
- Content filtering
"""

from typing import Optional, Dict, Any, List, Union, Pattern, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import re
import html
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class InjectionType(str, Enum):
    """Types of injection attacks."""
    SQL = "sql"
    XSS = "xss"
    COMMAND = "command"
    LDAP = "ldap"
    XPATH = "xpath"
    TEMPLATE = "template"
    HEADER = "header"
    PATH_TRAVERSAL = "path_traversal"


@dataclass
class ValidationIssue:
    """A validation issue found."""
    field: str
    message: str
    severity: ValidationSeverity
    injection_type: Optional[InjectionType] = None
    value_preview: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None

    def add_issue(
        self,
        field: str,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        injection_type: Optional[InjectionType] = None,
        value: Optional[Any] = None,
    ) -> None:
        """Add an issue."""
        preview = None
        if value is not None:
            preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)

        self.issues.append(ValidationIssue(
            field=field,
            message=message,
            severity=severity,
            injection_type=injection_type,
            value_preview=preview,
        ))

        if severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.is_valid = False

    def has_critical(self) -> bool:
        """Check for critical issues."""
        return any(i.severity == ValidationSeverity.CRITICAL for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": [
                {
                    "field": i.field,
                    "message": i.message,
                    "severity": i.severity.value,
                    "injection_type": i.injection_type.value if i.injection_type else None,
                }
                for i in self.issues
            ],
        }


@dataclass
class ValidationRule:
    """Validation rule configuration."""
    field: str
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    sanitize: bool = True
    check_injections: bool = True
    custom_validator: Optional[Callable[[Any], bool]] = None
    error_message: Optional[str] = None


class SQLInjectionDetector:
    """Detects SQL injection attempts."""

    # Common SQL injection patterns
    PATTERNS = [
        r"(\s|^)(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE|UNION|INTO)(\s|$)",
        r"--\s*$",
        r"/\*.*\*/",
        r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP)",
        r"'\s*(OR|AND)\s*'?\d*'?\s*=\s*'?\d*",
        r"'\s*(OR|AND)\s*'[^']*'\s*=\s*'[^']*'",
        r"BENCHMARK\s*\(",
        r"SLEEP\s*\(",
        r"WAITFOR\s+DELAY",
        r"';\s*SHUTDOWN",
        r"1\s*=\s*1",
        r"'\s*OR\s*''='",
        r"admin'\s*--",
        r"'\s*;\s*DROP\s+TABLE",
    ]

    def __init__(self):
        self._patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PATTERNS
        ]

    def detect(self, value: str) -> List[str]:
        """Detect SQL injection patterns."""
        findings = []
        for pattern in self._patterns:
            match = pattern.search(value)
            if match:
                findings.append(f"SQL pattern detected: {match.group()}")
        return findings

    def is_safe(self, value: str) -> bool:
        """Check if value is safe from SQL injection."""
        return len(self.detect(value)) == 0


class XSSDetector:
    """Detects cross-site scripting attempts."""

    # Common XSS patterns
    PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript\s*:",
        r"on\w+\s*=",
        r"<\s*img[^>]+onerror",
        r"<\s*iframe",
        r"<\s*object",
        r"<\s*embed",
        r"<\s*link[^>]+rel\s*=\s*['\"]?import",
        r"expression\s*\(",
        r"url\s*\(\s*['\"]?javascript",
        r"data\s*:\s*text/html",
        r"<\s*svg[^>]+onload",
        r"<\s*body[^>]+onload",
        r"<\s*input[^>]+onfocus",
        r"<!--.*?-->",
        r"<\s*meta[^>]+http-equiv",
    ]

    def __init__(self):
        self._patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.PATTERNS
        ]

    def detect(self, value: str) -> List[str]:
        """Detect XSS patterns."""
        findings = []
        for pattern in self._patterns:
            match = pattern.search(value)
            if match:
                findings.append(f"XSS pattern detected: {match.group()[:50]}")
        return findings

    def is_safe(self, value: str) -> bool:
        """Check if value is safe from XSS."""
        return len(self.detect(value)) == 0


class CommandInjectionDetector:
    """Detects command injection attempts."""

    # Common command injection patterns
    PATTERNS = [
        r";\s*(ls|cat|rm|wget|curl|bash|sh|python|perl|ruby|nc|netcat)",
        r"\|\s*(ls|cat|rm|wget|curl|bash|sh)",
        r"`[^`]+`",
        r"\$\([^)]+\)",
        r"\$\{[^}]+\}",
        r"&&\s*(ls|cat|rm|wget|curl)",
        r"\|\|\s*(ls|cat|rm|wget|curl)",
        r">\s*/",
        r"<\s*/",
        r"/etc/(passwd|shadow|hosts)",
        r"\.\./\.\./",
        r";\s*chmod",
        r";\s*chown",
        r"eval\s*\(",
    ]

    def __init__(self):
        self._patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PATTERNS
        ]

    def detect(self, value: str) -> List[str]:
        """Detect command injection patterns."""
        findings = []
        for pattern in self._patterns:
            match = pattern.search(value)
            if match:
                findings.append(f"Command injection pattern: {match.group()}")
        return findings

    def is_safe(self, value: str) -> bool:
        """Check if value is safe from command injection."""
        return len(self.detect(value)) == 0


class PathTraversalDetector:
    """Detects path traversal attempts."""

    PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e/",
        r"%2e%2e\\",
        r"\.\.%2f",
        r"\.\.%5c",
        r"%252e%252e/",
        r"/etc/",
        r"\\windows\\",
        r"c:\\",
        r"/proc/",
        r"/var/",
    ]

    def __init__(self):
        self._patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PATTERNS
        ]

    def detect(self, value: str) -> List[str]:
        """Detect path traversal patterns."""
        findings = []
        for pattern in self._patterns:
            match = pattern.search(value)
            if match:
                findings.append(f"Path traversal pattern: {match.group()}")
        return findings


class InputValidator:
    """
    Comprehensive input validation.

    Validates and sanitizes input data against various attacks.
    """

    def __init__(
        self,
        rules: Optional[List[ValidationRule]] = None,
        check_sql_injection: bool = True,
        check_xss: bool = True,
        check_command_injection: bool = True,
        check_path_traversal: bool = True,
    ):
        self.rules = {r.field: r for r in (rules or [])}
        self._sql_detector = SQLInjectionDetector() if check_sql_injection else None
        self._xss_detector = XSSDetector() if check_xss else None
        self._cmd_detector = CommandInjectionDetector() if check_command_injection else None
        self._path_detector = PathTraversalDetector() if check_path_traversal else None

    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule."""
        self.rules[rule.field] = rule

    def validate(
        self,
        data: Dict[str, Any],
        additional_rules: Optional[List[ValidationRule]] = None,
    ) -> ValidationResult:
        """Validate input data."""
        result = ValidationResult(is_valid=True)
        sanitized = {}

        # Merge rules
        rules = dict(self.rules)
        if additional_rules:
            for r in additional_rules:
                rules[r.field] = r

        # Validate each field
        for field_name, value in data.items():
            rule = rules.get(field_name)

            # Validate against rule if exists
            if rule:
                self._validate_field(field_name, value, rule, result)

            # Always check for injections in string values
            if isinstance(value, str):
                self._check_injections(field_name, value, result)

            # Sanitize value
            sanitized[field_name] = self._sanitize_value(value, rule)

        # Check required fields
        for field_name, rule in rules.items():
            if rule.required and field_name not in data:
                result.add_issue(
                    field_name,
                    rule.error_message or f"Field '{field_name}' is required",
                    ValidationSeverity.ERROR,
                )

        result.sanitized_data = sanitized
        return result

    def _validate_field(
        self,
        field_name: str,
        value: Any,
        rule: ValidationRule,
        result: ValidationResult,
    ) -> None:
        """Validate a single field."""
        if value is None:
            if rule.required:
                result.add_issue(
                    field_name,
                    f"Field '{field_name}' is required",
                    ValidationSeverity.ERROR,
                )
            return

        # String validations
        if isinstance(value, str):
            if rule.min_length and len(value) < rule.min_length:
                result.add_issue(
                    field_name,
                    f"Field '{field_name}' must be at least {rule.min_length} characters",
                    ValidationSeverity.ERROR,
                    value=value,
                )

            if rule.max_length and len(value) > rule.max_length:
                result.add_issue(
                    field_name,
                    f"Field '{field_name}' must be at most {rule.max_length} characters",
                    ValidationSeverity.ERROR,
                    value=value,
                )

            if rule.pattern:
                if not re.match(rule.pattern, value):
                    result.add_issue(
                        field_name,
                        rule.error_message or f"Field '{field_name}' has invalid format",
                        ValidationSeverity.ERROR,
                        value=value,
                    )

        # Allowed values
        if rule.allowed_values and value not in rule.allowed_values:
            result.add_issue(
                field_name,
                f"Field '{field_name}' must be one of: {rule.allowed_values}",
                ValidationSeverity.ERROR,
                value=value,
            )

        # Custom validator
        if rule.custom_validator and not rule.custom_validator(value):
            result.add_issue(
                field_name,
                rule.error_message or f"Field '{field_name}' failed custom validation",
                ValidationSeverity.ERROR,
                value=value,
            )

    def _check_injections(
        self,
        field_name: str,
        value: str,
        result: ValidationResult,
    ) -> None:
        """Check for injection attacks."""
        # SQL injection
        if self._sql_detector:
            findings = self._sql_detector.detect(value)
            for finding in findings:
                result.add_issue(
                    field_name,
                    finding,
                    ValidationSeverity.CRITICAL,
                    InjectionType.SQL,
                    value,
                )

        # XSS
        if self._xss_detector:
            findings = self._xss_detector.detect(value)
            for finding in findings:
                result.add_issue(
                    field_name,
                    finding,
                    ValidationSeverity.CRITICAL,
                    InjectionType.XSS,
                    value,
                )

        # Command injection
        if self._cmd_detector:
            findings = self._cmd_detector.detect(value)
            for finding in findings:
                result.add_issue(
                    field_name,
                    finding,
                    ValidationSeverity.CRITICAL,
                    InjectionType.COMMAND,
                    value,
                )

        # Path traversal
        if self._path_detector:
            findings = self._path_detector.detect(value)
            for finding in findings:
                result.add_issue(
                    field_name,
                    finding,
                    ValidationSeverity.CRITICAL,
                    InjectionType.PATH_TRAVERSAL,
                    value,
                )

    def _sanitize_value(
        self,
        value: Any,
        rule: Optional[ValidationRule] = None,
    ) -> Any:
        """Sanitize a value."""
        if value is None:
            return None

        if rule and not rule.sanitize:
            return value

        if isinstance(value, str):
            return self._sanitize_string(value)
        elif isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._sanitize_value(v) for v in value]

        return value

    def _sanitize_string(self, value: str) -> str:
        """Sanitize a string value."""
        # HTML escape
        value = html.escape(value)

        # Remove null bytes
        value = value.replace('\x00', '')

        # Normalize whitespace
        value = ' '.join(value.split())

        return value


class ContentFilter:
    """
    Content filtering for user-generated content.

    Filters profanity, spam, and inappropriate content.
    """

    def __init__(
        self,
        blocked_words: Optional[List[str]] = None,
        blocked_patterns: Optional[List[str]] = None,
        max_url_count: int = 5,
        max_repetition: int = 5,
    ):
        self.blocked_words = set(w.lower() for w in (blocked_words or []))
        self.blocked_patterns = [
            re.compile(p, re.IGNORECASE) for p in (blocked_patterns or [])
        ]
        self.max_url_count = max_url_count
        self.max_repetition = max_repetition

        # URL pattern
        self._url_pattern = re.compile(
            r'https?://[^\s<>"\']+|www\.[^\s<>"\']+',
            re.IGNORECASE
        )

    def filter(self, text: str) -> Dict[str, Any]:
        """Filter content and return analysis."""
        issues = []
        filtered_text = text

        # Check blocked words
        words = text.lower().split()
        for word in words:
            if word in self.blocked_words:
                issues.append(f"Blocked word: {word}")
                filtered_text = filtered_text.replace(word, '*' * len(word))

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            matches = pattern.findall(text)
            if matches:
                issues.append(f"Blocked pattern found: {matches}")
                filtered_text = pattern.sub('[FILTERED]', filtered_text)

        # Check URL count
        urls = self._url_pattern.findall(text)
        if len(urls) > self.max_url_count:
            issues.append(f"Too many URLs: {len(urls)}")

        # Check repetition (spam detection)
        if self._has_excessive_repetition(text):
            issues.append("Excessive character repetition detected")

        return {
            "original": text,
            "filtered": filtered_text,
            "issues": issues,
            "is_clean": len(issues) == 0,
            "url_count": len(urls),
        }

    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character repetition."""
        # Check for repeated characters
        for i in range(len(text) - self.max_repetition + 1):
            char = text[i]
            if text[i:i+self.max_repetition] == char * self.max_repetition:
                return True
        return False


# Convenience functions
def sanitize_input(value: Any) -> Any:
    """Sanitize input value."""
    validator = InputValidator()
    if isinstance(value, dict):
        result = validator.validate(value)
        return result.sanitized_data
    elif isinstance(value, str):
        return validator._sanitize_string(value)
    return value


def validate_input(
    data: Dict[str, Any],
    rules: Optional[List[ValidationRule]] = None,
) -> ValidationResult:
    """Validate input data."""
    validator = InputValidator(rules)
    return validator.validate(data)


# Decorator for automatic validation
def validate_request(rules: List[ValidationRule]):
    """
    Decorator for validating request data.

    Usage:
        @validate_request([
            ValidationRule("email", required=True, pattern=r"^[^@]+@[^@]+$"),
            ValidationRule("password", required=True, min_length=8),
        ])
        async def create_user(request_data: dict):
            ...
    """
    def decorator(func: Callable) -> Callable:
        validator = InputValidator(rules)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request data in kwargs
            data = kwargs.get('data') or kwargs.get('request_data') or kwargs.get('body')

            if data:
                result = validator.validate(data)
                if not result.is_valid:
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "Validation failed",
                            "errors": result.to_dict()["issues"],
                        }
                    )

                # Replace with sanitized data
                if 'data' in kwargs:
                    kwargs['data'] = result.sanitized_data
                elif 'request_data' in kwargs:
                    kwargs['request_data'] = result.sanitized_data
                elif 'body' in kwargs:
                    kwargs['body'] = result.sanitized_data

            return await func(*args, **kwargs)

        return wrapper
    return decorator
