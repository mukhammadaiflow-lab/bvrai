"""
Data Masking and PII Protection

Enterprise data protection with:
- PII detection and masking
- Configurable masking rules
- Tokenization/detokenization
- Field-level redaction
"""

from typing import Optional, Dict, Any, List, Union, Callable, Pattern
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import re
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of personally identifiable information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    NAME = "name"
    ADDRESS = "address"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    BANK_ACCOUNT = "bank_account"
    PASSWORD = "password"
    API_KEY = "api_key"
    CUSTOM = "custom"


class MaskingStrategy(str, Enum):
    """Masking strategies."""
    FULL = "full"  # Replace entirely
    PARTIAL = "partial"  # Show some characters
    HASH = "hash"  # One-way hash
    TOKENIZE = "tokenize"  # Reversible tokenization
    REDACT = "redact"  # Remove entirely
    ENCRYPT = "encrypt"  # Encrypt value


@dataclass
class MaskingRule:
    """Rule for masking data."""
    field_name: Optional[str] = None  # Field name to match
    pii_type: Optional[PIIType] = None  # PII type to match
    pattern: Optional[str] = None  # Regex pattern to match
    strategy: MaskingStrategy = MaskingStrategy.PARTIAL
    replacement: str = "***"  # Replacement for FULL strategy
    visible_chars: int = 4  # Chars to show for PARTIAL
    visible_position: str = "end"  # "start", "end", "both"
    hash_algorithm: str = "sha256"  # For HASH strategy
    case_sensitive: bool = False


# Default patterns for PII detection
PII_PATTERNS: Dict[PIIType, Pattern] = {
    PIIType.EMAIL: re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        re.IGNORECASE
    ),
    PIIType.PHONE: re.compile(
        r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
    ),
    PIIType.SSN: re.compile(
        r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'
    ),
    PIIType.CREDIT_CARD: re.compile(
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|'
        r'3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
    ),
    PIIType.IP_ADDRESS: re.compile(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    ),
    PIIType.DATE_OF_BIRTH: re.compile(
        r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-]'
        r'(?:19|20)\d{2}\b|\b(?:19|20)\d{2}[/-](?:0[1-9]|1[0-2])[/-]'
        r'(?:0[1-9]|[12][0-9]|3[01])\b'
    ),
    PIIType.API_KEY: re.compile(
        r'\b(?:sk_|pk_|api_|key_)[a-zA-Z0-9]{20,}\b'
    ),
}


class PIIDetector:
    """
    Detects PII in text and structured data.

    Uses pattern matching and heuristics to identify sensitive data.
    """

    def __init__(self, custom_patterns: Optional[Dict[str, Pattern]] = None):
        self.patterns = dict(PII_PATTERNS)
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self.patterns[PIIType.CUSTOM] = pattern

    def detect_in_text(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text."""
        findings = []

        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                findings.append({
                    "type": pii_type.value,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })

        return findings

    def detect_in_dict(
        self,
        data: Dict[str, Any],
        sensitive_fields: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Detect PII in dictionary."""
        findings = []
        sensitive = sensitive_fields or {
            "email", "phone", "ssn", "password", "secret",
            "credit_card", "card_number", "cvv", "api_key",
            "dob", "date_of_birth", "birth_date",
        }

        def check_field(field_name: str, value: Any, path: str = ""):
            current_path = f"{path}.{field_name}" if path else field_name

            # Check if field name is sensitive
            if any(s in field_name.lower() for s in sensitive):
                findings.append({
                    "type": "sensitive_field",
                    "field": current_path,
                    "has_value": value is not None,
                })

            # Check value for PII patterns
            if isinstance(value, str):
                text_findings = self.detect_in_text(value)
                for finding in text_findings:
                    finding["field"] = current_path
                    findings.append(finding)
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_field(k, v, current_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_field(f"[{i}]", item, current_path)

        for field_name, value in data.items():
            check_field(field_name, value)

        return findings

    def has_pii(self, text: str) -> bool:
        """Quick check if text contains PII."""
        for pattern in self.patterns.values():
            if pattern.search(text):
                return True
        return False


class DataMasker:
    """
    Data masking service.

    Applies masking rules to protect sensitive data.
    """

    def __init__(self, rules: Optional[List[MaskingRule]] = None):
        self.rules = rules or self._default_rules()
        self._detector = PIIDetector()
        self._token_store: Dict[str, str] = {}

    def _default_rules(self) -> List[MaskingRule]:
        """Default masking rules."""
        return [
            MaskingRule(
                pii_type=PIIType.EMAIL,
                strategy=MaskingStrategy.PARTIAL,
                visible_chars=3,
                visible_position="start",
            ),
            MaskingRule(
                pii_type=PIIType.PHONE,
                strategy=MaskingStrategy.PARTIAL,
                visible_chars=4,
                visible_position="end",
            ),
            MaskingRule(
                pii_type=PIIType.SSN,
                strategy=MaskingStrategy.PARTIAL,
                visible_chars=4,
                visible_position="end",
            ),
            MaskingRule(
                pii_type=PIIType.CREDIT_CARD,
                strategy=MaskingStrategy.PARTIAL,
                visible_chars=4,
                visible_position="end",
            ),
            MaskingRule(
                pii_type=PIIType.PASSWORD,
                strategy=MaskingStrategy.FULL,
                replacement="[REDACTED]",
            ),
            MaskingRule(
                pii_type=PIIType.API_KEY,
                strategy=MaskingStrategy.PARTIAL,
                visible_chars=8,
                visible_position="start",
            ),
        ]

    def mask_value(
        self,
        value: str,
        rule: MaskingRule,
    ) -> str:
        """Mask a value according to rule."""
        if not value:
            return value

        if rule.strategy == MaskingStrategy.FULL:
            return rule.replacement

        elif rule.strategy == MaskingStrategy.PARTIAL:
            return self._partial_mask(value, rule)

        elif rule.strategy == MaskingStrategy.HASH:
            return self._hash_value(value, rule.hash_algorithm)

        elif rule.strategy == MaskingStrategy.TOKENIZE:
            return self._tokenize(value)

        elif rule.strategy == MaskingStrategy.REDACT:
            return ""

        return value

    def _partial_mask(self, value: str, rule: MaskingRule) -> str:
        """Apply partial masking."""
        if len(value) <= rule.visible_chars:
            return '*' * len(value)

        mask_char = '*'

        if rule.visible_position == "start":
            visible = value[:rule.visible_chars]
            masked = mask_char * (len(value) - rule.visible_chars)
            return visible + masked

        elif rule.visible_position == "end":
            masked = mask_char * (len(value) - rule.visible_chars)
            visible = value[-rule.visible_chars:]
            return masked + visible

        elif rule.visible_position == "both":
            half = rule.visible_chars // 2
            start_visible = value[:half]
            end_visible = value[-half:] if half > 0 else ""
            masked = mask_char * (len(value) - half * 2)
            return start_visible + masked + end_visible

        return value

    def _hash_value(self, value: str, algorithm: str = "sha256") -> str:
        """Hash value with specified algorithm."""
        if algorithm == "sha256":
            return hashlib.sha256(value.encode()).hexdigest()[:16]
        elif algorithm == "sha512":
            return hashlib.sha512(value.encode()).hexdigest()[:16]
        elif algorithm == "md5":
            return hashlib.md5(value.encode()).hexdigest()[:16]
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def _tokenize(self, value: str) -> str:
        """Generate reversible token."""
        # Check if already tokenized
        for token, original in self._token_store.items():
            if original == value:
                return token

        # Generate new token
        token = f"TOK_{secrets.token_hex(8)}"
        self._token_store[token] = value
        return token

    def detokenize(self, token: str) -> Optional[str]:
        """Reverse tokenization."""
        return self._token_store.get(token)

    def mask_text(self, text: str) -> str:
        """Mask all PII in text."""
        findings = self._detector.detect_in_text(text)

        # Sort by position descending to avoid offset issues
        findings.sort(key=lambda f: f["start"], reverse=True)

        result = text
        for finding in findings:
            pii_type = PIIType(finding["type"]) if finding["type"] != "sensitive_field" else PIIType.CUSTOM

            # Find applicable rule
            rule = None
            for r in self.rules:
                if r.pii_type == pii_type:
                    rule = r
                    break

            if rule:
                masked = self.mask_value(finding["value"], rule)
                result = result[:finding["start"]] + masked + result[finding["end"]:]

        return result

    def mask_dict(
        self,
        data: Dict[str, Any],
        deep: bool = True,
    ) -> Dict[str, Any]:
        """Mask PII in dictionary."""
        def mask_field(field_name: str, value: Any) -> Any:
            if value is None:
                return None

            # Check for field-specific rules
            rule = None
            for r in self.rules:
                if r.field_name and r.field_name.lower() == field_name.lower():
                    rule = r
                    break

            if isinstance(value, str):
                if rule:
                    return self.mask_value(value, rule)
                return self.mask_text(value)
            elif isinstance(value, dict) and deep:
                return {k: mask_field(k, v) for k, v in value.items()}
            elif isinstance(value, list) and deep:
                return [mask_field(f"{field_name}[]", item) for item in value]

            return value

        return {k: mask_field(k, v) for k, v in data.items()}

    def add_rule(self, rule: MaskingRule) -> None:
        """Add a masking rule."""
        self.rules.append(rule)


class TokenVault:
    """
    Secure token vault for tokenization.

    Provides reversible tokenization with secure storage.
    """

    def __init__(self, encryption_key: Optional[bytes] = None):
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._encryption_key = encryption_key
        self._lock_async = None

    def tokenize(
        self,
        value: str,
        pii_type: Optional[PIIType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Tokenize a value."""
        # Create consistent hash for same values
        value_hash = hashlib.sha256(value.encode()).hexdigest()[:16]

        # Check for existing token
        for token, data in self._tokens.items():
            if data.get("hash") == value_hash:
                return token

        # Generate new token
        token = f"TOK_{secrets.token_hex(12)}"

        self._tokens[token] = {
            "hash": value_hash,
            "value": self._encrypt_value(value),
            "type": pii_type.value if pii_type else None,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        return token

    def detokenize(self, token: str) -> Optional[str]:
        """Reverse tokenization."""
        if token not in self._tokens:
            return None

        encrypted_value = self._tokens[token]["value"]
        return self._decrypt_value(encrypted_value)

    def _encrypt_value(self, value: str) -> str:
        """Encrypt value for storage."""
        if self._encryption_key:
            from app.security.encryption import AESEncryption
            aes = AESEncryption()
            encrypted = aes.encrypt(value.encode(), self._encryption_key)
            return encrypted.to_base64()
        return value

    def _decrypt_value(self, encrypted: str) -> str:
        """Decrypt stored value."""
        if self._encryption_key:
            from app.security.encryption import AESEncryption, EncryptedData
            aes = AESEncryption()
            data = EncryptedData.from_base64(encrypted)
            return aes.decrypt(data, self._encryption_key).decode()
        return encrypted

    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get token metadata without revealing value."""
        if token not in self._tokens:
            return None

        data = self._tokens[token]
        return {
            "token": token,
            "type": data.get("type"),
            "created_at": data.get("created_at"),
            "metadata": data.get("metadata"),
        }

    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False


# Convenience functions
def mask_pii(text: str, rules: Optional[List[MaskingRule]] = None) -> str:
    """Mask PII in text."""
    masker = DataMasker(rules)
    return masker.mask_text(text)


def redact_sensitive(
    data: Dict[str, Any],
    fields: Optional[set] = None,
) -> Dict[str, Any]:
    """Redact sensitive fields from dictionary."""
    sensitive = fields or {
        "password", "secret", "api_key", "token",
        "credit_card", "ssn", "cvv",
    }

    def redact(d: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in d.items():
            if any(s in k.lower() for s in sensitive):
                result[k] = "[REDACTED]"
            elif isinstance(v, dict):
                result[k] = redact(v)
            elif isinstance(v, list):
                result[k] = [
                    redact(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                result[k] = v
        return result

    return redact(data)


def tokenize(value: str, vault: Optional[TokenVault] = None) -> str:
    """Tokenize a value."""
    v = vault or TokenVault()
    return v.tokenize(value)


def detokenize(token: str, vault: Optional[TokenVault] = None) -> Optional[str]:
    """Detokenize a value."""
    v = vault or TokenVault()
    return v.detokenize(token)
