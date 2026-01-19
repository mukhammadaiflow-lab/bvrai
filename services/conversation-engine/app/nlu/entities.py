"""Entity extraction for NLU."""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of entities."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    ORDER_ID = "order_id"
    PRODUCT = "product"
    CUSTOM = "custom"


@dataclass
class Entity:
    """Extracted entity."""
    type: EntityType
    value: Any
    raw_value: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "value": self.value,
            "raw_value": self.raw_value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class EntityExtractor:
    """Base class for entity extractors."""

    async def extract(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Extract entities from text."""
        raise NotImplementedError


class PatternEntityExtractor(EntityExtractor):
    """
    Pattern-based entity extractor using regex.

    Usage:
        extractor = PatternEntityExtractor()
        extractor.add_pattern(
            EntityType.PHONE_NUMBER,
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        )

        entities = await extractor.extract("Call me at 555-123-4567")
    """

    def __init__(self):
        self._patterns: Dict[EntityType, List[Tuple[re.Pattern, callable]]] = {}
        self._setup_default_patterns()

    def _setup_default_patterns(self) -> None:
        """Setup default patterns for common entities."""
        # Phone numbers
        self.add_pattern(
            EntityType.PHONE_NUMBER,
            r"\b(\+?1?[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
            lambda m: re.sub(r"[^\d]", "", m.group()),
        )

        # Email
        self.add_pattern(
            EntityType.EMAIL,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        )

        # Numbers
        self.add_pattern(
            EntityType.NUMBER,
            r"\b\d+(?:\.\d+)?\b",
            lambda m: float(m.group()) if "." in m.group() else int(m.group()),
        )

        # Currency
        self.add_pattern(
            EntityType.CURRENCY,
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd)",
            lambda m: float(re.sub(r"[^\d.]", "", m.group())),
        )

        # Percentage
        self.add_pattern(
            EntityType.PERCENTAGE,
            r"\b\d+(?:\.\d+)?%",
            lambda m: float(m.group().rstrip("%")),
        )

        # Order ID (common formats)
        self.add_pattern(
            EntityType.ORDER_ID,
            r"\b(?:order|#|ord)[:\s]*([A-Z0-9]{6,12})\b",
            lambda m: m.group(1),
        )

    def add_pattern(
        self,
        entity_type: EntityType,
        pattern: str,
        transformer: Optional[callable] = None,
    ) -> None:
        """Add a pattern for an entity type."""
        if entity_type not in self._patterns:
            self._patterns[entity_type] = []

        compiled = re.compile(pattern, re.IGNORECASE)
        self._patterns[entity_type].append((compiled, transformer))

    async def extract(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Extract entities using patterns."""
        entities = []

        for entity_type, patterns in self._patterns.items():
            for pattern, transformer in patterns:
                for match in pattern.finditer(text):
                    raw_value = match.group()
                    value = transformer(match) if transformer else raw_value

                    entities.append(Entity(
                        type=entity_type,
                        value=value,
                        raw_value=raw_value,
                        start=match.start(),
                        end=match.end(),
                    ))

        # Sort by position
        entities.sort(key=lambda e: e.start)
        return entities


class DateTimeExtractor(EntityExtractor):
    """
    Specialized extractor for dates and times.

    Handles various date/time formats and relative expressions.
    """

    RELATIVE_DATES = {
        "today": 0,
        "tomorrow": 1,
        "yesterday": -1,
        "next week": 7,
        "next month": 30,
    }

    DAYS_OF_WEEK = [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ]

    TIME_PATTERNS = [
        (r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", "12h"),
        (r"\b(\d{1,2})\s*(am|pm)\b", "12h_short"),
        (r"\b(\d{2}):(\d{2})\b", "24h"),
    ]

    DATE_PATTERNS = [
        (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", "mdy"),
        (r"\b(\d{4})-(\d{2})-(\d{2})\b", "iso"),
        (r"\b(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?\b", "natural"),
    ]

    async def extract(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Extract dates and times from text."""
        entities = []
        text_lower = text.lower()

        # Extract relative dates
        for phrase, days in self.RELATIVE_DATES.items():
            if phrase in text_lower:
                start = text_lower.index(phrase)
                end = start + len(phrase)
                target_date = datetime.now().date() + timedelta(days=days)
                entities.append(Entity(
                    type=EntityType.DATE,
                    value=target_date.isoformat(),
                    raw_value=phrase,
                    start=start,
                    end=end,
                    metadata={"relative": True},
                ))

        # Extract day of week
        for i, day in enumerate(self.DAYS_OF_WEEK):
            if day in text_lower:
                start = text_lower.index(day)
                end = start + len(day)
                # Calculate next occurrence
                today = datetime.now().date()
                days_ahead = i - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target_date = today + timedelta(days=days_ahead)
                entities.append(Entity(
                    type=EntityType.DATE,
                    value=target_date.isoformat(),
                    raw_value=day,
                    start=start,
                    end=end,
                    metadata={"day_of_week": day},
                ))

        # Extract time patterns
        for pattern, format_type in self.TIME_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                time_str = self._parse_time(match, format_type)
                if time_str:
                    entities.append(Entity(
                        type=EntityType.TIME,
                        value=time_str,
                        raw_value=match.group(),
                        start=match.start(),
                        end=match.end(),
                    ))

        # Extract date patterns
        for pattern, format_type in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_str = self._parse_date(match, format_type)
                if date_str:
                    entities.append(Entity(
                        type=EntityType.DATE,
                        value=date_str,
                        raw_value=match.group(),
                        start=match.start(),
                        end=match.end(),
                    ))

        # Sort by position and remove duplicates
        entities.sort(key=lambda e: e.start)
        return self._remove_overlapping(entities)

    def _parse_time(self, match: re.Match, format_type: str) -> Optional[str]:
        """Parse time from regex match."""
        try:
            if format_type == "12h":
                hour = int(match.group(1))
                minute = int(match.group(2))
                period = match.group(3).lower() if match.group(3) else "am"
                if period == "pm" and hour < 12:
                    hour += 12
                elif period == "am" and hour == 12:
                    hour = 0
                return f"{hour:02d}:{minute:02d}"
            elif format_type == "12h_short":
                hour = int(match.group(1))
                period = match.group(2).lower()
                if period == "pm" and hour < 12:
                    hour += 12
                elif period == "am" and hour == 12:
                    hour = 0
                return f"{hour:02d}:00"
            elif format_type == "24h":
                hour = int(match.group(1))
                minute = int(match.group(2))
                return f"{hour:02d}:{minute:02d}"
        except (ValueError, IndexError):
            pass
        return None

    def _parse_date(self, match: re.Match, format_type: str) -> Optional[str]:
        """Parse date from regex match."""
        try:
            if format_type == "mdy":
                month = int(match.group(1))
                day = int(match.group(2))
                year = int(match.group(3))
                return f"{year}-{month:02d}-{day:02d}"
            elif format_type == "iso":
                return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            elif format_type == "natural":
                month_name = match.group(1).lower()
                months = [
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december",
                ]
                if month_name in months:
                    month = months.index(month_name) + 1
                    day = int(match.group(2))
                    year = int(match.group(3)) if match.group(3) else datetime.now().year
                    return f"{year}-{month:02d}-{day:02d}"
        except (ValueError, IndexError):
            pass
        return None

    def _remove_overlapping(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping higher confidence."""
        if not entities:
            return []

        result = [entities[0]]
        for entity in entities[1:]:
            if entity.start >= result[-1].end:
                result.append(entity)
            elif entity.confidence > result[-1].confidence:
                result[-1] = entity

        return result


class RuleBasedEntityExtractor(EntityExtractor):
    """
    Entity extractor using custom rules and dictionaries.

    Usage:
        extractor = RuleBasedEntityExtractor()
        extractor.add_dictionary("product", ["iPhone", "MacBook", "AirPods"])

        entities = await extractor.extract("I want to buy an iPhone")
    """

    def __init__(self):
        self._dictionaries: Dict[str, List[str]] = {}
        self._custom_extractors: List[callable] = []

    def add_dictionary(
        self,
        entity_type: str,
        values: List[str],
        case_sensitive: bool = False,
    ) -> None:
        """Add a dictionary of values for an entity type."""
        if not case_sensitive:
            values = [v.lower() for v in values]
        self._dictionaries[entity_type] = values

    def add_custom_extractor(self, extractor: callable) -> None:
        """Add a custom extraction function."""
        self._custom_extractors.append(extractor)

    async def extract(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Extract entities using rules."""
        entities = []
        text_lower = text.lower()

        # Extract from dictionaries
        for entity_type, values in self._dictionaries.items():
            for value in values:
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(value.lower(), start)
                    if pos == -1:
                        break
                    entities.append(Entity(
                        type=EntityType.CUSTOM,
                        value=value,
                        raw_value=text[pos:pos + len(value)],
                        start=pos,
                        end=pos + len(value),
                        metadata={"entity_type": entity_type},
                    ))
                    start = pos + 1

        # Run custom extractors
        for extractor in self._custom_extractors:
            try:
                custom_entities = await extractor(text, context)
                entities.extend(custom_entities)
            except Exception as e:
                logger.warning(f"Custom extractor failed: {e}")

        entities.sort(key=lambda e: e.start)
        return entities


class CompositeEntityExtractor(EntityExtractor):
    """
    Composite extractor combining multiple extractors.

    Usage:
        extractor = CompositeEntityExtractor()
        extractor.add_extractor(PatternEntityExtractor())
        extractor.add_extractor(DateTimeExtractor())

        entities = await extractor.extract("Call 555-1234 tomorrow at 3pm")
    """

    def __init__(self):
        self._extractors: List[EntityExtractor] = []

    def add_extractor(self, extractor: EntityExtractor) -> None:
        """Add an extractor to the composite."""
        self._extractors.append(extractor)

    async def extract(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Extract entities using all extractors."""
        all_entities = []

        for extractor in self._extractors:
            try:
                entities = await extractor.extract(text, context)
                all_entities.extend(entities)
            except Exception as e:
                logger.warning(f"Extractor {type(extractor).__name__} failed: {e}")

        # Sort and deduplicate
        all_entities.sort(key=lambda e: (e.start, -e.confidence))
        return self._deduplicate(all_entities)

    def _deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities at same position."""
        if not entities:
            return []

        seen_spans = set()
        result = []

        for entity in entities:
            span = (entity.start, entity.end)
            if span not in seen_spans:
                seen_spans.add(span)
                result.append(entity)

        return result


# Global entity extractor
_entity_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get or create the global entity extractor."""
    global _entity_extractor
    if _entity_extractor is None:
        composite = CompositeEntityExtractor()
        composite.add_extractor(PatternEntityExtractor())
        composite.add_extractor(DateTimeExtractor())
        _entity_extractor = composite
    return _entity_extractor


def setup_entity_extractor(extractor: EntityExtractor) -> None:
    """Set up the global entity extractor."""
    global _entity_extractor
    _entity_extractor = extractor
