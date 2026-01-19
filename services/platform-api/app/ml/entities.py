"""
Entity Extraction System

Named Entity Recognition with:
- Multiple entity types
- Custom entity definitions
- Span extraction
- Relationship detection
- Entity linking
"""

from typing import Optional, Dict, Any, List, Tuple, Set, Pattern
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import re
import logging

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Standard entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    URL = "URL"
    ADDRESS = "ADDRESS"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"
    LANGUAGE = "LANGUAGE"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    ORDER_NUMBER = "ORDER_NUMBER"
    CUSTOM = "CUSTOM"


@dataclass
class Entity:
    """Extracted entity."""
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    normalized_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> Tuple[int, int]:
        """Get entity span."""
        return (self.start, self.end)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "type": self.type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "normalized_value": self.normalized_value,
            "metadata": self.metadata,
        }


@dataclass
class SpanEntity:
    """Entity with context spans."""
    entity: Entity
    left_context: str = ""
    right_context: str = ""
    sentence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity": self.entity.to_dict(),
            "left_context": self.left_context,
            "right_context": self.right_context,
            "sentence": self.sentence,
        }


@dataclass
class EntityRelation:
    """Relationship between entities."""
    subject: Entity
    relation: str
    obj: Entity
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject": self.subject.to_dict(),
            "relation": self.relation,
            "object": self.obj.to_dict(),
            "confidence": self.confidence,
        }


@dataclass
class EntityResult:
    """Result from entity extraction."""
    text: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[EntityRelation] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model_version: str = "1.0.0"

    def get_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get entities of specific type."""
        return [e for e in self.entities if e.type == entity_type]

    def get_unique_values(self, entity_type: EntityType) -> Set[str]:
        """Get unique entity values of type."""
        return {e.text for e in self.get_by_type(entity_type)}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
        }


class EntityExtractor(ABC):
    """Abstract base for entity extractors."""

    @abstractmethod
    async def extract(self, text: str) -> EntityResult:
        """Extract entities from text."""
        pass

    @abstractmethod
    async def extract_batch(self, texts: List[str]) -> List[EntityResult]:
        """Extract entities from multiple texts."""
        pass


class RegexEntityExtractor(EntityExtractor):
    """
    Regex-based entity extractor.

    Uses patterns to identify entities.
    """

    def __init__(self):
        self._patterns: Dict[EntityType, List[Tuple[Pattern, Optional[str]]]] = {}
        self._setup_default_patterns()

    def _setup_default_patterns(self) -> None:
        """Setup default entity patterns."""
        # Email
        self._patterns[EntityType.EMAIL] = [
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.I), None),
        ]

        # Phone numbers
        self._patterns[EntityType.PHONE] = [
            (re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'), None),
            (re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'), None),
        ]

        # URLs
        self._patterns[EntityType.URL] = [
            (re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.I), None),
            (re.compile(r'www\.[^\s<>"{}|\\^`\[\]]+', re.I), None),
        ]

        # Money/Currency
        self._patterns[EntityType.MONEY] = [
            (re.compile(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?'), None),
            (re.compile(r'(?:USD|EUR|GBP|CAD|AUD)\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?', re.I), None),
            (re.compile(r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|euros?|pounds?)', re.I), None),
        ]

        # Percentages
        self._patterns[EntityType.PERCENT] = [
            (re.compile(r'\d+(?:\.\d+)?%'), None),
            (re.compile(r'\d+(?:\.\d+)?\s*percent', re.I), None),
        ]

        # Dates
        self._patterns[EntityType.DATE] = [
            # MM/DD/YYYY or DD/MM/YYYY
            (re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'), None),
            # YYYY-MM-DD
            (re.compile(r'\b\d{4}-\d{2}-\d{2}\b'), None),
            # Month DD, YYYY
            (re.compile(
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
                r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|'
                r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?'
                r'(?:\s*,?\s*\d{4})?\b',
                re.I
            ), None),
            # DD Month YYYY
            (re.compile(
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|'
                r'Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|'
                r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                r'(?:\s+\d{4})?\b',
                re.I
            ), None),
            # Relative dates
            (re.compile(r'\b(?:today|tomorrow|yesterday)\b', re.I), None),
            (re.compile(r'\b(?:next|last|this)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.I), None),
        ]

        # Times
        self._patterns[EntityType.TIME] = [
            (re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b'), None),
            (re.compile(r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b'), None),
            (re.compile(r'\b(?:noon|midnight|morning|afternoon|evening)\b', re.I), None),
        ]

        # Cardinals (numbers)
        self._patterns[EntityType.CARDINAL] = [
            (re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'), None),
            (re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                       r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                       r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                       r'eighty|ninety|hundred|thousand|million|billion)\b', re.I), None),
        ]

        # Ordinals
        self._patterns[EntityType.ORDINAL] = [
            (re.compile(r'\b\d+(?:st|nd|rd|th)\b', re.I), None),
            (re.compile(r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b', re.I), None),
        ]

        # Order numbers
        self._patterns[EntityType.ORDER_NUMBER] = [
            (re.compile(r'\b(?:order|ref|reference|confirmation|tracking)\s*#?\s*[A-Z0-9]{6,}', re.I), None),
            (re.compile(r'\b[A-Z]{2,3}-\d{6,}\b'), None),
        ]

        # Account numbers
        self._patterns[EntityType.ACCOUNT_NUMBER] = [
            (re.compile(r'\b(?:account|acct)\s*#?\s*\d{8,}\b', re.I), None),
        ]

        # Addresses (simplified)
        self._patterns[EntityType.ADDRESS] = [
            (re.compile(r'\b\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+'
                       r'(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|'
                       r'Lane|Ln|Way|Court|Ct|Circle|Cir)\b\.?', re.I), None),
        ]

    def add_pattern(
        self,
        entity_type: EntityType,
        pattern: str,
        normalizer: Optional[str] = None,
    ) -> None:
        """Add custom entity pattern."""
        if entity_type not in self._patterns:
            self._patterns[entity_type] = []
        self._patterns[entity_type].append((re.compile(pattern, re.I), normalizer))

    async def extract(self, text: str) -> EntityResult:
        """Extract entities from text."""
        import time
        start_time = time.time()

        entities = []
        seen_spans: Set[Tuple[int, int]] = set()

        for entity_type, patterns in self._patterns.items():
            for pattern, normalizer in patterns:
                for match in pattern.finditer(text):
                    span = (match.start(), match.end())

                    # Skip overlapping entities
                    if any(self._spans_overlap(span, seen) for seen in seen_spans):
                        continue

                    seen_spans.add(span)

                    entity_text = match.group()
                    normalized = self._normalize(entity_text, entity_type, normalizer)

                    entities.append(Entity(
                        text=entity_text,
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        normalized_value=normalized,
                    ))

        # Sort by position
        entities.sort(key=lambda e: e.start)

        return EntityResult(
            text=text,
            entities=entities,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version="regex-1.0.0",
        )

    def _spans_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """Check if two spans overlap."""
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])

    def _normalize(
        self,
        text: str,
        entity_type: EntityType,
        normalizer: Optional[str],
    ) -> Optional[str]:
        """Normalize entity value."""
        if normalizer:
            return normalizer

        if entity_type == EntityType.PHONE:
            # Remove non-digits
            digits = re.sub(r'\D', '', text)
            if len(digits) == 10:
                return f"+1{digits}"
            elif len(digits) == 11 and digits[0] == '1':
                return f"+{digits}"
            return digits

        if entity_type == EntityType.EMAIL:
            return text.lower()

        if entity_type == EntityType.URL:
            if not text.startswith('http'):
                return f"https://{text}"
            return text

        return None

    async def extract_batch(self, texts: List[str]) -> List[EntityResult]:
        """Extract entities from multiple texts."""
        return await asyncio.gather(*[self.extract(text) for text in texts])


class GazetteerEntityExtractor(EntityExtractor):
    """
    Gazetteer-based entity extractor.

    Uses predefined lists for entity matching.
    """

    def __init__(self):
        self._gazetteers: Dict[EntityType, Set[str]] = {}
        self._case_sensitive: Dict[EntityType, bool] = {}
        self._setup_default_gazetteers()

    def _setup_default_gazetteers(self) -> None:
        """Setup default gazetteers."""
        # Common person titles
        self._gazetteers[EntityType.PERSON] = {
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sir", "Madam",
        }
        self._case_sensitive[EntityType.PERSON] = True

        # Organizations
        self._gazetteers[EntityType.ORGANIZATION] = {
            "Google", "Apple", "Microsoft", "Amazon", "Facebook", "Meta",
            "Netflix", "Twitter", "LinkedIn", "IBM", "Oracle", "Salesforce",
            "Adobe", "Intel", "Cisco", "Samsung", "Sony", "Tesla",
            "Uber", "Airbnb", "Spotify", "Slack", "Zoom", "Shopify",
        }
        self._case_sensitive[EntityType.ORGANIZATION] = True

        # Locations
        self._gazetteers[EntityType.LOCATION] = {
            "United States", "USA", "UK", "United Kingdom", "Canada",
            "Australia", "Germany", "France", "Japan", "China", "India",
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "San Francisco", "Seattle", "Boston", "Miami", "Denver",
            "London", "Paris", "Tokyo", "Beijing", "Mumbai", "Sydney",
        }
        self._case_sensitive[EntityType.LOCATION] = True

        # Products (example)
        self._gazetteers[EntityType.PRODUCT] = {
            "iPhone", "iPad", "MacBook", "iMac", "AirPods", "Apple Watch",
            "Galaxy", "Pixel", "Surface", "Xbox", "PlayStation", "Nintendo",
            "Alexa", "Echo", "Kindle", "Fire TV",
        }
        self._case_sensitive[EntityType.PRODUCT] = True

        # Languages
        self._gazetteers[EntityType.LANGUAGE] = {
            "English", "Spanish", "French", "German", "Italian", "Portuguese",
            "Chinese", "Japanese", "Korean", "Arabic", "Hindi", "Russian",
        }
        self._case_sensitive[EntityType.LANGUAGE] = True

    def add_gazetteer(
        self,
        entity_type: EntityType,
        entries: Set[str],
        case_sensitive: bool = True,
    ) -> None:
        """Add entries to gazetteer."""
        if entity_type not in self._gazetteers:
            self._gazetteers[entity_type] = set()
        self._gazetteers[entity_type].update(entries)
        self._case_sensitive[entity_type] = case_sensitive

    async def extract(self, text: str) -> EntityResult:
        """Extract entities using gazetteers."""
        import time
        start_time = time.time()

        entities = []
        text_lower = text.lower()

        for entity_type, entries in self._gazetteers.items():
            case_sensitive = self._case_sensitive.get(entity_type, True)

            for entry in entries:
                search_text = text if case_sensitive else text_lower
                search_entry = entry if case_sensitive else entry.lower()

                # Find all occurrences
                start = 0
                while True:
                    pos = search_text.find(search_entry, start)
                    if pos == -1:
                        break

                    # Check word boundaries
                    if self._is_word_boundary(text, pos, len(entry)):
                        entities.append(Entity(
                            text=text[pos:pos + len(entry)],
                            type=entity_type,
                            start=pos,
                            end=pos + len(entry),
                            confidence=0.95,
                        ))

                    start = pos + 1

        # Remove duplicates and sort
        entities = self._deduplicate(entities)
        entities.sort(key=lambda e: e.start)

        return EntityResult(
            text=text,
            entities=entities,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version="gazetteer-1.0.0",
        )

    def _is_word_boundary(self, text: str, start: int, length: int) -> bool:
        """Check if match is at word boundary."""
        end = start + length

        # Check start
        if start > 0 and text[start - 1].isalnum():
            return False

        # Check end
        if end < len(text) and text[end].isalnum():
            return False

        return True

    def _deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities."""
        seen = set()
        unique = []
        for e in entities:
            key = (e.text, e.type, e.start, e.end)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    async def extract_batch(self, texts: List[str]) -> List[EntityResult]:
        """Extract entities from multiple texts."""
        return await asyncio.gather(*[self.extract(text) for text in texts])


class NERModel(EntityExtractor):
    """
    Neural NER model wrapper.

    Provides interface for ML-based NER.
    """

    def __init__(
        self,
        model_name: str = "default",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._label_map: Dict[int, str] = {}

    async def load_model(self) -> None:
        """Load NER model."""
        # Placeholder for actual model loading
        # In production, this would load a transformer model
        logger.info(f"Loading NER model: {self.model_name}")
        self._label_map = {
            0: "O",
            1: "B-PERSON", 2: "I-PERSON",
            3: "B-ORG", 4: "I-ORG",
            5: "B-LOC", 6: "I-LOC",
            7: "B-DATE", 8: "I-DATE",
            9: "B-TIME", 10: "I-TIME",
        }

    async def extract(self, text: str) -> EntityResult:
        """Extract entities using NER model."""
        import time
        start_time = time.time()

        # Fallback to regex if model not loaded
        if self._model is None:
            fallback = RegexEntityExtractor()
            result = await fallback.extract(text)
            result.model_version = f"ner-{self.model_name}-fallback"
            return result

        # In production, this would run inference
        entities = []

        return EntityResult(
            text=text,
            entities=entities,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version=f"ner-{self.model_name}",
        )

    async def extract_batch(self, texts: List[str]) -> List[EntityResult]:
        """Extract entities from multiple texts."""
        return await asyncio.gather(*[self.extract(text) for text in texts])


class CompositeEntityExtractor(EntityExtractor):
    """
    Composite entity extractor.

    Combines multiple extractors with conflict resolution.
    """

    def __init__(
        self,
        extractors: Optional[List[EntityExtractor]] = None,
        conflict_strategy: str = "prefer_first",
    ):
        self.extractors = extractors or [
            RegexEntityExtractor(),
            GazetteerEntityExtractor(),
        ]
        self.conflict_strategy = conflict_strategy

    async def extract(self, text: str) -> EntityResult:
        """Extract entities using all extractors."""
        import time
        start_time = time.time()

        # Get results from all extractors
        results = await asyncio.gather(*[
            extractor.extract(text) for extractor in self.extractors
        ])

        # Merge entities
        all_entities = []
        for result in results:
            all_entities.extend(result.entities)

        # Resolve conflicts
        resolved = self._resolve_conflicts(all_entities)

        # Sort by position
        resolved.sort(key=lambda e: e.start)

        return EntityResult(
            text=text,
            entities=resolved,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version="composite-1.0.0",
        )

    def _resolve_conflicts(self, entities: List[Entity]) -> List[Entity]:
        """Resolve overlapping entities."""
        if not entities:
            return []

        # Sort by start position, then by length (longer first)
        sorted_entities = sorted(
            entities,
            key=lambda e: (e.start, -(e.end - e.start))
        )

        resolved = []
        last_end = -1

        for entity in sorted_entities:
            if entity.start >= last_end:
                # No overlap
                resolved.append(entity)
                last_end = entity.end
            elif self.conflict_strategy == "prefer_longer":
                # Check if current is longer
                if entity.end - entity.start > resolved[-1].end - resolved[-1].start:
                    resolved[-1] = entity
                    last_end = entity.end
            elif self.conflict_strategy == "prefer_higher_confidence":
                if entity.confidence > resolved[-1].confidence:
                    resolved[-1] = entity
                    last_end = entity.end
            # else prefer_first: keep existing

        return resolved

    async def extract_batch(self, texts: List[str]) -> List[EntityResult]:
        """Extract entities from multiple texts."""
        return await asyncio.gather(*[self.extract(text) for text in texts])


class EntityLinker:
    """
    Entity linking and disambiguation.

    Links extracted entities to knowledge base entries.
    """

    def __init__(self):
        self._knowledge_base: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, str] = {}

    def add_entry(
        self,
        entity_id: str,
        canonical_name: str,
        entity_type: EntityType,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add knowledge base entry."""
        self._knowledge_base[entity_id] = {
            "canonical_name": canonical_name,
            "type": entity_type,
            "aliases": aliases or [],
            "metadata": metadata or {},
        }

        # Index aliases
        self._aliases[canonical_name.lower()] = entity_id
        for alias in aliases or []:
            self._aliases[alias.lower()] = entity_id

    async def link(self, entities: List[Entity]) -> List[Entity]:
        """Link entities to knowledge base."""
        linked = []
        for entity in entities:
            entity_id = self._aliases.get(entity.text.lower())
            if entity_id and entity_id in self._knowledge_base:
                kb_entry = self._knowledge_base[entity_id]
                linked.append(Entity(
                    text=entity.text,
                    type=entity.type,
                    start=entity.start,
                    end=entity.end,
                    confidence=entity.confidence,
                    normalized_value=kb_entry["canonical_name"],
                    metadata={
                        **entity.metadata,
                        "entity_id": entity_id,
                        "kb_metadata": kb_entry["metadata"],
                    },
                ))
            else:
                linked.append(entity)
        return linked


class RelationExtractor:
    """
    Extract relationships between entities.

    Identifies how entities are related.
    """

    def __init__(self):
        self._relation_patterns: List[Tuple[Pattern, str, str, str]] = []
        self._setup_patterns()

    def _setup_patterns(self) -> None:
        """Setup relation patterns."""
        # Pattern format: (regex, subject_group, relation, object_group)
        self._relation_patterns = [
            # "X is located in Y"
            (re.compile(r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+)?located\s+in\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.I),
             "1", "LOCATED_IN", "2"),
            # "X works at Y"
            (re.compile(r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+works?\s+(?:at|for)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.I),
             "1", "WORKS_FOR", "2"),
            # "X founded Y"
            (re.compile(r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+founded\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.I),
             "1", "FOUNDED", "2"),
            # "X acquired Y"
            (re.compile(r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+acquired\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.I),
             "1", "ACQUIRED", "2"),
        ]

    async def extract(
        self,
        text: str,
        entities: List[Entity],
    ) -> List[EntityRelation]:
        """Extract relations between entities."""
        relations = []

        for pattern, subj_group, relation, obj_group in self._relation_patterns:
            for match in pattern.finditer(text):
                subj_text = match.group(int(subj_group))
                obj_text = match.group(int(obj_group))

                # Find matching entities
                subj_entity = self._find_entity(subj_text, entities)
                obj_entity = self._find_entity(obj_text, entities)

                if subj_entity and obj_entity:
                    relations.append(EntityRelation(
                        subject=subj_entity,
                        relation=relation,
                        obj=obj_entity,
                        confidence=0.8,
                    ))

        return relations

    def _find_entity(
        self,
        text: str,
        entities: List[Entity],
    ) -> Optional[Entity]:
        """Find entity matching text."""
        text_lower = text.lower()
        for entity in entities:
            if entity.text.lower() == text_lower:
                return entity
        return None


class ContextualEntityExtractor:
    """
    Context-aware entity extraction.

    Uses conversation context for better extraction.
    """

    def __init__(
        self,
        base_extractor: Optional[EntityExtractor] = None,
    ):
        self.base_extractor = base_extractor or CompositeEntityExtractor()
        self._context_entities: List[Entity] = []
        self._co_reference_map: Dict[str, Entity] = {}

    def add_context(self, entities: List[Entity]) -> None:
        """Add entities from previous context."""
        self._context_entities.extend(entities)

        # Build co-reference map
        for entity in entities:
            if entity.type == EntityType.PERSON:
                # "he", "she", "they" might refer to last person
                self._co_reference_map["he"] = entity
                self._co_reference_map["she"] = entity
                self._co_reference_map["they"] = entity
            elif entity.type == EntityType.ORGANIZATION:
                self._co_reference_map["it"] = entity
                self._co_reference_map["they"] = entity

    async def extract(self, text: str) -> EntityResult:
        """Extract entities with context awareness."""
        result = await self.base_extractor.extract(text)

        # Resolve co-references
        resolved_entities = self._resolve_co_references(text, result.entities)

        return EntityResult(
            text=result.text,
            entities=resolved_entities,
            processing_time_ms=result.processing_time_ms,
            model_version=f"contextual-{result.model_version}",
        )

    def _resolve_co_references(
        self,
        text: str,
        entities: List[Entity],
    ) -> List[Entity]:
        """Resolve pronoun co-references."""
        # Find pronouns not already identified
        pronoun_pattern = re.compile(r'\b(he|she|it|they|him|her|them)\b', re.I)

        resolved = list(entities)

        for match in pronoun_pattern.finditer(text):
            pronoun = match.group().lower()
            if pronoun in self._co_reference_map:
                ref_entity = self._co_reference_map[pronoun]
                resolved.append(Entity(
                    text=match.group(),
                    type=ref_entity.type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                    normalized_value=ref_entity.text,
                    metadata={"co_reference": True, "refers_to": ref_entity.text},
                ))

        return resolved

    def clear_context(self) -> None:
        """Clear context."""
        self._context_entities.clear()
        self._co_reference_map.clear()


class EntityExtractionPipeline:
    """
    Complete entity extraction pipeline.

    Combines extraction, linking, and relation extraction.
    """

    def __init__(
        self,
        extractor: Optional[EntityExtractor] = None,
        linker: Optional[EntityLinker] = None,
        relation_extractor: Optional[RelationExtractor] = None,
        include_context: bool = True,
    ):
        self.extractor = extractor or CompositeEntityExtractor()
        self.linker = linker
        self.relation_extractor = relation_extractor or RelationExtractor()
        self.include_context = include_context
        self._context = ContextualEntityExtractor(self.extractor) if include_context else None

    async def extract(
        self,
        text: str,
        include_relations: bool = True,
    ) -> EntityResult:
        """Run full extraction pipeline."""
        import time
        start_time = time.time()

        # Extract entities
        if self._context:
            result = await self._context.extract(text)
        else:
            result = await self.extractor.extract(text)

        # Link entities
        if self.linker:
            result.entities = await self.linker.link(result.entities)

        # Extract relations
        if include_relations:
            result.relations = await self.relation_extractor.extract(
                text, result.entities
            )

        # Update context
        if self._context:
            self._context.add_context(result.entities)

        result.processing_time_ms = (time.time() - start_time) * 1000
        result.model_version = "pipeline-1.0.0"

        return result

    async def extract_batch(
        self,
        texts: List[str],
        include_relations: bool = True,
    ) -> List[EntityResult]:
        """Extract from multiple texts."""
        return await asyncio.gather(*[
            self.extract(text, include_relations) for text in texts
        ])

    def clear_context(self) -> None:
        """Clear extraction context."""
        if self._context:
            self._context.clear_context()


# Factory function
def create_entity_extractor(
    extractor_type: str = "composite",
    **kwargs,
) -> EntityExtractor:
    """Create entity extractor by type."""
    extractors = {
        "regex": RegexEntityExtractor,
        "gazetteer": GazetteerEntityExtractor,
        "ner": NERModel,
        "composite": CompositeEntityExtractor,
    }

    cls = extractors.get(extractor_type, CompositeEntityExtractor)
    return cls(**kwargs)
