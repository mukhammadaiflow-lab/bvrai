"""
Data Pipeline Processors
========================

Collection of reusable data processors for pipeline transformations.

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)

import structlog

from bvrai_core.pipeline.engine import DataRecord, Processor

logger = structlog.get_logger(__name__)


class TransformProcessor(Processor):
    """
    Transforms record values using a custom function.

    Usage:
        processor = TransformProcessor(
            "uppercase",
            lambda r: r.value.upper()
        )
    """

    def __init__(
        self,
        name: str,
        transform_func: Callable[[DataRecord], Any]
    ):
        super().__init__(name)
        self._transform_func = transform_func

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        new_value = self._transform_func(record)
        return record.with_value(new_value)


class AsyncTransformProcessor(Processor):
    """
    Transforms record values using an async function.
    """

    def __init__(
        self,
        name: str,
        transform_func: Callable[[DataRecord], Awaitable[Any]]
    ):
        super().__init__(name)
        self._transform_func = transform_func

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        new_value = await self._transform_func(record)
        return record.with_value(new_value)


class FilterProcessor(Processor):
    """
    Filters records based on a predicate.

    Usage:
        processor = FilterProcessor(
            "filter-errors",
            lambda r: r.value.get("status") != "error"
        )
    """

    def __init__(
        self,
        name: str,
        predicate: Callable[[DataRecord], bool]
    ):
        super().__init__(name)
        self._predicate = predicate

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        if self._predicate(record):
            return record
        return None


class MapProcessor(Processor):
    """
    Maps record fields to new structure.

    Usage:
        processor = MapProcessor(
            "map-fields",
            field_mapping={
                "user_id": "$.data.user.id",
                "timestamp": "$.timestamp"
            }
        )
    """

    def __init__(
        self,
        name: str,
        field_mapping: Dict[str, str],
        preserve_original: bool = False
    ):
        super().__init__(name)
        self._field_mapping = field_mapping
        self._preserve_original = preserve_original

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        new_value = {}

        if self._preserve_original:
            new_value = record.value.copy() if isinstance(record.value, dict) else {}

        for target_field, source_path in self._field_mapping.items():
            value = self._extract_value(record, source_path)
            if value is not None:
                new_value[target_field] = value

        return record.with_value(new_value)

    def _extract_value(self, record: DataRecord, path: str) -> Any:
        """Extract value using JSONPath-like syntax"""
        if path.startswith("$."):
            path = path[2:]

        parts = path.split(".")
        current = {"value": record.value, "metadata": record.metadata}

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current


class EnrichProcessor(Processor):
    """
    Enriches records with additional data from external sources.

    Usage:
        async def lookup_user(user_id: str) -> Dict:
            return await user_service.get(user_id)

        processor = EnrichProcessor(
            "enrich-user",
            enrichment_func=lookup_user,
            key_field="user_id",
            target_field="user_details"
        )
    """

    def __init__(
        self,
        name: str,
        enrichment_func: Callable[[Any], Awaitable[Dict[str, Any]]],
        key_field: str,
        target_field: str,
        cache_enabled: bool = True,
        cache_ttl_seconds: float = 300.0
    ):
        super().__init__(name)
        self._enrichment_func = enrichment_func
        self._key_field = key_field
        self._target_field = target_field
        self._cache_enabled = cache_enabled
        self._cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple] = {}

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        # Get key value
        key_value = self._get_field(record.value, self._key_field)
        if key_value is None:
            return record

        # Check cache
        enrichment_data = None
        if self._cache_enabled and key_value in self._cache:
            cached_value, cached_time = self._cache[key_value]
            if (datetime.utcnow() - cached_time).total_seconds() < self._cache_ttl:
                enrichment_data = cached_value

        # Fetch if not cached
        if enrichment_data is None:
            try:
                enrichment_data = await self._enrichment_func(key_value)
                if self._cache_enabled:
                    self._cache[key_value] = (enrichment_data, datetime.utcnow())
            except Exception as e:
                self._logger.warning(
                    "enrichment_failed",
                    key=key_value,
                    error=str(e)
                )
                return record

        # Add enrichment to record
        if isinstance(record.value, dict):
            new_value = record.value.copy()
            new_value[self._target_field] = enrichment_data
            return record.with_value(new_value)

        return record.with_metadata(self._target_field, enrichment_data)

    def _get_field(self, data: Any, field: str) -> Any:
        if isinstance(data, dict):
            return data.get(field)
        return None


class AggregateProcessor(Processor):
    """
    Aggregates records over time windows.

    Usage:
        processor = AggregateProcessor(
            "count-calls",
            window_seconds=60,
            aggregation_func=lambda records: len(records),
            group_by="agent_id"
        )
    """

    def __init__(
        self,
        name: str,
        window_seconds: float,
        aggregation_func: Callable[[List[DataRecord]], Any],
        group_by: Optional[str] = None,
        emit_empty: bool = False
    ):
        super().__init__(name)
        self._window_seconds = window_seconds
        self._aggregation_func = aggregation_func
        self._group_by = group_by
        self._emit_empty = emit_empty
        self._buffers: Dict[str, List[DataRecord]] = {}
        self._last_emit: Dict[str, datetime] = {}

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        # Get group key
        group_key = "default"
        if self._group_by and isinstance(record.value, dict):
            group_key = str(record.value.get(self._group_by, "default"))

        # Initialize buffer
        if group_key not in self._buffers:
            self._buffers[group_key] = []
            self._last_emit[group_key] = datetime.utcnow()

        self._buffers[group_key].append(record)

        # Check if window expired
        elapsed = (datetime.utcnow() - self._last_emit[group_key]).total_seconds()
        if elapsed >= self._window_seconds:
            records = self._buffers[group_key]
            self._buffers[group_key] = []
            self._last_emit[group_key] = datetime.utcnow()

            if records or self._emit_empty:
                aggregated_value = self._aggregation_func(records)
                return DataRecord(
                    key=group_key,
                    value={
                        "group": group_key,
                        "window_start": self._last_emit[group_key].isoformat(),
                        "count": len(records),
                        "result": aggregated_value
                    },
                    metadata={"aggregation": self.name}
                )

        return None


class RouteProcessor(Processor):
    """
    Routes records to different outputs based on conditions.

    Usage:
        processor = RouteProcessor(
            "route-by-type",
            routes={
                "calls": lambda r: r.value.get("type") == "call",
                "messages": lambda r: r.value.get("type") == "message"
            },
            default_route="other"
        )
    """

    def __init__(
        self,
        name: str,
        routes: Dict[str, Callable[[DataRecord], bool]],
        default_route: str = "default"
    ):
        super().__init__(name)
        self._routes = routes
        self._default_route = default_route

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        for route_name, condition in self._routes.items():
            if condition(record):
                return record.with_metadata("route", route_name)

        return record.with_metadata("route", self._default_route)


class FlatMapProcessor(Processor):
    """
    Flattens and maps records to multiple output records.

    Usage:
        processor = FlatMapProcessor(
            "split-words",
            lambda r: [{"word": w} for w in r.value.split()]
        )
    """

    def __init__(
        self,
        name: str,
        flat_map_func: Callable[[DataRecord], List[Any]]
    ):
        super().__init__(name)
        self._flat_map_func = flat_map_func
        self._output_queue: List[DataRecord] = []

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        results = self._flat_map_func(record)

        if not results:
            return None

        # Return first result, queue the rest
        for i, result in enumerate(results[1:]):
            self._output_queue.append(record.with_value(result))

        return record.with_value(results[0])

    async def process_batch(
        self,
        records: List[DataRecord]
    ) -> List[DataRecord]:
        output = []

        for record in records:
            results = self._flat_map_func(record)
            for result in results:
                output.append(record.with_value(result))

        return output


class DeduplicateProcessor(Processor):
    """
    Deduplicates records based on a key.

    Usage:
        processor = DeduplicateProcessor(
            "dedupe-calls",
            key_func=lambda r: r.value.get("call_id"),
            window_seconds=3600
        )
    """

    def __init__(
        self,
        name: str,
        key_func: Callable[[DataRecord], str],
        window_seconds: float = 3600.0
    ):
        super().__init__(name)
        self._key_func = key_func
        self._window_seconds = window_seconds
        self._seen: Dict[str, datetime] = {}

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        key = self._key_func(record)

        # Clean old entries
        now = datetime.utcnow()
        self._seen = {
            k: v for k, v in self._seen.items()
            if (now - v).total_seconds() < self._window_seconds
        }

        # Check if seen
        if key in self._seen:
            return None

        self._seen[key] = now
        return record


class ValidateProcessor(Processor):
    """
    Validates records against a schema.

    Usage:
        processor = ValidateProcessor(
            "validate-call",
            required_fields=["call_id", "agent_id", "timestamp"],
            field_types={"call_id": str, "duration": int}
        )
    """

    def __init__(
        self,
        name: str,
        required_fields: Optional[List[str]] = None,
        field_types: Optional[Dict[str, type]] = None,
        custom_validator: Optional[Callable[[DataRecord], bool]] = None,
        reject_invalid: bool = True
    ):
        super().__init__(name)
        self._required_fields = required_fields or []
        self._field_types = field_types or {}
        self._custom_validator = custom_validator
        self._reject_invalid = reject_invalid

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        errors = []

        if isinstance(record.value, dict):
            # Check required fields
            for field in self._required_fields:
                if field not in record.value:
                    errors.append(f"Missing required field: {field}")

            # Check field types
            for field, expected_type in self._field_types.items():
                if field in record.value:
                    if not isinstance(record.value[field], expected_type):
                        errors.append(
                            f"Invalid type for {field}: expected {expected_type.__name__}"
                        )

        # Custom validation
        if self._custom_validator and not self._custom_validator(record):
            errors.append("Custom validation failed")

        if errors:
            if self._reject_invalid:
                record.errors.extend(errors)
                return None
            else:
                return record.with_metadata("validation_errors", errors)

        return record.with_metadata("validated", True)


class JSONParseProcessor(Processor):
    """
    Parses JSON string values into objects.
    """

    def __init__(self, name: str, strict: bool = False):
        super().__init__(name)
        self._strict = strict

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        if isinstance(record.value, str):
            try:
                parsed = json.loads(record.value)
                return record.with_value(parsed)
            except json.JSONDecodeError as e:
                if self._strict:
                    return None
                record.errors.append(f"JSON parse error: {e}")
                return record

        return record


class TimestampProcessor(Processor):
    """
    Parses and normalizes timestamps.
    """

    def __init__(
        self,
        name: str,
        source_field: str,
        target_field: str = "parsed_timestamp",
        formats: Optional[List[str]] = None
    ):
        super().__init__(name)
        self._source_field = source_field
        self._target_field = target_field
        self._formats = formats or [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ]

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        if isinstance(record.value, dict):
            timestamp_str = record.value.get(self._source_field)
            if timestamp_str:
                parsed = self._parse_timestamp(timestamp_str)
                if parsed:
                    new_value = record.value.copy()
                    new_value[self._target_field] = parsed.isoformat()
                    return record.with_value(new_value)

        return record

    def _parse_timestamp(self, value: str) -> Optional[datetime]:
        for fmt in self._formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None


class SentimentProcessor(Processor):
    """
    Analyzes sentiment of text fields.
    """

    def __init__(
        self,
        name: str,
        text_field: str,
        output_field: str = "sentiment"
    ):
        super().__init__(name)
        self._text_field = text_field
        self._output_field = output_field

        # Simple lexicon-based sentiment
        self._positive_words = {
            "good", "great", "excellent", "amazing", "wonderful",
            "fantastic", "happy", "love", "perfect", "best",
            "thank", "thanks", "appreciate", "helpful", "awesome"
        }
        self._negative_words = {
            "bad", "terrible", "awful", "horrible", "hate",
            "worst", "poor", "disappointed", "angry", "frustrated",
            "problem", "issue", "wrong", "error", "fail"
        }

    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        if isinstance(record.value, dict):
            text = record.value.get(self._text_field, "")
            if text:
                sentiment = self._analyze(text)
                new_value = record.value.copy()
                new_value[self._output_field] = sentiment
                return record.with_value(new_value)

        return record

    def _analyze(self, text: str) -> Dict[str, Any]:
        words = set(text.lower().split())
        positive_count = len(words & self._positive_words)
        negative_count = len(words & self._negative_words)
        total = positive_count + negative_count

        if total == 0:
            score = 0.0
            label = "neutral"
        else:
            score = (positive_count - negative_count) / total
            if score > 0.2:
                label = "positive"
            elif score < -0.2:
                label = "negative"
            else:
                label = "neutral"

        return {
            "score": score,
            "label": label,
            "positive_count": positive_count,
            "negative_count": negative_count
        }
