"""
Query Optimization System

Advanced query optimization with:
- Query analysis and explain plans
- Query caching with intelligent invalidation
- Slow query logging and alerting
- Index recommendations
- Query rewriting
"""

from typing import Optional, Dict, Any, List, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import asyncio
import hashlib
import logging
import re
import time
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of SQL queries."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    OTHER = "OTHER"


class ScanType(str, Enum):
    """Query scan types from EXPLAIN."""
    SEQ_SCAN = "Seq Scan"
    INDEX_SCAN = "Index Scan"
    INDEX_ONLY_SCAN = "Index Only Scan"
    BITMAP_SCAN = "Bitmap Index Scan"
    NESTED_LOOP = "Nested Loop"
    HASH_JOIN = "Hash Join"
    MERGE_JOIN = "Merge Join"


@dataclass
class QueryPlan:
    """Parsed query execution plan."""
    query: str
    plan_json: Dict[str, Any]
    total_cost: float = 0.0
    startup_cost: float = 0.0
    rows_estimate: int = 0
    width: int = 0
    actual_time_ms: Optional[float] = None
    actual_rows: Optional[int] = None
    scan_types: List[str] = field(default_factory=list)
    tables_accessed: List[str] = field(default_factory=list)
    indexes_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query[:200],  # Truncate for display
            "total_cost": self.total_cost,
            "startup_cost": self.startup_cost,
            "rows_estimate": self.rows_estimate,
            "actual_time_ms": self.actual_time_ms,
            "actual_rows": self.actual_rows,
            "scan_types": self.scan_types,
            "tables_accessed": self.tables_accessed,
            "indexes_used": self.indexes_used,
            "warnings": self.warnings,
        }


@dataclass
class SlowQuery:
    """A slow query record."""
    query_hash: str
    query: str
    execution_time_ms: float
    timestamp: datetime
    database: str
    params: Optional[Dict[str, Any]] = None
    plan: Optional[QueryPlan] = None
    call_count: int = 1
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_hash": self.query_hash,
            "query": self.query[:500],
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "database": self.database,
            "call_count": self.call_count,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.total_time_ms / self.call_count if self.call_count > 0 else 0,
        }


@dataclass
class IndexRecommendation:
    """A recommended index."""
    table: str
    columns: List[str]
    index_type: str = "btree"
    unique: bool = False
    reason: str = ""
    estimated_improvement: float = 0.0
    query_count: int = 0

    @property
    def name(self) -> str:
        """Generate index name."""
        cols = "_".join(self.columns)
        return f"ix_{self.table}_{cols}"

    def to_sql(self) -> str:
        """Generate CREATE INDEX statement."""
        unique = "UNIQUE " if self.unique else ""
        columns = ", ".join(f'"{c}"' for c in self.columns)
        return f'CREATE {unique}INDEX "{self.name}" ON "{self.table}" USING {self.index_type} ({columns})'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "table": self.table,
            "columns": self.columns,
            "index_type": self.index_type,
            "unique": self.unique,
            "reason": self.reason,
            "estimated_improvement": self.estimated_improvement,
            "sql": self.to_sql(),
        }


class QueryAnalyzer:
    """
    Analyzes query execution plans and performance.

    Features:
    - EXPLAIN ANALYZE execution
    - Plan parsing and optimization suggestions
    - Sequential scan detection
    - Join analysis
    """

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory

    async def analyze(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        execute: bool = False,
    ) -> QueryPlan:
        """
        Analyze a query using EXPLAIN.

        Args:
            query: SQL query to analyze
            params: Query parameters
            execute: If True, use EXPLAIN ANALYZE (actually runs query)
        """
        explain_query = f"EXPLAIN (FORMAT JSON, COSTS, VERBOSE"
        if execute:
            explain_query += ", ANALYZE, BUFFERS"
        explain_query += f") {query}"

        async with self._session_factory() as session:
            if params:
                result = await session.execute(text(explain_query), params)
            else:
                result = await session.execute(text(explain_query))

            plan_data = result.fetchone()[0]

        return self._parse_plan(query, plan_data)

    def _parse_plan(self, query: str, plan_data: Any) -> QueryPlan:
        """Parse EXPLAIN JSON output."""
        if isinstance(plan_data, str):
            plan_data = json.loads(plan_data)

        plan = plan_data[0]["Plan"] if isinstance(plan_data, list) else plan_data["Plan"]

        query_plan = QueryPlan(
            query=query,
            plan_json=plan,
            total_cost=plan.get("Total Cost", 0),
            startup_cost=plan.get("Startup Cost", 0),
            rows_estimate=plan.get("Plan Rows", 0),
            width=plan.get("Plan Width", 0),
        )

        # Parse actual execution stats if available
        if "Actual Total Time" in plan:
            query_plan.actual_time_ms = plan["Actual Total Time"]
        if "Actual Rows" in plan:
            query_plan.actual_rows = plan["Actual Rows"]

        # Extract scan types and tables
        self._extract_plan_details(plan, query_plan)

        # Generate warnings
        self._generate_warnings(query_plan)

        return query_plan

    def _extract_plan_details(self, node: Dict[str, Any], query_plan: QueryPlan) -> None:
        """Recursively extract details from plan nodes."""
        node_type = node.get("Node Type", "")
        query_plan.scan_types.append(node_type)

        # Extract table name
        if "Relation Name" in node:
            query_plan.tables_accessed.append(node["Relation Name"])

        # Extract index name
        if "Index Name" in node:
            query_plan.indexes_used.append(node["Index Name"])

        # Process child nodes
        for child in node.get("Plans", []):
            self._extract_plan_details(child, query_plan)

    def _generate_warnings(self, query_plan: QueryPlan) -> None:
        """Generate optimization warnings."""
        # Check for sequential scans on large tables
        if ScanType.SEQ_SCAN.value in query_plan.scan_types:
            if query_plan.rows_estimate > 10000:
                query_plan.warnings.append(
                    f"Sequential scan on {query_plan.rows_estimate} estimated rows - consider adding an index"
                )

        # Check for nested loops with many rows
        if ScanType.NESTED_LOOP.value in query_plan.scan_types:
            if query_plan.rows_estimate > 100000:
                query_plan.warnings.append(
                    "Nested loop join with large row estimate - consider query restructuring"
                )

        # Check for high cost
        if query_plan.total_cost > 10000:
            query_plan.warnings.append(
                f"High query cost ({query_plan.total_cost:.0f}) - review query optimization"
            )

    async def suggest_indexes(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[IndexRecommendation]:
        """Suggest indexes based on query analysis."""
        plan = await self.analyze(query, params)
        recommendations = []

        # Check for sequential scans
        if ScanType.SEQ_SCAN.value in plan.scan_types:
            for table in plan.tables_accessed:
                # Extract WHERE clause columns
                where_columns = self._extract_where_columns(query, table)
                if where_columns:
                    recommendations.append(IndexRecommendation(
                        table=table,
                        columns=where_columns,
                        reason="Sequential scan detected in WHERE clause",
                        estimated_improvement=plan.total_cost * 0.8,
                    ))

        # Check for ORDER BY without index
        order_columns = self._extract_order_columns(query)
        for table, columns in order_columns.items():
            if table in plan.tables_accessed:
                if not any(idx for idx in plan.indexes_used if any(c in idx for c in columns)):
                    recommendations.append(IndexRecommendation(
                        table=table,
                        columns=columns,
                        reason="ORDER BY without supporting index",
                        estimated_improvement=plan.total_cost * 0.5,
                    ))

        return recommendations

    def _extract_where_columns(self, query: str, table: str) -> List[str]:
        """Extract columns from WHERE clause for a table."""
        # Simplified extraction - real implementation would use SQL parser
        columns = []
        where_match = re.search(r"WHERE\s+(.+?)(?:ORDER|GROUP|LIMIT|$)", query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            # Find column references
            col_matches = re.findall(r'["\']?(\w+)["\']?\s*[=<>!]', where_clause)
            columns = list(set(col_matches))
        return columns[:3]  # Limit to 3 columns for composite index

    def _extract_order_columns(self, query: str) -> Dict[str, List[str]]:
        """Extract ORDER BY columns."""
        result = {}
        order_match = re.search(r"ORDER\s+BY\s+(.+?)(?:LIMIT|$)", query, re.IGNORECASE)
        if order_match:
            order_clause = order_match.group(1)
            # Parse table.column or column references
            for col_expr in order_clause.split(","):
                col_expr = col_expr.strip().split()[0]  # Remove ASC/DESC
                if "." in col_expr:
                    table, col = col_expr.replace('"', '').split(".")
                    if table not in result:
                        result[table] = []
                    result[table].append(col)
        return result


@dataclass
class CacheEntry:
    """A cached query result."""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)


class QueryCache:
    """
    Intelligent query result cache.

    Features:
    - TTL-based expiration
    - Tag-based invalidation
    - LRU eviction
    - Size-based limits
    """

    def __init__(
        self,
        max_size_mb: int = 100,
        default_ttl_seconds: int = 300,
        max_entries: int = 10000,
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = timedelta(seconds=default_ttl_seconds)
        self.max_entries = max_entries

        self._cache: Dict[str, CacheEntry] = {}
        self._tags: Dict[str, Set[str]] = defaultdict(set)  # tag -> keys
        self._current_size = 0
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()

    def _generate_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from query and params."""
        key_data = query
        if params:
            key_data += json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    async def get(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Get cached result for a query."""
        key = self._generate_key(query, params)

        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check expiration
            if datetime.utcnow() > entry.expires_at:
                await self._remove_entry(key)
                self._misses += 1
                return None

            entry.hit_count += 1
            self._hits += 1
            return entry.value

    async def set(
        self,
        query: str,
        value: Any,
        params: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Cache a query result."""
        key = self._generate_key(query, params)
        ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else self.default_ttl

        # Estimate size
        size_bytes = len(json.dumps(value, default=str).encode())

        async with self._lock:
            # Evict if necessary
            await self._ensure_space(size_bytes)

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + ttl,
                size_bytes=size_bytes,
                tags=set(tags) if tags else set(),
            )

            # Add to cache
            if key in self._cache:
                self._current_size -= self._cache[key].size_bytes
            self._cache[key] = entry
            self._current_size += size_bytes

            # Update tags
            for tag in entry.tags:
                self._tags[tag].add(key)

    async def invalidate(self, query: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """Invalidate a specific cached query."""
        key = self._generate_key(query, params)

        async with self._lock:
            return await self._remove_entry(key)

    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all cache entries with a tag."""
        async with self._lock:
            keys = list(self._tags.get(tag, set()))
            count = 0
            for key in keys:
                if await self._remove_entry(key):
                    count += 1
            return count

    async def invalidate_by_table(self, table: str) -> int:
        """Invalidate all cache entries for a table."""
        return await self.invalidate_by_tag(f"table:{table}")

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate entries matching a pattern."""
        async with self._lock:
            regex = re.compile(pattern)
            keys_to_remove = [k for k in self._cache.keys() if regex.search(k)]
            count = 0
            for key in keys_to_remove:
                if await self._remove_entry(key):
                    count += 1
            return count

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()
            self._tags.clear()
            self._current_size = 0

    async def _remove_entry(self, key: str) -> bool:
        """Remove a cache entry."""
        entry = self._cache.pop(key, None)
        if entry is None:
            return False

        self._current_size -= entry.size_bytes

        # Remove from tags
        for tag in entry.tags:
            self._tags[tag].discard(key)
            if not self._tags[tag]:
                del self._tags[tag]

        return True

    async def _ensure_space(self, needed_bytes: int) -> None:
        """Ensure there's space for new entry."""
        # Evict by entry count
        while len(self._cache) >= self.max_entries:
            await self._evict_lru()

        # Evict by size
        while self._current_size + needed_bytes > self.max_size_bytes:
            if not await self._evict_lru():
                break

    async def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False

        # Find LRU entry (lowest hit count, oldest)
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].hit_count, self._cache[k].created_at)
        )
        return await self._remove_entry(lru_key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "entries": len(self._cache),
            "size_bytes": self._current_size,
            "size_mb": self._current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "tags": len(self._tags),
        }


class IndexAdvisor:
    """
    Analyzes query patterns and recommends indexes.

    Features:
    - Query pattern analysis
    - Index impact estimation
    - Unused index detection
    - Index consolidation suggestions
    """

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory
        self._query_patterns: Dict[str, Dict[str, Any]] = {}
        self._analyzer = QueryAnalyzer(session_factory)

    async def record_query(self, query: str, execution_time_ms: float) -> None:
        """Record a query for pattern analysis."""
        # Normalize query
        normalized = self._normalize_query(query)
        query_hash = hashlib.md5(normalized.encode()).hexdigest()

        if query_hash not in self._query_patterns:
            self._query_patterns[query_hash] = {
                "query": normalized,
                "original": query,
                "count": 0,
                "total_time_ms": 0,
                "tables": self._extract_tables(query),
            }

        self._query_patterns[query_hash]["count"] += 1
        self._query_patterns[query_hash]["total_time_ms"] += execution_time_ms

    def _normalize_query(self, query: str) -> str:
        """Normalize query by replacing literals."""
        # Replace string literals
        normalized = re.sub(r"'[^']*'", "'?'", query)
        # Replace numeric literals
        normalized = re.sub(r"\b\d+\b", "?", normalized)
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return normalized

    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query."""
        # Simplified extraction
        tables = []
        from_match = re.search(r"FROM\s+([^\s,]+)", query, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1).strip('"'))

        join_matches = re.findall(r"JOIN\s+([^\s]+)", query, re.IGNORECASE)
        for match in join_matches:
            tables.append(match.strip('"'))

        return tables

    async def get_recommendations(
        self,
        min_query_count: int = 10,
        min_avg_time_ms: float = 100,
    ) -> List[IndexRecommendation]:
        """Get index recommendations based on query patterns."""
        recommendations = []

        for query_hash, pattern in self._query_patterns.items():
            if pattern["count"] < min_query_count:
                continue

            avg_time = pattern["total_time_ms"] / pattern["count"]
            if avg_time < min_avg_time_ms:
                continue

            # Analyze query
            try:
                suggestions = await self._analyzer.suggest_indexes(pattern["original"])
                for suggestion in suggestions:
                    suggestion.query_count = pattern["count"]
                    recommendations.append(suggestion)
            except Exception as e:
                logger.debug(f"Could not analyze query: {e}")

        # Deduplicate recommendations
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            key = f"{rec.table}:{','.join(rec.columns)}"
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)

        return sorted(unique_recommendations, key=lambda r: -r.estimated_improvement)

    async def get_unused_indexes(self) -> List[Dict[str, Any]]:
        """Get indexes that haven't been used."""
        async with self._session_factory() as session:
            result = await session.execute(text("""
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0
                AND indexrelname NOT LIKE 'pg_%'
                ORDER BY pg_relation_size(indexrelid) DESC
            """))

            return [
                {
                    "schema": row[0],
                    "table": row[1],
                    "index": row[2],
                    "scans": row[3],
                    "tuples_read": row[4],
                    "tuples_fetched": row[5],
                    "size": row[6],
                }
                for row in result.fetchall()
            ]


class SlowQueryLog:
    """
    Logs and analyzes slow queries.

    Features:
    - Configurable threshold
    - Query aggregation
    - Alerting integration
    - Historical analysis
    """

    def __init__(
        self,
        threshold_ms: float = 1000,
        max_entries: int = 1000,
        alert_callback: Optional[Callable[[SlowQuery], None]] = None,
    ):
        self.threshold_ms = threshold_ms
        self.max_entries = max_entries
        self.alert_callback = alert_callback

        self._queries: Dict[str, SlowQuery] = {}
        self._lock = asyncio.Lock()

    async def log(
        self,
        query: str,
        execution_time_ms: float,
        database: str = "default",
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[SlowQuery]:
        """Log a slow query."""
        if execution_time_ms < self.threshold_ms:
            return None

        # Normalize and hash query
        normalized = self._normalize_query(query)
        query_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]

        async with self._lock:
            if query_hash in self._queries:
                # Update existing entry
                entry = self._queries[query_hash]
                entry.call_count += 1
                entry.total_time_ms += execution_time_ms
                if execution_time_ms > entry.execution_time_ms:
                    entry.execution_time_ms = execution_time_ms
                    entry.timestamp = datetime.utcnow()
                    entry.params = params
            else:
                # Create new entry
                entry = SlowQuery(
                    query_hash=query_hash,
                    query=normalized,
                    execution_time_ms=execution_time_ms,
                    timestamp=datetime.utcnow(),
                    database=database,
                    params=params,
                    total_time_ms=execution_time_ms,
                )
                self._queries[query_hash] = entry

                # Evict if needed
                if len(self._queries) > self.max_entries:
                    self._evict_oldest()

        # Trigger alert
        if self.alert_callback:
            try:
                self.alert_callback(entry)
            except Exception as e:
                logger.error(f"Slow query alert callback failed: {e}")

        logger.warning(
            f"Slow query ({execution_time_ms:.2f}ms): {normalized[:100]}..."
        )

        return entry

    def _normalize_query(self, query: str) -> str:
        """Normalize query for grouping."""
        normalized = re.sub(r"'[^']*'", "'?'", query)
        normalized = re.sub(r"\b\d+\b", "?", normalized)
        normalized = " ".join(normalized.split())
        return normalized

    def _evict_oldest(self) -> None:
        """Evict oldest slow query entry."""
        if not self._queries:
            return

        oldest_hash = min(
            self._queries.keys(),
            key=lambda h: self._queries[h].timestamp
        )
        del self._queries[oldest_hash]

    def get_top_slow_queries(
        self,
        limit: int = 20,
        order_by: str = "avg_time",  # "avg_time", "total_time", "count", "max_time"
    ) -> List[SlowQuery]:
        """Get top slow queries."""
        queries = list(self._queries.values())

        if order_by == "avg_time":
            queries.sort(key=lambda q: q.total_time_ms / q.call_count, reverse=True)
        elif order_by == "total_time":
            queries.sort(key=lambda q: q.total_time_ms, reverse=True)
        elif order_by == "count":
            queries.sort(key=lambda q: q.call_count, reverse=True)
        else:  # max_time
            queries.sort(key=lambda q: q.execution_time_ms, reverse=True)

        return queries[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get slow query statistics."""
        if not self._queries:
            return {
                "total_slow_queries": 0,
                "unique_queries": 0,
            }

        queries = list(self._queries.values())
        total_calls = sum(q.call_count for q in queries)
        total_time = sum(q.total_time_ms for q in queries)

        return {
            "total_slow_queries": total_calls,
            "unique_queries": len(queries),
            "total_time_ms": total_time,
            "avg_time_ms": total_time / total_calls if total_calls > 0 else 0,
            "threshold_ms": self.threshold_ms,
        }

    def clear(self) -> None:
        """Clear all slow query logs."""
        self._queries.clear()
