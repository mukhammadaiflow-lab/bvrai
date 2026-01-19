"""
Database Sharding System

Horizontal scaling with:
- Hash-based sharding
- Range-based sharding
- Shard routing
- Cross-shard queries
- Shard rebalancing
"""

from typing import Optional, Dict, Any, List, Callable, TypeVar, Generic, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import hashlib
import logging
import struct

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import ConnectionPool, DatabaseConfig, ConnectionManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ShardState(str, Enum):
    """Shard operational state."""
    ACTIVE = "active"
    READONLY = "readonly"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class ShardConfig:
    """Configuration for a single shard."""
    shard_id: str
    database_config: DatabaseConfig
    weight: int = 100  # For weighted routing
    state: ShardState = ShardState.ACTIVE
    min_key: Optional[Any] = None  # For range sharding
    max_key: Optional[Any] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shard_id": self.shard_id,
            "database": self.database_config.to_dict(),
            "weight": self.weight,
            "state": self.state.value,
            "min_key": self.min_key,
            "max_key": self.max_key,
            "tags": self.tags,
        }


@dataclass
class ShardKey:
    """A sharding key."""
    value: Any
    hash_value: Optional[int] = None

    def __post_init__(self):
        if self.hash_value is None:
            self.hash_value = self._compute_hash()

    def _compute_hash(self) -> int:
        """Compute consistent hash for the key."""
        key_bytes = str(self.value).encode("utf-8")
        hash_bytes = hashlib.md5(key_bytes).digest()
        # Use first 4 bytes as integer
        return struct.unpack(">I", hash_bytes[:4])[0]


class ShardingStrategy(ABC):
    """
    Base class for sharding strategies.

    Determines how to route requests to shards.
    """

    @abstractmethod
    def get_shard(self, key: ShardKey, shards: List[ShardConfig]) -> ShardConfig:
        """Get the shard for a given key."""
        pass

    @abstractmethod
    def get_shards_for_range(
        self,
        start_key: ShardKey,
        end_key: ShardKey,
        shards: List[ShardConfig],
    ) -> List[ShardConfig]:
        """Get shards that may contain keys in the range."""
        pass


class HashSharding(ShardingStrategy):
    """
    Consistent hash-based sharding.

    Distributes data evenly across shards using consistent hashing.
    Good for point queries, less optimal for range queries.
    """

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self._ring: List[Tuple[int, str]] = []
        self._shard_map: Dict[str, ShardConfig] = {}

    def build_ring(self, shards: List[ShardConfig]) -> None:
        """Build the consistent hash ring."""
        self._ring = []
        self._shard_map = {}

        for shard in shards:
            if shard.state == ShardState.OFFLINE:
                continue

            self._shard_map[shard.shard_id] = shard

            # Create virtual nodes
            for i in range(self.virtual_nodes * shard.weight // 100):
                virtual_key = f"{shard.shard_id}:{i}"
                hash_value = ShardKey(virtual_key).hash_value
                self._ring.append((hash_value, shard.shard_id))

        # Sort by hash value
        self._ring.sort(key=lambda x: x[0])

    def get_shard(self, key: ShardKey, shards: List[ShardConfig]) -> ShardConfig:
        """Get the shard for a given key using consistent hashing."""
        if not self._ring:
            self.build_ring(shards)

        if not self._ring:
            raise RuntimeError("No active shards available")

        # Find the first node with hash >= key hash
        for hash_value, shard_id in self._ring:
            if hash_value >= key.hash_value:
                return self._shard_map[shard_id]

        # Wrap around to first node
        return self._shard_map[self._ring[0][1]]

    def get_shards_for_range(
        self,
        start_key: ShardKey,
        end_key: ShardKey,
        shards: List[ShardConfig],
    ) -> List[ShardConfig]:
        """For hash sharding, range queries may hit all shards."""
        # Hash sharding doesn't preserve key ordering
        # Return all active shards for range queries
        return [s for s in shards if s.state in (ShardState.ACTIVE, ShardState.READONLY)]


class RangeSharding(ShardingStrategy):
    """
    Range-based sharding.

    Assigns key ranges to specific shards.
    Good for range queries, may lead to hotspots.
    """

    def get_shard(self, key: ShardKey, shards: List[ShardConfig]) -> ShardConfig:
        """Get the shard for a given key based on range."""
        for shard in shards:
            if shard.state == ShardState.OFFLINE:
                continue

            if shard.min_key is not None and key.value < shard.min_key:
                continue
            if shard.max_key is not None and key.value >= shard.max_key:
                continue

            return shard

        raise RuntimeError(f"No shard found for key {key.value}")

    def get_shards_for_range(
        self,
        start_key: ShardKey,
        end_key: ShardKey,
        shards: List[ShardConfig],
    ) -> List[ShardConfig]:
        """Get shards that contain keys in the range."""
        result = []

        for shard in shards:
            if shard.state == ShardState.OFFLINE:
                continue

            # Check if shard range overlaps with query range
            shard_min = shard.min_key
            shard_max = shard.max_key

            if shard_max is not None and start_key.value >= shard_max:
                continue
            if shard_min is not None and end_key.value < shard_min:
                continue

            result.append(shard)

        return result


class DirectorySharding(ShardingStrategy):
    """
    Directory-based sharding.

    Uses a lookup table to map keys to shards.
    Maximum flexibility but requires directory management.
    """

    def __init__(self):
        self._directory: Dict[Any, str] = {}
        self._shard_map: Dict[str, ShardConfig] = {}
        self._default_shard: Optional[str] = None

    def set_mapping(self, key: Any, shard_id: str) -> None:
        """Set the shard mapping for a key."""
        self._directory[key] = shard_id

    def set_mappings(self, mappings: Dict[Any, str]) -> None:
        """Set multiple shard mappings."""
        self._directory.update(mappings)

    def set_default_shard(self, shard_id: str) -> None:
        """Set the default shard for unmapped keys."""
        self._default_shard = shard_id

    def get_shard(self, key: ShardKey, shards: List[ShardConfig]) -> ShardConfig:
        """Get the shard for a given key from directory."""
        self._shard_map = {s.shard_id: s for s in shards}

        shard_id = self._directory.get(key.value)
        if shard_id is None:
            shard_id = self._default_shard

        if shard_id is None:
            raise RuntimeError(f"No shard mapping for key {key.value}")

        shard = self._shard_map.get(shard_id)
        if shard is None or shard.state == ShardState.OFFLINE:
            raise RuntimeError(f"Shard {shard_id} not available")

        return shard

    def get_shards_for_range(
        self,
        start_key: ShardKey,
        end_key: ShardKey,
        shards: List[ShardConfig],
    ) -> List[ShardConfig]:
        """Get all shards that might contain keys in range."""
        # For directory sharding, we need to check all mappings
        shard_ids = set()

        for key, shard_id in self._directory.items():
            if start_key.value <= key <= end_key.value:
                shard_ids.add(shard_id)

        self._shard_map = {s.shard_id: s for s in shards}
        return [
            self._shard_map[sid] for sid in shard_ids
            if sid in self._shard_map and self._shard_map[sid].state != ShardState.OFFLINE
        ]


class ShardRouter:
    """
    Routes queries to appropriate shards.

    Handles:
    - Single-shard routing
    - Multi-shard scatter-gather
    - Shard affinity
    """

    def __init__(
        self,
        strategy: ShardingStrategy,
        shards: List[ShardConfig],
    ):
        self.strategy = strategy
        self.shards = shards
        self._shard_pools: Dict[str, ConnectionPool] = {}

    async def initialize(self) -> None:
        """Initialize connection pools for all shards."""
        for shard in self.shards:
            pool = ConnectionPool(shard.database_config, shard.shard_id)
            await pool.initialize()
            self._shard_pools[shard.shard_id] = pool

    async def close(self) -> None:
        """Close all shard connections."""
        for pool in self._shard_pools.values():
            await pool.close()

    def get_shard_for_key(self, key: Any) -> ShardConfig:
        """Get the shard for a key."""
        shard_key = ShardKey(key)
        return self.strategy.get_shard(shard_key, self.shards)

    def get_shards_for_keys(self, keys: List[Any]) -> Dict[str, List[Any]]:
        """Group keys by their target shard."""
        shard_keys: Dict[str, List[Any]] = {}

        for key in keys:
            shard = self.get_shard_for_key(key)
            if shard.shard_id not in shard_keys:
                shard_keys[shard.shard_id] = []
            shard_keys[shard.shard_id].append(key)

        return shard_keys

    async def execute_on_shard(
        self,
        shard_id: str,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a query on a specific shard."""
        pool = self._shard_pools.get(shard_id)
        if pool is None:
            raise RuntimeError(f"Shard {shard_id} not found")

        return await pool.execute(query, params)

    async def execute_on_key(
        self,
        key: Any,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a query on the shard for a key."""
        shard = self.get_shard_for_key(key)
        return await self.execute_on_shard(shard.shard_id, query, params)

    async def scatter_gather(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        shard_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, Any]]:
        """
        Execute a query on multiple shards and gather results.

        Returns list of (shard_id, result) tuples.
        """
        if shard_ids is None:
            shard_ids = [s.shard_id for s in self.shards if s.state != ShardState.OFFLINE]

        async def execute_on_shard(shard_id: str):
            try:
                result = await self.execute_on_shard(shard_id, query, params)
                return (shard_id, result)
            except Exception as e:
                logger.error(f"Scatter-gather failed on shard {shard_id}: {e}")
                return (shard_id, None)

        results = await asyncio.gather(*[
            execute_on_shard(sid) for sid in shard_ids
        ])

        return list(results)

    async def execute_partitioned(
        self,
        keys: List[Any],
        query_template: str,
        key_param: str = "keys",
    ) -> Dict[str, Any]:
        """
        Execute a query partitioned by shard keys.

        The query template should have a placeholder for the keys.
        """
        shard_keys = self.get_shards_for_keys(keys)
        results = {}

        async def execute_on_shard(shard_id: str, shard_key_list: List[Any]):
            try:
                # Replace key placeholder with actual keys
                params = {key_param: tuple(shard_key_list)}
                result = await self.execute_on_shard(shard_id, query_template, params)
                return (shard_id, result)
            except Exception as e:
                logger.error(f"Partitioned query failed on shard {shard_id}: {e}")
                return (shard_id, None)

        gather_results = await asyncio.gather(*[
            execute_on_shard(sid, keys) for sid, keys in shard_keys.items()
        ])

        for shard_id, result in gather_results:
            results[shard_id] = result

        return results

    def get_pool(self, shard_id: str) -> Optional[ConnectionPool]:
        """Get connection pool for a shard."""
        return self._shard_pools.get(shard_id)

    def get_pool_for_key(self, key: Any) -> ConnectionPool:
        """Get connection pool for a key's shard."""
        shard = self.get_shard_for_key(key)
        pool = self._shard_pools.get(shard.shard_id)
        if pool is None:
            raise RuntimeError(f"Pool for shard {shard.shard_id} not found")
        return pool


class ShardManager:
    """
    Manages shard lifecycle and operations.

    Handles:
    - Shard provisioning
    - Rebalancing
    - Failover
    - Metrics collection
    """

    def __init__(self, router: ShardRouter):
        self.router = router
        self._rebalance_lock = asyncio.Lock()

    async def add_shard(self, shard: ShardConfig) -> None:
        """Add a new shard to the cluster."""
        self.router.shards.append(shard)

        pool = ConnectionPool(shard.database_config, shard.shard_id)
        await pool.initialize()
        self.router._shard_pools[shard.shard_id] = pool

        # Rebuild hash ring if using hash sharding
        if isinstance(self.router.strategy, HashSharding):
            self.router.strategy.build_ring(self.router.shards)

        logger.info(f"Added shard {shard.shard_id}")

    async def remove_shard(self, shard_id: str) -> None:
        """Remove a shard from the cluster."""
        # Find and remove shard
        self.router.shards = [s for s in self.router.shards if s.shard_id != shard_id]

        # Close connection pool
        pool = self.router._shard_pools.pop(shard_id, None)
        if pool:
            await pool.close()

        # Rebuild hash ring if using hash sharding
        if isinstance(self.router.strategy, HashSharding):
            self.router.strategy.build_ring(self.router.shards)

        logger.info(f"Removed shard {shard_id}")

    async def set_shard_state(self, shard_id: str, state: ShardState) -> None:
        """Update shard state."""
        for shard in self.router.shards:
            if shard.shard_id == shard_id:
                shard.state = state
                break

        # Rebuild hash ring if needed
        if isinstance(self.router.strategy, HashSharding):
            self.router.strategy.build_ring(self.router.shards)

        logger.info(f"Shard {shard_id} state changed to {state.value}")

    async def drain_shard(
        self,
        shard_id: str,
        target_shard_id: str,
        batch_size: int = 1000,
    ) -> int:
        """
        Drain data from one shard to another.

        Returns number of records moved.
        """
        async with self._rebalance_lock:
            await self.set_shard_state(shard_id, ShardState.DRAINING)

            source_pool = self.router.get_pool(shard_id)
            target_pool = self.router.get_pool(target_shard_id)

            if not source_pool or not target_pool:
                raise RuntimeError("Source or target pool not found")

            # This is a simplified implementation
            # Real implementation would need table-specific logic
            moved = 0
            logger.info(f"Draining shard {shard_id} to {target_shard_id}")

            # Mark as offline after draining
            await self.set_shard_state(shard_id, ShardState.OFFLINE)

            return moved

    async def health_check_all(self) -> Dict[str, bool]:
        """Run health checks on all shards."""
        results = {}

        for shard in self.router.shards:
            pool = self.router.get_pool(shard.shard_id)
            if pool:
                results[shard.shard_id] = await pool.health_check()
            else:
                results[shard.shard_id] = False

        return results

    def get_shard_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all shards."""
        metrics = {}

        for shard in self.router.shards:
            pool = self.router.get_pool(shard.shard_id)
            if pool:
                pool_metrics = pool.get_metrics()
                metrics[shard.shard_id] = {
                    "state": shard.state.value,
                    "weight": shard.weight,
                    **pool_metrics.to_dict(),
                }
            else:
                metrics[shard.shard_id] = {
                    "state": shard.state.value,
                    "weight": shard.weight,
                    "is_healthy": False,
                }

        return metrics

    def get_key_distribution(self, sample_keys: List[Any]) -> Dict[str, int]:
        """Get distribution of sample keys across shards."""
        distribution: Dict[str, int] = {}

        for key in sample_keys:
            shard = self.router.get_shard_for_key(key)
            distribution[shard.shard_id] = distribution.get(shard.shard_id, 0) + 1

        return distribution
