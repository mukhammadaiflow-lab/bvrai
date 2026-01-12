"""
Distributed Cache System

Enterprise features:
- Consistent hash ring for node distribution
- Replication strategies
- Failure detection and recovery
- Cross-datacenter caching
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import hashlib
import struct
import logging
import time

logger = logging.getLogger(__name__)


class NodeState(str, Enum):
    """State of a cache node."""
    ACTIVE = "active"
    SUSPECT = "suspect"
    FAILED = "failed"
    DRAINING = "draining"


class ReplicationStrategy(str, Enum):
    """Data replication strategy."""
    NONE = "none"
    SYNC = "sync"  # Wait for all replicas
    ASYNC = "async"  # Fire-and-forget to replicas
    QUORUM = "quorum"  # Wait for majority


@dataclass
class CacheNode:
    """A node in the distributed cache cluster."""
    node_id: str
    host: str
    port: int
    weight: int = 100
    state: NodeState = NodeState.ACTIVE
    datacenter: str = "default"
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def address(self) -> str:
        """Get node address."""
        return f"{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "state": self.state.value,
            "datacenter": self.datacenter,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "failure_count": self.failure_count,
        }


class ConsistentHashRing:
    """
    Consistent hash ring for distributing keys across nodes.

    Uses virtual nodes for better distribution.
    """

    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self._ring: List[Tuple[int, str]] = []
        self._nodes: Dict[str, CacheNode] = {}
        self._lock = asyncio.Lock()

    def _hash(self, key: str) -> int:
        """Compute hash for a key."""
        key_bytes = key.encode('utf-8')
        hash_bytes = hashlib.md5(key_bytes).digest()
        return struct.unpack('>I', hash_bytes[:4])[0]

    async def add_node(self, node: CacheNode) -> None:
        """Add a node to the ring."""
        async with self._lock:
            self._nodes[node.node_id] = node

            # Add virtual nodes
            for i in range(self.virtual_nodes * node.weight // 100):
                virtual_key = f"{node.node_id}:{i}"
                hash_value = self._hash(virtual_key)
                self._ring.append((hash_value, node.node_id))

            # Sort ring
            self._ring.sort(key=lambda x: x[0])

        logger.info(f"Added node {node.node_id} to hash ring")

    async def remove_node(self, node_id: str) -> None:
        """Remove a node from the ring."""
        async with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]

            # Remove virtual nodes
            self._ring = [
                (h, nid) for h, nid in self._ring
                if nid != node_id
            ]

        logger.info(f"Removed node {node_id} from hash ring")

    def get_node(self, key: str) -> Optional[CacheNode]:
        """Get the node responsible for a key."""
        if not self._ring:
            return None

        hash_value = self._hash(key)

        # Binary search for first node with hash >= key hash
        for ring_hash, node_id in self._ring:
            if ring_hash >= hash_value:
                node = self._nodes.get(node_id)
                if node and node.state == NodeState.ACTIVE:
                    return node

        # Wrap around to first node
        for ring_hash, node_id in self._ring:
            node = self._nodes.get(node_id)
            if node and node.state == NodeState.ACTIVE:
                return node

        return None

    def get_nodes(self, key: str, count: int = 3) -> List[CacheNode]:
        """Get multiple nodes for a key (for replication)."""
        if not self._ring:
            return []

        hash_value = self._hash(key)
        nodes = []
        seen_ids: Set[str] = set()

        # Find nodes starting from key position
        for ring_hash, node_id in self._ring:
            if ring_hash >= hash_value and node_id not in seen_ids:
                node = self._nodes.get(node_id)
                if node and node.state == NodeState.ACTIVE:
                    nodes.append(node)
                    seen_ids.add(node_id)
                    if len(nodes) >= count:
                        return nodes

        # Wrap around if needed
        for ring_hash, node_id in self._ring:
            if node_id not in seen_ids:
                node = self._nodes.get(node_id)
                if node and node.state == NodeState.ACTIVE:
                    nodes.append(node)
                    seen_ids.add(node_id)
                    if len(nodes) >= count:
                        return nodes

        return nodes

    def get_all_nodes(self) -> List[CacheNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_active_nodes(self) -> List[CacheNode]:
        """Get all active nodes."""
        return [n for n in self._nodes.values() if n.state == NodeState.ACTIVE]


class DistributedCache:
    """
    Distributed cache with consistent hashing and replication.

    Features:
    - Consistent hashing for key distribution
    - Configurable replication
    - Automatic failover
    - Cross-datacenter support
    """

    def __init__(
        self,
        replication: ReplicationStrategy = ReplicationStrategy.ASYNC,
        replica_count: int = 2,
        read_from_replica: bool = True,
        failure_threshold: int = 3,
        heartbeat_interval: float = 5.0,
    ):
        self.replication = replication
        self.replica_count = replica_count
        self.read_from_replica = read_from_replica
        self.failure_threshold = failure_threshold
        self.heartbeat_interval = heartbeat_interval

        self._ring = ConsistentHashRing()
        self._node_clients: Dict[str, Any] = {}  # node_id -> Redis client
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def add_node(
        self,
        node: CacheNode,
        client: Any,
    ) -> None:
        """Add a cache node."""
        await self._ring.add_node(node)
        self._node_clients[node.node_id] = client
        logger.info(f"Added distributed cache node: {node.node_id}")

    async def remove_node(self, node_id: str) -> None:
        """Remove a cache node."""
        await self._ring.remove_node(node_id)
        if node_id in self._node_clients:
            del self._node_clients[node_id]
        logger.info(f"Removed distributed cache node: {node_id}")

    async def start(self) -> None:
        """Start the distributed cache."""
        self._running = True
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info("Distributed cache started")

    async def stop(self) -> None:
        """Stop the distributed cache."""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        logger.info("Distributed cache stopped")

    async def get(
        self,
        key: str,
        default: Any = None,
    ) -> Optional[Any]:
        """Get a value from the distributed cache."""
        if self.read_from_replica:
            nodes = self._ring.get_nodes(key, self.replica_count + 1)
        else:
            node = self._ring.get_node(key)
            nodes = [node] if node else []

        for node in nodes:
            client = self._node_clients.get(node.node_id)
            if client:
                try:
                    value = await client.get(key)
                    if value is not None:
                        return value
                except Exception as e:
                    logger.warning(f"Read from node {node.node_id} failed: {e}")
                    await self._mark_node_failure(node)

        return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in the distributed cache."""
        nodes = self._ring.get_nodes(key, self.replica_count + 1)

        if not nodes:
            logger.error("No nodes available for write")
            return False

        primary = nodes[0]
        replicas = nodes[1:]

        # Write to primary
        primary_client = self._node_clients.get(primary.node_id)
        if not primary_client:
            return False

        try:
            if ttl:
                await primary_client.setex(key, ttl, value)
            else:
                await primary_client.set(key, value)
        except Exception as e:
            logger.error(f"Write to primary {primary.node_id} failed: {e}")
            await self._mark_node_failure(primary)
            return False

        # Replicate based on strategy
        if self.replication == ReplicationStrategy.NONE:
            return True
        elif self.replication == ReplicationStrategy.SYNC:
            return await self._replicate_sync(key, value, ttl, replicas)
        elif self.replication == ReplicationStrategy.ASYNC:
            asyncio.create_task(self._replicate_async(key, value, ttl, replicas))
            return True
        elif self.replication == ReplicationStrategy.QUORUM:
            return await self._replicate_quorum(key, value, ttl, replicas)

        return True

    async def delete(self, key: str) -> bool:
        """Delete a key from all nodes."""
        nodes = self._ring.get_nodes(key, self.replica_count + 1)
        success = False

        for node in nodes:
            client = self._node_clients.get(node.node_id)
            if client:
                try:
                    await client.delete(key)
                    success = True
                except Exception as e:
                    logger.warning(f"Delete from node {node.node_id} failed: {e}")

        return success

    async def _replicate_sync(
        self,
        key: str,
        value: Any,
        ttl: Optional[int],
        replicas: List[CacheNode],
    ) -> bool:
        """Synchronous replication - wait for all replicas."""
        tasks = []
        for node in replicas:
            client = self._node_clients.get(node.node_id)
            if client:
                tasks.append(self._write_to_node(client, key, value, ttl, node))

        if not tasks:
            return True

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(r is True for r in results)

    async def _replicate_async(
        self,
        key: str,
        value: Any,
        ttl: Optional[int],
        replicas: List[CacheNode],
    ) -> None:
        """Asynchronous replication - fire and forget."""
        for node in replicas:
            client = self._node_clients.get(node.node_id)
            if client:
                try:
                    if ttl:
                        await client.setex(key, ttl, value)
                    else:
                        await client.set(key, value)
                except Exception as e:
                    logger.warning(f"Async replication to {node.node_id} failed: {e}")

    async def _replicate_quorum(
        self,
        key: str,
        value: Any,
        ttl: Optional[int],
        replicas: List[CacheNode],
    ) -> bool:
        """Quorum replication - wait for majority."""
        tasks = []
        for node in replicas:
            client = self._node_clients.get(node.node_id)
            if client:
                tasks.append(self._write_to_node(client, key, value, ttl, node))

        if not tasks:
            return True

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)

        # Quorum is majority of total (primary + replicas)
        total = len(replicas) + 1
        quorum = (total // 2) + 1

        # Primary write already succeeded, so we need (quorum - 1) replicas
        return success_count >= (quorum - 1)

    async def _write_to_node(
        self,
        client: Any,
        key: str,
        value: Any,
        ttl: Optional[int],
        node: CacheNode,
    ) -> bool:
        """Write to a specific node."""
        try:
            if ttl:
                await client.setex(key, ttl, value)
            else:
                await client.set(key, value)
            return True
        except Exception as e:
            logger.warning(f"Write to node {node.node_id} failed: {e}")
            await self._mark_node_failure(node)
            return False

    async def _mark_node_failure(self, node: CacheNode) -> None:
        """Mark a node as potentially failed."""
        node.failure_count += 1

        if node.failure_count >= self.failure_threshold:
            node.state = NodeState.FAILED
            logger.warning(f"Node {node.node_id} marked as failed")

    async def _health_check_loop(self) -> None:
        """Periodic health check of nodes."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                for node in self._ring.get_all_nodes():
                    client = self._node_clients.get(node.node_id)
                    if client:
                        try:
                            await client.ping()
                            node.last_heartbeat = datetime.utcnow()
                            node.failure_count = 0
                            if node.state == NodeState.SUSPECT:
                                node.state = NodeState.ACTIVE
                                logger.info(f"Node {node.node_id} recovered")
                        except Exception:
                            if node.state == NodeState.ACTIVE:
                                node.state = NodeState.SUSPECT
                                logger.warning(f"Node {node.node_id} marked as suspect")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of the cache cluster."""
        nodes = self._ring.get_all_nodes()
        return {
            "total_nodes": len(nodes),
            "active_nodes": len([n for n in nodes if n.state == NodeState.ACTIVE]),
            "suspect_nodes": len([n for n in nodes if n.state == NodeState.SUSPECT]),
            "failed_nodes": len([n for n in nodes if n.state == NodeState.FAILED]),
            "replication_strategy": self.replication.value,
            "replica_count": self.replica_count,
            "nodes": [n.to_dict() for n in nodes],
        }


class CrossDatacenterCache:
    """
    Cache with cross-datacenter replication.

    Features:
    - Local read preference
    - Async cross-DC replication
    - Conflict resolution
    """

    def __init__(
        self,
        local_datacenter: str,
        local_cache: DistributedCache,
    ):
        self.local_datacenter = local_datacenter
        self.local_cache = local_cache
        self._remote_caches: Dict[str, DistributedCache] = {}

    def add_remote_datacenter(
        self,
        datacenter: str,
        cache: DistributedCache,
    ) -> None:
        """Add a remote datacenter."""
        self._remote_caches[datacenter] = cache

    async def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Get from local datacenter first."""
        value = await self.local_cache.get(key)
        if value is not None:
            return value
        return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        replicate_globally: bool = True,
    ) -> bool:
        """Set in local and optionally replicate globally."""
        # Write to local
        success = await self.local_cache.set(key, value, ttl)

        # Async replication to remote DCs
        if replicate_globally and self._remote_caches:
            asyncio.create_task(self._replicate_to_remote_dcs(key, value, ttl))

        return success

    async def _replicate_to_remote_dcs(
        self,
        key: str,
        value: Any,
        ttl: Optional[int],
    ) -> None:
        """Replicate to all remote datacenters."""
        for dc, cache in self._remote_caches.items():
            try:
                await cache.set(key, value, ttl)
            except Exception as e:
                logger.warning(f"Replication to DC {dc} failed: {e}")

    async def delete(self, key: str, delete_globally: bool = True) -> bool:
        """Delete from local and optionally globally."""
        success = await self.local_cache.delete(key)

        if delete_globally and self._remote_caches:
            for dc, cache in self._remote_caches.items():
                try:
                    await cache.delete(key)
                except Exception as e:
                    logger.warning(f"Delete from DC {dc} failed: {e}")

        return success
