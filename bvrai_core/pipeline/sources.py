"""
Data Pipeline Sources
=====================

Collection of data source connectors for pipeline ingestion.

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

import structlog

from bvrai_core.pipeline.engine import DataRecord, DataSource

logger = structlog.get_logger(__name__)


class KafkaSource(DataSource):
    """
    Kafka consumer data source.
    """

    def __init__(
        self,
        name: str,
        topics: List[str],
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "pipeline-consumer",
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = False,
        max_poll_records: int = 500
    ):
        super().__init__(name)
        self._topics = topics
        self._bootstrap_servers = bootstrap_servers
        self._group_id = group_id
        self._auto_offset_reset = auto_offset_reset
        self._enable_auto_commit = enable_auto_commit
        self._max_poll_records = max_poll_records
        self._consumer = None
        self._pending_commits: Dict[str, int] = {}

    async def connect(self) -> None:
        try:
            from aiokafka import AIOKafkaConsumer

            self._consumer = AIOKafkaConsumer(
                *self._topics,
                bootstrap_servers=self._bootstrap_servers,
                group_id=self._group_id,
                auto_offset_reset=self._auto_offset_reset,
                enable_auto_commit=self._enable_auto_commit,
                max_poll_records=self._max_poll_records
            )
            await self._consumer.start()
            self._running = True
            self._logger.info("kafka_source_connected", topics=self._topics)

        except ImportError:
            self._logger.warning("aiokafka_not_installed")
            self._running = True  # Continue with mock mode

    async def disconnect(self) -> None:
        if self._consumer:
            await self._consumer.stop()
        self._running = False
        self._logger.info("kafka_source_disconnected")

    async def read(self) -> AsyncGenerator[DataRecord, None]:
        if not self._consumer:
            # Mock mode for development
            while self._running:
                await asyncio.sleep(1)
                yield DataRecord(
                    key="mock",
                    value={"type": "mock", "timestamp": datetime.utcnow().isoformat()},
                    source=self.name
                )
            return

        try:
            async for msg in self._consumer:
                if not self._running:
                    break

                record = DataRecord(
                    id=f"{msg.topic}-{msg.partition}-{msg.offset}",
                    key=msg.key.decode() if msg.key else None,
                    value=json.loads(msg.value.decode()) if msg.value else None,
                    timestamp=datetime.fromtimestamp(msg.timestamp / 1000),
                    partition=msg.partition,
                    offset=msg.offset,
                    source=self.name,
                    headers={k: v.decode() for k, v in msg.headers} if msg.headers else {}
                )

                self._pending_commits[f"{msg.topic}-{msg.partition}"] = msg.offset
                yield record

        except Exception as e:
            self._logger.error("kafka_read_error", error=str(e))

    async def commit(self, record: DataRecord) -> None:
        if self._consumer and not self._enable_auto_commit:
            await self._consumer.commit()


class RedisSource(DataSource):
    """
    Redis Streams/Pub-Sub data source.
    """

    def __init__(
        self,
        name: str,
        url: str = "redis://localhost:6379",
        channels: Optional[List[str]] = None,
        streams: Optional[List[str]] = None,
        consumer_group: str = "pipeline",
        consumer_name: Optional[str] = None,
        block_ms: int = 1000
    ):
        super().__init__(name)
        self._url = url
        self._channels = channels or []
        self._streams = streams or []
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name or f"consumer-{uuid4().hex[:8]}"
        self._block_ms = block_ms
        self._redis = None
        self._pubsub = None

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis

            self._redis = await aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True
            )

            if self._channels:
                self._pubsub = self._redis.pubsub()
                await self._pubsub.subscribe(*self._channels)

            # Create consumer groups for streams
            for stream in self._streams:
                try:
                    await self._redis.xgroup_create(
                        stream,
                        self._consumer_group,
                        id="0",
                        mkstream=True
                    )
                except Exception:
                    pass  # Group may already exist

            self._running = True
            self._logger.info("redis_source_connected")

        except ImportError:
            self._logger.warning("redis_not_installed")
            self._running = True

    async def disconnect(self) -> None:
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        self._running = False
        self._logger.info("redis_source_disconnected")

    async def read(self) -> AsyncGenerator[DataRecord, None]:
        if not self._redis:
            while self._running:
                await asyncio.sleep(1)
                yield DataRecord(
                    key="mock",
                    value={"source": "redis_mock"},
                    source=self.name
                )
            return

        # Read from pub/sub channels
        if self._pubsub:
            async for message in self._read_pubsub():
                yield message

        # Read from streams
        if self._streams:
            async for message in self._read_streams():
                yield message

    async def _read_pubsub(self) -> AsyncGenerator[DataRecord, None]:
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                if message and message["type"] == "message":
                    yield DataRecord(
                        key=message["channel"],
                        value=json.loads(message["data"]) if message["data"] else None,
                        source=self.name,
                        metadata={"channel": message["channel"]}
                    )
            except Exception as e:
                self._logger.error("pubsub_read_error", error=str(e))
                await asyncio.sleep(1)

    async def _read_streams(self) -> AsyncGenerator[DataRecord, None]:
        streams_dict = {s: ">" for s in self._streams}

        while self._running:
            try:
                messages = await self._redis.xreadgroup(
                    self._consumer_group,
                    self._consumer_name,
                    streams_dict,
                    count=100,
                    block=self._block_ms
                )

                if messages:
                    for stream_name, stream_messages in messages:
                        for msg_id, data in stream_messages:
                            yield DataRecord(
                                id=msg_id,
                                key=stream_name,
                                value=data,
                                source=self.name,
                                metadata={"stream": stream_name, "msg_id": msg_id}
                            )

            except Exception as e:
                self._logger.error("stream_read_error", error=str(e))
                await asyncio.sleep(1)

    async def commit(self, record: DataRecord) -> None:
        if self._redis and "msg_id" in record.metadata:
            stream = record.metadata.get("stream")
            msg_id = record.metadata.get("msg_id")
            if stream and msg_id:
                await self._redis.xack(stream, self._consumer_group, msg_id)


class WebSocketSource(DataSource):
    """
    WebSocket data source for real-time streaming.
    """

    def __init__(
        self,
        name: str,
        url: str,
        reconnect_interval: float = 5.0,
        ping_interval: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(name)
        self._url = url
        self._reconnect_interval = reconnect_interval
        self._ping_interval = ping_interval
        self._headers = headers or {}
        self._websocket = None

    async def connect(self) -> None:
        try:
            import websockets

            self._websocket = await websockets.connect(
                self._url,
                extra_headers=self._headers,
                ping_interval=self._ping_interval
            )
            self._running = True
            self._logger.info("websocket_source_connected", url=self._url)

        except ImportError:
            self._logger.warning("websockets_not_installed")
            self._running = True
        except Exception as e:
            self._logger.error("websocket_connect_error", error=str(e))
            self._running = True

    async def disconnect(self) -> None:
        if self._websocket:
            await self._websocket.close()
        self._running = False
        self._logger.info("websocket_source_disconnected")

    async def read(self) -> AsyncGenerator[DataRecord, None]:
        while self._running:
            try:
                if self._websocket:
                    async for message in self._websocket:
                        if not self._running:
                            break

                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            data = {"raw": message}

                        yield DataRecord(
                            value=data,
                            source=self.name
                        )
                else:
                    # Mock mode
                    await asyncio.sleep(1)
                    yield DataRecord(
                        value={"source": "websocket_mock"},
                        source=self.name
                    )

            except Exception as e:
                self._logger.error("websocket_read_error", error=str(e))
                await asyncio.sleep(self._reconnect_interval)
                await self.connect()

    async def commit(self, record: DataRecord) -> None:
        pass  # WebSocket doesn't have commits


class HTTPSource(DataSource):
    """
    HTTP polling data source.
    """

    def __init__(
        self,
        name: str,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        poll_interval_seconds: float = 10.0,
        timeout_seconds: float = 30.0
    ):
        super().__init__(name)
        self._url = url
        self._method = method
        self._headers = headers or {}
        self._params = params or {}
        self._poll_interval = poll_interval_seconds
        self._timeout = timeout_seconds
        self._session = None

    async def connect(self) -> None:
        try:
            import aiohttp

            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
            self._running = True
            self._logger.info("http_source_connected", url=self._url)

        except ImportError:
            self._logger.warning("aiohttp_not_installed")
            self._running = True

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
        self._running = False
        self._logger.info("http_source_disconnected")

    async def read(self) -> AsyncGenerator[DataRecord, None]:
        while self._running:
            try:
                if self._session:
                    async with self._session.request(
                        self._method,
                        self._url,
                        params=self._params
                    ) as response:
                        data = await response.json()

                        # Handle array response
                        if isinstance(data, list):
                            for item in data:
                                yield DataRecord(
                                    value=item,
                                    source=self.name,
                                    metadata={"status": response.status}
                                )
                        else:
                            yield DataRecord(
                                value=data,
                                source=self.name,
                                metadata={"status": response.status}
                            )
                else:
                    yield DataRecord(
                        value={"source": "http_mock"},
                        source=self.name
                    )

                await asyncio.sleep(self._poll_interval)

            except Exception as e:
                self._logger.error("http_poll_error", error=str(e))
                await asyncio.sleep(self._poll_interval)

    async def commit(self, record: DataRecord) -> None:
        pass


class InMemorySource(DataSource):
    """
    In-memory data source for testing and development.
    """

    def __init__(
        self,
        name: str,
        data: Optional[List[Any]] = None,
        emit_interval_ms: float = 100.0
    ):
        super().__init__(name)
        self._data = deque(data or [])
        self._emit_interval = emit_interval_ms / 1000

    async def connect(self) -> None:
        self._running = True
        self._logger.info("memory_source_connected")

    async def disconnect(self) -> None:
        self._running = False
        self._logger.info("memory_source_disconnected")

    async def read(self) -> AsyncGenerator[DataRecord, None]:
        while self._running:
            if self._data:
                item = self._data.popleft()
                yield DataRecord(
                    value=item,
                    source=self.name
                )
            await asyncio.sleep(self._emit_interval)

    async def commit(self, record: DataRecord) -> None:
        pass

    def push(self, data: Any) -> None:
        """Push data to the source"""
        self._data.append(data)

    def push_many(self, items: List[Any]) -> None:
        """Push multiple items"""
        self._data.extend(items)
