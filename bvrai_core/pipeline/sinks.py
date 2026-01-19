"""
Data Pipeline Sinks
===================

Collection of data sink connectors for pipeline output.

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
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

from bvrai_core.pipeline.engine import DataRecord, DataSink

logger = structlog.get_logger(__name__)


class KafkaSink(DataSink):
    """
    Kafka producer data sink.
    """

    def __init__(
        self,
        name: str,
        topic: str,
        bootstrap_servers: str = "localhost:9092",
        key_field: Optional[str] = None,
        batch_size: int = 100,
        linger_ms: int = 10,
        compression_type: str = "gzip"
    ):
        super().__init__(name)
        self._topic = topic
        self._bootstrap_servers = bootstrap_servers
        self._key_field = key_field
        self._batch_size = batch_size
        self._linger_ms = linger_ms
        self._compression_type = compression_type
        self._producer = None

    async def connect(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer

            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                linger_ms=self._linger_ms,
                compression_type=self._compression_type
            )
            await self._producer.start()
            self._running = True
            self._logger.info("kafka_sink_connected", topic=self._topic)

        except ImportError:
            self._logger.warning("aiokafka_not_installed")
            self._running = True

    async def disconnect(self) -> None:
        if self._producer:
            await self._producer.stop()
        self._running = False
        self._logger.info("kafka_sink_disconnected")

    async def write(self, record: DataRecord) -> None:
        if not self._producer:
            return

        key = None
        if self._key_field and isinstance(record.value, dict):
            key = str(record.value.get(self._key_field, "")).encode()

        value = json.dumps(record.value).encode() if record.value else None

        await self._producer.send(
            self._topic,
            value=value,
            key=key,
            headers=[(k, v.encode()) for k, v in record.headers.items()]
        )

    async def write_batch(self, records: List[DataRecord]) -> None:
        if not self._producer:
            return

        for record in records:
            await self.write(record)

    async def flush(self) -> None:
        if self._producer:
            await self._producer.flush()


class RedisSink(DataSink):
    """
    Redis data sink (supports streams, lists, and pub/sub).
    """

    def __init__(
        self,
        name: str,
        url: str = "redis://localhost:6379",
        mode: str = "stream",  # stream, list, pubsub, hash
        key_prefix: str = "pipeline",
        max_len: int = 10000,
        ttl_seconds: Optional[int] = None
    ):
        super().__init__(name)
        self._url = url
        self._mode = mode
        self._key_prefix = key_prefix
        self._max_len = max_len
        self._ttl_seconds = ttl_seconds
        self._redis = None

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis

            self._redis = await aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True
            )
            self._running = True
            self._logger.info("redis_sink_connected", mode=self._mode)

        except ImportError:
            self._logger.warning("redis_not_installed")
            self._running = True

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.close()
        self._running = False
        self._logger.info("redis_sink_disconnected")

    async def write(self, record: DataRecord) -> None:
        if not self._redis:
            return

        key = f"{self._key_prefix}:{record.key or 'default'}"
        value = json.dumps(record.value) if isinstance(record.value, (dict, list)) else str(record.value)

        if self._mode == "stream":
            await self._redis.xadd(
                key,
                {"data": value, "timestamp": record.timestamp.isoformat()},
                maxlen=self._max_len
            )

        elif self._mode == "list":
            await self._redis.rpush(key, value)
            await self._redis.ltrim(key, -self._max_len, -1)

        elif self._mode == "pubsub":
            await self._redis.publish(key, value)

        elif self._mode == "hash":
            await self._redis.hset(key, record.id, value)

        if self._ttl_seconds:
            await self._redis.expire(key, self._ttl_seconds)

    async def write_batch(self, records: List[DataRecord]) -> None:
        if not self._redis:
            return

        async with self._redis.pipeline(transaction=True) as pipe:
            for record in records:
                key = f"{self._key_prefix}:{record.key or 'default'}"
                value = json.dumps(record.value) if isinstance(record.value, (dict, list)) else str(record.value)

                if self._mode == "stream":
                    pipe.xadd(
                        key,
                        {"data": value, "timestamp": record.timestamp.isoformat()},
                        maxlen=self._max_len
                    )
                elif self._mode == "list":
                    pipe.rpush(key, value)

            await pipe.execute()

    async def flush(self) -> None:
        pass  # Redis operations are synchronous


class DatabaseSink(DataSink):
    """
    SQL database data sink.
    """

    def __init__(
        self,
        name: str,
        connection_string: str,
        table: str,
        column_mapping: Optional[Dict[str, str]] = None,
        batch_size: int = 100,
        upsert: bool = False,
        upsert_key: Optional[str] = None
    ):
        super().__init__(name)
        self._connection_string = connection_string
        self._table = table
        self._column_mapping = column_mapping or {}
        self._batch_size = batch_size
        self._upsert = upsert
        self._upsert_key = upsert_key
        self._engine = None
        self._buffer: List[DataRecord] = []

    async def connect(self) -> None:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine

            self._engine = create_async_engine(
                self._connection_string,
                pool_size=5,
                max_overflow=10
            )
            self._running = True
            self._logger.info("database_sink_connected", table=self._table)

        except ImportError:
            self._logger.warning("sqlalchemy_not_installed")
            self._running = True

    async def disconnect(self) -> None:
        if self._engine:
            await self._engine.dispose()
        self._running = False
        self._logger.info("database_sink_disconnected")

    async def write(self, record: DataRecord) -> None:
        self._buffer.append(record)

        if len(self._buffer) >= self._batch_size:
            await self._flush_buffer()

    async def write_batch(self, records: List[DataRecord]) -> None:
        self._buffer.extend(records)

        while len(self._buffer) >= self._batch_size:
            await self._flush_buffer()

    async def flush(self) -> None:
        if self._buffer:
            await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        if not self._engine or not self._buffer:
            self._buffer = []
            return

        batch = self._buffer[:self._batch_size]
        self._buffer = self._buffer[self._batch_size:]

        # Convert records to rows
        rows = []
        for record in batch:
            if isinstance(record.value, dict):
                row = {}
                for col, field in self._column_mapping.items():
                    row[col] = record.value.get(field)
                row["id"] = record.id
                row["created_at"] = record.timestamp
                rows.append(row)

        if not rows:
            return

        try:
            from sqlalchemy import text

            async with self._engine.begin() as conn:
                # Build insert statement
                columns = list(rows[0].keys())
                placeholders = ", ".join([f":{col}" for col in columns])
                column_names = ", ".join(columns)

                sql = f"INSERT INTO {self._table} ({column_names}) VALUES ({placeholders})"

                for row in rows:
                    await conn.execute(text(sql), row)

        except Exception as e:
            self._logger.error("database_write_error", error=str(e))
            raise


class WebhookSink(DataSink):
    """
    HTTP webhook data sink.
    """

    def __init__(
        self,
        name: str,
        url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 30.0,
        retry_count: int = 3,
        retry_delay_seconds: float = 1.0,
        batch_endpoint: Optional[str] = None
    ):
        super().__init__(name)
        self._url = url
        self._method = method
        self._headers = headers or {"Content-Type": "application/json"}
        self._timeout = timeout_seconds
        self._retry_count = retry_count
        self._retry_delay = retry_delay_seconds
        self._batch_endpoint = batch_endpoint
        self._session = None

    async def connect(self) -> None:
        try:
            import aiohttp

            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
            self._running = True
            self._logger.info("webhook_sink_connected", url=self._url)

        except ImportError:
            self._logger.warning("aiohttp_not_installed")
            self._running = True

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
        self._running = False
        self._logger.info("webhook_sink_disconnected")

    async def write(self, record: DataRecord) -> None:
        if not self._session:
            return

        payload = {
            "id": record.id,
            "timestamp": record.timestamp.isoformat(),
            "data": record.value,
            "metadata": record.metadata
        }

        for attempt in range(self._retry_count):
            try:
                async with self._session.request(
                    self._method,
                    self._url,
                    json=payload
                ) as response:
                    if response.status < 300:
                        return
                    elif response.status >= 500:
                        raise Exception(f"Server error: {response.status}")
                    else:
                        self._logger.warning(
                            "webhook_non_success",
                            status=response.status
                        )
                        return

            except Exception as e:
                if attempt < self._retry_count - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                else:
                    raise

    async def write_batch(self, records: List[DataRecord]) -> None:
        if self._batch_endpoint:
            # Send as batch
            payload = {
                "records": [
                    {
                        "id": r.id,
                        "timestamp": r.timestamp.isoformat(),
                        "data": r.value,
                        "metadata": r.metadata
                    }
                    for r in records
                ]
            }

            if self._session:
                async with self._session.request(
                    self._method,
                    self._batch_endpoint,
                    json=payload
                ) as response:
                    if response.status >= 400:
                        raise Exception(f"Batch webhook failed: {response.status}")
        else:
            # Send individually
            for record in records:
                await self.write(record)

    async def flush(self) -> None:
        pass


class FileSink(DataSink):
    """
    File system data sink (JSON lines, CSV).
    """

    def __init__(
        self,
        name: str,
        path: str,
        format: str = "jsonl",  # jsonl, csv, json
        rotate_size_mb: int = 100,
        rotate_interval_hours: int = 24,
        compression: Optional[str] = None  # gzip, None
    ):
        super().__init__(name)
        self._path = path
        self._format = format
        self._rotate_size = rotate_size_mb * 1024 * 1024
        self._rotate_interval = rotate_interval_hours * 3600
        self._compression = compression
        self._file = None
        self._bytes_written = 0
        self._last_rotate = datetime.utcnow()

    async def connect(self) -> None:
        await self._open_file()
        self._running = True
        self._logger.info("file_sink_connected", path=self._path)

    async def disconnect(self) -> None:
        await self._close_file()
        self._running = False
        self._logger.info("file_sink_disconnected")

    async def _open_file(self) -> None:
        import aiofiles

        mode = "a"
        if self._compression == "gzip":
            import gzip
            self._file = gzip.open(f"{self._path}.gz", "at")
        else:
            self._file = await aiofiles.open(self._path, mode)

    async def _close_file(self) -> None:
        if self._file:
            if hasattr(self._file, "close"):
                if asyncio.iscoroutinefunction(self._file.close):
                    await self._file.close()
                else:
                    self._file.close()

    async def write(self, record: DataRecord) -> None:
        await self._check_rotation()

        if self._format == "jsonl":
            line = json.dumps({
                "id": record.id,
                "timestamp": record.timestamp.isoformat(),
                "data": record.value
            }) + "\n"
        elif self._format == "csv":
            # Simple CSV (flatten dict)
            if isinstance(record.value, dict):
                line = ",".join(str(v) for v in record.value.values()) + "\n"
            else:
                line = str(record.value) + "\n"
        else:
            line = json.dumps(record.value) + "\n"

        if self._file:
            if asyncio.iscoroutinefunction(self._file.write):
                await self._file.write(line)
            else:
                self._file.write(line)
            self._bytes_written += len(line.encode())

    async def write_batch(self, records: List[DataRecord]) -> None:
        for record in records:
            await self.write(record)

    async def flush(self) -> None:
        if self._file and hasattr(self._file, "flush"):
            if asyncio.iscoroutinefunction(self._file.flush):
                await self._file.flush()
            else:
                self._file.flush()

    async def _check_rotation(self) -> None:
        should_rotate = (
            self._bytes_written >= self._rotate_size or
            (datetime.utcnow() - self._last_rotate).total_seconds() >= self._rotate_interval
        )

        if should_rotate:
            await self._rotate()

    async def _rotate(self) -> None:
        await self._close_file()

        # Rename current file
        import os
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        new_name = f"{self._path}.{timestamp}"

        if os.path.exists(self._path):
            os.rename(self._path, new_name)

        await self._open_file()
        self._bytes_written = 0
        self._last_rotate = datetime.utcnow()
        self._logger.info("file_rotated", new_file=new_name)


class InMemorySink(DataSink):
    """
    In-memory data sink for testing.
    """

    def __init__(self, name: str, max_size: int = 10000):
        super().__init__(name)
        self._max_size = max_size
        self._records: deque = deque(maxlen=max_size)

    async def connect(self) -> None:
        self._running = True

    async def disconnect(self) -> None:
        self._running = False

    async def write(self, record: DataRecord) -> None:
        self._records.append(record)

    async def write_batch(self, records: List[DataRecord]) -> None:
        self._records.extend(records)

    async def flush(self) -> None:
        pass

    def get_records(self) -> List[DataRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()
