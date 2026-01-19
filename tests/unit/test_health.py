"""
Unit Tests for Health Check System

Tests for health check endpoints and service status monitoring.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from platform.api.health import (
    HealthStatus,
    ComponentType,
    ComponentHealth,
    SystemHealth,
    HealthCheckService,
    check_postgres,
    check_redis,
    check_rabbitmq,
    check_external_api,
)


# =============================================================================
# Health Status Tests
# =============================================================================


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"


class TestComponentType:
    """Tests for ComponentType enum."""

    def test_component_types(self):
        """Test that all expected component types exist."""
        assert ComponentType.DATABASE == "database"
        assert ComponentType.CACHE == "cache"
        assert ComponentType.QUEUE == "queue"
        assert ComponentType.STORAGE == "storage"
        assert ComponentType.EXTERNAL_SERVICE == "external_service"
        assert ComponentType.INTERNAL_SERVICE == "internal_service"


# =============================================================================
# Component Health Tests
# =============================================================================


class TestComponentHealth:
    """Tests for ComponentHealth model."""

    def test_creation(self):
        """Test creating a ComponentHealth instance."""
        health = ComponentHealth(
            name="PostgreSQL",
            status=HealthStatus.HEALTHY,
            component_type=ComponentType.DATABASE,
            response_time_ms=5.2,
            message="Connected",
        )

        assert health.name == "PostgreSQL"
        assert health.status == HealthStatus.HEALTHY
        assert health.component_type == ComponentType.DATABASE
        assert health.response_time_ms == 5.2
        assert health.message == "Connected"
        assert health.last_check is not None

    def test_unhealthy_component(self):
        """Test creating an unhealthy component."""
        health = ComponentHealth(
            name="Redis",
            status=HealthStatus.UNHEALTHY,
            component_type=ComponentType.CACHE,
            message="Connection refused",
            details={"error": "ECONNREFUSED"},
        )

        assert health.status == HealthStatus.UNHEALTHY
        assert health.details["error"] == "ECONNREFUSED"


# =============================================================================
# System Health Tests
# =============================================================================


class TestSystemHealth:
    """Tests for SystemHealth model."""

    def test_creation(self):
        """Test creating a SystemHealth instance."""
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            version="1.0.0",
            environment="test",
            uptime_seconds=3600,
        )

        assert health.status == HealthStatus.HEALTHY
        assert health.version == "1.0.0"
        assert health.environment == "test"
        assert health.uptime_seconds == 3600
        assert health.timestamp is not None

    def test_with_components(self):
        """Test system health with components."""
        db_health = ComponentHealth(
            name="PostgreSQL",
            status=HealthStatus.HEALTHY,
            component_type=ComponentType.DATABASE,
        )

        cache_health = ComponentHealth(
            name="Redis",
            status=HealthStatus.DEGRADED,
            component_type=ComponentType.CACHE,
        )

        health = SystemHealth(
            status=HealthStatus.DEGRADED,
            version="1.0.0",
            environment="test",
            uptime_seconds=3600,
            components={
                "postgres": db_health,
                "redis": cache_health,
            },
            total_components=2,
            healthy_components=1,
            degraded_components=1,
            unhealthy_components=0,
        )

        assert health.total_components == 2
        assert health.healthy_components == 1
        assert health.degraded_components == 1


# =============================================================================
# Health Check Service Tests
# =============================================================================


class TestHealthCheckService:
    """Tests for HealthCheckService."""

    @pytest.fixture
    def service(self):
        """Create a test health check service."""
        return HealthCheckService(version="1.0.0")

    def test_initialization(self, service):
        """Test service initialization."""
        assert service.version == "1.0.0"
        assert service.start_time > 0

    def test_register_check(self, service):
        """Test registering a health check."""
        async def dummy_check():
            return ComponentHealth(
                name="Dummy",
                status=HealthStatus.HEALTHY,
                component_type=ComponentType.INTERNAL_SERVICE,
            )

        service.register_check("dummy", dummy_check)
        assert "dummy" in service._checks

    @pytest.mark.asyncio
    async def test_run_check(self, service):
        """Test running a single check."""
        async def test_check():
            return ComponentHealth(
                name="Test",
                status=HealthStatus.HEALTHY,
                component_type=ComponentType.INTERNAL_SERVICE,
                message="All good",
            )

        service.register_check("test", test_check)
        result = await service.run_check("test")

        assert result is not None
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_run_nonexistent_check(self, service):
        """Test running a nonexistent check."""
        result = await service.run_check("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_run_all_checks(self, service):
        """Test running all checks."""
        async def healthy_check():
            return ComponentHealth(
                name="Healthy",
                status=HealthStatus.HEALTHY,
                component_type=ComponentType.INTERNAL_SERVICE,
            )

        async def degraded_check():
            return ComponentHealth(
                name="Degraded",
                status=HealthStatus.DEGRADED,
                component_type=ComponentType.INTERNAL_SERVICE,
            )

        service.register_check("healthy", healthy_check)
        service.register_check("degraded", degraded_check)

        health = await service.run_all_checks()

        assert health.total_components == 2
        assert health.healthy_components == 1
        assert health.degraded_components == 1
        assert health.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_overall_status_unhealthy(self, service):
        """Test that unhealthy component makes overall status unhealthy."""
        async def unhealthy_check():
            return ComponentHealth(
                name="Unhealthy",
                status=HealthStatus.UNHEALTHY,
                component_type=ComponentType.DATABASE,
            )

        async def healthy_check():
            return ComponentHealth(
                name="Healthy",
                status=HealthStatus.HEALTHY,
                component_type=ComponentType.CACHE,
            )

        service.register_check("unhealthy", unhealthy_check)
        service.register_check("healthy", healthy_check)

        health = await service.run_all_checks()

        assert health.status == HealthStatus.UNHEALTHY
        assert health.unhealthy_components == 1

    @pytest.mark.asyncio
    async def test_liveness_check(self, service):
        """Test liveness check."""
        result = await service.liveness_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_error_handling(self, service):
        """Test that check errors are handled gracefully."""
        async def failing_check():
            raise Exception("Check failed!")

        service.register_check("failing", failing_check)
        result = await service.run_check("failing")

        assert result is not None
        assert result.status == HealthStatus.UNHEALTHY
        assert "exception" in result.message.lower()


# =============================================================================
# Postgres Check Tests
# =============================================================================


class TestCheckPostgres:
    """Tests for PostgreSQL health check."""

    @pytest.mark.asyncio
    async def test_check_returns_component_health(self):
        """Test that check returns ComponentHealth."""
        with patch("platform.api.health.create_async_engine") as mock_engine:
            # Setup mock
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = (1,)
            mock_conn.execute = AsyncMock(return_value=mock_result)

            mock_engine_instance = AsyncMock()
            mock_engine_instance.connect = AsyncMock()
            mock_engine_instance.connect.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_engine_instance.connect.return_value.__aexit__ = AsyncMock()
            mock_engine_instance.dispose = AsyncMock()
            mock_engine.return_value = mock_engine_instance

            result = await check_postgres(timeout=1.0)

            assert isinstance(result, ComponentHealth)
            assert result.component_type == ComponentType.DATABASE

    @pytest.mark.asyncio
    async def test_check_timeout(self):
        """Test that timeout is handled."""
        with patch("platform.api.health.create_async_engine") as mock_engine:
            import asyncio
            mock_engine.side_effect = asyncio.TimeoutError()

            result = await check_postgres(timeout=0.001)

            assert result.status == HealthStatus.UNHEALTHY
            assert "timeout" in result.message.lower()


# =============================================================================
# Redis Check Tests
# =============================================================================


class TestCheckRedis:
    """Tests for Redis health check."""

    @pytest.mark.asyncio
    async def test_check_healthy(self):
        """Test healthy Redis check."""
        with patch("platform.api.health.redis") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.info = AsyncMock(return_value={"redis_version": "7.0.0"})
            mock_client.close = AsyncMock()
            mock_redis.asyncio.from_url.return_value = mock_client

            result = await check_redis()

            assert result.status == HealthStatus.HEALTHY
            assert result.component_type == ComponentType.CACHE

    @pytest.mark.asyncio
    async def test_check_import_error(self):
        """Test handling of missing redis package."""
        with patch.dict("sys.modules", {"redis.asyncio": None, "redis": None}):
            with patch("platform.api.health.redis", None):
                # This will cause ImportError
                result = await check_redis()

                assert result.status == HealthStatus.DEGRADED


# =============================================================================
# External API Check Tests
# =============================================================================


class TestCheckExternalApi:
    """Tests for external API health check."""

    @pytest.mark.asyncio
    async def test_check_healthy_api(self):
        """Test healthy external API."""
        with patch("platform.api.health.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_httpx.AsyncClient.return_value = mock_client

            result = await check_external_api(
                name="TestAPI",
                url="https://api.example.com/health",
            )

            assert result.status == HealthStatus.HEALTHY
            assert result.component_type == ComponentType.EXTERNAL_SERVICE

    @pytest.mark.asyncio
    async def test_check_unhealthy_api(self):
        """Test unhealthy external API (wrong status)."""
        with patch("platform.api.health.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 500

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_httpx.AsyncClient.return_value = mock_client

            result = await check_external_api(
                name="TestAPI",
                url="https://api.example.com/health",
                expected_status=200,
            )

            assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_api_timeout(self):
        """Test API timeout handling."""
        import asyncio

        with patch("platform.api.health.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_httpx.AsyncClient.return_value = mock_client

            result = await check_external_api(
                name="TestAPI",
                url="https://api.example.com/health",
                timeout=0.001,
            )

            assert result.status == HealthStatus.DEGRADED
            assert "timeout" in result.message.lower()
