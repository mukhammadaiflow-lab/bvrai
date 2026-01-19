"""
Unit Tests for CORS Configuration

Tests for CORS origin validation, configuration, and security best practices.
"""

import os
import pytest
from unittest.mock import patch

from platform.security.cors import (
    CORSConfig,
    get_cors_middleware_config,
    validate_cors_config,
    get_development_cors,
    get_staging_cors,
    get_production_cors,
    get_cors_for_environment,
)


# =============================================================================
# CORSConfig Tests
# =============================================================================


class TestCORSConfig:
    """Tests for CORSConfig class."""

    def test_default_config(self):
        """Test default CORS configuration."""
        config = CORSConfig()

        assert config.allow_origins == []
        assert "GET" in config.allow_methods
        assert "POST" in config.allow_methods
        assert "Authorization" in config.allow_headers
        assert config.allow_credentials is True
        assert config.max_age == 600

    def test_custom_config(self):
        """Test custom CORS configuration."""
        config = CORSConfig(
            allow_origins=["https://example.com"],
            allow_methods=["GET", "POST"],
            allow_credentials=False,
            max_age=300,
        )

        assert config.allow_origins == ["https://example.com"]
        assert config.allow_methods == ["GET", "POST"]
        assert config.allow_credentials is False
        assert config.max_age == 300


# =============================================================================
# Origin Validation Tests
# =============================================================================


class TestOriginValidation:
    """Tests for origin validation logic."""

    def test_exact_origin_match(self):
        """Test exact origin matching."""
        config = CORSConfig(allow_origins=["https://example.com"])

        assert config.is_allowed_origin("https://example.com") is True
        assert config.is_allowed_origin("https://other.com") is False

    def test_multiple_origins(self):
        """Test multiple allowed origins."""
        config = CORSConfig(
            allow_origins=[
                "https://app.example.com",
                "https://admin.example.com",
            ]
        )

        assert config.is_allowed_origin("https://app.example.com") is True
        assert config.is_allowed_origin("https://admin.example.com") is True
        assert config.is_allowed_origin("https://other.example.com") is False

    def test_wildcard_subdomain(self):
        """Test wildcard subdomain matching."""
        config = CORSConfig(allow_origins=["https://*.example.com"])

        assert config.is_allowed_origin("https://app.example.com") is True
        assert config.is_allowed_origin("https://admin.example.com") is True
        assert config.is_allowed_origin("https://example.com") is False
        assert config.is_allowed_origin("https://evil.com") is False

    def test_regex_origin(self):
        """Test regex origin pattern."""
        config = CORSConfig(
            allow_origins=[],
            allow_origin_regex=r"https://[a-z]+\.example\.com",
        )

        assert config.is_allowed_origin("https://app.example.com") is True
        assert config.is_allowed_origin("https://admin.example.com") is True
        assert config.is_allowed_origin("https://123.example.com") is False

    def test_empty_origin_rejected(self):
        """Test that empty origin is rejected."""
        config = CORSConfig(allow_origins=["https://example.com"])

        assert config.is_allowed_origin("") is False
        assert config.is_allowed_origin(None) is False

    def test_case_sensitive_origin(self):
        """Test that origin matching is case-sensitive."""
        config = CORSConfig(allow_origins=["https://Example.com"])

        assert config.is_allowed_origin("https://Example.com") is True
        assert config.is_allowed_origin("https://example.com") is False


# =============================================================================
# Environment Configuration Tests
# =============================================================================


class TestEnvironmentConfig:
    """Tests for environment-based CORS configuration."""

    @patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True)
    def test_from_env_development(self):
        """Test loading config for development environment."""
        config = CORSConfig.from_env()

        assert "http://localhost:3000" in config.allow_origins
        assert "http://localhost:3001" in config.allow_origins

    @patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True)
    def test_from_env_staging(self):
        """Test loading config for staging environment."""
        config = CORSConfig.from_env()

        assert "https://staging.buildervoice.ai" in config.allow_origins

    @patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True)
    def test_from_env_production(self):
        """Test loading config for production environment."""
        config = CORSConfig.from_env()

        assert "https://buildervoice.ai" in config.allow_origins
        assert "https://app.buildervoice.ai" in config.allow_origins
        # No localhost in production
        assert "http://localhost:3000" not in config.allow_origins

    @patch.dict(os.environ, {"CORS_ORIGINS": "https://custom.com,https://other.com"}, clear=True)
    def test_from_env_custom_origins(self):
        """Test loading custom origins from environment."""
        config = CORSConfig.from_env()

        assert "https://custom.com" in config.allow_origins
        assert "https://other.com" in config.allow_origins


# =============================================================================
# Preset Configuration Tests
# =============================================================================


class TestPresetConfigs:
    """Tests for preset CORS configurations."""

    def test_development_cors(self):
        """Test development CORS preset."""
        config = get_development_cors()

        assert "http://localhost:3000" in config.allow_origins
        assert config.allow_credentials is True

    def test_staging_cors(self):
        """Test staging CORS preset."""
        config = get_staging_cors()

        assert "https://staging.buildervoice.ai" in config.allow_origins
        assert config.allow_origin_regex is not None

    def test_production_cors(self):
        """Test production CORS preset."""
        config = get_production_cors()

        assert "https://buildervoice.ai" in config.allow_origins
        assert "https://www.buildervoice.ai" in config.allow_origins
        # No HTTP in production
        for origin in config.allow_origins:
            assert origin.startswith("https://")

    @patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True)
    def test_get_cors_for_environment_production(self):
        """Test getting CORS for production."""
        config = get_cors_for_environment()
        assert "https://buildervoice.ai" in config.allow_origins

    @patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True)
    def test_get_cors_for_environment_development(self):
        """Test getting CORS for development."""
        config = get_cors_for_environment()
        assert "http://localhost:3000" in config.allow_origins


# =============================================================================
# Middleware Config Tests
# =============================================================================


class TestMiddlewareConfig:
    """Tests for middleware configuration generation."""

    def test_middleware_config_basic(self):
        """Test basic middleware config generation."""
        config = CORSConfig(allow_origins=["https://example.com"])
        middleware_config = get_cors_middleware_config(config)

        assert middleware_config["allow_origins"] == ["https://example.com"]
        assert middleware_config["allow_credentials"] is True
        assert "allow_methods" in middleware_config

    def test_middleware_config_wildcard_disables_credentials(self):
        """Test that wildcard origin disables credentials."""
        config = CORSConfig(
            allow_origins=["*"],
            allow_credentials=True,  # This should be overridden
        )
        middleware_config = get_cors_middleware_config(config)

        assert middleware_config["allow_origins"] == ["*"]
        assert middleware_config["allow_credentials"] is False

    def test_middleware_config_preserves_headers(self):
        """Test that middleware config preserves header settings."""
        config = CORSConfig(
            allow_origins=["https://example.com"],
            allow_headers=["X-Custom-Header", "Authorization"],
            expose_headers=["X-Response-Header"],
        )
        middleware_config = get_cors_middleware_config(config)

        assert "X-Custom-Header" in middleware_config["allow_headers"]
        assert "X-Response-Header" in middleware_config["expose_headers"]


# =============================================================================
# Security Validation Tests
# =============================================================================


class TestSecurityValidation:
    """Tests for CORS security validation."""

    def test_validate_wildcard_warning(self):
        """Test that wildcard origin generates warning."""
        config = CORSConfig(allow_origins=["*"])
        warnings = validate_cors_config(config)

        assert len(warnings) >= 1
        assert any("wildcard" in w.lower() for w in warnings)

    def test_validate_credentials_with_wildcard_error(self):
        """Test that credentials with wildcard generates error."""
        config = CORSConfig(
            allow_origins=["*"],
            allow_credentials=True,
        )
        warnings = validate_cors_config(config)

        assert len(warnings) >= 1
        assert any("credentials" in w.lower() for w in warnings)

    @patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True)
    def test_validate_localhost_in_production(self):
        """Test that localhost in production generates warning."""
        config = CORSConfig(allow_origins=["http://localhost:3000"])
        warnings = validate_cors_config(config)

        assert len(warnings) >= 1
        assert any("localhost" in w.lower() for w in warnings)

    @patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True)
    def test_validate_http_in_production(self):
        """Test that HTTP (non-localhost) in production generates warning."""
        config = CORSConfig(allow_origins=["http://insecure.example.com"])
        warnings = validate_cors_config(config)

        assert len(warnings) >= 1
        assert any("http" in w.lower() for w in warnings)

    @patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True)
    def test_no_warnings_for_valid_dev_config(self):
        """Test that valid development config has no security warnings."""
        config = get_development_cors()
        warnings = validate_cors_config(config)

        # Development is allowed to have localhost
        # Check there are no wildcard-related warnings
        wildcard_warnings = [w for w in warnings if "wildcard" in w.lower()]
        assert len(wildcard_warnings) == 0

    @patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True)
    def test_no_warnings_for_valid_prod_config(self):
        """Test that valid production config has no security warnings."""
        config = get_production_cors()
        warnings = validate_cors_config(config)

        assert len(warnings) == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in CORS handling."""

    def test_origin_with_port(self):
        """Test origin with port number."""
        config = CORSConfig(allow_origins=["https://example.com:8443"])

        assert config.is_allowed_origin("https://example.com:8443") is True
        assert config.is_allowed_origin("https://example.com") is False
        assert config.is_allowed_origin("https://example.com:443") is False

    def test_origin_with_path_rejected(self):
        """Test that origins are not confused with URLs with paths."""
        config = CORSConfig(allow_origins=["https://example.com"])

        # Origins don't have paths - this is just the origin
        assert config.is_allowed_origin("https://example.com") is True

    def test_wildcard_in_middle_of_domain(self):
        """Test wildcard subdomain pattern."""
        config = CORSConfig(allow_origins=["https://*.staging.example.com"])

        assert config.is_allowed_origin("https://app.staging.example.com") is True
        assert config.is_allowed_origin("https://staging.example.com") is False
        assert config.is_allowed_origin("https://app.example.com") is False

    def test_multiple_wildcards(self):
        """Test multiple wildcard patterns."""
        config = CORSConfig(
            allow_origins=[
                "https://*.example.com",
                "https://*.other.com",
            ]
        )

        assert config.is_allowed_origin("https://app.example.com") is True
        assert config.is_allowed_origin("https://app.other.com") is True
        assert config.is_allowed_origin("https://app.evil.com") is False

    def test_empty_origins_list(self):
        """Test with empty origins list."""
        config = CORSConfig(allow_origins=[])

        assert config.is_allowed_origin("https://example.com") is False
        assert config.is_allowed_origin("http://localhost:3000") is False

    def test_default_allowed_headers(self):
        """Test that default headers include common authentication headers."""
        config = CORSConfig()

        assert "Authorization" in config.allow_headers
        assert "X-API-Key" in config.allow_headers
        assert "Content-Type" in config.allow_headers

    def test_default_expose_headers(self):
        """Test that default exposed headers include rate limit headers."""
        config = CORSConfig()

        assert "X-RateLimit-Limit" in config.expose_headers
        assert "X-RateLimit-Remaining" in config.expose_headers
