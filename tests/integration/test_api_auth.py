"""Integration tests for authentication API."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient


class TestAuthAPI:
    """Tests for authentication endpoints."""

    @pytest.mark.asyncio
    async def test_register_user(self, async_client: AsyncClient):
        """Test user registration."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "SecurePass123!",
                "name": "New User",
                "company": "Test Company",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "user" in data
        assert data["user"]["email"] == "newuser@example.com"

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, async_client: AsyncClient, test_user):
        """Test registration with existing email fails."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": test_user.email,
                "password": "SecurePass123!",
                "name": "Duplicate User",
            },
        )

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_register_weak_password(self, async_client: AsyncClient):
        """Test registration with weak password fails."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "weak@example.com",
                "password": "123",
                "name": "Weak Password User",
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_login_success(self, async_client: AsyncClient, test_user):
        """Test successful login."""
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "password123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, async_client: AsyncClient, test_user):
        """Test login with invalid credentials."""
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "wrongpassword",
            },
        )

        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, async_client: AsyncClient):
        """Test login with non-existent user."""
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user(self, authenticated_client: AsyncClient, test_user):
        """Test getting current user info."""
        response = await authenticated_client.get("/api/v1/auth/me")

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["id"] == test_user.id

    @pytest.mark.asyncio
    async def test_get_current_user_unauthorized(self, async_client: AsyncClient):
        """Test getting user info without auth."""
        response = await async_client.get("/api/v1/auth/me")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token(self, async_client: AsyncClient, test_user):
        """Test refreshing access token."""
        # First login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "password123",
            },
        )
        tokens = login_response.json()

        # Refresh token
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] != tokens["access_token"]

    @pytest.mark.asyncio
    async def test_refresh_invalid_token(self, async_client: AsyncClient):
        """Test refreshing with invalid token."""
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid_token"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_change_password(self, authenticated_client: AsyncClient):
        """Test changing password."""
        response = await authenticated_client.post(
            "/api/v1/auth/password/change",
            json={
                "current_password": "password123",
                "new_password": "NewSecurePass456!",
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_change_password_wrong_current(self, authenticated_client: AsyncClient):
        """Test changing password with wrong current password."""
        response = await authenticated_client.post(
            "/api/v1/auth/password/change",
            json={
                "current_password": "wrongpassword",
                "new_password": "NewSecurePass456!",
            },
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_logout(self, authenticated_client: AsyncClient):
        """Test logout."""
        response = await authenticated_client.post("/api/v1/auth/logout")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_request_password_reset(self, async_client: AsyncClient, test_user):
        """Test requesting password reset."""
        with patch("app.services.email.send_password_reset_email", new_callable=AsyncMock):
            response = await async_client.post(
                "/api/v1/auth/password/reset/request",
                json={"email": test_user.email},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_key_authentication(self, async_client: AsyncClient, test_user, db_session):
        """Test API key authentication."""
        # Create an API key
        from app.models.api_key import ApiKey
        from app.core.security import generate_api_key

        key, hashed = generate_api_key()
        api_key = ApiKey(
            id="key_123",
            name="Test Key",
            hashed_key=hashed,
            prefix=key[:8],
            user_id=test_user.id,
            tenant_id=test_user.tenant_id,
            scopes=["read", "write"],
        )
        db_session.add(api_key)
        await db_session.commit()

        # Use API key for authentication
        response = await async_client.get(
            "/api/v1/auth/me",
            headers={"X-API-Key": key},
        )

        assert response.status_code == 200


class TestAPIKeyManagement:
    """Tests for API key management."""

    @pytest.mark.asyncio
    async def test_create_api_key(self, authenticated_client: AsyncClient):
        """Test creating an API key."""
        response = await authenticated_client.post(
            "/api/v1/auth/api-keys",
            json={
                "name": "Production Key",
                "scopes": ["read", "write", "calls"],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "key" in data
        assert data["name"] == "Production Key"

    @pytest.mark.asyncio
    async def test_list_api_keys(self, authenticated_client: AsyncClient):
        """Test listing API keys."""
        # Create a key first
        await authenticated_client.post(
            "/api/v1/auth/api-keys",
            json={"name": "Test Key", "scopes": ["read"]},
        )

        response = await authenticated_client.get("/api/v1/auth/api-keys")

        assert response.status_code == 200
        data = response.json()
        assert "keys" in data
        assert len(data["keys"]) >= 1

    @pytest.mark.asyncio
    async def test_delete_api_key(self, authenticated_client: AsyncClient):
        """Test deleting an API key."""
        # Create a key
        create_response = await authenticated_client.post(
            "/api/v1/auth/api-keys",
            json={"name": "To Delete", "scopes": ["read"]},
        )
        key_id = create_response.json()["id"]

        # Delete it
        response = await authenticated_client.delete(f"/api/v1/auth/api-keys/{key_id}")

        assert response.status_code == 204


class TestOAuth:
    """Tests for OAuth authentication."""

    @pytest.mark.asyncio
    async def test_google_oauth_redirect(self, async_client: AsyncClient):
        """Test Google OAuth redirect."""
        response = await async_client.get(
            "/api/v1/auth/oauth/google",
            follow_redirects=False,
        )

        # Should redirect to Google
        assert response.status_code == 307 or response.status_code == 302

    @pytest.mark.asyncio
    async def test_google_oauth_callback(self, async_client: AsyncClient):
        """Test Google OAuth callback."""
        with patch("app.services.oauth.google.exchange_code") as mock_exchange, \
             patch("app.services.oauth.google.get_user_info") as mock_user_info:

            mock_exchange.return_value = {"access_token": "google_token"}
            mock_user_info.return_value = {
                "email": "googleuser@gmail.com",
                "name": "Google User",
                "picture": "https://example.com/avatar.jpg",
            }

            response = await async_client.get(
                "/api/v1/auth/oauth/google/callback",
                params={"code": "test_auth_code"},
            )

            # Should either create user and return tokens or redirect
            assert response.status_code in [200, 307]
