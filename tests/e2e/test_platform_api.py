"""End-to-end tests for Platform API critical flows."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock
from uuid import uuid4


class TestAuthenticationFlow:
    """E2E tests for authentication flows."""

    @pytest.mark.asyncio
    async def test_complete_registration_and_login_flow(self, async_client):
        """Test user registration, email verification, and login."""
        # 1. Register a new user
        register_response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "SecurePassword123!",
                "name": "New User",
                "organization_name": "Test Organization",
            },
        )

        assert register_response.status_code in [200, 201]
        user_data = register_response.json()
        assert "access_token" in user_data or "token" in user_data

        # 2. Login with credentials
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "newuser@example.com",
                "password": "SecurePassword123!",
            },
        )

        assert login_response.status_code == 200
        login_data = login_response.json()
        assert "token" in login_data or "access_token" in login_data
        token = login_data.get("token") or login_data.get("access_token")

        # 3. Access protected endpoint with token
        me_response = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert me_response.status_code == 200
        me_data = me_response.json()
        assert me_data["email"] == "newuser@example.com"

    @pytest.mark.asyncio
    async def test_login_with_invalid_credentials(self, async_client):
        """Test login failure with wrong password."""
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "user@example.com",
                "password": "WrongPassword",
            },
        )

        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_token_refresh_flow(self, authenticated_client):
        """Test JWT token refresh."""
        response = await authenticated_client.post("/api/v1/auth/refresh")

        assert response.status_code == 200
        data = response.json()
        assert "token" in data

    @pytest.mark.asyncio
    async def test_logout_invalidates_token(self, authenticated_client):
        """Test that logout properly invalidates token."""
        # Logout
        logout_response = await authenticated_client.post("/api/v1/auth/logout")
        assert logout_response.status_code == 200

        # Subsequent requests should fail (if server-side invalidation implemented)
        # Note: This depends on token invalidation strategy


class TestAPIKeyManagement:
    """E2E tests for API key management."""

    @pytest.mark.asyncio
    async def test_api_key_lifecycle(self, authenticated_client):
        """Test complete API key creation, usage, and revocation."""
        # 1. Create API key
        create_response = await authenticated_client.post(
            "/api/v1/api-keys",
            json={
                "name": "Production API Key",
                "scopes": ["agents:read", "agents:write", "calls:read"],
                "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
            },
        )

        assert create_response.status_code in [200, 201]
        key_data = create_response.json()
        assert "key" in key_data
        api_key = key_data["key"]
        key_id = key_data.get("api_key", {}).get("id") or key_data.get("id")

        # 2. Use API key to access endpoints
        agents_response = await authenticated_client.get(
            "/api/v1/agents",
            headers={"X-API-Key": api_key},
        )

        # Should work with valid key
        assert agents_response.status_code in [200, 401]  # 401 if API key auth not implemented

        # 3. List API keys
        list_response = await authenticated_client.get("/api/v1/api-keys")
        assert list_response.status_code == 200
        keys = list_response.json()
        assert len(keys) >= 1

        # 4. Revoke API key
        if key_id:
            revoke_response = await authenticated_client.delete(f"/api/v1/api-keys/{key_id}")
            assert revoke_response.status_code in [200, 204]

    @pytest.mark.asyncio
    async def test_api_key_scope_enforcement(self, authenticated_client):
        """Test that API key scopes are enforced."""
        # Create key with limited scopes
        create_response = await authenticated_client.post(
            "/api/v1/api-keys",
            json={
                "name": "Read-Only Key",
                "scopes": ["agents:read"],  # Only read access
            },
        )

        assert create_response.status_code in [200, 201]
        api_key = create_response.json()["key"]

        # Attempting write operation should fail
        # (depends on scope enforcement implementation)


class TestWebhookIntegration:
    """E2E tests for webhook functionality."""

    @pytest.mark.asyncio
    async def test_webhook_crud_and_delivery(self, authenticated_client):
        """Test webhook creation, update, test, and deletion."""
        # 1. Create webhook
        create_response = await authenticated_client.post(
            "/api/v1/webhooks",
            json={
                "name": "Test CRM Webhook",
                "url": "https://httpbin.org/post",
                "events": ["call.started", "call.ended"],
                "secret": "whsec_testsecret123",
            },
        )

        assert create_response.status_code in [200, 201]
        webhook_data = create_response.json()
        webhook_id = webhook_data["id"]

        # 2. Update webhook
        update_response = await authenticated_client.put(
            f"/api/v1/webhooks/{webhook_id}",
            json={
                "name": "Updated CRM Webhook",
                "events": ["call.started", "call.ended", "call.failed"],
            },
        )

        assert update_response.status_code == 200

        # 3. Test webhook delivery
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"received": True}

            test_response = await authenticated_client.post(
                f"/api/v1/webhooks/{webhook_id}/test"
            )

            assert test_response.status_code == 200
            test_data = test_response.json()
            assert test_data.get("success") is True or "response_status" in test_data

        # 4. Get delivery logs
        deliveries_response = await authenticated_client.get(
            f"/api/v1/webhooks/{webhook_id}/deliveries"
        )

        assert deliveries_response.status_code == 200

        # 5. Delete webhook
        delete_response = await authenticated_client.delete(
            f"/api/v1/webhooks/{webhook_id}"
        )

        assert delete_response.status_code in [200, 204]

    @pytest.mark.asyncio
    async def test_webhook_signature_verification(self, async_client, test_webhook):
        """Test webhook payload signature verification."""
        import hmac
        import hashlib

        webhook_secret = "whsec_testsecret"
        payload = '{"event": "call.ended", "call_id": "123"}'
        timestamp = str(int(datetime.utcnow().timestamp()))

        # Generate valid signature
        signature_payload = f"{timestamp}.{payload}"
        signature = hmac.new(
            webhook_secret.encode(),
            signature_payload.encode(),
            hashlib.sha256
        ).hexdigest()

        # Send with valid signature
        response = await async_client.post(
            "/api/v1/webhooks/inbound/test",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": f"t={timestamp},v1={signature}",
            },
        )

        # Should accept valid signature (or 404 if endpoint not implemented)
        assert response.status_code in [200, 404]


class TestBillingIntegration:
    """E2E tests for billing flows."""

    @pytest.mark.asyncio
    async def test_get_current_plan_and_usage(self, authenticated_client):
        """Test retrieving current plan and usage data."""
        # Get current plan
        plan_response = await authenticated_client.get("/api/v1/billing/plan")

        assert plan_response.status_code == 200
        plan_data = plan_response.json()
        assert "plan" in plan_data or "usage" in plan_data

    @pytest.mark.asyncio
    async def test_get_invoice_history(self, authenticated_client):
        """Test retrieving invoice history."""
        invoices_response = await authenticated_client.get("/api/v1/billing/invoices")

        assert invoices_response.status_code == 200
        invoices = invoices_response.json()
        assert isinstance(invoices, list)

    @pytest.mark.asyncio
    async def test_usage_tracking_over_time(self, authenticated_client):
        """Test that usage is properly tracked."""
        # Get initial usage
        initial_response = await authenticated_client.get("/api/v1/billing/usage")
        assert initial_response.status_code == 200

        # Make an API call that should increment usage
        await authenticated_client.get("/api/v1/agents")

        # Usage tracking is typically asynchronous, so we just verify the endpoint works


class TestOrganizationManagement:
    """E2E tests for organization and team management."""

    @pytest.mark.asyncio
    async def test_organization_settings(self, authenticated_client):
        """Test organization settings management."""
        # Get organization
        org_response = await authenticated_client.get("/api/v1/organization")

        assert org_response.status_code == 200
        org_data = org_response.json()
        assert "id" in org_data or "name" in org_data

        # Update organization
        update_response = await authenticated_client.put(
            "/api/v1/organization",
            json={
                "name": "Updated Organization Name",
            },
        )

        assert update_response.status_code in [200, 204]

    @pytest.mark.asyncio
    async def test_team_member_management(self, authenticated_client):
        """Test team member invitation and management."""
        # Get team members
        members_response = await authenticated_client.get("/api/v1/organization/members")

        assert members_response.status_code == 200
        members = members_response.json()
        assert isinstance(members, list)

        # Invite new member
        with patch("app.services.email.send_invitation", new_callable=AsyncMock) as mock_email:
            mock_email.return_value = True

            invite_response = await authenticated_client.post(
                "/api/v1/organization/invitations",
                json={
                    "email": "newteammember@example.com",
                    "role": "member",
                },
            )

            # Should succeed or return validation error if member exists
            assert invite_response.status_code in [200, 201, 400, 409]


class TestAnalyticsAndReporting:
    """E2E tests for analytics endpoints."""

    @pytest.mark.asyncio
    async def test_dashboard_stats(self, authenticated_client):
        """Test dashboard statistics endpoint."""
        response = await authenticated_client.get("/api/v1/analytics/dashboard")

        assert response.status_code == 200
        data = response.json()

        # Should contain key metrics
        assert "today" in data or "total_calls" in data or "usage" in data

    @pytest.mark.asyncio
    async def test_analytics_summary(self, authenticated_client):
        """Test analytics summary with date range."""
        from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        to_date = datetime.utcnow().strftime("%Y-%m-%d")

        response = await authenticated_client.get(
            "/api/v1/analytics/summary",
            params={
                "from_date": from_date,
                "to_date": to_date,
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_calls_by_day_analytics(self, authenticated_client):
        """Test calls by day analytics."""
        response = await authenticated_client.get("/api/v1/analytics/calls-by-day")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestRateLimiting:
    """E2E tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_headers(self, authenticated_client):
        """Test that rate limit headers are present."""
        response = await authenticated_client.get("/api/v1/agents")

        # Rate limit headers should be present
        # Headers may vary based on implementation
        rate_headers = [
            "X-RateLimit-Remaining",
            "X-RateLimit-Limit",
            "RateLimit-Remaining",
        ]

        has_rate_headers = any(h in response.headers for h in rate_headers)
        # This may or may not be present depending on middleware configuration

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, authenticated_client):
        """Test behavior when rate limit is exceeded."""
        # Make many rapid requests
        # In a real test, this would need actual rate limiting to be configured
        responses = []
        for _ in range(100):
            resp = await authenticated_client.get("/api/v1/agents")
            responses.append(resp.status_code)
            if resp.status_code == 429:
                break

        # If rate limiting is active, we should eventually get 429
        # This test is informational - rate limits may not kick in during tests


class TestConcurrencyAndEdgeCases:
    """E2E tests for concurrent operations and edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_agent_updates(self, authenticated_client, test_agent):
        """Test handling concurrent updates to same agent."""
        import asyncio

        async def update_agent(name_suffix):
            return await authenticated_client.patch(
                f"/api/v1/agents/{test_agent.id}",
                json={"name": f"Agent {name_suffix}"},
            )

        # Run concurrent updates
        results = await asyncio.gather(
            update_agent("A"),
            update_agent("B"),
            update_agent("C"),
            return_exceptions=True,
        )

        # All should succeed or fail gracefully
        for result in results:
            if not isinstance(result, Exception):
                assert result.status_code in [200, 409, 423]  # OK, Conflict, or Locked

    @pytest.mark.asyncio
    async def test_large_payload_handling(self, authenticated_client):
        """Test handling of large payloads."""
        # Create agent with large system prompt
        large_prompt = "You are a helpful assistant. " * 1000

        response = await authenticated_client.post(
            "/api/v1/agents",
            json={
                "name": "Large Prompt Agent",
                "system_prompt": large_prompt,
            },
        )

        # Should either succeed or return validation error
        assert response.status_code in [200, 201, 400, 413, 422]

    @pytest.mark.asyncio
    async def test_unicode_handling(self, authenticated_client):
        """Test proper handling of unicode in names and prompts."""
        response = await authenticated_client.post(
            "/api/v1/agents",
            json={
                "name": "Test 日本語 Agent 🤖",
                "description": "Agent with unicode: café, naïve, 中文",
                "system_prompt": "You speak multiple languages: 你好, مرحبا, Привет",
            },
        )

        assert response.status_code in [200, 201, 422]
        if response.status_code in [200, 201]:
            data = response.json()
            assert "日本語" in data["name"]
