"""API client wrapper for the CLI."""

import json
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx


class CLIClient:
    """Simple HTTP client for CLI operations."""

    def __init__(self, api_key: str, base_url: str = "https://api.builderengine.io"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request."""
        response = self._client.request(
            method=method,
            url=path,
            params=params,
            json=json_data,
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
            except json.JSONDecodeError:
                error_msg = response.text
            raise Exception(f"API Error ({response.status_code}): {error_msg}")

        if response.status_code == 204:
            return {}

        return response.json()

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request."""
        return self._request("GET", path, params=params)

    def post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a POST request."""
        return self._request("POST", path, json_data=data)

    def patch(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a PATCH request."""
        return self._request("PATCH", path, json_data=data)

    def delete(self, path: str) -> Dict[str, Any]:
        """Make a DELETE request."""
        return self._request("DELETE", path)

    # Resource shortcuts
    @property
    def agents(self) -> "AgentsResource":
        return AgentsResource(self)

    @property
    def calls(self) -> "CallsResource":
        return CallsResource(self)

    @property
    def campaigns(self) -> "CampaignsResource":
        return CampaignsResource(self)

    @property
    def phone_numbers(self) -> "PhoneNumbersResource":
        return PhoneNumbersResource(self)

    @property
    def webhooks(self) -> "WebhooksResource":
        return WebhooksResource(self)

    @property
    def analytics(self) -> "AnalyticsResource":
        return AnalyticsResource(self)

    @property
    def users(self) -> "UsersResource":
        return UsersResource(self)


class AgentsResource:
    """Agents API resource."""

    def __init__(self, client: CLIClient):
        self.client = client

    def list(self, **params) -> Dict[str, Any]:
        return self.client.get("/api/v1/agents", params=params)

    def get(self, agent_id: str) -> Dict[str, Any]:
        return self.client.get(f"/api/v1/agents/{agent_id}")

    def create(self, **data) -> Dict[str, Any]:
        return self.client.post("/api/v1/agents", data=data)

    def update(self, agent_id: str, **data) -> Dict[str, Any]:
        return self.client.patch(f"/api/v1/agents/{agent_id}", data=data)

    def delete(self, agent_id: str) -> None:
        self.client.delete(f"/api/v1/agents/{agent_id}")

    def duplicate(self, agent_id: str, name: str) -> Dict[str, Any]:
        return self.client.post(f"/api/v1/agents/{agent_id}/duplicate", data={"name": name})


class CallsResource:
    """Calls API resource."""

    def __init__(self, client: CLIClient):
        self.client = client

    def list(self, **params) -> Dict[str, Any]:
        return self.client.get("/api/v1/calls", params=params)

    def get(self, call_id: str) -> Dict[str, Any]:
        return self.client.get(f"/api/v1/calls/{call_id}")

    def create(self, **data) -> Dict[str, Any]:
        return self.client.post("/api/v1/calls", data=data)

    def end(self, call_id: str) -> None:
        self.client.post(f"/api/v1/calls/{call_id}/end")

    def get_transcript(self, call_id: str) -> Dict[str, Any]:
        return self.client.get(f"/api/v1/calls/{call_id}/transcript")

    def get_recording(self, call_id: str) -> Dict[str, Any]:
        return self.client.get(f"/api/v1/calls/{call_id}/recording")


class CampaignsResource:
    """Campaigns API resource."""

    def __init__(self, client: CLIClient):
        self.client = client

    def list(self, **params) -> Dict[str, Any]:
        return self.client.get("/api/v1/campaigns", params=params)

    def get(self, campaign_id: str) -> Dict[str, Any]:
        return self.client.get(f"/api/v1/campaigns/{campaign_id}")

    def create(self, **data) -> Dict[str, Any]:
        return self.client.post("/api/v1/campaigns", data=data)

    def update(self, campaign_id: str, **data) -> Dict[str, Any]:
        return self.client.patch(f"/api/v1/campaigns/{campaign_id}", data=data)

    def delete(self, campaign_id: str) -> None:
        self.client.delete(f"/api/v1/campaigns/{campaign_id}")

    def start(self, campaign_id: str) -> Dict[str, Any]:
        return self.client.post(f"/api/v1/campaigns/{campaign_id}/start")

    def pause(self, campaign_id: str) -> Dict[str, Any]:
        return self.client.post(f"/api/v1/campaigns/{campaign_id}/pause")

    def resume(self, campaign_id: str) -> Dict[str, Any]:
        return self.client.post(f"/api/v1/campaigns/{campaign_id}/resume")

    def cancel(self, campaign_id: str) -> Dict[str, Any]:
        return self.client.post(f"/api/v1/campaigns/{campaign_id}/cancel")

    def get_progress(self, campaign_id: str) -> Dict[str, Any]:
        return self.client.get(f"/api/v1/campaigns/{campaign_id}/progress")


class PhoneNumbersResource:
    """Phone numbers API resource."""

    def __init__(self, client: CLIClient):
        self.client = client

    def list(self, **params) -> Dict[str, Any]:
        return self.client.get("/api/v1/phone-numbers", params=params)

    def get(self, number_id: str) -> Dict[str, Any]:
        return self.client.get(f"/api/v1/phone-numbers/{number_id}")

    def list_available(self, **params) -> Dict[str, Any]:
        return self.client.get("/api/v1/phone-numbers/available", params=params)

    def purchase(self, phone_number: str, **data) -> Dict[str, Any]:
        data["phone_number"] = phone_number
        return self.client.post("/api/v1/phone-numbers/purchase", data=data)

    def update(self, number_id: str, **data) -> Dict[str, Any]:
        return self.client.patch(f"/api/v1/phone-numbers/{number_id}", data=data)

    def release(self, number_id: str) -> None:
        self.client.delete(f"/api/v1/phone-numbers/{number_id}")


class WebhooksResource:
    """Webhooks API resource."""

    def __init__(self, client: CLIClient):
        self.client = client

    def list(self, **params) -> Dict[str, Any]:
        return self.client.get("/api/v1/webhooks", params=params)

    def get(self, webhook_id: str) -> Dict[str, Any]:
        return self.client.get(f"/api/v1/webhooks/{webhook_id}")

    def create(self, **data) -> Dict[str, Any]:
        return self.client.post("/api/v1/webhooks", data=data)

    def update(self, webhook_id: str, **data) -> Dict[str, Any]:
        return self.client.patch(f"/api/v1/webhooks/{webhook_id}", data=data)

    def delete(self, webhook_id: str) -> None:
        self.client.delete(f"/api/v1/webhooks/{webhook_id}")

    def test(self, webhook_id: str) -> Dict[str, Any]:
        return self.client.post(f"/api/v1/webhooks/{webhook_id}/test")


class AnalyticsResource:
    """Analytics API resource."""

    def __init__(self, client: CLIClient):
        self.client = client

    def get_overview(self, **params) -> Dict[str, Any]:
        return self.client.get("/api/v1/analytics/overview", params=params)

    def get_usage(self, **params) -> Dict[str, Any]:
        return self.client.get("/api/v1/analytics/usage", params=params)

    def get_costs(self, **params) -> Dict[str, Any]:
        return self.client.get("/api/v1/analytics/costs", params=params)


class UsersResource:
    """Users API resource."""

    def __init__(self, client: CLIClient):
        self.client = client

    def get_me(self) -> Dict[str, Any]:
        return self.client.get("/api/v1/users/me")


def get_client(api_key: str, base_url: str = "https://api.builderengine.io") -> CLIClient:
    """Create and return a CLI client instance."""
    return CLIClient(api_key, base_url)
