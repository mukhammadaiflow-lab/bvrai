"""Tests for Builder Engine CLI main commands."""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from builderengine_cli.main import cli


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_client():
    """Create mock API client."""
    client = MagicMock()

    # Mock user endpoint
    client.users.get_me.return_value = {
        "id": "user_123",
        "email": "test@example.com",
        "organization_id": "org_456",
        "role": "admin",
    }

    # Mock agents endpoint
    client.agents.list.return_value = {
        "data": [
            {"id": "agent_1", "name": "Test Agent", "status": "active"},
        ],
        "total": 1,
    }

    client.agents.get.return_value = {
        "id": "agent_1",
        "name": "Test Agent",
        "voice": "nova",
        "model": "gpt-4-turbo",
        "status": "active",
    }

    # Mock calls endpoint
    client.calls.list.return_value = {
        "data": [
            {"id": "call_1", "status": "completed", "duration": 120},
        ],
        "total": 1,
    }

    # Mock analytics endpoint
    client.analytics.get_overview.return_value = {
        "total_calls": 100,
        "success_rate": 0.95,
        "total_minutes": 500,
    }

    return client


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_version(self, runner):
        """Test version command."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_help(self, runner):
        """Test help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Builder Engine CLI" in result.output
        assert "agents" in result.output
        assert "calls" in result.output
        assert "campaigns" in result.output


class TestLoginLogout:
    """Test authentication commands."""

    @patch("builderengine_cli.main.get_client")
    @patch("builderengine_cli.main.save_config")
    def test_login_success(self, mock_save, mock_get_client, runner, mock_client):
        """Test successful login."""
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["login", "--api-key", "sk_test_123"])

        assert result.exit_code == 0
        assert "Logged in as" in result.output
        mock_save.assert_called_once()

    @patch("builderengine_cli.main.get_client")
    def test_login_failure(self, mock_get_client, runner):
        """Test failed login."""
        mock_get_client.side_effect = Exception("Invalid API key")

        result = runner.invoke(cli, ["login", "--api-key", "invalid"])

        assert result.exit_code == 1
        assert "Login failed" in result.output

    @patch("os.path.exists")
    @patch("os.remove")
    def test_logout(self, mock_remove, mock_exists, runner):
        """Test logout command."""
        mock_exists.return_value = True

        result = runner.invoke(cli, ["logout"])

        assert result.exit_code == 0
        assert "Logged out" in result.output


class TestWhoami:
    """Test whoami command."""

    @patch("builderengine_cli.main.get_client")
    def test_whoami(self, mock_get_client, runner, mock_client):
        """Test whoami with logged in user."""
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "whoami"])

        assert result.exit_code == 0
        assert "test@example.com" in result.output

    def test_whoami_not_logged_in(self, runner):
        """Test whoami without authentication."""
        result = runner.invoke(cli, ["whoami"])

        assert result.exit_code == 1
        assert "Not logged in" in result.output


class TestStatus:
    """Test status command."""

    @patch("builderengine_cli.main.get_client")
    def test_status(self, mock_get_client, runner, mock_client):
        """Test status command."""
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "status"])

        assert result.exit_code == 0
        assert "Platform Status" in result.output
        assert "API is operational" in result.output


class TestOutputFormats:
    """Test output format options."""

    @patch("builderengine_cli.main.get_client")
    def test_json_output(self, mock_get_client, runner, mock_client):
        """Test JSON output format."""
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "--output", "json", "agents", "list"])

        # Should be valid JSON
        assert result.exit_code == 0

    @patch("builderengine_cli.main.get_client")
    def test_table_output(self, mock_get_client, runner, mock_client):
        """Test table output format."""
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "--output", "table", "agents", "list"])

        assert result.exit_code == 0


class TestAgentCommands:
    """Test agent commands."""

    @patch("builderengine_cli.commands.agents.get_client")
    def test_agents_list(self, mock_get_client, runner, mock_client):
        """Test agents list command."""
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "agents", "list"])

        assert result.exit_code == 0
        mock_client.agents.list.assert_called_once()

    @patch("builderengine_cli.commands.agents.get_client")
    def test_agents_get(self, mock_get_client, runner, mock_client):
        """Test agents get command."""
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "agents", "get", "agent_123"])

        assert result.exit_code == 0
        mock_client.agents.get.assert_called_once_with("agent_123")

    @patch("builderengine_cli.commands.agents.get_client")
    def test_agents_create(self, mock_get_client, runner, mock_client):
        """Test agents create command."""
        mock_client.agents.create.return_value = {
            "id": "agent_new",
            "name": "New Agent",
            "voice": "nova",
        }
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, [
            "--api-key", "sk_test",
            "agents", "create",
            "--name", "New Agent",
            "--voice", "nova",
        ])

        assert result.exit_code == 0
        mock_client.agents.create.assert_called_once()

    @patch("builderengine_cli.commands.agents.get_client")
    @patch("builderengine_cli.commands.agents.confirm")
    def test_agents_delete(self, mock_confirm, mock_get_client, runner, mock_client):
        """Test agents delete command."""
        mock_confirm.return_value = True
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, [
            "--api-key", "sk_test",
            "agents", "delete", "agent_123",
        ])

        assert result.exit_code == 0
        mock_client.agents.delete.assert_called_once_with("agent_123")


class TestCallCommands:
    """Test call commands."""

    @patch("builderengine_cli.commands.calls.get_client")
    def test_calls_list(self, mock_get_client, runner, mock_client):
        """Test calls list command."""
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "calls", "list"])

        assert result.exit_code == 0
        mock_client.calls.list.assert_called()

    @patch("builderengine_cli.commands.calls.get_client")
    def test_calls_create(self, mock_get_client, runner, mock_client):
        """Test calls create command."""
        mock_client.calls.create.return_value = {
            "id": "call_new",
            "status": "queued",
            "to_number": "+14155551234",
        }
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, [
            "--api-key", "sk_test",
            "calls", "create",
            "--agent-id", "agent_123",
            "--to", "+14155551234",
        ])

        assert result.exit_code == 0
        mock_client.calls.create.assert_called_once()


class TestCampaignCommands:
    """Test campaign commands."""

    @patch("builderengine_cli.commands.campaigns.get_client")
    def test_campaigns_list(self, mock_get_client, runner, mock_client):
        """Test campaigns list command."""
        mock_client.campaigns.list.return_value = {
            "data": [{"id": "camp_1", "name": "Test", "status": "draft"}],
            "total": 1,
        }
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "campaigns", "list"])

        assert result.exit_code == 0


class TestPhoneNumberCommands:
    """Test phone number commands."""

    @patch("builderengine_cli.commands.numbers.get_client")
    def test_numbers_list(self, mock_get_client, runner, mock_client):
        """Test numbers list command."""
        mock_client.phone_numbers.list.return_value = {
            "data": [{"id": "pn_1", "phone_number": "+14155551234"}],
            "total": 1,
        }
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "numbers", "list"])

        assert result.exit_code == 0

    @patch("builderengine_cli.commands.numbers.get_client")
    def test_numbers_search(self, mock_get_client, runner, mock_client):
        """Test numbers search command."""
        mock_client.phone_numbers.list_available.return_value = {
            "data": [
                {"phone_number": "+14155551111", "region": "CA", "type": "local"},
                {"phone_number": "+14155552222", "region": "CA", "type": "local"},
            ],
        }
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, [
            "--api-key", "sk_test",
            "numbers", "search",
            "--country", "US",
            "--area-code", "415",
        ])

        assert result.exit_code == 0


class TestWebhookCommands:
    """Test webhook commands."""

    @patch("builderengine_cli.commands.webhooks.get_client")
    def test_webhooks_list(self, mock_get_client, runner, mock_client):
        """Test webhooks list command."""
        mock_client.webhooks.list.return_value = {
            "data": [{"id": "wh_1", "url": "https://example.com/webhook"}],
            "total": 1,
        }
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "webhooks", "list"])

        assert result.exit_code == 0

    def test_webhooks_events(self, runner):
        """Test webhooks events command."""
        result = runner.invoke(cli, ["webhooks", "events"])

        assert result.exit_code == 0
        assert "call.started" in result.output
        assert "call.ended" in result.output


class TestAnalyticsCommands:
    """Test analytics commands."""

    @patch("builderengine_cli.commands.analytics.get_client")
    def test_analytics_overview(self, mock_get_client, runner, mock_client):
        """Test analytics overview command."""
        mock_client.analytics.get_overview.return_value = {
            "calls": {"total": 100, "completed": 95},
            "usage": {"minutes": 500},
            "costs": {"total": 50.00},
        }
        mock_get_client.return_value = mock_client

        result = runner.invoke(cli, ["--api-key", "sk_test", "analytics", "overview"])

        assert result.exit_code == 0


class TestConfigCommands:
    """Test config commands."""

    @patch("builderengine_cli.commands.config.get_config")
    def test_config_show(self, mock_get_config, runner):
        """Test config show command."""
        mock_get_config.return_value = {
            "profile": "default",
            "output": "table",
            "base_url": "https://api.builderengine.io",
        }

        result = runner.invoke(cli, ["config", "show"])

        assert result.exit_code == 0

    @patch("builderengine_cli.commands.config.get_config")
    @patch("builderengine_cli.commands.config.save_config")
    def test_config_set(self, mock_save, mock_get_config, runner):
        """Test config set command."""
        mock_get_config.return_value = {"output": "table"}

        result = runner.invoke(cli, ["config", "set", "output", "json"])

        assert result.exit_code == 0
        mock_save.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
