"""Builder Engine CLI - Main entry point."""

import os
import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .utils.config import Config, load_config, save_config
from .utils.client import get_client
from .commands import (
    agents,
    calls,
    campaigns,
    numbers,
    webhooks,
    analytics,
    config as config_cmd,
)

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="builderengine")
@click.option("--api-key", envvar="BUILDERENGINE_API_KEY", help="API key for authentication")
@click.option("--base-url", envvar="BUILDERENGINE_BASE_URL", default="https://api.builderengine.io",
              help="API base URL")
@click.option("--output", "-o", type=click.Choice(["table", "json", "yaml"]), default="table",
              help="Output format")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def cli(ctx: click.Context, api_key: Optional[str], base_url: str, output: str, debug: bool):
    """Builder Engine CLI - Manage AI voice agents from the command line.

    \b
    Examples:
      builderengine agents list
      builderengine calls create --agent-id agent_123 --to +14155551234
      builderengine campaigns start camp_abc
    """
    ctx.ensure_object(dict)

    # Load config from file if API key not provided
    config = load_config()
    if not api_key and config.api_key:
        api_key = config.api_key
    if config.base_url and base_url == "https://api.builderengine.io":
        base_url = config.base_url

    ctx.obj["api_key"] = api_key
    ctx.obj["base_url"] = base_url
    ctx.obj["output"] = output
    ctx.obj["debug"] = debug
    ctx.obj["config"] = config


@cli.command("login")
@click.option("--api-key", "-k", required=True, prompt=True, hide_input=True,
              help="Your Builder Engine API key")
@click.option("--base-url", "-u", default="https://api.builderengine.io",
              help="API base URL (for self-hosted)")
def login(api_key: str, base_url: str):
    """Authenticate with Builder Engine.

    Your API key will be stored securely in ~/.builderengine/config.json
    """
    try:
        # Verify the API key works
        client = get_client(api_key, base_url)
        user = client.users.get_me()

        # Save config
        config = Config(api_key=api_key, base_url=base_url)
        save_config(config)

        console.print(f"[green]✓[/green] Logged in as [bold]{user.get('email', 'Unknown')}[/bold]")
        console.print(f"  Organization: {user.get('organization_id', 'N/A')}")
        console.print(f"  Config saved to: ~/.builderengine/config.json")
    except Exception as e:
        console.print(f"[red]✗[/red] Login failed: {e}")
        sys.exit(1)


@cli.command("logout")
def logout():
    """Log out and remove stored credentials."""
    config_path = os.path.expanduser("~/.builderengine/config.json")
    if os.path.exists(config_path):
        os.remove(config_path)
        console.print("[green]✓[/green] Logged out successfully")
    else:
        console.print("Not logged in")


@cli.command("whoami")
@click.pass_context
def whoami(ctx: click.Context):
    """Show current user information."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        console.print("[red]✗[/red] Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        user = client.users.get_me()

        table = Table(title="Current User")
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        table.add_row("Email", user.get("email", "N/A"))
        table.add_row("ID", user.get("id", "N/A"))
        table.add_row("Organization", user.get("organization_id", "N/A"))
        table.add_row("Role", user.get("role", "N/A"))

        console.print(table)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        sys.exit(1)


@cli.command("status")
@click.pass_context
def status(ctx: click.Context):
    """Show platform status and health."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        console.print("[red]✗[/red] Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])

        # Get various stats
        agents = client.agents.list(limit=1)
        calls = client.calls.list(limit=1)
        analytics = client.analytics.get_overview(period="day")

        table = Table(title="Platform Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Agents", str(agents.get("total", 0)))
        table.add_row("Total Calls (Today)", str(analytics.get("total_calls", 0)))
        table.add_row("Active Calls", str(calls.get("total", 0)))
        table.add_row("Success Rate", f"{analytics.get('success_rate', 0):.1%}")
        table.add_row("Total Minutes", f"{analytics.get('total_minutes', 0):.1f}")

        console.print(table)
        console.print("\n[green]✓[/green] API is operational")
    except Exception as e:
        console.print(f"[red]✗[/red] Error checking status: {e}")
        sys.exit(1)


# Register command groups
cli.add_command(agents)
cli.add_command(calls)
cli.add_command(campaigns)
cli.add_command(numbers)
cli.add_command(webhooks)
cli.add_command(analytics)
cli.add_command(config_cmd)


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
