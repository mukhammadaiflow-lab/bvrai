"""Webhook management commands."""

import sys
import json
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

from ..utils.client import get_client
from ..utils.output import format_output, print_success, print_error, confirm

console = Console()

# Available webhook event types
EVENT_TYPES = [
    "call.started",
    "call.ringing",
    "call.answered",
    "call.ended",
    "call.failed",
    "transcription.partial",
    "transcription.final",
    "agent.speech.start",
    "agent.speech.end",
    "agent.thinking",
    "conversation.created",
    "conversation.updated",
    "campaign.started",
    "campaign.completed",
    "campaign.paused",
    "phone_number.purchased",
    "phone_number.released",
]


@click.group()
def webhooks():
    """Manage webhooks.

    \b
    Examples:
      builderengine webhooks list
      builderengine webhooks create --url https://example.com/webhook --events call.ended
      builderengine webhooks test webhook_abc123
      builderengine webhooks logs webhook_abc123
    """
    pass


@webhooks.command("list")
@click.option("--status", type=click.Choice(["active", "disabled", "failed"]),
              help="Filter by status")
@click.option("--limit", "-l", type=int, default=20, help="Maximum results")
@click.pass_context
def list_webhooks(ctx: click.Context, status: Optional[str], limit: int):
    """List webhooks."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {"limit": limit}
        if status:
            params["status"] = status

        result = client.webhooks.list(**params)
        format_output(
            result,
            ctx.obj["output"],
            columns=["id", "url", "events", "status", "last_triggered", "created_at"],
        )
    except Exception as e:
        print_error(f"Failed to list webhooks: {e}")
        sys.exit(1)


@webhooks.command("get")
@click.argument("webhook_id")
@click.pass_context
def get_webhook(ctx: click.Context, webhook_id: str):
    """Get details for a specific webhook."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.webhooks.get(webhook_id)
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to get webhook: {e}")
        sys.exit(1)


@webhooks.command("create")
@click.option("--url", "-u", required=True, help="Webhook endpoint URL")
@click.option("--events", "-e", multiple=True, required=True,
              help="Event types to subscribe to (can specify multiple)")
@click.option("--secret", "-s", help="Signing secret for verification")
@click.option("--description", "-d", help="Description of this webhook")
@click.option("--headers", help="Custom headers as JSON object")
@click.option("--retry", is_flag=True, default=True, help="Enable automatic retries")
@click.option("--max-retries", type=int, default=3, help="Maximum retry attempts")
@click.pass_context
def create_webhook(
    ctx: click.Context,
    url: str,
    events: tuple,
    secret: Optional[str],
    description: Optional[str],
    headers: Optional[str],
    retry: bool,
    max_retries: int,
):
    """Create a new webhook."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    # Validate events
    for event in events:
        if event not in EVENT_TYPES and not event.endswith("*"):
            console.print(f"[yellow]Warning: Unknown event type '{event}'[/yellow]")

    try:
        client = get_client(api_key, ctx.obj["base_url"])

        data = {
            "url": url,
            "events": list(events),
            "retry_enabled": retry,
            "max_retries": max_retries,
        }

        if secret:
            data["secret"] = secret
        if description:
            data["description"] = description
        if headers:
            try:
                data["headers"] = json.loads(headers)
            except json.JSONDecodeError:
                print_error("Invalid JSON for headers")
                sys.exit(1)

        result = client.webhooks.create(**data)
        print_success(f"Webhook created: {result['id']}")
        console.print(f"  URL: {result.get('url')}")
        console.print(f"  Events: {', '.join(result.get('events', []))}")

        if not secret:
            generated_secret = result.get("secret")
            if generated_secret:
                console.print(f"\n[yellow]Signing secret (save this): {generated_secret}[/yellow]")

        console.print(f"\nUse 'builderengine webhooks test {result['id']}' to send a test event")

    except Exception as e:
        print_error(f"Failed to create webhook: {e}")
        sys.exit(1)


@webhooks.command("update")
@click.argument("webhook_id")
@click.option("--url", "-u", help="Webhook endpoint URL")
@click.option("--events", "-e", multiple=True, help="Event types to subscribe to")
@click.option("--description", "-d", help="Description")
@click.option("--headers", help="Custom headers as JSON object")
@click.option("--enable", is_flag=True, help="Enable webhook")
@click.option("--disable", is_flag=True, help="Disable webhook")
@click.pass_context
def update_webhook(
    ctx: click.Context,
    webhook_id: str,
    url: Optional[str],
    events: tuple,
    description: Optional[str],
    headers: Optional[str],
    enable: bool,
    disable: bool,
):
    """Update a webhook."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])

        data = {}
        if url:
            data["url"] = url
        if events:
            data["events"] = list(events)
        if description:
            data["description"] = description
        if headers:
            try:
                data["headers"] = json.loads(headers)
            except json.JSONDecodeError:
                print_error("Invalid JSON for headers")
                sys.exit(1)
        if enable:
            data["status"] = "active"
        if disable:
            data["status"] = "disabled"

        if not data:
            print_error("No updates specified")
            sys.exit(1)

        result = client.webhooks.update(webhook_id, **data)
        print_success(f"Webhook updated: {result['id']}")
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to update webhook: {e}")
        sys.exit(1)


@webhooks.command("delete")
@click.argument("webhook_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_webhook(ctx: click.Context, webhook_id: str, force: bool):
    """Delete a webhook."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    if not force:
        if not confirm(f"Delete webhook {webhook_id}?"):
            console.print("Canceled")
            return

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        client.webhooks.delete(webhook_id)
        print_success(f"Webhook deleted: {webhook_id}")
    except Exception as e:
        print_error(f"Failed to delete webhook: {e}")
        sys.exit(1)


@webhooks.command("test")
@click.argument("webhook_id")
@click.option("--event", "-e", default="test.ping", help="Event type to send")
@click.pass_context
def test_webhook(ctx: click.Context, webhook_id: str, event: str):
    """Send a test event to a webhook."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.webhooks.test(webhook_id, event_type=event)

        if result.get("success"):
            print_success("Test event delivered successfully!")
            console.print(f"  Response status: {result.get('response_status', 'N/A')}")
            console.print(f"  Response time: {result.get('response_time_ms', 'N/A')}ms")
        else:
            print_error("Test event delivery failed")
            console.print(f"  Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print_error(f"Failed to test webhook: {e}")
        sys.exit(1)


@webhooks.command("logs")
@click.argument("webhook_id")
@click.option("--status", type=click.Choice(["success", "failed", "pending"]),
              help="Filter by delivery status")
@click.option("--event", "-e", help="Filter by event type")
@click.option("--limit", "-l", type=int, default=20, help="Maximum results")
@click.option("--show-payload", is_flag=True, help="Show full payload in output")
@click.pass_context
def webhook_logs(
    ctx: click.Context,
    webhook_id: str,
    status: Optional[str],
    event: Optional[str],
    limit: int,
    show_payload: bool,
):
    """View webhook delivery logs."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {"limit": limit}
        if status:
            params["status"] = status
        if event:
            params["event_type"] = event

        result = client.webhooks.get_logs(webhook_id, **params)

        if ctx.obj["output"] == "table" and not show_payload:
            table = Table(title=f"Webhook Logs: {webhook_id}")
            table.add_column("ID", style="dim")
            table.add_column("Event", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Response", style="white")
            table.add_column("Attempts", style="white")
            table.add_column("Timestamp", style="dim")

            for log in result.get("data", []):
                status_style = "green" if log.get("status") == "success" else "red"
                table.add_row(
                    log.get("id", "")[:12],
                    log.get("event_type", ""),
                    f"[{status_style}]{log.get('status', 'unknown')}[/{status_style}]",
                    str(log.get("response_status", "N/A")),
                    str(log.get("attempts", 0)),
                    log.get("created_at", "")[:19],
                )

            console.print(table)
        else:
            format_output(result, ctx.obj["output"])

    except Exception as e:
        print_error(f"Failed to get webhook logs: {e}")
        sys.exit(1)


@webhooks.command("log")
@click.argument("webhook_id")
@click.argument("log_id")
@click.pass_context
def get_log_detail(ctx: click.Context, webhook_id: str, log_id: str):
    """Get detailed information for a specific delivery attempt."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.webhooks.get_log_detail(webhook_id, log_id)

        if ctx.obj["output"] == "table":
            console.print(f"\n[bold]Webhook Delivery: {log_id}[/bold]\n")

            table = Table(show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Event Type", result.get("event_type", ""))
            table.add_row("Status", result.get("status", ""))
            table.add_row("Response Status", str(result.get("response_status", "N/A")))
            table.add_row("Response Time", f"{result.get('response_time_ms', 'N/A')}ms")
            table.add_row("Attempts", str(result.get("attempts", 0)))
            table.add_row("Timestamp", result.get("created_at", ""))

            console.print(table)

            # Show request payload
            if result.get("payload"):
                console.print("\n[bold]Request Payload:[/bold]")
                payload_json = json.dumps(result["payload"], indent=2)
                syntax = Syntax(payload_json, "json", theme="monokai", line_numbers=False)
                console.print(syntax)

            # Show response body
            if result.get("response_body"):
                console.print("\n[bold]Response Body:[/bold]")
                try:
                    response_json = json.dumps(json.loads(result["response_body"]), indent=2)
                    syntax = Syntax(response_json, "json", theme="monokai", line_numbers=False)
                    console.print(syntax)
                except json.JSONDecodeError:
                    console.print(result["response_body"])

            # Show error if any
            if result.get("error"):
                console.print(f"\n[red]Error: {result['error']}[/red]")

        else:
            format_output(result, ctx.obj["output"])

    except Exception as e:
        print_error(f"Failed to get log detail: {e}")
        sys.exit(1)


@webhooks.command("retry")
@click.argument("webhook_id")
@click.argument("log_id")
@click.pass_context
def retry_delivery(ctx: click.Context, webhook_id: str, log_id: str):
    """Manually retry a failed webhook delivery."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.webhooks.retry_delivery(webhook_id, log_id)

        if result.get("success"):
            print_success("Webhook delivery retried successfully!")
            console.print(f"  Response status: {result.get('response_status', 'N/A')}")
        else:
            print_error("Retry failed")
            console.print(f"  Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print_error(f"Failed to retry delivery: {e}")
        sys.exit(1)


@webhooks.command("regenerate-secret")
@click.argument("webhook_id")
@click.pass_context
def regenerate_secret(ctx: click.Context, webhook_id: str):
    """Regenerate the signing secret for a webhook."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    if not confirm("Regenerate webhook secret? The old secret will no longer work."):
        console.print("Canceled")
        return

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.webhooks.regenerate_secret(webhook_id)

        print_success("Webhook secret regenerated!")
        console.print(f"\n[yellow]New signing secret: {result.get('secret')}[/yellow]")
        console.print("\n[dim]Update your server to use the new secret.[/dim]")

    except Exception as e:
        print_error(f"Failed to regenerate secret: {e}")
        sys.exit(1)


@webhooks.command("events")
def list_events():
    """List all available webhook event types."""
    table = Table(title="Available Webhook Events")
    table.add_column("Event Type", style="cyan")
    table.add_column("Description", style="white")

    event_descriptions = {
        "call.started": "Call has been initiated",
        "call.ringing": "Phone is ringing",
        "call.answered": "Call was answered",
        "call.ended": "Call has ended",
        "call.failed": "Call failed to connect",
        "transcription.partial": "Partial transcription available",
        "transcription.final": "Final transcription for utterance",
        "agent.speech.start": "Agent started speaking",
        "agent.speech.end": "Agent stopped speaking",
        "agent.thinking": "Agent is processing response",
        "conversation.created": "New conversation started",
        "conversation.updated": "Conversation was updated",
        "campaign.started": "Campaign started processing",
        "campaign.completed": "Campaign finished all calls",
        "campaign.paused": "Campaign was paused",
        "phone_number.purchased": "Phone number was purchased",
        "phone_number.released": "Phone number was released",
    }

    for event in EVENT_TYPES:
        table.add_row(event, event_descriptions.get(event, ""))

    console.print(table)
    console.print("\n[dim]Use wildcards like 'call.*' to subscribe to multiple events[/dim]")
