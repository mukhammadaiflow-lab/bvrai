"""Campaign management commands."""

import sys
import json
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from ..utils.client import get_client
from ..utils.output import format_output, print_success, print_error, confirm

console = Console()


@click.group()
def campaigns():
    """Manage batch call campaigns.

    \b
    Examples:
      builderengine campaigns list
      builderengine campaigns create --name "Q1 Outreach" --agent-id agent_123
      builderengine campaigns start campaign_abc
      builderengine campaigns progress campaign_abc
    """
    pass


@campaigns.command("list")
@click.option("--status", type=click.Choice(["draft", "scheduled", "running", "paused", "completed", "canceled"]),
              help="Filter by status")
@click.option("--agent-id", help="Filter by agent ID")
@click.option("--limit", "-l", type=int, default=20, help="Maximum results")
@click.pass_context
def list_campaigns(
    ctx: click.Context,
    status: Optional[str],
    agent_id: Optional[str],
    limit: int,
):
    """List campaigns."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {"limit": limit}
        if status:
            params["status"] = status
        if agent_id:
            params["agent_id"] = agent_id

        result = client.campaigns.list(**params)
        format_output(
            result,
            ctx.obj["output"],
            columns=["id", "name", "agent_id", "status", "total_contacts", "completed", "created_at"],
        )
    except Exception as e:
        print_error(f"Failed to list campaigns: {e}")
        sys.exit(1)


@campaigns.command("get")
@click.argument("campaign_id")
@click.pass_context
def get_campaign(ctx: click.Context, campaign_id: str):
    """Get details for a specific campaign."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.campaigns.get(campaign_id)
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to get campaign: {e}")
        sys.exit(1)


@campaigns.command("create")
@click.option("--name", "-n", required=True, help="Campaign name")
@click.option("--agent-id", "-a", required=True, help="Agent ID to use")
@click.option("--from-number", "-f", help="Caller ID / from number")
@click.option("--phone-number-id", help="Phone number ID to use")
@click.option("--contacts-file", "-c", type=click.Path(exists=True),
              help="JSON file with contacts array")
@click.option("--max-concurrent", type=int, default=5, help="Maximum concurrent calls")
@click.option("--start-time", help="Schedule start time (ISO 8601)")
@click.option("--end-time", help="Schedule end time (ISO 8601)")
@click.option("--timezone", default="UTC", help="Schedule timezone")
@click.option("--retry-failed", is_flag=True, help="Auto-retry failed calls")
@click.option("--max-retries", type=int, default=2, help="Maximum retry attempts")
@click.pass_context
def create_campaign(
    ctx: click.Context,
    name: str,
    agent_id: str,
    from_number: Optional[str],
    phone_number_id: Optional[str],
    contacts_file: Optional[str],
    max_concurrent: int,
    start_time: Optional[str],
    end_time: Optional[str],
    timezone: str,
    retry_failed: bool,
    max_retries: int,
):
    """Create a new campaign."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])

        # Load contacts from file if provided
        contacts = []
        if contacts_file:
            with open(contacts_file, "r") as f:
                contacts = json.load(f)
                if not isinstance(contacts, list):
                    print_error("Contacts file must contain a JSON array")
                    sys.exit(1)

        data = {
            "name": name,
            "agent_id": agent_id,
            "contacts": contacts,
            "settings": {
                "max_concurrent": max_concurrent,
                "retry_failed": retry_failed,
                "max_retries": max_retries,
            },
        }

        if from_number:
            data["from_number"] = from_number
        if phone_number_id:
            data["phone_number_id"] = phone_number_id

        if start_time or end_time:
            data["schedule"] = {
                "timezone": timezone,
            }
            if start_time:
                data["schedule"]["start_time"] = start_time
            if end_time:
                data["schedule"]["end_time"] = end_time

        result = client.campaigns.create(**data)
        print_success(f"Campaign created: {result['id']}")
        console.print(f"  Name: {result.get('name')}")
        console.print(f"  Contacts: {len(contacts)}")
        console.print(f"  Status: {result.get('status', 'draft')}")

        if not contacts:
            console.print("\n[yellow]Note: Add contacts with 'builderengine campaigns add-contacts'[/yellow]")

    except Exception as e:
        print_error(f"Failed to create campaign: {e}")
        sys.exit(1)


@campaigns.command("update")
@click.argument("campaign_id")
@click.option("--name", "-n", help="Campaign name")
@click.option("--agent-id", "-a", help="Agent ID to use")
@click.option("--from-number", "-f", help="Caller ID / from number")
@click.option("--max-concurrent", type=int, help="Maximum concurrent calls")
@click.option("--start-time", help="Schedule start time (ISO 8601)")
@click.option("--end-time", help="Schedule end time (ISO 8601)")
@click.option("--timezone", help="Schedule timezone")
@click.pass_context
def update_campaign(
    ctx: click.Context,
    campaign_id: str,
    name: Optional[str],
    agent_id: Optional[str],
    from_number: Optional[str],
    max_concurrent: Optional[int],
    start_time: Optional[str],
    end_time: Optional[str],
    timezone: Optional[str],
):
    """Update a campaign (only in draft status)."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])

        data = {}
        if name:
            data["name"] = name
        if agent_id:
            data["agent_id"] = agent_id
        if from_number:
            data["from_number"] = from_number
        if max_concurrent:
            data["settings"] = {"max_concurrent": max_concurrent}

        if start_time or end_time or timezone:
            data["schedule"] = {}
            if start_time:
                data["schedule"]["start_time"] = start_time
            if end_time:
                data["schedule"]["end_time"] = end_time
            if timezone:
                data["schedule"]["timezone"] = timezone

        if not data:
            print_error("No updates specified")
            sys.exit(1)

        result = client.campaigns.update(campaign_id, **data)
        print_success(f"Campaign updated: {result['id']}")
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to update campaign: {e}")
        sys.exit(1)


@campaigns.command("delete")
@click.argument("campaign_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_campaign(ctx: click.Context, campaign_id: str, force: bool):
    """Delete a campaign."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    if not force:
        if not confirm(f"Delete campaign {campaign_id}?"):
            console.print("Canceled")
            return

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        client.campaigns.delete(campaign_id)
        print_success(f"Campaign deleted: {campaign_id}")
    except Exception as e:
        print_error(f"Failed to delete campaign: {e}")
        sys.exit(1)


@campaigns.command("add-contacts")
@click.argument("campaign_id")
@click.option("--file", "-f", "contacts_file", type=click.Path(exists=True),
              help="JSON file with contacts array")
@click.option("--number", "-n", multiple=True, help="Phone number to add")
@click.pass_context
def add_contacts(
    ctx: click.Context,
    campaign_id: str,
    contacts_file: Optional[str],
    number: tuple,
):
    """Add contacts to a campaign."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    if not contacts_file and not number:
        print_error("Provide either --file or --number")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])

        contacts = []
        if contacts_file:
            with open(contacts_file, "r") as f:
                contacts = json.load(f)
                if not isinstance(contacts, list):
                    print_error("Contacts file must contain a JSON array")
                    sys.exit(1)

        # Add individual numbers
        for n in number:
            contacts.append({"phone_number": n})

        result = client.campaigns.add_contacts(campaign_id, contacts)
        print_success(f"Added {len(contacts)} contacts to campaign")
        console.print(f"  Total contacts: {result.get('total_contacts', 0)}")
    except Exception as e:
        print_error(f"Failed to add contacts: {e}")
        sys.exit(1)


@campaigns.command("start")
@click.argument("campaign_id")
@click.pass_context
def start_campaign(ctx: click.Context, campaign_id: str):
    """Start a campaign."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.campaigns.start(campaign_id)
        print_success(f"Campaign started: {campaign_id}")
        console.print(f"  Status: {result.get('status', 'running')}")
    except Exception as e:
        print_error(f"Failed to start campaign: {e}")
        sys.exit(1)


@campaigns.command("pause")
@click.argument("campaign_id")
@click.pass_context
def pause_campaign(ctx: click.Context, campaign_id: str):
    """Pause a running campaign."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.campaigns.pause(campaign_id)
        print_success(f"Campaign paused: {campaign_id}")
        console.print(f"  Status: {result.get('status', 'paused')}")
    except Exception as e:
        print_error(f"Failed to pause campaign: {e}")
        sys.exit(1)


@campaigns.command("resume")
@click.argument("campaign_id")
@click.pass_context
def resume_campaign(ctx: click.Context, campaign_id: str):
    """Resume a paused campaign."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.campaigns.resume(campaign_id)
        print_success(f"Campaign resumed: {campaign_id}")
        console.print(f"  Status: {result.get('status', 'running')}")
    except Exception as e:
        print_error(f"Failed to resume campaign: {e}")
        sys.exit(1)


@campaigns.command("cancel")
@click.argument("campaign_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def cancel_campaign(ctx: click.Context, campaign_id: str, force: bool):
    """Cancel a campaign."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    if not force:
        if not confirm(f"Cancel campaign {campaign_id}? This cannot be undone."):
            console.print("Canceled")
            return

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.campaigns.cancel(campaign_id)
        print_success(f"Campaign canceled: {campaign_id}")
    except Exception as e:
        print_error(f"Failed to cancel campaign: {e}")
        sys.exit(1)


@campaigns.command("progress")
@click.argument("campaign_id")
@click.option("--watch", "-w", is_flag=True, help="Watch progress in real-time")
@click.option("--interval", "-i", type=int, default=5, help="Update interval in seconds")
@click.pass_context
def campaign_progress(ctx: click.Context, campaign_id: str, watch: bool, interval: int):
    """Get campaign progress."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])

        def show_progress(progress_data: dict):
            total = progress_data.get("total", 0)
            completed = progress_data.get("completed", 0)
            successful = progress_data.get("successful", 0)
            failed = progress_data.get("failed", 0)
            pending = progress_data.get("pending", 0)
            in_progress = progress_data.get("in_progress", 0)

            table = Table(title=f"Campaign Progress: {campaign_id}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Total Contacts", str(total))
            table.add_row("Completed", f"[green]{completed}[/green]")
            table.add_row("Successful", f"[green]{successful}[/green]")
            table.add_row("Failed", f"[red]{failed}[/red]")
            table.add_row("Pending", f"[yellow]{pending}[/yellow]")
            table.add_row("In Progress", f"[blue]{in_progress}[/blue]")

            if total > 0:
                pct = (completed / total) * 100
                table.add_row("Progress", f"{pct:.1f}%")
                success_rate = (successful / completed * 100) if completed > 0 else 0
                table.add_row("Success Rate", f"{success_rate:.1f}%")

            console.print(table)

        if watch:
            import time
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress_bar:
                task = progress_bar.add_task("Campaign progress...", total=100)

                while True:
                    result = client.campaigns.get_progress(campaign_id)
                    total = result.get("total", 1)
                    completed = result.get("completed", 0)
                    pct = (completed / total) * 100 if total > 0 else 0

                    progress_bar.update(task, completed=pct)

                    status = result.get("status", "unknown")
                    if status in ["completed", "canceled"]:
                        break

                    time.sleep(interval)

            # Show final stats
            result = client.campaigns.get_progress(campaign_id)
            show_progress(result)
        else:
            result = client.campaigns.get_progress(campaign_id)
            if ctx.obj["output"] == "table":
                show_progress(result)
            else:
                format_output(result, ctx.obj["output"])

    except Exception as e:
        print_error(f"Failed to get campaign progress: {e}")
        sys.exit(1)


@campaigns.command("contacts")
@click.argument("campaign_id")
@click.option("--status", type=click.Choice(["pending", "in_progress", "completed", "failed"]),
              help="Filter by status")
@click.option("--limit", "-l", type=int, default=50, help="Maximum results")
@click.pass_context
def list_contacts(
    ctx: click.Context,
    campaign_id: str,
    status: Optional[str],
    limit: int,
):
    """List contacts in a campaign."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {"limit": limit}
        if status:
            params["status"] = status

        result = client.campaigns.list_contacts(campaign_id, **params)
        format_output(
            result,
            ctx.obj["output"],
            columns=["phone_number", "name", "status", "call_id", "attempts", "last_attempt"],
        )
    except Exception as e:
        print_error(f"Failed to list contacts: {e}")
        sys.exit(1)


@campaigns.command("export")
@click.argument("campaign_id")
@click.option("--output", "-o", "output_path", help="Output file path")
@click.option("--format", "-f", "export_format", type=click.Choice(["json", "csv"]),
              default="csv", help="Export format")
@click.pass_context
def export_campaign(
    ctx: click.Context,
    campaign_id: str,
    output_path: Optional[str],
    export_format: str,
):
    """Export campaign results."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.campaigns.export(campaign_id, format=export_format)

        output_file = output_path or f"campaign_{campaign_id}.{export_format}"

        if export_format == "json":
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
        else:
            # CSV export - write content directly
            with open(output_file, "w") as f:
                f.write(result.get("content", ""))

        print_success(f"Campaign exported to {output_file}")
    except Exception as e:
        print_error(f"Failed to export campaign: {e}")
        sys.exit(1)
