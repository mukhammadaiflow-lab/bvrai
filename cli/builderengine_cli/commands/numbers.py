"""Phone number management commands."""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from ..utils.client import get_client
from ..utils.output import format_output, print_success, print_error, confirm

console = Console()


@click.group()
def numbers():
    """Manage phone numbers.

    \b
    Examples:
      builderengine numbers list
      builderengine numbers search --country US --area-code 415
      builderengine numbers purchase +14155551234
      builderengine numbers configure pn_abc123 --agent-id agent_123
    """
    pass


@numbers.command("list")
@click.option("--agent-id", help="Filter by agent ID")
@click.option("--status", type=click.Choice(["active", "pending", "released"]),
              help="Filter by status")
@click.option("--limit", "-l", type=int, default=20, help="Maximum results")
@click.pass_context
def list_numbers(
    ctx: click.Context,
    agent_id: Optional[str],
    status: Optional[str],
    limit: int,
):
    """List your phone numbers."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {"limit": limit}
        if agent_id:
            params["agent_id"] = agent_id
        if status:
            params["status"] = status

        result = client.phone_numbers.list(**params)
        format_output(
            result,
            ctx.obj["output"],
            columns=["id", "phone_number", "friendly_name", "agent_id", "status", "capabilities"],
        )
    except Exception as e:
        print_error(f"Failed to list phone numbers: {e}")
        sys.exit(1)


@numbers.command("get")
@click.argument("phone_number_id")
@click.pass_context
def get_number(ctx: click.Context, phone_number_id: str):
    """Get details for a specific phone number."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.phone_numbers.get(phone_number_id)
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to get phone number: {e}")
        sys.exit(1)


@numbers.command("search")
@click.option("--country", "-c", default="US", help="Country code (e.g., US, CA, GB)")
@click.option("--area-code", "-a", help="Area code to search")
@click.option("--contains", help="Search for numbers containing this pattern")
@click.option("--type", "-t", "number_type", type=click.Choice(["local", "toll_free", "mobile"]),
              default="local", help="Number type")
@click.option("--voice", is_flag=True, default=True, help="Voice capable")
@click.option("--sms", is_flag=True, help="SMS capable")
@click.option("--limit", "-l", type=int, default=20, help="Maximum results")
@click.pass_context
def search_numbers(
    ctx: click.Context,
    country: str,
    area_code: Optional[str],
    contains: Optional[str],
    number_type: str,
    voice: bool,
    sms: bool,
    limit: int,
):
    """Search for available phone numbers to purchase."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {
            "country": country,
            "type": number_type,
            "voice_capable": voice,
            "limit": limit,
        }
        if area_code:
            params["area_code"] = area_code
        if contains:
            params["contains"] = contains
        if sms:
            params["sms_capable"] = True

        result = client.phone_numbers.list_available(**params)

        if ctx.obj["output"] == "table":
            table = Table(title="Available Phone Numbers")
            table.add_column("Phone Number", style="cyan")
            table.add_column("Region", style="white")
            table.add_column("Type", style="white")
            table.add_column("Capabilities", style="green")
            table.add_column("Monthly Cost", style="yellow")

            for num in result.get("data", []):
                capabilities = []
                if num.get("voice_capable"):
                    capabilities.append("Voice")
                if num.get("sms_capable"):
                    capabilities.append("SMS")
                if num.get("mms_capable"):
                    capabilities.append("MMS")

                table.add_row(
                    num.get("phone_number", ""),
                    num.get("region", ""),
                    num.get("type", ""),
                    ", ".join(capabilities),
                    f"${num.get('monthly_cost', 0):.2f}",
                )

            console.print(table)
            console.print(f"\n[dim]Use 'builderengine numbers purchase <number>' to purchase[/dim]")
        else:
            format_output(result, ctx.obj["output"])

    except Exception as e:
        print_error(f"Failed to search numbers: {e}")
        sys.exit(1)


@numbers.command("purchase")
@click.argument("phone_number")
@click.option("--name", "-n", help="Friendly name for the number")
@click.option("--agent-id", "-a", help="Agent ID to configure with")
@click.pass_context
def purchase_number(
    ctx: click.Context,
    phone_number: str,
    name: Optional[str],
    agent_id: Optional[str],
):
    """Purchase a phone number."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    # Confirm purchase
    if not confirm(f"Purchase phone number {phone_number}? Monthly charges will apply."):
        console.print("Canceled")
        return

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        data = {"phone_number": phone_number}
        if name:
            data["friendly_name"] = name
        if agent_id:
            data["agent_id"] = agent_id

        result = client.phone_numbers.purchase(**data)
        print_success(f"Phone number purchased: {result['id']}")
        console.print(f"  Number: {result.get('phone_number')}")
        console.print(f"  Status: {result.get('status', 'pending')}")

        if agent_id:
            console.print(f"  Configured with agent: {agent_id}")
        else:
            console.print("\n[yellow]Note: Configure with 'builderengine numbers configure'[/yellow]")

    except Exception as e:
        print_error(f"Failed to purchase number: {e}")
        sys.exit(1)


@numbers.command("configure")
@click.argument("phone_number_id")
@click.option("--agent-id", "-a", help="Agent ID to handle calls")
@click.option("--name", "-n", help="Friendly name")
@click.option("--voicemail", is_flag=True, help="Enable voicemail")
@click.option("--no-voicemail", is_flag=True, help="Disable voicemail")
@click.option("--voicemail-greeting", help="Voicemail greeting text")
@click.option("--forward-to", help="Forwarding number when agent unavailable")
@click.option("--webhook-url", help="Webhook URL for call events")
@click.pass_context
def configure_number(
    ctx: click.Context,
    phone_number_id: str,
    agent_id: Optional[str],
    name: Optional[str],
    voicemail: bool,
    no_voicemail: bool,
    voicemail_greeting: Optional[str],
    forward_to: Optional[str],
    webhook_url: Optional[str],
):
    """Configure a phone number."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])

        data = {}
        if agent_id:
            data["agent_id"] = agent_id
        if name:
            data["friendly_name"] = name
        if voicemail:
            data["voicemail_enabled"] = True
        if no_voicemail:
            data["voicemail_enabled"] = False
        if voicemail_greeting:
            data["voicemail_greeting"] = voicemail_greeting
        if forward_to:
            data["forward_to"] = forward_to
        if webhook_url:
            data["webhook_url"] = webhook_url

        if not data:
            print_error("No configuration options specified")
            sys.exit(1)

        result = client.phone_numbers.update(phone_number_id, **data)
        print_success(f"Phone number configured: {result['id']}")
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to configure number: {e}")
        sys.exit(1)


@numbers.command("release")
@click.argument("phone_number_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def release_number(ctx: click.Context, phone_number_id: str, force: bool):
    """Release a phone number."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    if not force:
        if not confirm(f"Release phone number {phone_number_id}? This cannot be undone."):
            console.print("Canceled")
            return

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        client.phone_numbers.release(phone_number_id)
        print_success(f"Phone number released: {phone_number_id}")
    except Exception as e:
        print_error(f"Failed to release number: {e}")
        sys.exit(1)


@numbers.command("verify")
@click.argument("phone_number")
@click.option("--method", "-m", type=click.Choice(["sms", "call"]),
              default="sms", help="Verification method")
@click.pass_context
def verify_number(ctx: click.Context, phone_number: str, method: str):
    """Verify a phone number for caller ID."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.phone_numbers.start_verification(phone_number, method=method)

        print_success(f"Verification started for {phone_number}")
        console.print(f"  Method: {method}")
        console.print(f"  Verification ID: {result.get('verification_id')}")

        if method == "sms":
            console.print("\n[yellow]Check your SMS for the verification code.[/yellow]")
        else:
            console.print("\n[yellow]You will receive a call with the verification code.[/yellow]")

        console.print(f"\nUse 'builderengine numbers confirm {result.get('verification_id')} <code>' to complete verification")

    except Exception as e:
        print_error(f"Failed to start verification: {e}")
        sys.exit(1)


@numbers.command("confirm")
@click.argument("verification_id")
@click.argument("code")
@click.pass_context
def confirm_verification(ctx: click.Context, verification_id: str, code: str):
    """Confirm phone number verification with code."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.phone_numbers.confirm_verification(verification_id, code)
        print_success("Phone number verified successfully!")
        console.print(f"  Number: {result.get('phone_number')}")
        console.print(f"  Can be used as caller ID")
    except Exception as e:
        print_error(f"Failed to confirm verification: {e}")
        sys.exit(1)


@numbers.command("verified")
@click.pass_context
def list_verified(ctx: click.Context):
    """List verified caller IDs."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.phone_numbers.list_verified()
        format_output(
            result,
            ctx.obj["output"],
            columns=["phone_number", "verified_at", "status"],
        )
    except Exception as e:
        print_error(f"Failed to list verified numbers: {e}")
        sys.exit(1)


@numbers.command("stats")
@click.argument("phone_number_id")
@click.option("--period", "-p", type=click.Choice(["day", "week", "month"]),
              default="week", help="Statistics period")
@click.pass_context
def number_stats(ctx: click.Context, phone_number_id: str, period: str):
    """Get usage statistics for a phone number."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.phone_numbers.get_stats(phone_number_id, period=period)

        if ctx.obj["output"] == "table":
            table = Table(title=f"Phone Number Statistics ({period})")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Total Calls", str(result.get("total_calls", 0)))
            table.add_row("Inbound Calls", str(result.get("inbound_calls", 0)))
            table.add_row("Outbound Calls", str(result.get("outbound_calls", 0)))
            table.add_row("Total Duration", f"{result.get('total_duration', 0)} sec")
            table.add_row("Average Duration", f"{result.get('avg_duration', 0):.1f} sec")
            table.add_row("Successful Calls", str(result.get("successful_calls", 0)))
            table.add_row("Failed Calls", str(result.get("failed_calls", 0)))

            console.print(table)
        else:
            format_output(result, ctx.obj["output"])

    except Exception as e:
        print_error(f"Failed to get number stats: {e}")
        sys.exit(1)
