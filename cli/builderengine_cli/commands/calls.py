"""Call management commands."""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

from ..utils.client import get_client
from ..utils.output import format_output, print_success, print_error, confirm

console = Console()


@click.group()
def calls():
    """Manage phone calls.

    \b
    Examples:
      builderengine calls list
      builderengine calls create --agent-id agent_123 --to +14155551234
      builderengine calls end call_abc123
    """
    pass


@calls.command("list")
@click.option("--agent-id", help="Filter by agent ID")
@click.option("--status", type=click.Choice(["queued", "ringing", "in_progress", "completed", "failed"]),
              help="Filter by status")
@click.option("--direction", type=click.Choice(["inbound", "outbound"]), help="Filter by direction")
@click.option("--from-date", help="Filter by start date (YYYY-MM-DD)")
@click.option("--to-date", help="Filter by end date (YYYY-MM-DD)")
@click.option("--limit", "-l", type=int, default=20, help="Maximum results")
@click.pass_context
def list_calls(
    ctx: click.Context,
    agent_id: Optional[str],
    status: Optional[str],
    direction: Optional[str],
    from_date: Optional[str],
    to_date: Optional[str],
    limit: int,
):
    """List calls."""
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
        if direction:
            params["direction"] = direction
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        result = client.calls.list(**params)
        format_output(
            result,
            ctx.obj["output"],
            columns=["id", "agent_id", "to_number", "from_number", "status", "duration", "created_at"],
        )
    except Exception as e:
        print_error(f"Failed to list calls: {e}")
        sys.exit(1)


@calls.command("get")
@click.argument("call_id")
@click.pass_context
def get_call(ctx: click.Context, call_id: str):
    """Get details for a specific call."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.calls.get(call_id)
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to get call: {e}")
        sys.exit(1)


@calls.command("create")
@click.option("--agent-id", "-a", required=True, help="Agent ID to use")
@click.option("--to", "-t", "to_number", required=True, help="Phone number to call")
@click.option("--from", "-f", "from_number", help="Caller ID / from number")
@click.option("--phone-number-id", help="Phone number ID to use as caller")
@click.option("--record", is_flag=True, help="Record the call")
@click.option("--max-duration", type=int, help="Maximum call duration in seconds")
@click.option("--wait", "-w", is_flag=True, help="Wait for call to complete")
@click.pass_context
def create_call(
    ctx: click.Context,
    agent_id: str,
    to_number: str,
    from_number: Optional[str],
    phone_number_id: Optional[str],
    record: bool,
    max_duration: Optional[int],
    wait: bool,
):
    """Create an outbound call."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        data = {
            "agent_id": agent_id,
            "to_number": to_number,
        }
        if from_number:
            data["from_number"] = from_number
        if phone_number_id:
            data["phone_number_id"] = phone_number_id
        if record:
            data["record"] = True
        if max_duration:
            data["max_duration"] = max_duration

        result = client.calls.create(**data)
        print_success(f"Call created: {result['id']}")
        console.print(f"  Status: {result.get('status', 'unknown')}")
        console.print(f"  To: {result.get('to_number')}")

        if wait:
            import time
            console.print("\nWaiting for call to complete...")
            while True:
                time.sleep(2)
                call = client.calls.get(result["id"])
                status = call.get("status", "unknown")
                console.print(f"  Status: {status}")
                if status in ["completed", "failed", "busy", "no_answer", "canceled"]:
                    break

            format_output(call, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to create call: {e}")
        sys.exit(1)


@calls.command("end")
@click.argument("call_id")
@click.pass_context
def end_call(ctx: click.Context, call_id: str):
    """End an active call."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        client.calls.end(call_id)
        print_success(f"Call ended: {call_id}")
    except Exception as e:
        print_error(f"Failed to end call: {e}")
        sys.exit(1)


@calls.command("transcript")
@click.argument("call_id")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]),
              default="text", help="Output format")
@click.pass_context
def get_transcript(ctx: click.Context, call_id: str, output_format: str):
    """Get call transcript."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.calls.get_transcript(call_id)

        if output_format == "json":
            format_output(result, "json")
        else:
            console.print(f"\n[bold]Transcript for {call_id}[/bold]\n")
            for msg in result.get("messages", []):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "assistant":
                    console.print(f"[blue]Agent:[/blue] {content}")
                else:
                    console.print(f"[green]User:[/green] {content}")
            if result.get("summary"):
                console.print(f"\n[dim]Summary: {result['summary']}[/dim]")
    except Exception as e:
        print_error(f"Failed to get transcript: {e}")
        sys.exit(1)


@calls.command("recording")
@click.argument("call_id")
@click.option("--download", "-d", is_flag=True, help="Download the recording")
@click.option("--output", "-o", "output_path", help="Output file path")
@click.pass_context
def get_recording(ctx: click.Context, call_id: str, download: bool, output_path: Optional[str]):
    """Get call recording URL or download."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.calls.get_recording(call_id)

        if download:
            import httpx
            url = result.get("url")
            if not url:
                print_error("No recording URL available")
                sys.exit(1)

            output_file = output_path or f"{call_id}.mp3"
            console.print(f"Downloading to {output_file}...")

            with httpx.stream("GET", url) as response:
                with open(output_file, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)

            print_success(f"Recording saved to {output_file}")
        else:
            format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to get recording: {e}")
        sys.exit(1)
