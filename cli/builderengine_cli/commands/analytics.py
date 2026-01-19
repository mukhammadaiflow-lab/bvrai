"""Analytics commands."""

import sys
import json
from typing import Optional
from datetime import datetime, timedelta

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..utils.client import get_client
from ..utils.output import format_output, print_success, print_error

console = Console()


@click.group()
def analytics():
    """View analytics and reports.

    \b
    Examples:
      builderengine analytics overview
      builderengine analytics calls --period week
      builderengine analytics usage --period month
      builderengine analytics export --type calls --format csv
    """
    pass


@analytics.command("overview")
@click.option("--period", "-p", type=click.Choice(["day", "week", "month", "quarter"]),
              default="week", help="Time period")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--agent-id", "-a", help="Filter by agent ID")
@click.pass_context
def overview(
    ctx: click.Context,
    period: str,
    start_date: Optional[str],
    end_date: Optional[str],
    agent_id: Optional[str],
):
    """Get analytics overview dashboard."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {"period": period}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if agent_id:
            params["agent_id"] = agent_id

        result = client.analytics.get_overview(**params)

        if ctx.obj["output"] == "table":
            # Main metrics panel
            console.print(Panel.fit(
                f"[bold cyan]Analytics Overview[/bold cyan] ({period})",
                border_style="blue",
            ))

            # Calls metrics
            table = Table(title="Call Metrics", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            table.add_column("Change", style="white")

            calls = result.get("calls", {})
            table.add_row(
                "Total Calls",
                str(calls.get("total", 0)),
                _format_change(calls.get("change_percent", 0)),
            )
            table.add_row(
                "Completed",
                str(calls.get("completed", 0)),
                "",
            )
            table.add_row(
                "Failed",
                str(calls.get("failed", 0)),
                "",
            )
            table.add_row(
                "Success Rate",
                f"{calls.get('success_rate', 0):.1f}%",
                _format_change(calls.get("success_rate_change", 0)),
            )
            table.add_row(
                "Avg Duration",
                f"{calls.get('avg_duration', 0):.1f}s",
                _format_change(calls.get("avg_duration_change", 0)),
            )

            console.print(table)
            console.print()

            # Usage metrics
            table2 = Table(title="Usage Metrics", show_header=True)
            table2.add_column("Metric", style="cyan")
            table2.add_column("Value", style="white")
            table2.add_column("Change", style="white")

            usage = result.get("usage", {})
            table2.add_row(
                "Total Minutes",
                f"{usage.get('minutes', 0):.1f}",
                _format_change(usage.get("minutes_change", 0)),
            )
            table2.add_row(
                "API Requests",
                _format_number(usage.get("api_requests", 0)),
                _format_change(usage.get("api_requests_change", 0)),
            )
            table2.add_row(
                "Transcription Minutes",
                f"{usage.get('transcription_minutes', 0):.1f}",
                "",
            )
            table2.add_row(
                "LLM Tokens",
                _format_number(usage.get("llm_tokens", 0)),
                "",
            )

            console.print(table2)
            console.print()

            # Cost breakdown
            table3 = Table(title="Cost Summary", show_header=True)
            table3.add_column("Category", style="cyan")
            table3.add_column("Amount", style="yellow")

            costs = result.get("costs", {})
            table3.add_row("Voice Minutes", f"${costs.get('voice', 0):.2f}")
            table3.add_row("Telephony", f"${costs.get('telephony', 0):.2f}")
            table3.add_row("LLM Usage", f"${costs.get('llm', 0):.2f}")
            table3.add_row("Transcription", f"${costs.get('transcription', 0):.2f}")
            table3.add_row("[bold]Total[/bold]", f"[bold]${costs.get('total', 0):.2f}[/bold]")

            console.print(table3)

        else:
            format_output(result, ctx.obj["output"])

    except Exception as e:
        print_error(f"Failed to get overview: {e}")
        sys.exit(1)


@analytics.command("calls")
@click.option("--period", "-p", type=click.Choice(["day", "week", "month", "quarter"]),
              default="week", help="Time period")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--agent-id", "-a", help="Filter by agent ID")
@click.option("--group-by", "-g", type=click.Choice(["day", "hour", "agent", "status"]),
              default="day", help="Group results by")
@click.pass_context
def calls_analytics(
    ctx: click.Context,
    period: str,
    start_date: Optional[str],
    end_date: Optional[str],
    agent_id: Optional[str],
    group_by: str,
):
    """Get detailed call analytics."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {"period": period, "group_by": group_by}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if agent_id:
            params["agent_id"] = agent_id

        result = client.analytics.get_calls(**params)

        if ctx.obj["output"] == "table":
            console.print(Panel.fit(
                f"[bold cyan]Call Analytics[/bold cyan] ({period}, grouped by {group_by})",
                border_style="blue",
            ))

            table = Table()

            if group_by == "day":
                table.add_column("Date", style="cyan")
            elif group_by == "hour":
                table.add_column("Hour", style="cyan")
            elif group_by == "agent":
                table.add_column("Agent", style="cyan")
            else:
                table.add_column("Status", style="cyan")

            table.add_column("Total", style="white")
            table.add_column("Completed", style="green")
            table.add_column("Failed", style="red")
            table.add_column("Avg Duration", style="white")
            table.add_column("Total Minutes", style="white")

            for row in result.get("data", []):
                table.add_row(
                    str(row.get("key", "")),
                    str(row.get("total", 0)),
                    str(row.get("completed", 0)),
                    str(row.get("failed", 0)),
                    f"{row.get('avg_duration', 0):.1f}s",
                    f"{row.get('total_minutes', 0):.1f}",
                )

            console.print(table)

        else:
            format_output(result, ctx.obj["output"])

    except Exception as e:
        print_error(f"Failed to get call analytics: {e}")
        sys.exit(1)


@analytics.command("usage")
@click.option("--period", "-p", type=click.Choice(["day", "week", "month", "quarter"]),
              default="month", help="Time period")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed breakdown")
@click.pass_context
def usage_analytics(ctx: click.Context, period: str, detailed: bool):
    """Get usage metrics and billing information."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.analytics.get_usage(period=period, detailed=detailed)

        if ctx.obj["output"] == "table":
            console.print(Panel.fit(
                f"[bold cyan]Usage Report[/bold cyan] ({period})",
                border_style="blue",
            ))

            # Main usage
            table = Table(title="Resource Usage")
            table.add_column("Resource", style="cyan")
            table.add_column("Usage", style="white")
            table.add_column("Limit", style="dim")
            table.add_column("Remaining", style="white")

            usage = result.get("usage", {})
            limits = result.get("limits", {})

            for resource, data in usage.items():
                limit = limits.get(resource, {}).get("limit", "unlimited")
                remaining = limits.get(resource, {}).get("remaining", "-")

                if isinstance(data, dict):
                    value = data.get("value", 0)
                    unit = data.get("unit", "")
                    formatted = f"{value:,.1f} {unit}" if unit else f"{value:,.0f}"
                else:
                    formatted = f"{data:,.0f}" if isinstance(data, (int, float)) else str(data)

                table.add_row(
                    _format_resource_name(resource),
                    formatted,
                    str(limit) if limit != "unlimited" else "Unlimited",
                    str(remaining) if remaining != "-" else "-",
                )

            console.print(table)

            if detailed and result.get("breakdown"):
                console.print()
                table2 = Table(title="Cost Breakdown")
                table2.add_column("Category", style="cyan")
                table2.add_column("Usage", style="white")
                table2.add_column("Rate", style="dim")
                table2.add_column("Cost", style="yellow")

                for item in result.get("breakdown", []):
                    table2.add_row(
                        item.get("category", ""),
                        str(item.get("usage", "")),
                        item.get("rate", ""),
                        f"${item.get('cost', 0):.2f}",
                    )

                console.print(table2)

            # Billing summary
            billing = result.get("billing", {})
            if billing:
                console.print()
                console.print(Panel(
                    f"[bold]Current Period:[/bold] {billing.get('period_start', '')} - {billing.get('period_end', '')}\n"
                    f"[bold]Current Charges:[/bold] ${billing.get('current_charges', 0):.2f}\n"
                    f"[bold]Credits Remaining:[/bold] ${billing.get('credits_remaining', 0):.2f}",
                    title="Billing Summary",
                    border_style="yellow",
                ))

        else:
            format_output(result, ctx.obj["output"])

    except Exception as e:
        print_error(f"Failed to get usage analytics: {e}")
        sys.exit(1)


@analytics.command("agents")
@click.option("--period", "-p", type=click.Choice(["day", "week", "month", "quarter"]),
              default="week", help="Time period")
@click.option("--sort", "-s", type=click.Choice(["calls", "minutes", "success_rate", "cost"]),
              default="calls", help="Sort by metric")
@click.option("--limit", "-l", type=int, default=10, help="Number of agents to show")
@click.pass_context
def agent_analytics(ctx: click.Context, period: str, sort: str, limit: int):
    """Get per-agent analytics."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.analytics.get_agent_stats(period=period, sort_by=sort, limit=limit)

        if ctx.obj["output"] == "table":
            console.print(Panel.fit(
                f"[bold cyan]Agent Performance[/bold cyan] ({period})",
                border_style="blue",
            ))

            table = Table()
            table.add_column("Agent", style="cyan")
            table.add_column("Total Calls", style="white")
            table.add_column("Completed", style="green")
            table.add_column("Success Rate", style="white")
            table.add_column("Avg Duration", style="white")
            table.add_column("Total Minutes", style="white")
            table.add_column("Cost", style="yellow")

            for agent in result.get("data", []):
                success_rate = agent.get("success_rate", 0)
                rate_style = "green" if success_rate >= 90 else "yellow" if success_rate >= 70 else "red"

                table.add_row(
                    f"{agent.get('name', 'Unknown')}\n[dim]{agent.get('id', '')}[/dim]",
                    str(agent.get("total_calls", 0)),
                    str(agent.get("completed_calls", 0)),
                    f"[{rate_style}]{success_rate:.1f}%[/{rate_style}]",
                    f"{agent.get('avg_duration', 0):.1f}s",
                    f"{agent.get('total_minutes', 0):.1f}",
                    f"${agent.get('cost', 0):.2f}",
                )

            console.print(table)

        else:
            format_output(result, ctx.obj["output"])

    except Exception as e:
        print_error(f"Failed to get agent analytics: {e}")
        sys.exit(1)


@analytics.command("export")
@click.option("--type", "-t", "report_type", required=True,
              type=click.Choice(["calls", "usage", "agents", "conversations", "campaigns"]),
              help="Report type to export")
@click.option("--format", "-f", "export_format", type=click.Choice(["csv", "json", "xlsx"]),
              default="csv", help="Export format")
@click.option("--period", "-p", type=click.Choice(["day", "week", "month", "quarter"]),
              default="month", help="Time period")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--output", "-o", "output_path", help="Output file path")
@click.pass_context
def export_report(
    ctx: click.Context,
    report_type: str,
    export_format: str,
    period: str,
    start_date: Optional[str],
    end_date: Optional[str],
    output_path: Optional[str],
):
    """Export analytics report."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {
            "report_type": report_type,
            "format": export_format,
            "period": period,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        result = client.analytics.export(**params)

        # Determine output file
        output_file = output_path or f"{report_type}_report_{datetime.now().strftime('%Y%m%d')}.{export_format}"

        # Handle different export methods
        if result.get("download_url"):
            # Download from URL
            import httpx
            console.print(f"Downloading report...")

            with httpx.stream("GET", result["download_url"]) as response:
                with open(output_file, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)

            print_success(f"Report exported to {output_file}")

        elif result.get("content"):
            # Direct content
            if export_format == "json":
                with open(output_file, "w") as f:
                    if isinstance(result["content"], str):
                        f.write(result["content"])
                    else:
                        json.dump(result["content"], f, indent=2)
            else:
                with open(output_file, "w") as f:
                    f.write(result["content"])

            print_success(f"Report exported to {output_file}")

        else:
            print_error("Export failed: No content received")
            sys.exit(1)

    except Exception as e:
        print_error(f"Failed to export report: {e}")
        sys.exit(1)


@analytics.command("realtime")
@click.option("--agent-id", "-a", help="Filter by agent ID")
@click.pass_context
def realtime_stats(ctx: click.Context, agent_id: Optional[str]):
    """Get real-time statistics (current active calls, etc.)."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {}
        if agent_id:
            params["agent_id"] = agent_id

        result = client.analytics.get_realtime(**params)

        if ctx.obj["output"] == "table":
            console.print(Panel.fit(
                "[bold cyan]Real-time Statistics[/bold cyan]",
                border_style="blue",
            ))

            table = Table(show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Active Calls", f"[bold]{result.get('active_calls', 0)}[/bold]")
            table.add_row("Calls in Queue", str(result.get('queued_calls', 0)))
            table.add_row("Active Campaigns", str(result.get('active_campaigns', 0)))
            table.add_row("Concurrent Limit", str(result.get('concurrent_limit', 'N/A')))
            table.add_row("Today's Calls", str(result.get('calls_today', 0)))
            table.add_row("Today's Minutes", f"{result.get('minutes_today', 0):.1f}")

            console.print(table)

            # Show active calls if any
            active = result.get("active_calls_detail", [])
            if active:
                console.print()
                table2 = Table(title="Active Calls")
                table2.add_column("Call ID", style="dim")
                table2.add_column("Agent", style="cyan")
                table2.add_column("Direction", style="white")
                table2.add_column("Duration", style="white")
                table2.add_column("Status", style="green")

                for call in active[:10]:
                    table2.add_row(
                        call.get("id", "")[:12],
                        call.get("agent_name", "Unknown"),
                        call.get("direction", ""),
                        f"{call.get('duration', 0)}s",
                        call.get("status", ""),
                    )

                console.print(table2)

        else:
            format_output(result, ctx.obj["output"])

    except Exception as e:
        print_error(f"Failed to get real-time stats: {e}")
        sys.exit(1)


def _format_change(change: float) -> str:
    """Format a percentage change value."""
    if change > 0:
        return f"[green]+{change:.1f}%[/green]"
    elif change < 0:
        return f"[red]{change:.1f}%[/red]"
    else:
        return "[dim]0%[/dim]"


def _format_number(value: int) -> str:
    """Format a large number with K/M suffix."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return str(value)


def _format_resource_name(name: str) -> str:
    """Format a resource name for display."""
    return name.replace("_", " ").title()
