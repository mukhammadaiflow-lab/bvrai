"""Output formatting utilities."""

import json
from typing import Any, Dict, List, Optional

import yaml
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

console = Console()


def format_output(data: Any, format_type: str = "table", columns: Optional[List[str]] = None) -> None:
    """Format and print data based on format type."""
    if format_type == "json":
        print_json(data)
    elif format_type == "yaml":
        print_yaml(data)
    else:
        if isinstance(data, list):
            print_table(data, columns)
        elif isinstance(data, dict):
            if "data" in data:
                print_table(data["data"], columns)
                if "total" in data:
                    console.print(f"\nTotal: {data['total']}")
            else:
                print_table([data], columns)
        else:
            console.print(data)


def print_json(data: Any) -> None:
    """Print data as formatted JSON."""
    json_str = json.dumps(data, indent=2, default=str)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
    console.print(syntax)


def print_yaml(data: Any) -> None:
    """Print data as formatted YAML."""
    yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
    console.print(syntax)


def print_table(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> None:
    """Print data as a formatted table."""
    if not data:
        console.print("[dim]No data to display[/dim]")
        return

    table = Table(title=title, show_header=True, header_style="bold cyan")

    # Determine columns to display
    if columns:
        display_columns = columns
    else:
        # Use all keys from first item, prioritizing common fields
        all_keys = list(data[0].keys())
        priority = ["id", "name", "status", "created_at", "updated_at"]
        display_columns = [k for k in priority if k in all_keys]
        display_columns.extend([k for k in all_keys if k not in display_columns])
        # Limit to reasonable number
        display_columns = display_columns[:8]

    # Add columns
    for col in display_columns:
        table.add_column(col.replace("_", " ").title())

    # Add rows
    for item in data:
        row = []
        for col in display_columns:
            value = item.get(col, "")
            if value is None:
                value = "-"
            elif isinstance(value, bool):
                value = "✓" if value else "✗"
            elif isinstance(value, (list, dict)):
                value = json.dumps(value)[:50] + "..." if len(json.dumps(value)) > 50 else json.dumps(value)
            else:
                value = str(value)
                if len(value) > 50:
                    value = value[:47] + "..."
            row.append(value)
        table.add_row(*row)

    console.print(table)


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]![/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]i[/blue] {message}")


def confirm(message: str, default: bool = False) -> bool:
    """Ask for confirmation."""
    suffix = " [Y/n]" if default else " [y/N]"
    result = console.input(f"{message}{suffix} ")
    if not result:
        return default
    return result.lower() in ("y", "yes")
