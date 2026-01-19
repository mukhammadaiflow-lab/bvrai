"""Agent management commands."""

import sys
from typing import Optional

import click
from rich.console import Console

from ..utils.client import get_client
from ..utils.output import format_output, print_success, print_error, confirm

console = Console()


@click.group()
def agents():
    """Manage AI voice agents.

    \b
    Examples:
      builderengine agents list
      builderengine agents get agent_abc123
      builderengine agents create --name "Support Bot" --voice nova
    """
    pass


@agents.command("list")
@click.option("--status", type=click.Choice(["active", "inactive", "draft"]),
              help="Filter by status")
@click.option("--limit", "-l", type=int, default=20, help="Maximum results")
@click.option("--search", "-s", help="Search by name")
@click.pass_context
def list_agents(ctx: click.Context, status: Optional[str], limit: int, search: Optional[str]):
    """List all agents."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        params = {"limit": limit}
        if status:
            params["status"] = status
        if search:
            params["search"] = search

        result = client.agents.list(**params)
        format_output(
            result,
            ctx.obj["output"],
            columns=["id", "name", "voice", "status", "total_calls", "success_rate", "created_at"],
        )
    except Exception as e:
        print_error(f"Failed to list agents: {e}")
        sys.exit(1)


@agents.command("get")
@click.argument("agent_id")
@click.pass_context
def get_agent(ctx: click.Context, agent_id: str):
    """Get details for a specific agent."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.agents.get(agent_id)
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to get agent: {e}")
        sys.exit(1)


@agents.command("create")
@click.option("--name", "-n", required=True, help="Agent name")
@click.option("--voice", "-v", required=True, help="Voice ID or name")
@click.option("--prompt", "-p", required=True, help="System prompt")
@click.option("--model", "-m", default="gpt-4-turbo", help="LLM model")
@click.option("--language", default="en-US", help="Language code")
@click.option("--first-message", help="Initial greeting message")
@click.option("--temperature", type=float, default=0.7, help="Temperature (0-1)")
@click.pass_context
def create_agent(
    ctx: click.Context,
    name: str,
    voice: str,
    prompt: str,
    model: str,
    language: str,
    first_message: Optional[str],
    temperature: float,
):
    """Create a new agent."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        data = {
            "name": name,
            "voice": voice,
            "system_prompt": prompt,
            "model": model,
            "language": language,
            "temperature": temperature,
        }
        if first_message:
            data["first_message"] = first_message

        result = client.agents.create(**data)
        print_success(f"Agent created: {result['id']}")
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to create agent: {e}")
        sys.exit(1)


@agents.command("update")
@click.argument("agent_id")
@click.option("--name", "-n", help="Agent name")
@click.option("--voice", "-v", help="Voice ID or name")
@click.option("--prompt", "-p", help="System prompt")
@click.option("--status", type=click.Choice(["active", "inactive"]), help="Agent status")
@click.option("--temperature", type=float, help="Temperature (0-1)")
@click.pass_context
def update_agent(
    ctx: click.Context,
    agent_id: str,
    name: Optional[str],
    voice: Optional[str],
    prompt: Optional[str],
    status: Optional[str],
    temperature: Optional[float],
):
    """Update an agent."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        data = {}
        if name:
            data["name"] = name
        if voice:
            data["voice"] = voice
        if prompt:
            data["system_prompt"] = prompt
        if status:
            data["status"] = status
        if temperature is not None:
            data["temperature"] = temperature

        if not data:
            print_error("No updates specified")
            sys.exit(1)

        result = client.agents.update(agent_id, **data)
        print_success(f"Agent updated: {agent_id}")
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to update agent: {e}")
        sys.exit(1)


@agents.command("delete")
@click.argument("agent_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_agent(ctx: click.Context, agent_id: str, force: bool):
    """Delete an agent."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    if not force:
        if not confirm(f"Are you sure you want to delete agent {agent_id}?"):
            console.print("Cancelled")
            return

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        client.agents.delete(agent_id)
        print_success(f"Agent deleted: {agent_id}")
    except Exception as e:
        print_error(f"Failed to delete agent: {e}")
        sys.exit(1)


@agents.command("duplicate")
@click.argument("agent_id")
@click.option("--name", "-n", required=True, help="Name for the new agent")
@click.pass_context
def duplicate_agent(ctx: click.Context, agent_id: str, name: str):
    """Duplicate an agent."""
    api_key = ctx.obj.get("api_key")
    if not api_key:
        print_error("Not logged in. Run 'builderengine login' first.")
        sys.exit(1)

    try:
        client = get_client(api_key, ctx.obj["base_url"])
        result = client.agents.duplicate(agent_id, name)
        print_success(f"Agent duplicated: {result['id']}")
        format_output(result, ctx.obj["output"])
    except Exception as e:
        print_error(f"Failed to duplicate agent: {e}")
        sys.exit(1)
