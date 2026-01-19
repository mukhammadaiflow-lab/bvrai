"""Configuration management commands."""

import sys
import json
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from ..utils.config import (
    get_config,
    save_config,
    get_profile,
    save_profile,
    list_profiles,
    delete_profile,
    set_default_profile,
    get_config_path,
)
from ..utils.output import print_success, print_error, confirm

console = Console()


@click.group()
def config():
    """Manage CLI configuration.

    \b
    Examples:
      builderengine config show
      builderengine config set output json
      builderengine config profiles
      builderengine config use production
    """
    pass


@config.command("show")
@click.option("--all", "-a", "show_all", is_flag=True, help="Show all settings including secrets")
@click.pass_context
def show_config(ctx: click.Context, show_all: bool):
    """Show current configuration."""
    cfg = get_config()

    if ctx.obj["output"] == "json":
        if not show_all:
            # Mask sensitive data
            if cfg.get("api_key"):
                cfg["api_key"] = cfg["api_key"][:8] + "..." + cfg["api_key"][-4:]
        console.print(json.dumps(cfg, indent=2))
    else:
        console.print(Panel.fit("[bold cyan]CLI Configuration[/bold cyan]", border_style="blue"))

        table = Table(show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        # Profile info
        table.add_row("Active Profile", cfg.get("profile", "default"))
        table.add_row("Config Path", get_config_path())

        # API settings
        api_key = cfg.get("api_key", "Not set")
        if api_key != "Not set" and not show_all:
            api_key = api_key[:8] + "..." + api_key[-4:]
        table.add_row("API Key", api_key)
        table.add_row("Base URL", cfg.get("base_url", "https://api.builderengine.io"))

        # Output settings
        table.add_row("Output Format", cfg.get("output", "table"))
        table.add_row("Color", str(cfg.get("color", True)))

        # Other settings
        if cfg.get("default_agent_id"):
            table.add_row("Default Agent", cfg.get("default_agent_id"))
        if cfg.get("default_phone_number_id"):
            table.add_row("Default Phone Number", cfg.get("default_phone_number_id"))

        console.print(table)


@config.command("set")
@click.argument("key")
@click.argument("value")
def set_config(key: str, value: str):
    """Set a configuration value.

    \b
    Available keys:
      output          - Output format (table, json, yaml)
      color           - Enable color output (true/false)
      base_url        - API base URL
      default_agent_id      - Default agent ID for commands
      default_phone_number_id - Default phone number ID
    """
    cfg = get_config()

    # Validate and convert value
    valid_keys = ["output", "color", "base_url", "default_agent_id", "default_phone_number_id"]

    if key not in valid_keys:
        print_error(f"Unknown configuration key: {key}")
        console.print(f"[dim]Valid keys: {', '.join(valid_keys)}[/dim]")
        sys.exit(1)

    # Type conversion
    if key == "color":
        value = value.lower() in ("true", "yes", "1", "on")
    elif key == "output":
        if value not in ("table", "json", "yaml"):
            print_error("Output must be one of: table, json, yaml")
            sys.exit(1)

    cfg[key] = value
    save_config(cfg)
    print_success(f"Configuration updated: {key} = {value}")


@config.command("unset")
@click.argument("key")
def unset_config(key: str):
    """Unset a configuration value."""
    cfg = get_config()

    if key not in cfg:
        print_error(f"Configuration key not found: {key}")
        sys.exit(1)

    # Don't allow unsetting critical keys
    if key in ("api_key", "profile"):
        print_error(f"Cannot unset {key}. Use 'builderengine logout' instead.")
        sys.exit(1)

    del cfg[key]
    save_config(cfg)
    print_success(f"Configuration key removed: {key}")


@config.command("profiles")
def list_profiles_cmd():
    """List all configuration profiles."""
    profiles = list_profiles()
    cfg = get_config()
    current = cfg.get("profile", "default")

    if not profiles:
        console.print("[dim]No profiles found.[/dim]")
        return

    table = Table(title="Configuration Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Base URL", style="white")
    table.add_column("API Key", style="dim")
    table.add_column("Default", style="green")

    for name in profiles:
        profile = get_profile(name)
        is_current = "✓" if name == current else ""

        api_key = profile.get("api_key", "Not set")
        if api_key != "Not set":
            api_key = api_key[:8] + "..." if len(api_key) > 8 else api_key

        table.add_row(
            name,
            profile.get("base_url", "https://api.builderengine.io"),
            api_key,
            is_current,
        )

    console.print(table)


@config.command("create-profile")
@click.argument("name")
@click.option("--api-key", "-k", help="API key for this profile")
@click.option("--base-url", "-u", help="Base URL for this profile")
@click.option("--copy-from", "-c", help="Copy settings from existing profile")
def create_profile(
    name: str,
    api_key: Optional[str],
    base_url: Optional[str],
    copy_from: Optional[str],
):
    """Create a new configuration profile."""
    # Check if profile already exists
    profiles = list_profiles()
    if name in profiles:
        print_error(f"Profile '{name}' already exists")
        sys.exit(1)

    # Create profile data
    if copy_from:
        if copy_from not in profiles:
            print_error(f"Profile '{copy_from}' not found")
            sys.exit(1)
        profile_data = get_profile(copy_from).copy()
    else:
        profile_data = {}

    # Override with provided values
    if api_key:
        profile_data["api_key"] = api_key
    if base_url:
        profile_data["base_url"] = base_url

    save_profile(name, profile_data)
    print_success(f"Profile created: {name}")

    if not api_key:
        console.print(f"\n[yellow]Note: Run 'builderengine config use {name}' and then 'builderengine login'[/yellow]")


@config.command("use")
@click.argument("name")
def use_profile(name: str):
    """Switch to a different profile."""
    profiles = list_profiles()

    if name not in profiles:
        print_error(f"Profile '{name}' not found")
        console.print(f"[dim]Available profiles: {', '.join(profiles)}[/dim]")
        sys.exit(1)

    set_default_profile(name)
    print_success(f"Switched to profile: {name}")


@config.command("delete-profile")
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def delete_profile_cmd(name: str, force: bool):
    """Delete a configuration profile."""
    profiles = list_profiles()

    if name not in profiles:
        print_error(f"Profile '{name}' not found")
        sys.exit(1)

    if name == "default":
        print_error("Cannot delete the default profile")
        sys.exit(1)

    cfg = get_config()
    if cfg.get("profile") == name:
        print_error(f"Cannot delete active profile. Switch to another profile first.")
        sys.exit(1)

    if not force:
        if not confirm(f"Delete profile '{name}'?"):
            console.print("Canceled")
            return

    delete_profile(name)
    print_success(f"Profile deleted: {name}")


@config.command("edit")
def edit_config():
    """Open configuration file in editor."""
    import os
    import subprocess

    config_path = get_config_path()

    # Determine editor
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))

    try:
        subprocess.run([editor, config_path])
        print_success(f"Configuration file: {config_path}")
    except FileNotFoundError:
        print_error(f"Editor not found: {editor}")
        console.print(f"[dim]Set EDITOR environment variable or edit {config_path} manually[/dim]")
        sys.exit(1)


@config.command("path")
def show_path():
    """Show configuration file path."""
    console.print(get_config_path())


@config.command("reset")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def reset_config(force: bool):
    """Reset configuration to defaults."""
    if not force:
        if not confirm("Reset all configuration to defaults? This will log you out."):
            console.print("Canceled")
            return

    # Create default config
    default_cfg = {
        "profile": "default",
        "output": "table",
        "color": True,
        "base_url": "https://api.builderengine.io",
    }

    save_config(default_cfg)
    print_success("Configuration reset to defaults")


@config.command("export")
@click.option("--output", "-o", "output_path", help="Output file path")
@click.option("--include-secrets", is_flag=True, help="Include API keys in export")
def export_config(output_path: Optional[str], include_secrets: bool):
    """Export configuration to file."""
    cfg = get_config()

    # Remove secrets unless explicitly included
    if not include_secrets:
        export_cfg = cfg.copy()
        if "api_key" in export_cfg:
            del export_cfg["api_key"]
    else:
        export_cfg = cfg

    output_file = output_path or "builderengine-config.json"

    with open(output_file, "w") as f:
        json.dump(export_cfg, f, indent=2)

    print_success(f"Configuration exported to {output_file}")

    if include_secrets:
        console.print("[yellow]Warning: Export contains API key. Store securely.[/yellow]")


@config.command("import")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--merge", "-m", is_flag=True, help="Merge with existing config")
def import_config(file_path: str, merge: bool):
    """Import configuration from file."""
    try:
        with open(file_path, "r") as f:
            import_cfg = json.load(f)
    except json.JSONDecodeError:
        print_error("Invalid JSON file")
        sys.exit(1)

    if merge:
        cfg = get_config()
        cfg.update(import_cfg)
        save_config(cfg)
        print_success("Configuration merged successfully")
    else:
        save_config(import_cfg)
        print_success("Configuration imported successfully")


@config.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def shell_completion(shell: str):
    """Generate shell completion script.

    \b
    Installation:
      Bash: builderengine config completion bash >> ~/.bashrc
      Zsh:  builderengine config completion zsh >> ~/.zshrc
      Fish: builderengine config completion fish > ~/.config/fish/completions/builderengine.fish
    """
    import os

    if shell == "bash":
        script = '''
# Bash completion for builderengine
_builderengine_completion() {
    local IFS=$'\\n'
    local response

    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD _BUILDERENGINE_COMPLETE=bash_complete builderengine)

    for completion in $response; do
        IFS=',' read type value <<< "$completion"

        if [[ $type == 'dir' ]]; then
            COMPREPLY=()
            compopt -o dirnames
        elif [[ $type == 'file' ]]; then
            COMPREPLY=()
            compopt -o default
        elif [[ $type == 'plain' ]]; then
            COMPREPLY+=($value)
        fi
    done

    return 0
}

complete -o nosort -F _builderengine_completion builderengine
'''
    elif shell == "zsh":
        script = '''
#compdef builderengine

_builderengine() {
    local -a completions
    local -a completions_with_descriptions
    local -a response
    (( ! $+commands[builderengine] )) && return 1

    response=("${(@f)$(env COMP_WORDS="${words[*]}" COMP_CWORD=$((CURRENT-1)) _BUILDERENGINE_COMPLETE=zsh_complete builderengine)}")

    for key descr in ${(kv)response}; do
      if [[ "$descr" == "_" ]]; then
          completions+=("$key")
      else
          completions_with_descriptions+=("$key":"$descr")
      fi
    done

    if [ -n "$completions_with_descriptions" ]; then
        _describe -V unsorted completions_with_descriptions -U
    fi

    if [ -n "$completions" ]; then
        compadd -U -V unsorted -a completions
    fi
}

compdef _builderengine builderengine
'''
    else:  # fish
        script = '''
function __fish_builderengine_complete
    set -l response (env _BUILDERENGINE_COMPLETE=fish_complete COMP_WORDS=(commandline -cp) COMP_CWORD=(commandline -t) builderengine)

    for completion in $response
        set -l metadata (string split "," -- $completion)

        if test $metadata[1] = "dir"
            __fish_complete_directories $metadata[2]
        else if test $metadata[1] = "file"
            __fish_complete_path $metadata[2]
        else if test $metadata[1] = "plain"
            echo $metadata[2]
        end
    end
end

complete -c builderengine -f -a "(__fish_builderengine_complete)"
'''

    console.print(script.strip())
