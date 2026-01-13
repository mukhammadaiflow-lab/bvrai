# Builder Engine CLI

Official command-line interface for the Builder Engine AI Voice Agent Platform.

[![PyPI version](https://badge.fury.io/py/builderengine-cli.svg)](https://pypi.org/project/builderengine-cli/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install builderengine-cli
```

Or with pipx for isolated installation:

```bash
pipx install builderengine-cli
```

## Quick Start

```bash
# Login with your API key
builderengine login

# List your agents
builderengine agents list

# Create a new agent
builderengine agents create --name "Support Agent" --voice nova

# Make an outbound call
builderengine calls create --agent-id agent_abc123 --to +14155551234

# View call details
builderengine calls get call_xyz789

# Get analytics overview
builderengine analytics overview
```

## Features

- **Full API Coverage**: Access all Builder Engine APIs from the command line
- **Multiple Output Formats**: Table, JSON, and YAML output
- **Configuration Profiles**: Manage multiple environments (dev, staging, production)
- **Shell Completion**: Tab completion for bash, zsh, and fish
- **Rich Output**: Beautiful colored output with progress indicators

## Commands

### Agents

```bash
# List all agents
builderengine agents list

# Get agent details
builderengine agents get agent_abc123

# Create an agent
builderengine agents create \
  --name "Support Agent" \
  --voice nova \
  --model gpt-4-turbo \
  --system-prompt "You are a helpful customer support agent."

# Update an agent
builderengine agents update agent_abc123 \
  --temperature 0.8 \
  --max-tokens 500

# Delete an agent
builderengine agents delete agent_abc123

# Duplicate an agent
builderengine agents duplicate agent_abc123 --name "Support Agent v2"
```

### Calls

```bash
# List recent calls
builderengine calls list --limit 50

# List calls by status
builderengine calls list --status completed

# Get call details
builderengine calls get call_xyz789

# Make an outbound call
builderengine calls create \
  --agent-id agent_abc123 \
  --to +14155551234 \
  --from +14155550000

# Make a call and wait for completion
builderengine calls create \
  --agent-id agent_abc123 \
  --to +14155551234 \
  --wait

# End an active call
builderengine calls end call_xyz789

# Get call transcript
builderengine calls transcript call_xyz789

# Download call recording
builderengine calls recording call_xyz789 --download -o recording.mp3
```

### Campaigns

```bash
# List campaigns
builderengine campaigns list

# Create a campaign
builderengine campaigns create \
  --name "January Outreach" \
  --agent-id agent_abc123 \
  --contacts-file contacts.json \
  --max-concurrent 10

# Add contacts to campaign
builderengine campaigns add-contacts campaign_abc \
  --file more-contacts.json

# Start a campaign
builderengine campaigns start campaign_abc

# Watch campaign progress
builderengine campaigns progress campaign_abc --watch

# Pause/resume campaign
builderengine campaigns pause campaign_abc
builderengine campaigns resume campaign_abc

# Cancel campaign
builderengine campaigns cancel campaign_abc

# Export campaign results
builderengine campaigns export campaign_abc -o results.csv
```

### Phone Numbers

```bash
# List your phone numbers
builderengine numbers list

# Search available numbers
builderengine numbers search --country US --area-code 415

# Purchase a number
builderengine numbers purchase +14155551234 --name "Support Line"

# Configure number with agent
builderengine numbers configure pn_abc123 --agent-id agent_abc123

# Enable voicemail
builderengine numbers configure pn_abc123 \
  --voicemail \
  --voicemail-greeting "Leave a message after the tone."

# Release a number
builderengine numbers release pn_abc123

# Get number statistics
builderengine numbers stats pn_abc123 --period month
```

### Webhooks

```bash
# List webhooks
builderengine webhooks list

# Create a webhook
builderengine webhooks create \
  --url https://your-server.com/webhook \
  --events call.started call.ended transcription.final

# Test a webhook
builderengine webhooks test webhook_abc123

# View delivery logs
builderengine webhooks logs webhook_abc123

# View available event types
builderengine webhooks events

# Update webhook
builderengine webhooks update webhook_abc123 \
  --events call.started call.ended call.failed

# Regenerate signing secret
builderengine webhooks regenerate-secret webhook_abc123

# Delete webhook
builderengine webhooks delete webhook_abc123
```

### Analytics

```bash
# Overview dashboard
builderengine analytics overview

# Call analytics
builderengine analytics calls --period month --group-by day

# Usage metrics
builderengine analytics usage --period month --detailed

# Per-agent stats
builderengine analytics agents --sort success_rate

# Real-time stats
builderengine analytics realtime

# Export reports
builderengine analytics export \
  --type calls \
  --format csv \
  --period month \
  -o call-report.csv
```

### Configuration

```bash
# Show current configuration
builderengine config show

# Set output format
builderengine config set output json

# List profiles
builderengine config profiles

# Create a profile
builderengine config create-profile production --api-key sk_prod_xxx

# Switch profiles
builderengine config use production

# Export configuration
builderengine config export -o config.json

# Generate shell completion
builderengine config completion bash >> ~/.bashrc
builderengine config completion zsh >> ~/.zshrc
```

## Output Formats

Control output format with the `--output` flag:

```bash
# Table output (default)
builderengine agents list

# JSON output
builderengine agents list --output json

# YAML output
builderengine agents list --output yaml
```

Or set a default format:

```bash
builderengine config set output json
```

## Configuration Profiles

Manage multiple environments with profiles:

```bash
# Create profiles
builderengine config create-profile development --api-key sk_dev_xxx
builderengine config create-profile staging --api-key sk_staging_xxx
builderengine config create-profile production --api-key sk_prod_xxx

# Switch between profiles
builderengine config use development
builderengine agents list  # Uses development API key

builderengine config use production
builderengine agents list  # Uses production API key
```

## Environment Variables

The CLI supports the following environment variables:

| Variable | Description |
|----------|-------------|
| `BUILDERENGINE_API_KEY` | API key (overrides config) |
| `BUILDERENGINE_BASE_URL` | API base URL |
| `BUILDERENGINE_PROFILE` | Profile to use |
| `NO_COLOR` | Disable colored output |

```bash
# Use environment variable for API key
export BUILDERENGINE_API_KEY=sk_xxx
builderengine agents list

# Override profile for a single command
BUILDERENGINE_PROFILE=production builderengine agents list
```

## Shell Completion

Enable tab completion for your shell:

### Bash

```bash
builderengine config completion bash >> ~/.bashrc
source ~/.bashrc
```

### Zsh

```bash
builderengine config completion zsh >> ~/.zshrc
source ~/.zshrc
```

### Fish

```bash
builderengine config completion fish > ~/.config/fish/completions/builderengine.fish
```

## Contacts File Format

For campaigns, contacts should be in JSON format:

```json
[
  {
    "phone_number": "+14155551111",
    "name": "John Doe",
    "email": "john@example.com",
    "custom_field": "value1"
  },
  {
    "phone_number": "+14155552222",
    "name": "Jane Smith",
    "custom_field": "value2"
  }
]
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Authentication error |
| 4 | Resource not found |
| 5 | Rate limit exceeded |

## Aliases

The CLI can also be invoked as `bvr` for convenience:

```bash
bvr agents list
bvr calls create --agent-id agent_123 --to +14155551234
```

## Development

```bash
# Clone the repository
git clone https://github.com/builderengine/cli
cd cli

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy builderengine_cli
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- Documentation: https://docs.builderengine.io/cli
- API Reference: https://docs.builderengine.io/api
- Discord: https://discord.gg/builderengine
- Email: support@builderengine.io
