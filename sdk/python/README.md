# BVRAI Python SDK

Official Python SDK for the Builder Voice AI Platform.

## Installation

```bash
pip install bvrai
```

## Quick Start

```python
from bvrai import BVRAIClient

# Initialize the client
client = BVRAIClient(api_key="bvr_your_api_key")

# List your agents
agents = client.agents.list()
for agent in agents.items:
    print(f"Agent: {agent.name} ({agent.status})")

# Create a new agent
agent = client.agents.create(
    name="Customer Support Agent",
    description="Handles customer inquiries",
    system_prompt="You are a helpful customer support agent.",
)

# Start an outbound call
call = client.calls.create(
    agent_id=agent.id,
    to_number="+1234567890",
    from_number="+0987654321",
)
```

## Async Usage

```python
import asyncio
from bvrai import AsyncBVRAIClient

async def main():
    client = AsyncBVRAIClient(api_key="bvr_your_api_key")

    agents = await client.agents.list()
    for agent in agents.items:
        print(f"Agent: {agent.name}")

    await client.close()

asyncio.run(main())
```

## Documentation

Full documentation is available at [docs.bvrai.com](https://docs.bvrai.com)

## License

MIT License
