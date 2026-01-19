"""
Builder Voice AI Python SDK

Official Python client for the Builder Voice AI Platform.

Quick Start:

    from bvrai import Client

    # Initialize client
    client = Client(api_key="your_api_key")

    # Create an agent
    agent = client.agents.create(
        name="Sales Agent",
        system_prompt="You are a helpful sales assistant.",
        llm_config={"provider": "openai", "model": "gpt-4o"},
    )

    # List all agents
    agents = client.agents.list()

    # Get call history
    calls = client.calls.list(status="completed")

    # Async usage
    import asyncio
    from bvrai import AsyncClient

    async def main():
        async with AsyncClient(api_key="your_api_key") as client:
            agents = await client.agents.list()
            print(agents)

    asyncio.run(main())

"""

__version__ = "1.0.0"
__author__ = "Builder Voice AI"

from .client import Client, AsyncClient
from .exceptions import (
    BVRAIError,
    APIError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ValidationError,
)
from .types import (
    Agent,
    Call,
    Conversation,
    Message,
    VoiceConfiguration,
    Webhook,
    AnalyticsSummary,
)

__all__ = [
    # Version
    "__version__",
    # Clients
    "Client",
    "AsyncClient",
    # Exceptions
    "BVRAIError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "ValidationError",
    # Types
    "Agent",
    "Call",
    "Conversation",
    "Message",
    "VoiceConfiguration",
    "Webhook",
    "AnalyticsSummary",
]
