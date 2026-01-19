# Builder Engine Python SDK

The official Python SDK for [Builder Engine](https://builderengine.io) - the AI Voice Agent Platform.

Build, deploy, and manage AI voice agents that can make and receive phone calls, handle conversations, and integrate with your existing systems.

## Features

- 🤖 **AI Voice Agents** - Create and manage conversational AI agents
- 📞 **Voice Calls** - Make and receive phone calls programmatically
- 🎙️ **Real-time Transcription** - Get live transcriptions of conversations
- 🔊 **Text-to-Speech** - Multiple TTS providers (ElevenLabs, OpenAI, Azure)
- 🧠 **LLM Integration** - Connect with GPT-4, Claude, and other LLMs
- 📚 **Knowledge Bases** - RAG-powered contextual responses
- 🔄 **Workflows** - Automate post-call actions
- 📊 **Analytics** - Comprehensive call and usage analytics
- 🌐 **Webhooks** - Real-time event notifications

## Installation

```bash
pip install builderengine
```

For async support with WebSockets:

```bash
pip install builderengine[async]
```

## Quick Start

```python
from builderengine import BuilderEngine

# Initialize the client
client = BuilderEngine(api_key="your-api-key")

# Create an AI agent
agent = client.agents.create(
    name="Customer Support Agent",
    system_prompt="You are a helpful customer support agent for Acme Corp.",
    voice_id="voice_abc123",
    first_message="Hello! Thank you for calling Acme Corp. How can I help you today?",
    llm_config={
        "model": "gpt-4-turbo",
        "temperature": 0.7
    }
)

# Make an outbound call
call = client.calls.create(
    agent_id=agent.id,
    to_number="+1234567890",
    context={
        "customer_name": "John Doe",
        "account_id": "12345"
    }
)

print(f"Call initiated: {call.id}")
```

## Usage Examples

### Managing Agents

```python
# List all agents
agents = client.agents.list()
for agent in agents:
    print(f"{agent.name}: {agent.total_calls} calls")

# Get agent details
agent = client.agents.get("agent_abc123")

# Update an agent
agent = client.agents.update(
    agent_id="agent_abc123",
    name="Updated Agent Name",
    config={
        "llm": {"temperature": 0.5}
    }
)

# Delete an agent
client.agents.delete("agent_abc123")
```

### Making Calls

```python
# Create an outbound call
call = client.calls.create(
    agent_id="agent_abc123",
    to_number="+1234567890",
    metadata={"campaign": "outreach"}
)

# Wait for call to complete
completed_call = client.calls.wait_for_completion(call.id)
print(f"Call duration: {completed_call.duration_seconds}s")

# Get call transcript
transcript = client.calls.get_transcript(call.id)
for message in transcript["messages"]:
    print(f"{message['role']}: {message['content']}")
```

### Real-time Streaming

```python
import asyncio
from builderengine import AsyncBuilderEngine, StreamEventType

async def main():
    async with AsyncBuilderEngine(api_key="your-api-key") as client:
        streaming = client.streaming()

        @streaming.on(StreamEventType.TRANSCRIPTION_FINAL)
        async def on_transcription(event):
            print(f"User said: {event.data['text']}")

        @streaming.on(StreamEventType.AGENT_SPEECH_START)
        async def on_agent_speech(event):
            print(f"Agent speaking...")

        await streaming.connect(call_id="call_abc123")
        await streaming.listen()

asyncio.run(main())
```

### Knowledge Bases

```python
# Create a knowledge base
kb = client.knowledge_base.create(
    name="Product Documentation",
    description="All product docs and FAQs"
)

# Add documents
client.knowledge_base.add_document(
    knowledge_base_id=kb.id,
    name="Getting Started Guide",
    content="Welcome to our product..."
)

# Upload a file
with open("manual.pdf", "rb") as f:
    client.knowledge_base.add_document(
        knowledge_base_id=kb.id,
        name="User Manual",
        file=f
    )

# Query the knowledge base
results = client.knowledge_base.query(
    knowledge_base_id=kb.id,
    query="How do I reset my password?",
    top_k=3
)
```

### Webhooks

```python
from builderengine import WebhookEvent

# Create a webhook
webhook = client.webhooks.create(
    url="https://api.example.com/webhooks/builderengine",
    events=[
        WebhookEvent.CALL_STARTED,
        WebhookEvent.CALL_ENDED,
        WebhookEvent.TRANSCRIPTION_READY
    ]
)

# Verify webhook signatures in your endpoint
from builderengine.resources.webhooks import WebhooksResource

@app.route("/webhook", methods=["POST"])
def handle_webhook():
    is_valid = WebhooksResource.verify_signature(
        payload=request.data,
        signature=request.headers.get("X-BuilderEngine-Signature"),
        secret=WEBHOOK_SECRET
    )
    if not is_valid:
        return "Invalid signature", 401

    # Process the webhook...
    return "OK", 200
```

### Campaigns

```python
# Create a call campaign
campaign = client.campaigns.create(
    name="Customer Survey",
    agent_id="agent_abc123",
    contacts=[
        {"phone_number": "+1111111111", "name": "Alice", "custom_data": {"order_id": "123"}},
        {"phone_number": "+2222222222", "name": "Bob", "custom_data": {"order_id": "456"}},
    ],
    max_concurrent_calls=5,
    calling_hours_start="09:00",
    calling_hours_end="17:00",
    calling_days=["mon", "tue", "wed", "thu", "fri"]
)

# Start the campaign
client.campaigns.start(campaign.id)

# Monitor progress
analytics = client.campaigns.get_analytics(campaign.id)
print(f"Completed: {analytics['completed_contacts']}/{analytics['total_contacts']}")
```

### Analytics

```python
# Get overview analytics
overview = client.analytics.get_overview(period="month")
print(f"Total calls: {overview.call_metrics.total_calls}")
print(f"Success rate: {overview.call_metrics.success_rate:.1%}")
print(f"Total cost: ${overview.usage_metrics.total_cost:.2f}")

# Get agent performance
top_agents = client.analytics.get_agent_performance(
    period="week",
    sort_by="success_rate",
    limit=5
)
for agent in top_agents:
    print(f"{agent['name']}: {agent['success_rate']:.1%}")
```

## Async Support

The SDK provides full async support:

```python
from builderengine import AsyncBuilderEngine

async def main():
    async with AsyncBuilderEngine(api_key="your-api-key") as client:
        # All methods are async
        agents = await client.agents.list()
        call = await client.calls.create(
            agent_id="agent_abc123",
            to_number="+1234567890"
        )

asyncio.run(main())
```

## Error Handling

```python
from builderengine.exceptions import (
    BuilderEngineError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
)

try:
    call = client.calls.create(agent_id="invalid", to_number="+1234567890")
except AuthenticationError:
    print("Invalid API key")
except ValidationError as e:
    print(f"Validation error: {e.field_errors}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except NotFoundError:
    print("Agent not found")
except BuilderEngineError as e:
    print(f"API error: {e}")
```

## Configuration

```python
client = BuilderEngine(
    api_key="your-api-key",
    base_url="https://api.builderengine.io",  # Custom base URL
    timeout=30.0,  # Request timeout
    max_retries=3,  # Retry attempts
    organization_id="org_123",  # Multi-tenant org ID
    debug=True,  # Enable debug logging
)
```

Environment variables:
- `BUILDERENGINE_API_KEY` - API key
- `BUILDERENGINE_BASE_URL` - Custom base URL

## Documentation

- [API Documentation](https://docs.builderengine.io)
- [Examples](https://github.com/builderengine/python-sdk/tree/main/examples)
- [Changelog](https://github.com/builderengine/python-sdk/blob/main/CHANGELOG.md)

## Support

- Email: support@builderengine.io
- Discord: [Join our community](https://discord.gg/builderengine)
- GitHub Issues: [Report a bug](https://github.com/builderengine/python-sdk/issues)

## License

MIT License - see [LICENSE](LICENSE) for details.
