# Builder Engine TypeScript SDK

Official TypeScript/JavaScript SDK for the Builder Engine AI Voice Agent Platform.

[![npm version](https://badge.fury.io/js/%40builderengine%2Fsdk.svg)](https://www.npmjs.com/package/@builderengine/sdk)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
npm install @builderengine/sdk
# or
yarn add @builderengine/sdk
# or
pnpm add @builderengine/sdk
```

## Quick Start

```typescript
import { BuilderEngine } from '@builderengine/sdk';

const client = new BuilderEngine({
  apiKey: 'your-api-key',
});

// Create an AI voice agent
const agent = await client.agents.create({
  name: 'Support Agent',
  voice: 'nova',
  systemPrompt: 'You are a helpful customer support agent.',
  model: 'gpt-4-turbo',
  language: 'en-US',
});

// Make an outbound call
const call = await client.calls.create({
  agentId: agent.id,
  toNumber: '+14155551234',
  fromNumber: '+14155550000',
});

console.log(`Call started: ${call.id}`);
```

## Features

- **Full API Coverage**: Access all Builder Engine APIs
- **TypeScript Native**: Full type definitions included
- **Browser & Node.js**: Works in both environments
- **Real-time Streaming**: WebSocket support for live events
- **Automatic Retries**: Built-in retry logic for reliability
- **Error Handling**: Typed exceptions for all error scenarios

## Usage Examples

### Managing Agents

```typescript
// List all agents
const agents = await client.agents.list({ limit: 10 });

// Get a specific agent
const agent = await client.agents.get('agent_abc123');

// Update an agent
const updated = await client.agents.update('agent_abc123', {
  systemPrompt: 'Updated prompt...',
  temperature: 0.8,
});

// Delete an agent
await client.agents.delete('agent_abc123');
```

### Making Calls

```typescript
// Create an outbound call
const call = await client.calls.create({
  agentId: 'agent_abc123',
  toNumber: '+14155551234',
  fromNumber: '+14155550000',
  metadata: { customerId: 'cust_123' },
});

// Get call status
const status = await client.calls.get(call.id);

// List recent calls
const calls = await client.calls.list({
  status: 'completed',
  limit: 50,
});

// Get call transcript
const transcript = await client.calls.getTranscript(call.id);

// Get call recording URL
const recording = await client.calls.getRecording(call.id);

// End an active call
await client.calls.end(call.id);
```

### Real-time Streaming

```typescript
import { StreamingClient } from '@builderengine/sdk/streaming';

// Create streaming client
const streaming = client.streaming();

// Listen for events
streaming.on('transcription.final', (event) => {
  console.log('User said:', event.data.text);
});

streaming.on('agent.speech.start', (event) => {
  console.log('Agent is speaking...');
});

streaming.on('call.ended', (event) => {
  console.log('Call ended:', event.data.reason);
});

// Connect and subscribe to a call
await streaming.connect({ callId: 'call_abc123' });

// Or connect and subscribe to all events
await streaming.connect({ subscribeAll: true });

// Send audio data
streaming.sendAudio('call_abc123', audioBuffer);

// Inject text for agent to speak
streaming.injectText('call_abc123', 'Please hold while I check that.', true);

// Disconnect when done
streaming.disconnect();
```

### Phone Numbers

```typescript
// List available numbers
const available = await client.phoneNumbers.listAvailable({
  country: 'US',
  areaCode: '415',
});

// Purchase a number
const number = await client.phoneNumbers.purchase({
  phoneNumber: '+14155551234',
  friendlyName: 'Support Line',
});

// Configure a number with an agent
await client.phoneNumbers.update(number.id, {
  agentId: 'agent_abc123',
  voicemailEnabled: true,
});

// Release a number
await client.phoneNumbers.release(number.id);
```

### Conversations

```typescript
// List conversations
const conversations = await client.conversations.list({
  agentId: 'agent_abc123',
  limit: 20,
});

// Get conversation with messages
const conversation = await client.conversations.get('conv_abc123');

// Get full message history
const messages = await client.conversations.listMessages('conv_abc123');

// Add a message programmatically
await client.conversations.addMessage('conv_abc123', {
  role: 'system',
  content: 'Context: Customer is a VIP member.',
});
```

### Webhooks

```typescript
// Create a webhook
const webhook = await client.webhooks.create({
  url: 'https://your-server.com/webhooks',
  events: ['call.started', 'call.ended', 'transcription.final'],
  secret: 'your-signing-secret',
});

// List webhooks
const webhooks = await client.webhooks.list();

// Test a webhook
await client.webhooks.test(webhook.id);

// Delete a webhook
await client.webhooks.delete(webhook.id);
```

### Knowledge Base

```typescript
// Create a knowledge base
const kb = await client.knowledgeBase.create({
  name: 'Product Documentation',
  description: 'Company product docs and FAQs',
});

// Upload a document
const doc = await client.knowledgeBase.uploadDocument(kb.id, {
  file: documentBuffer,
  filename: 'product-guide.pdf',
  contentType: 'application/pdf',
});

// Add text content
await client.knowledgeBase.addContent(kb.id, {
  title: 'Return Policy',
  content: 'Our return policy allows returns within 30 days...',
  metadata: { category: 'policies' },
});

// Search the knowledge base
const results = await client.knowledgeBase.search(kb.id, {
  query: 'How do I return a product?',
  limit: 5,
});

// Attach to an agent
await client.agents.update('agent_abc123', {
  knowledgeBaseIds: [kb.id],
});
```

### Campaigns (Batch Calls)

```typescript
// Create a campaign
const campaign = await client.campaigns.create({
  name: 'January Outreach',
  agentId: 'agent_abc123',
  fromNumber: '+14155550000',
  contacts: [
    { phoneNumber: '+14155551111', name: 'John Doe', customField: 'value1' },
    { phoneNumber: '+14155552222', name: 'Jane Smith', customField: 'value2' },
  ],
  schedule: {
    startTime: '2024-01-15T09:00:00Z',
    endTime: '2024-01-15T17:00:00Z',
    timezone: 'America/Los_Angeles',
    maxConcurrent: 10,
  },
});

// Start the campaign
await client.campaigns.start(campaign.id);

// Get campaign progress
const progress = await client.campaigns.getProgress(campaign.id);
console.log(`Completed: ${progress.completed}/${progress.total}`);

// Pause/resume campaign
await client.campaigns.pause(campaign.id);
await client.campaigns.resume(campaign.id);

// Cancel campaign
await client.campaigns.cancel(campaign.id);
```

### Analytics

```typescript
// Get overview analytics
const overview = await client.analytics.getOverview({
  period: 'week',
  startDate: '2024-01-01',
  endDate: '2024-01-07',
});

console.log(`Total calls: ${overview.totalCalls}`);
console.log(`Success rate: ${overview.successRate}%`);

// Get usage metrics
const usage = await client.analytics.getUsage('month');

// Get cost breakdown
const costs = await client.analytics.getCostBreakdown({
  period: 'month',
  groupBy: 'agent',
});

// Export a report
const report = await client.analytics.export({
  reportType: 'calls',
  period: 'month',
  format: 'csv',
});
console.log(`Download: ${report.downloadUrl}`);
```

### Organizations & Users

```typescript
// Get current organization
const org = await client.organizations.getCurrent();

// Update organization
await client.organizations.update(org.id, {
  name: 'Acme Corp',
  billingEmail: 'billing@acme.com',
});

// List organization members
const members = await client.organizations.listMembers(org.id);

// Invite a new member
await client.organizations.inviteMember(org.id, 'newuser@acme.com', 'admin');

// Get current user
const me = await client.users.getMe();

// Update profile
await client.users.updateProfile({
  firstName: 'John',
  lastName: 'Doe',
});
```

### API Keys

```typescript
// List API keys
const keys = await client.apiKeys.list();

// Create a new API key
const newKey = await client.apiKeys.create({
  name: 'Production Key',
  permissions: ['calls:read', 'calls:write', 'agents:read'],
  rateLimit: 1000,
});
console.log(`New key: ${newKey.key}`); // Only shown once!

// Regenerate a key
const regenerated = await client.apiKeys.regenerate(newKey.id);

// Delete a key
await client.apiKeys.delete(newKey.id);
```

## Error Handling

The SDK provides typed exceptions for different error scenarios:

```typescript
import {
  BuilderEngineError,
  AuthenticationError,
  RateLimitError,
  ValidationError,
  NotFoundError,
  InsufficientCreditsError,
} from '@builderengine/sdk';

try {
  await client.calls.create({
    agentId: 'invalid',
    toNumber: '+14155551234',
  });
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error('Invalid API key');
  } else if (error instanceof RateLimitError) {
    console.error(`Rate limited. Retry after ${error.retryAfter}s`);
  } else if (error instanceof ValidationError) {
    console.error('Invalid parameters:', error.errors);
  } else if (error instanceof NotFoundError) {
    console.error('Resource not found');
  } else if (error instanceof InsufficientCreditsError) {
    console.error('Add credits to your account');
  } else if (error instanceof BuilderEngineError) {
    console.error(`API error: ${error.message}`);
  }
}
```

## Configuration Options

```typescript
const client = new BuilderEngine({
  apiKey: 'your-api-key',

  // Custom base URL (for enterprise/self-hosted)
  baseUrl: 'https://api.your-domain.com',

  // Request timeout in milliseconds
  timeout: 30000,

  // Maximum retry attempts
  maxRetries: 3,
});
```

## TypeScript Support

The SDK is written in TypeScript and includes full type definitions:

```typescript
import type {
  Agent,
  Call,
  CallStatus,
  Conversation,
  PhoneNumber,
  Voice,
  Webhook,
  Campaign,
  KnowledgeBase,
} from '@builderengine/sdk';

// All types are exported and available
const agent: Agent = await client.agents.get('agent_abc123');
const status: CallStatus = call.status;
```

## Browser Usage

The SDK works in modern browsers. For real-time streaming, the native WebSocket API is used automatically:

```html
<script type="module">
  import { BuilderEngine } from 'https://esm.sh/@builderengine/sdk';

  const client = new BuilderEngine({ apiKey: 'your-api-key' });

  const calls = await client.calls.list();
  console.log(calls);
</script>
```

## Node.js Usage

For Node.js environments, install the `ws` package for WebSocket support:

```bash
npm install ws
```

The SDK will automatically use `ws` when the native WebSocket is not available.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- Documentation: https://docs.builderengine.io
- API Reference: https://docs.builderengine.io/api
- Discord: https://discord.gg/builderengine
- Email: support@builderengine.io
