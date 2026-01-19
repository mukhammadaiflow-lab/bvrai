/**
 * Builder Engine TypeScript/JavaScript SDK
 *
 * A comprehensive SDK for the Builder Engine AI Voice Agent Platform.
 *
 * @example
 * ```typescript
 * import { BuilderEngine } from '@builderengine/sdk';
 *
 * const client = new BuilderEngine({ apiKey: 'your-api-key' });
 *
 * const agent = await client.agents.create({
 *   name: 'Sales Agent',
 *   voiceId: 'voice_abc123',
 *   llmConfig: { model: 'gpt-4-turbo', temperature: 0.7 }
 * });
 *
 * const call = await client.calls.create({
 *   agentId: agent.id,
 *   toNumber: '+1234567890'
 * });
 * ```
 *
 * @packageDocumentation
 */

// Main client
export { BuilderEngine } from './client';
export type { BuilderEngineConfig } from './client';

// Models
export * from './models';

// Resources
export { AgentsResource } from './resources/agents';
export { CallsResource } from './resources/calls';
export { ConversationsResource } from './resources/conversations';
export { PhoneNumbersResource } from './resources/phoneNumbers';
export { VoicesResource } from './resources/voices';
export { WebhooksResource } from './resources/webhooks';
export { KnowledgeBaseResource } from './resources/knowledgeBase';
export { WorkflowsResource } from './resources/workflows';
export { CampaignsResource } from './resources/campaigns';
export { AnalyticsResource } from './resources/analytics';
export { OrganizationsResource } from './resources/organizations';
export { UsersResource } from './resources/users';
export { APIKeysResource } from './resources/apiKeys';
export { BillingResource } from './resources/billing';

// Exceptions
export * from './exceptions';

// Streaming
export { StreamingClient } from './streaming';
export type { StreamEvent, StreamEventType, EventHandler } from './streaming';

// Utilities
export { verifyWebhookSignature } from './utils/webhooks';
export type { PaginatedResponse, PaginationParams } from './utils/pagination';

// Version
export const VERSION = '1.0.0';
