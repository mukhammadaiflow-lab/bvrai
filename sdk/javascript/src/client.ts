/**
 * Main client for Builder Engine SDK
 */

import { HttpClient } from './utils/http';
import { AgentsAPI } from './api/agents';
import { CallsAPI } from './api/calls';
import { KnowledgeAPI } from './api/knowledge';
import { AnalyticsAPI } from './api/analytics';
import { WebhooksAPI } from './api/webhooks';
import { PhoneNumbersAPI } from './api/phone-numbers';
import { StreamingConnection, StreamingSession } from './streaming';
import type { ClientConfig } from './types';

export class BvraiClient {
  private readonly http: HttpClient;

  /**
   * Agents API for managing voice agents
   */
  public readonly agents: AgentsAPI;

  /**
   * Calls API for managing voice calls
   */
  public readonly calls: CallsAPI;

  /**
   * Knowledge API for managing RAG knowledge bases
   */
  public readonly knowledge: KnowledgeAPI;

  /**
   * Analytics API for metrics and reporting
   */
  public readonly analytics: AnalyticsAPI;

  /**
   * Webhooks API for event subscriptions
   */
  public readonly webhooks: WebhooksAPI;

  /**
   * Phone Numbers API for managing phone numbers
   */
  public readonly phoneNumbers: PhoneNumbersAPI;

  /**
   * API key for authentication
   */
  private readonly apiKey: string;

  /**
   * Base WebSocket URL for streaming
   */
  private readonly wsBaseUrl: string;

  /**
   * Create a new BVRAI client
   *
   * @param config - Client configuration
   *
   * @example
   * ```typescript
   * const client = new BvraiClient({ apiKey: 'your-api-key' });
   *
   * // Create an agent
   * const agent = await client.agents.create({
   *   name: 'Sales Assistant',
   *   systemPrompt: 'You are a helpful sales assistant...',
   * });
   *
   * // Make an outbound call
   * const call = await client.calls.create({
   *   agentId: agent.id,
   *   toNumber: '+15551234567',
   * });
   * ```
   */
  constructor(config: ClientConfig) {
    this.apiKey = config.apiKey;
    this.http = new HttpClient(config);

    // Determine WebSocket URL from HTTP base URL
    const baseUrl = config.baseUrl || 'https://api.bvrai.com';
    this.wsBaseUrl = baseUrl.replace('https://', 'wss://').replace('http://', 'ws://');

    // Initialize API modules
    this.agents = new AgentsAPI(this.http);
    this.calls = new CallsAPI(this.http);
    this.knowledge = new KnowledgeAPI(this.http);
    this.analytics = new AnalyticsAPI(this.http);
    this.webhooks = new WebhooksAPI(this.http);
    this.phoneNumbers = new PhoneNumbersAPI(this.http);
  }

  /**
   * Create a streaming connection for real-time events
   *
   * @param options - Connection options
   * @returns StreamingConnection instance
   *
   * @example
   * ```typescript
   * const stream = client.createStreamingConnection({ callId: 'call_123' });
   * await stream.connect();
   *
   * stream.on('transcript.final', (event) => {
   *   console.log('Transcript:', event.data.text);
   * });
   * ```
   */
  createStreamingConnection(options?: {
    callId?: string;
    sessionId?: string;
    autoReconnect?: boolean;
  }): StreamingConnection {
    return new StreamingConnection({
      apiKey: this.apiKey,
      baseUrl: this.wsBaseUrl,
      ...options,
    });
  }

  /**
   * Create a streaming session for a voice conversation
   *
   * @param agentId - Agent ID to use for the session
   * @returns StreamingSession instance
   *
   * @example
   * ```typescript
   * const session = client.createStreamingSession('agent_123');
   * await session.connect();
   *
   * session.onTranscript((text, isFinal, speaker) => {
   *   console.log(`[${speaker}]: ${text}`);
   * });
   *
   * session.onFunctionCall(async (name, args) => {
   *   if (name === 'get_weather') {
   *     return { temperature: 72, condition: 'sunny' };
   *   }
   *   return {};
   * });
   *
   * const callId = await session.startCall({ toNumber: '+15551234567' });
   * await session.waitForEnd();
   * ```
   */
  createStreamingSession(agentId: string): StreamingSession {
    return new StreamingSession(this.apiKey, agentId, this.wsBaseUrl);
  }

  /**
   * Check API health
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.http.get<{ status: string }>('/health');
      return response.status === 'healthy';
    } catch {
      return false;
    }
  }

  /**
   * Get API information
   */
  async getInfo(): Promise<{
    version: string;
    environment: string;
    timestamp: string;
  }> {
    return this.http.get('/info');
  }
}

/**
 * Create a new BVRAI client
 *
 * @param apiKey - Your API key
 * @param options - Additional configuration options
 * @returns BvraiClient instance
 *
 * @example
 * ```typescript
 * const client = createClient('your-api-key');
 * ```
 */
export function createClient(
  apiKey: string,
  options?: Omit<ClientConfig, 'apiKey'>
): BvraiClient {
  return new BvraiClient({ apiKey, ...options });
}
