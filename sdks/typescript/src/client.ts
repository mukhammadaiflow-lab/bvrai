/**
 * Builder Engine TypeScript SDK - Main Client
 *
 * This module provides the main BuilderEngine client class.
 */

import { AgentsResource } from './resources/agents';
import { CallsResource } from './resources/calls';
import { ConversationsResource } from './resources/conversations';
import { PhoneNumbersResource } from './resources/phoneNumbers';
import { VoicesResource } from './resources/voices';
import { WebhooksResource } from './resources/webhooks';
import { KnowledgeBaseResource } from './resources/knowledgeBase';
import { WorkflowsResource } from './resources/workflows';
import { CampaignsResource } from './resources/campaigns';
import { AnalyticsResource } from './resources/analytics';
import { OrganizationsResource } from './resources/organizations';
import { UsersResource } from './resources/users';
import { APIKeysResource } from './resources/apiKeys';
import { BillingResource } from './resources/billing';
import { StreamingClient } from './streaming';
import {
  BuilderEngineError,
  AuthenticationError,
  RateLimitError,
  ValidationError,
  NotFoundError,
  ConflictError,
  ServerError,
} from './exceptions';

/**
 * Configuration options for the BuilderEngine client.
 */
export interface BuilderEngineConfig {
  /** Your Builder Engine API key */
  apiKey: string;
  /** Base URL for the API (default: https://api.builderengine.io) */
  baseUrl?: string;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Maximum number of retry attempts (default: 3) */
  maxRetries?: number;
  /** Organization ID for multi-tenant requests */
  organizationId?: string;
  /** Enable debug logging */
  debug?: boolean;
  /** Custom fetch implementation */
  fetch?: typeof fetch;
}

/**
 * Request options for API calls.
 */
export interface RequestOptions {
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  path: string;
  params?: Record<string, any>;
  body?: Record<string, any>;
  headers?: Record<string, string>;
  timeout?: number;
}

/**
 * Main client for interacting with the Builder Engine API.
 *
 * This client provides access to all Builder Engine resources including
 * agents, calls, conversations, phone numbers, and more.
 *
 * @example
 * ```typescript
 * const client = new BuilderEngine({ apiKey: 'your-api-key' });
 *
 * // Create an agent
 * const agent = await client.agents.create({
 *   name: 'Customer Support Agent',
 *   systemPrompt: 'You are a helpful customer support agent...',
 *   voiceId: 'voice_abc123'
 * });
 *
 * // Make a call
 * const call = await client.calls.create({
 *   agentId: agent.id,
 *   toNumber: '+1234567890'
 * });
 * ```
 */
export class BuilderEngine {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly timeout: number;
  private readonly maxRetries: number;
  private readonly organizationId?: string;
  private readonly debug: boolean;
  private readonly fetchImpl: typeof fetch;

  /** Resource for managing AI agents */
  public readonly agents: AgentsResource;
  /** Resource for managing voice calls */
  public readonly calls: CallsResource;
  /** Resource for managing conversations */
  public readonly conversations: ConversationsResource;
  /** Resource for managing phone numbers */
  public readonly phoneNumbers: PhoneNumbersResource;
  /** Resource for managing voice configurations */
  public readonly voices: VoicesResource;
  /** Resource for managing webhooks */
  public readonly webhooks: WebhooksResource;
  /** Resource for managing knowledge bases */
  public readonly knowledgeBase: KnowledgeBaseResource;
  /** Resource for managing workflows */
  public readonly workflows: WorkflowsResource;
  /** Resource for managing campaigns */
  public readonly campaigns: CampaignsResource;
  /** Resource for accessing analytics */
  public readonly analytics: AnalyticsResource;
  /** Resource for managing organizations */
  public readonly organizations: OrganizationsResource;
  /** Resource for managing users */
  public readonly users: UsersResource;
  /** Resource for managing API keys */
  public readonly apiKeys: APIKeysResource;
  /** Resource for billing and subscriptions */
  public readonly billing: BillingResource;

  /**
   * Create a new BuilderEngine client.
   *
   * @param config - Configuration options
   * @throws {AuthenticationError} If API key is not provided
   */
  constructor(config: BuilderEngineConfig) {
    if (!config.apiKey) {
      throw new AuthenticationError(
        'API key is required. Provide it in the config or set BUILDERENGINE_API_KEY environment variable.'
      );
    }

    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || process.env.BUILDERENGINE_BASE_URL || 'https://api.builderengine.io';
    this.timeout = config.timeout || 30000;
    this.maxRetries = config.maxRetries || 3;
    this.organizationId = config.organizationId;
    this.debug = config.debug || false;
    this.fetchImpl = config.fetch || globalThis.fetch;

    // Initialize resources
    this.agents = new AgentsResource(this);
    this.calls = new CallsResource(this);
    this.conversations = new ConversationsResource(this);
    this.phoneNumbers = new PhoneNumbersResource(this);
    this.voices = new VoicesResource(this);
    this.webhooks = new WebhooksResource(this);
    this.knowledgeBase = new KnowledgeBaseResource(this);
    this.workflows = new WorkflowsResource(this);
    this.campaigns = new CampaignsResource(this);
    this.analytics = new AnalyticsResource(this);
    this.organizations = new OrganizationsResource(this);
    this.users = new UsersResource(this);
    this.apiKeys = new APIKeysResource(this);
    this.billing = new BillingResource(this);

    if (this.debug) {
      console.log(`BuilderEngine client initialized with base URL: ${this.baseUrl}`);
    }
  }

  /**
   * Make an HTTP request to the API.
   *
   * @param options - Request options
   * @returns Response data
   * @throws {BuilderEngineError} On API errors
   */
  async request<T = any>(options: RequestOptions): Promise<T> {
    const { method, path, params, body, headers: customHeaders, timeout = this.timeout } = options;

    // Build URL with query params
    let url = path.startsWith('http') ? path : `${this.baseUrl}${path}`;
    if (params) {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.append(key, String(value));
        }
      });
      const queryString = searchParams.toString();
      if (queryString) {
        url += `?${queryString}`;
      }
    }

    // Build headers
    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': 'builderengine-typescript/1.0.0',
      ...customHeaders,
    };

    if (this.organizationId) {
      headers['X-Organization-ID'] = this.organizationId;
    }

    // Build request
    const requestInit: RequestInit = {
      method,
      headers,
    };

    if (body && method !== 'GET') {
      requestInit.body = JSON.stringify(body);
    }

    // Execute with retries
    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        if (this.debug) {
          console.log(`${method} ${url}`, body ? JSON.stringify(body) : '');
        }

        // Create abort controller for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        requestInit.signal = controller.signal;

        const response = await this.fetchImpl(url, requestInit);
        clearTimeout(timeoutId);

        return await this.handleResponse<T>(response);
      } catch (error) {
        lastError = error as Error;

        // Don't retry on client errors
        if (error instanceof AuthenticationError ||
            error instanceof ValidationError ||
            error instanceof NotFoundError ||
            error instanceof ConflictError) {
          throw error;
        }

        // Retry on server errors and rate limits
        if (attempt < this.maxRetries) {
          const delay = Math.pow(2, attempt) * 1000;
          if (this.debug) {
            console.log(`Retrying in ${delay}ms (attempt ${attempt + 1}/${this.maxRetries})`);
          }
          await this.sleep(delay);
        }
      }
    }

    throw lastError || new BuilderEngineError('Request failed after retries');
  }

  /**
   * Handle API response and throw appropriate errors.
   */
  private async handleResponse<T>(response: Response): Promise<T> {
    if (this.debug) {
      console.log(`Response status: ${response.status}`);
    }

    // Success responses
    if (response.status >= 200 && response.status < 300) {
      if (response.status === 204) {
        return {} as T;
      }

      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        return await response.json();
      }

      return { data: await response.text() } as T;
    }

    // Error responses
    let errorMessage: string;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorData.message || JSON.stringify(errorData);
    } catch {
      errorMessage = await response.text() || `HTTP ${response.status}`;
    }

    switch (response.status) {
      case 401:
        throw new AuthenticationError(errorMessage);
      case 403:
        throw new AuthenticationError(`Forbidden: ${errorMessage}`);
      case 404:
        throw new NotFoundError(errorMessage);
      case 409:
        throw new ConflictError(errorMessage);
      case 422:
        throw new ValidationError(errorMessage);
      case 429:
        const retryAfter = response.headers.get('Retry-After');
        throw new RateLimitError(errorMessage, retryAfter ? parseInt(retryAfter) : undefined);
      default:
        if (response.status >= 500) {
          throw new ServerError(errorMessage);
        }
        throw new BuilderEngineError(`HTTP ${response.status}: ${errorMessage}`);
    }
  }

  /**
   * Get a streaming client for real-time events.
   *
   * @returns StreamingClient for WebSocket connections
   *
   * @example
   * ```typescript
   * const streaming = client.streaming();
   *
   * streaming.on('transcription.final', (event) => {
   *   console.log('User said:', event.data.text);
   * });
   *
   * await streaming.connect({ callId: 'call_abc123' });
   * ```
   */
  streaming(): StreamingClient {
    const wsUrl = this.baseUrl.replace('https://', 'wss://').replace('http://', 'ws://');
    return new StreamingClient({
      apiKey: this.apiKey,
      baseUrl: wsUrl,
    });
  }

  /**
   * Sleep helper for retries.
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

export default BuilderEngine;
