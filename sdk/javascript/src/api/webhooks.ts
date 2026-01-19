/**
 * Webhooks API for Builder Engine SDK
 */

import type { HttpClient } from '../utils/http';
import type {
  Webhook,
  CreateWebhookRequest,
  WebhookDelivery,
  WebhookPayload,
  WebhookEvent,
  WebhookStatus,
  RequestOptions,
  PaginatedResponse,
} from '../types';
import { createHmac, timingSafeEqual } from 'crypto';

export class WebhooksAPI {
  constructor(private readonly http: HttpClient) {}

  /**
   * Create a new webhook
   */
  async create(request: CreateWebhookRequest, options?: RequestOptions): Promise<Webhook> {
    const response = await this.http.post<Webhook>(
      '/v1/webhooks',
      {
        url: request.url,
        events: request.events,
        description: request.description,
        headers: request.headers,
        retry_count: request.retryCount,
        timeout_seconds: request.timeoutSeconds,
      },
      options
    );
    return this.transformWebhook(response);
  }

  /**
   * Get a webhook by ID
   */
  async get(webhookId: string, options?: RequestOptions): Promise<Webhook> {
    const response = await this.http.get<Webhook>(
      `/v1/webhooks/${webhookId}`,
      undefined,
      options
    );
    return this.transformWebhook(response);
  }

  /**
   * List all webhooks
   */
  async list(
    params?: {
      limit?: number;
      offset?: number;
    },
    options?: RequestOptions
  ): Promise<PaginatedResponse<Webhook>> {
    const response = await this.http.get<{
      webhooks: Webhook[];
      total: number;
      limit: number;
      offset: number;
    }>('/v1/webhooks', params, options);

    return {
      items: response.webhooks.map((w) => this.transformWebhook(w)),
      total: response.total,
      limit: response.limit,
      offset: response.offset,
      hasMore: response.offset + response.webhooks.length < response.total,
    };
  }

  /**
   * Update a webhook
   */
  async update(
    webhookId: string,
    request: Partial<CreateWebhookRequest> & { status?: WebhookStatus },
    options?: RequestOptions
  ): Promise<Webhook> {
    const response = await this.http.patch<Webhook>(
      `/v1/webhooks/${webhookId}`,
      {
        url: request.url,
        events: request.events,
        description: request.description,
        headers: request.headers,
        retry_count: request.retryCount,
        timeout_seconds: request.timeoutSeconds,
        status: request.status,
      },
      options
    );
    return this.transformWebhook(response);
  }

  /**
   * Delete a webhook
   */
  async delete(webhookId: string, options?: RequestOptions): Promise<void> {
    await this.http.delete(`/v1/webhooks/${webhookId}`, options);
  }

  /**
   * Enable a webhook
   */
  async enable(webhookId: string, options?: RequestOptions): Promise<Webhook> {
    return this.update(webhookId, { status: 'active' }, options);
  }

  /**
   * Disable a webhook
   */
  async disable(webhookId: string, options?: RequestOptions): Promise<Webhook> {
    return this.update(webhookId, { status: 'disabled' }, options);
  }

  /**
   * Rotate the webhook secret
   */
  async rotateSecret(webhookId: string, options?: RequestOptions): Promise<Webhook> {
    const response = await this.http.post<Webhook>(
      `/v1/webhooks/${webhookId}/rotate-secret`,
      undefined,
      options
    );
    return this.transformWebhook(response);
  }

  /**
   * Send a test event to a webhook
   */
  async test(
    webhookId: string,
    eventType?: WebhookEvent,
    options?: RequestOptions
  ): Promise<{
    success: boolean;
    statusCode?: number;
    responseBody?: string;
    durationMs: number;
    errorMessage?: string;
  }> {
    const response = await this.http.post<{
      success: boolean;
      status_code?: number;
      response_body?: string;
      duration_ms: number;
      error_message?: string;
    }>(
      `/v1/webhooks/${webhookId}/test`,
      eventType ? { event_type: eventType } : undefined,
      options
    );
    return {
      success: response.success,
      statusCode: response.status_code,
      responseBody: response.response_body,
      durationMs: response.duration_ms,
      errorMessage: response.error_message,
    };
  }

  /**
   * Get delivery history for a webhook
   */
  async getDeliveries(
    webhookId: string,
    params?: {
      limit?: number;
      offset?: number;
      success?: boolean;
    },
    options?: RequestOptions
  ): Promise<PaginatedResponse<WebhookDelivery>> {
    const response = await this.http.get<{
      deliveries: WebhookDelivery[];
      total: number;
      limit: number;
      offset: number;
    }>(
      `/v1/webhooks/${webhookId}/deliveries`,
      {
        limit: params?.limit,
        offset: params?.offset,
        success: params?.success?.toString(),
      },
      options
    );

    return {
      items: response.deliveries.map((d) => this.transformDelivery(d)),
      total: response.total,
      limit: response.limit,
      offset: response.offset,
      hasMore: response.offset + response.deliveries.length < response.total,
    };
  }

  /**
   * Retry a failed delivery
   */
  async retryDelivery(
    webhookId: string,
    deliveryId: string,
    options?: RequestOptions
  ): Promise<WebhookDelivery> {
    const response = await this.http.post<WebhookDelivery>(
      `/v1/webhooks/${webhookId}/deliveries/${deliveryId}/retry`,
      undefined,
      options
    );
    return this.transformDelivery(response);
  }

  // Convenience Methods

  /**
   * Subscribe to all call events
   */
  async subscribeToCallEvents(
    url: string,
    description?: string,
    options?: RequestOptions
  ): Promise<Webhook> {
    return this.create(
      {
        url,
        events: [
          'call.started',
          'call.ringing',
          'call.answered',
          'call.ended',
          'call.failed',
          'call.transferred',
        ],
        description: description || 'Call events webhook',
      },
      options
    );
  }

  /**
   * Subscribe to all conversation events
   */
  async subscribeToConversationEvents(
    url: string,
    description?: string,
    options?: RequestOptions
  ): Promise<Webhook> {
    return this.create(
      {
        url,
        events: [
          'conversation.started',
          'conversation.turn',
          'conversation.ended',
          'transcription.final',
        ],
        description: description || 'Conversation events webhook',
      },
      options
    );
  }

  /**
   * Subscribe to all events
   */
  async subscribeToAllEvents(
    url: string,
    description?: string,
    options?: RequestOptions
  ): Promise<Webhook> {
    return this.create(
      {
        url,
        events: [
          'call.started',
          'call.ringing',
          'call.answered',
          'call.ended',
          'call.failed',
          'call.transferred',
          'conversation.started',
          'conversation.turn',
          'conversation.ended',
          'transcription.partial',
          'transcription.final',
          'agent.created',
          'agent.updated',
          'agent.deleted',
          'function.called',
          'function.completed',
          'recording.ready',
          'call.analyzed',
          'sentiment.detected',
        ],
        description: description || 'All events webhook',
      },
      options
    );
  }

  // Private helpers

  private transformWebhook(data: Record<string, unknown>): Webhook {
    return {
      id: data.id as string,
      url: data.url as string,
      events: data.events as WebhookEvent[],
      status: data.status as WebhookStatus,
      secret: data.secret as string,
      description: data.description as string | undefined,
      headers: data.headers as Record<string, string> | undefined,
      retryCount: (data.retryCount || data.retry_count) as number,
      timeoutSeconds: (data.timeoutSeconds || data.timeout_seconds) as number,
      createdAt: (data.createdAt || data.created_at) as string,
      updatedAt: (data.updatedAt || data.updated_at) as string,
      lastTriggeredAt: (data.lastTriggeredAt || data.last_triggered_at) as string | undefined,
      failureCount: (data.failureCount || data.failure_count) as number,
    };
  }

  private transformDelivery(data: Record<string, unknown>): WebhookDelivery {
    return {
      id: data.id as string,
      webhookId: (data.webhookId || data.webhook_id) as string,
      eventType: (data.eventType || data.event_type) as WebhookEvent,
      payload: data.payload as Record<string, unknown>,
      statusCode: (data.statusCode || data.status_code) as number | undefined,
      responseBody: (data.responseBody || data.response_body) as string | undefined,
      durationMs: (data.durationMs || data.duration_ms) as number,
      success: data.success as boolean,
      attemptNumber: (data.attemptNumber || data.attempt_number) as number,
      createdAt: (data.createdAt || data.created_at) as string,
      errorMessage: (data.errorMessage || data.error_message) as string | undefined,
    };
  }
}

/**
 * Webhook signature verifier
 */
export class WebhookSignatureVerifier {
  constructor(private readonly secret: string) {}

  /**
   * Compute the expected signature for a payload
   */
  computeSignature(payload: string | Buffer, timestamp: string): string {
    const message = `${timestamp}.${payload.toString()}`;
    const signature = createHmac('sha256', this.secret).update(message).digest('hex');
    return `v1=${signature}`;
  }

  /**
   * Verify a webhook signature
   */
  verify(
    payload: string | Buffer,
    signature: string,
    timestamp: string,
    toleranceSeconds = 300
  ): boolean {
    // Check timestamp is recent
    const ts = parseInt(timestamp, 10);
    const now = Math.floor(Date.now() / 1000);
    if (Math.abs(now - ts) > toleranceSeconds) {
      return false;
    }

    // Compute expected signature
    const expected = this.computeSignature(payload, timestamp);

    // Compare signatures (timing-safe)
    try {
      return timingSafeEqual(Buffer.from(expected), Buffer.from(signature));
    } catch {
      return false;
    }
  }

  /**
   * Parse and verify webhook payload
   */
  parsePayload(
    body: string | Buffer,
    signature: string,
    timestamp: string
  ): WebhookPayload | null {
    if (!this.verify(body, signature, timestamp)) {
      return null;
    }

    const data = JSON.parse(body.toString());
    return {
      id: data.id,
      eventType: data.event_type,
      timestamp: data.timestamp,
      data: data.data || {},
      metadata: data.metadata,
    };
  }
}

/**
 * Webhook handler for processing events
 */
export class WebhookHandler {
  private handlers: Map<WebhookEvent, Array<(payload: WebhookPayload) => void | Promise<void>>> =
    new Map();
  private verifier: WebhookSignatureVerifier;

  constructor(secret: string) {
    this.verifier = new WebhookSignatureVerifier(secret);
  }

  /**
   * Register a handler for an event
   */
  on(
    event: WebhookEvent,
    handler: (payload: WebhookPayload) => void | Promise<void>
  ): this {
    const handlers = this.handlers.get(event) || [];
    handlers.push(handler);
    this.handlers.set(event, handlers);
    return this;
  }

  /**
   * Remove a handler for an event
   */
  off(
    event: WebhookEvent,
    handler: (payload: WebhookPayload) => void | Promise<void>
  ): this {
    const handlers = this.handlers.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
    return this;
  }

  /**
   * Handle an incoming webhook request
   */
  async handle(
    body: string | Buffer,
    signature: string,
    timestamp: string
  ): Promise<{ status: string; message: string }> {
    // Verify signature
    const payload = this.verifier.parsePayload(body, signature, timestamp);
    if (!payload) {
      return { status: 'error', message: 'Invalid signature' };
    }

    // Get handlers for this event
    const handlers = this.handlers.get(payload.eventType) || [];
    if (handlers.length === 0) {
      return { status: 'ok', message: 'No handlers registered' };
    }

    // Run all handlers
    const errors: string[] = [];
    for (const handler of handlers) {
      try {
        await handler(payload);
      } catch (err) {
        errors.push(err instanceof Error ? err.message : String(err));
      }
    }

    if (errors.length > 0) {
      return {
        status: 'partial',
        message: `Some handlers failed: ${errors.join(', ')}`,
      };
    }

    return { status: 'ok', message: 'Processed successfully' };
  }
}

/**
 * Builder for creating webhooks
 */
export class WebhookBuilder {
  private _url?: string;
  private _events: WebhookEvent[] = [];
  private _description?: string;
  private _headers: Record<string, string> = {};
  private _retryCount = 3;
  private _timeoutSeconds = 30;

  url(url: string): this {
    this._url = url;
    return this;
  }

  description(description: string): this {
    this._description = description;
    return this;
  }

  event(event: WebhookEvent): this {
    if (!this._events.includes(event)) {
      this._events.push(event);
    }
    return this;
  }

  events(events: WebhookEvent[]): this {
    events.forEach((e) => this.event(e));
    return this;
  }

  onCallEvents(): this {
    return this.events([
      'call.started',
      'call.ringing',
      'call.answered',
      'call.ended',
      'call.failed',
      'call.transferred',
    ]);
  }

  onConversationEvents(): this {
    return this.events([
      'conversation.started',
      'conversation.turn',
      'conversation.ended',
    ]);
  }

  onTranscriptionEvents(): this {
    return this.events(['transcription.partial', 'transcription.final']);
  }

  onAllEvents(): this {
    return this.events([
      'call.started',
      'call.ringing',
      'call.answered',
      'call.ended',
      'call.failed',
      'call.transferred',
      'conversation.started',
      'conversation.turn',
      'conversation.ended',
      'transcription.partial',
      'transcription.final',
      'agent.created',
      'agent.updated',
      'agent.deleted',
      'function.called',
      'function.completed',
      'recording.ready',
      'call.analyzed',
      'sentiment.detected',
    ]);
  }

  withHeader(key: string, value: string): this {
    this._headers[key] = value;
    return this;
  }

  withHeaders(headers: Record<string, string>): this {
    Object.assign(this._headers, headers);
    return this;
  }

  retryCount(count: number): this {
    this._retryCount = count;
    return this;
  }

  timeout(seconds: number): this {
    this._timeoutSeconds = seconds;
    return this;
  }

  build(): CreateWebhookRequest {
    if (!this._url) {
      throw new Error('URL is required');
    }
    if (this._events.length === 0) {
      throw new Error('At least one event is required');
    }

    return {
      url: this._url,
      events: this._events,
      description: this._description,
      headers: Object.keys(this._headers).length > 0 ? this._headers : undefined,
      retryCount: this._retryCount,
      timeoutSeconds: this._timeoutSeconds,
    };
  }

  async create(api: WebhooksAPI): Promise<Webhook> {
    return api.create(this.build());
  }
}
