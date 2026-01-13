import type { BuilderEngine } from '../client';
import type { Webhook, WebhookEvent, CreateWebhookRequest, PaginatedResponse, ListParams } from '../models';

export class WebhooksResource {
  constructor(private readonly client: BuilderEngine) {}

  async list(params: ListParams = {}): Promise<PaginatedResponse<Webhook>> {
    return this.client.request({ method: 'GET', path: '/api/v1/webhooks', params });
  }

  async get(webhookId: string): Promise<Webhook> {
    return this.client.request({ method: 'GET', path: `/api/v1/webhooks/${webhookId}` });
  }

  async create(data: CreateWebhookRequest): Promise<Webhook> {
    return this.client.request({ method: 'POST', path: '/api/v1/webhooks', body: { url: data.url, events: data.events, enabled: data.enabled ?? true } });
  }

  async delete(webhookId: string): Promise<void> {
    await this.client.request({ method: 'DELETE', path: `/api/v1/webhooks/${webhookId}` });
  }

  async test(webhookId: string, event?: WebhookEvent): Promise<Record<string, any>> {
    return this.client.request({ method: 'POST', path: `/api/v1/webhooks/${webhookId}/test`, body: event ? { event } : undefined });
  }

  async rotateSecret(webhookId: string): Promise<Webhook> {
    return this.client.request({ method: 'POST', path: `/api/v1/webhooks/${webhookId}/rotate-secret` });
  }
}
