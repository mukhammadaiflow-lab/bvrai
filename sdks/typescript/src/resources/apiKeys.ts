import type { BuilderEngine } from '../client';
import type { APIKey, PaginatedResponse, ListParams } from '../models';

export class APIKeysResource {
  constructor(private readonly client: BuilderEngine) {}

  async list(params: ListParams = {}): Promise<PaginatedResponse<APIKey>> {
    return this.client.request({ method: 'GET', path: '/api/v1/api-keys', params });
  }

  async create(data: { name: string; permissions?: string[]; rateLimit?: number; expiresAt?: string }): Promise<APIKey & { key: string }> {
    return this.client.request({ method: 'POST', path: '/api/v1/api-keys', body: data });
  }

  async delete(apiKeyId: string): Promise<void> {
    await this.client.request({ method: 'DELETE', path: `/api/v1/api-keys/${apiKeyId}` });
  }

  async regenerate(apiKeyId: string): Promise<APIKey & { key: string }> {
    return this.client.request({ method: 'POST', path: `/api/v1/api-keys/${apiKeyId}/regenerate` });
  }
}
