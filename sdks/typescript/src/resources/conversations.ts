import type { BuilderEngine } from '../client';
import type { Conversation, Message, PaginatedResponse, ListParams } from '../models';

export class ConversationsResource {
  constructor(private readonly client: BuilderEngine) {}

  async list(params: ListParams = {}): Promise<PaginatedResponse<Conversation>> {
    return this.client.request({ method: 'GET', path: '/api/v1/conversations', params });
  }

  async get(conversationId: string): Promise<Conversation> {
    return this.client.request({ method: 'GET', path: `/api/v1/conversations/${conversationId}` });
  }

  async getMessages(conversationId: string, params: ListParams = {}): Promise<PaginatedResponse<Message>> {
    return this.client.request({ method: 'GET', path: `/api/v1/conversations/${conversationId}/messages`, params });
  }

  async getSummary(conversationId: string): Promise<Record<string, any>> {
    return this.client.request({ method: 'GET', path: `/api/v1/conversations/${conversationId}/summary` });
  }
}
