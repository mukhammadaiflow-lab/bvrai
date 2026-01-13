import type { BuilderEngine } from '../client';
import type { KnowledgeBase, Document, PaginatedResponse, ListParams } from '../models';

export class KnowledgeBaseResource {
  constructor(private readonly client: BuilderEngine) {}

  async list(params: ListParams = {}): Promise<PaginatedResponse<KnowledgeBase>> {
    return this.client.request({ method: 'GET', path: '/api/v1/knowledge-bases', params });
  }

  async get(knowledgeBaseId: string): Promise<KnowledgeBase> {
    return this.client.request({ method: 'GET', path: `/api/v1/knowledge-bases/${knowledgeBaseId}` });
  }

  async create(data: { name: string; description?: string; chunkSize?: number }): Promise<KnowledgeBase> {
    return this.client.request({ method: 'POST', path: '/api/v1/knowledge-bases', body: data });
  }

  async delete(knowledgeBaseId: string): Promise<void> {
    await this.client.request({ method: 'DELETE', path: `/api/v1/knowledge-bases/${knowledgeBaseId}` });
  }

  async addDocument(knowledgeBaseId: string, data: { name: string; content?: string; fileUrl?: string }): Promise<Document> {
    return this.client.request({ method: 'POST', path: `/api/v1/knowledge-bases/${knowledgeBaseId}/documents`, body: data });
  }

  async query(knowledgeBaseId: string, query: string, topK: number = 5): Promise<any[]> {
    const response = await this.client.request<{ results: any[] }>({ method: 'POST', path: `/api/v1/knowledge-bases/${knowledgeBaseId}/query`, body: { query, top_k: topK } });
    return response.results;
  }
}
