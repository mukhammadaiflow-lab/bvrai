/**
 * Knowledge API for Builder Engine SDK
 */

import type { HttpClient } from '../utils/http';
import type {
  KnowledgeBase,
  CreateKnowledgeBaseRequest,
  KnowledgeSearchResult,
  RequestOptions,
  PaginatedResponse,
  Metadata,
} from '../types';

export class KnowledgeAPI {
  constructor(private readonly http: HttpClient) {}

  /**
   * Create a new knowledge base
   */
  async create(
    request: CreateKnowledgeBaseRequest,
    options?: RequestOptions
  ): Promise<KnowledgeBase> {
    const response = await this.http.post<KnowledgeBase>(
      '/v1/knowledge',
      {
        agent_id: request.agentId,
        name: request.name,
        description: request.description,
      },
      options
    );
    return this.transformKnowledgeBase(response);
  }

  /**
   * Get a knowledge base by ID
   */
  async get(kbId: string, options?: RequestOptions): Promise<KnowledgeBase> {
    const response = await this.http.get<KnowledgeBase>(
      `/v1/knowledge/${kbId}`,
      undefined,
      options
    );
    return this.transformKnowledgeBase(response);
  }

  /**
   * List knowledge bases
   */
  async list(
    params?: {
      agentId?: string;
      limit?: number;
      offset?: number;
    },
    options?: RequestOptions
  ): Promise<PaginatedResponse<KnowledgeBase>> {
    const response = await this.http.get<{
      knowledge_bases: KnowledgeBase[];
      total: number;
      limit: number;
      offset: number;
    }>(
      '/v1/knowledge',
      {
        agent_id: params?.agentId,
        limit: params?.limit,
        offset: params?.offset,
      },
      options
    );

    return {
      items: response.knowledge_bases.map((kb) => this.transformKnowledgeBase(kb)),
      total: response.total,
      limit: response.limit,
      offset: response.offset,
      hasMore: response.offset + response.knowledge_bases.length < response.total,
    };
  }

  /**
   * Delete a knowledge base
   */
  async delete(kbId: string, options?: RequestOptions): Promise<void> {
    await this.http.delete(`/v1/knowledge/${kbId}`, options);
  }

  /**
   * Add text to knowledge base
   */
  async addText(
    kbId: string,
    text: string,
    metadata?: Metadata,
    options?: RequestOptions
  ): Promise<{ documentId: string; chunks: number }> {
    const response = await this.http.post<{ document_id: string; chunks: number }>(
      `/v1/knowledge/${kbId}/documents`,
      { text, metadata: metadata || {} },
      options
    );
    return {
      documentId: response.document_id,
      chunks: response.chunks,
    };
  }

  /**
   * Add URL content to knowledge base
   */
  async addUrl(
    kbId: string,
    url: string,
    options?: {
      crawlLinks?: boolean;
      maxDepth?: number;
      metadata?: Metadata;
    } & RequestOptions
  ): Promise<{ documentId: string; status: string }> {
    const response = await this.http.post<{ document_id: string; status: string }>(
      `/v1/knowledge/${kbId}/urls`,
      {
        url,
        crawl_links: options?.crawlLinks ?? false,
        max_depth: options?.maxDepth ?? 1,
        metadata: options?.metadata || {},
      },
      options
    );
    return {
      documentId: response.document_id,
      status: response.status,
    };
  }

  /**
   * Add file to knowledge base (not implemented - requires multipart upload)
   */
  async addFile(
    _kbId: string,
    _file: File | Blob,
    _options?: RequestOptions
  ): Promise<never> {
    throw new Error('File upload not implemented. Use addText or addUrl instead.');
  }

  /**
   * Search knowledge base
   */
  async search(
    kbId: string,
    query: string,
    options?: {
      limit?: number;
      threshold?: number;
      filter?: Metadata;
    } & RequestOptions
  ): Promise<KnowledgeSearchResult[]> {
    const response = await this.http.get<{ results: KnowledgeSearchResult[] }>(
      `/v1/knowledge/${kbId}/search`,
      {
        query,
        limit: options?.limit ?? 5,
        threshold: options?.threshold,
        filter: options?.filter ? JSON.stringify(options.filter) : undefined,
      },
      options
    );
    return response.results;
  }

  /**
   * Trigger sync for knowledge base
   */
  async sync(kbId: string, options?: RequestOptions): Promise<{ status: string }> {
    return this.http.post<{ status: string }>(
      `/v1/knowledge/${kbId}/sync`,
      undefined,
      options
    );
  }

  /**
   * Get documents in a knowledge base
   */
  async listDocuments(
    kbId: string,
    params?: {
      limit?: number;
      offset?: number;
    },
    options?: RequestOptions
  ): Promise<
    PaginatedResponse<{
      id: string;
      sourceType: string;
      sourceUrl?: string;
      chunkCount: number;
      createdAt: string;
      metadata: Metadata;
    }>
  > {
    const response = await this.http.get<{
      documents: Array<{
        id: string;
        source_type: string;
        source_url?: string;
        chunk_count: number;
        created_at: string;
        metadata: Metadata;
      }>;
      total: number;
      limit: number;
      offset: number;
    }>(`/v1/knowledge/${kbId}/documents`, params, options);

    return {
      items: response.documents.map((doc) => ({
        id: doc.id,
        sourceType: doc.source_type,
        sourceUrl: doc.source_url,
        chunkCount: doc.chunk_count,
        createdAt: doc.created_at,
        metadata: doc.metadata,
      })),
      total: response.total,
      limit: response.limit,
      offset: response.offset,
      hasMore: response.offset + response.documents.length < response.total,
    };
  }

  /**
   * Delete a document from knowledge base
   */
  async deleteDocument(
    kbId: string,
    documentId: string,
    options?: RequestOptions
  ): Promise<void> {
    await this.http.delete(`/v1/knowledge/${kbId}/documents/${documentId}`, options);
  }

  /**
   * Get knowledge base statistics
   */
  async getStats(
    kbId: string,
    options?: RequestOptions
  ): Promise<{
    totalDocuments: number;
    totalChunks: number;
    totalTokens: number;
    lastUpdated: string;
  }> {
    const response = await this.http.get<{
      total_documents: number;
      total_chunks: number;
      total_tokens: number;
      last_updated: string;
    }>(`/v1/knowledge/${kbId}/stats`, undefined, options);

    return {
      totalDocuments: response.total_documents,
      totalChunks: response.total_chunks,
      totalTokens: response.total_tokens,
      lastUpdated: response.last_updated,
    };
  }

  // Private helpers

  private transformKnowledgeBase(data: Record<string, unknown>): KnowledgeBase {
    return {
      id: data.id as string,
      agentId: (data.agentId || data.agent_id) as string,
      name: data.name as string,
      description: data.description as string | undefined,
      sourceType: (data.sourceType || data.source_type) as KnowledgeBase['sourceType'],
      chunkCount: (data.chunkCount || data.chunk_count) as number,
      status: data.status as KnowledgeBase['status'],
      createdAt: (data.createdAt || data.created_at) as string,
      updatedAt: (data.updatedAt || data.updated_at) as string,
    };
  }
}
