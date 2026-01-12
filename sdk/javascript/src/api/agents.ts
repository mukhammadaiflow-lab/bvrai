/**
 * Agents API for Builder Engine SDK
 */

import type { HttpClient } from '../utils/http';
import type {
  Agent,
  CreateAgentRequest,
  UpdateAgentRequest,
  Tool,
  ToolParameter,
  RequestOptions,
  PaginatedResponse,
} from '../types';

export class AgentsAPI {
  constructor(private readonly http: HttpClient) {}

  /**
   * Create a new agent
   */
  async create(request: CreateAgentRequest, options?: RequestOptions): Promise<Agent> {
    const response = await this.http.post<Agent>('/v1/agents', request, options);
    return this.transformAgent(response);
  }

  /**
   * Get an agent by ID
   */
  async get(agentId: string, options?: RequestOptions): Promise<Agent> {
    const response = await this.http.get<Agent>(`/v1/agents/${agentId}`, undefined, options);
    return this.transformAgent(response);
  }

  /**
   * List all agents
   */
  async list(
    params?: {
      limit?: number;
      offset?: number;
      status?: 'active' | 'inactive' | 'draft';
    },
    options?: RequestOptions
  ): Promise<PaginatedResponse<Agent>> {
    const response = await this.http.get<{
      agents: Agent[];
      total: number;
      limit: number;
      offset: number;
    }>('/v1/agents', params, options);

    return {
      items: response.agents.map((a) => this.transformAgent(a)),
      total: response.total,
      limit: response.limit,
      offset: response.offset,
      hasMore: response.offset + response.agents.length < response.total,
    };
  }

  /**
   * Update an agent
   */
  async update(
    agentId: string,
    request: UpdateAgentRequest,
    options?: RequestOptions
  ): Promise<Agent> {
    const response = await this.http.patch<Agent>(`/v1/agents/${agentId}`, request, options);
    return this.transformAgent(response);
  }

  /**
   * Delete an agent
   */
  async delete(agentId: string, options?: RequestOptions): Promise<void> {
    await this.http.delete(`/v1/agents/${agentId}`, options);
  }

  /**
   * Duplicate an agent
   */
  async duplicate(
    agentId: string,
    name?: string,
    options?: RequestOptions
  ): Promise<Agent> {
    const response = await this.http.post<Agent>(
      `/v1/agents/${agentId}/duplicate`,
      { name },
      options
    );
    return this.transformAgent(response);
  }

  /**
   * Activate an agent
   */
  async activate(agentId: string, options?: RequestOptions): Promise<Agent> {
    return this.update(agentId, { status: 'active' }, options);
  }

  /**
   * Deactivate an agent
   */
  async deactivate(agentId: string, options?: RequestOptions): Promise<Agent> {
    return this.update(agentId, { status: 'inactive' }, options);
  }

  // Tool Management

  /**
   * Add a tool to an agent
   */
  async addTool(
    agentId: string,
    tool: Tool,
    options?: RequestOptions
  ): Promise<Agent> {
    const response = await this.http.post<Agent>(
      `/v1/agents/${agentId}/tools`,
      tool,
      options
    );
    return this.transformAgent(response);
  }

  /**
   * Update a tool on an agent
   */
  async updateTool(
    agentId: string,
    toolName: string,
    tool: Partial<Tool>,
    options?: RequestOptions
  ): Promise<Agent> {
    const response = await this.http.patch<Agent>(
      `/v1/agents/${agentId}/tools/${toolName}`,
      tool,
      options
    );
    return this.transformAgent(response);
  }

  /**
   * Remove a tool from an agent
   */
  async removeTool(
    agentId: string,
    toolName: string,
    options?: RequestOptions
  ): Promise<Agent> {
    const response = await this.http.delete<Agent>(
      `/v1/agents/${agentId}/tools/${toolName}`,
      options
    );
    return this.transformAgent(response);
  }

  /**
   * List tools on an agent
   */
  async listTools(agentId: string, options?: RequestOptions): Promise<Tool[]> {
    const response = await this.http.get<{ tools: Tool[] }>(
      `/v1/agents/${agentId}/tools`,
      undefined,
      options
    );
    return response.tools;
  }

  // Versions

  /**
   * List agent versions
   */
  async listVersions(
    agentId: string,
    options?: RequestOptions
  ): Promise<Array<{ version: string; createdAt: string; description?: string }>> {
    const response = await this.http.get<{
      versions: Array<{ version: string; createdAt: string; description?: string }>;
    }>(`/v1/agents/${agentId}/versions`, undefined, options);
    return response.versions;
  }

  /**
   * Create a new version of an agent
   */
  async createVersion(
    agentId: string,
    description?: string,
    options?: RequestOptions
  ): Promise<{ version: string; createdAt: string }> {
    return this.http.post<{ version: string; createdAt: string }>(
      `/v1/agents/${agentId}/versions`,
      { description },
      options
    );
  }

  /**
   * Rollback to a previous version
   */
  async rollbackToVersion(
    agentId: string,
    version: string,
    options?: RequestOptions
  ): Promise<Agent> {
    const response = await this.http.post<Agent>(
      `/v1/agents/${agentId}/versions/${version}/rollback`,
      undefined,
      options
    );
    return this.transformAgent(response);
  }

  // Testing

  /**
   * Test an agent with a message
   */
  async test(
    agentId: string,
    message: string,
    options?: RequestOptions
  ): Promise<{ response: string; latencyMs: number }> {
    return this.http.post<{ response: string; latencyMs: number }>(
      `/v1/agents/${agentId}/test`,
      { message },
      options
    );
  }

  // Private helpers

  private transformAgent(data: Record<string, unknown>): Agent {
    return {
      id: data.id as string,
      name: data.name as string,
      systemPrompt: (data.systemPrompt || data.system_prompt) as string,
      config: this.transformConfig(data.config as Record<string, unknown>),
      status: data.status as 'active' | 'inactive' | 'draft',
      createdAt: (data.createdAt || data.created_at) as string,
      updatedAt: (data.updatedAt || data.updated_at) as string,
      metadata: data.metadata as Record<string, string | number | boolean | null>,
    };
  }

  private transformConfig(config: Record<string, unknown> | undefined): Agent['config'] {
    if (!config) return {};

    return {
      voice: config.voice as Agent['config']['voice'],
      llm: config.llm as Agent['config']['llm'],
      asr: config.asr as Agent['config']['asr'],
      interruptionThreshold: (config.interruptionThreshold || config.interruption_threshold) as number,
      silenceTimeout: (config.silenceTimeout || config.silence_timeout) as number,
      maxDuration: (config.maxDuration || config.max_duration) as number,
      recordCalls: (config.recordCalls || config.record_calls) as boolean,
      transcribeCalls: (config.transcribeCalls || config.transcribe_calls) as boolean,
      tools: config.tools as Tool[],
      firstMessage: (config.firstMessage || config.first_message) as string,
      endCallMessage: (config.endCallMessage || config.end_call_message) as string,
      transferNumber: (config.transferNumber || config.transfer_number) as string,
    };
  }
}

/**
 * Builder for creating tools
 */
export class ToolBuilder {
  private name = '';
  private description = '';
  private parameters: ToolParameter[] = [];
  private webhookUrl?: string;

  setName(name: string): this {
    this.name = name;
    return this;
  }

  setDescription(description: string): this {
    this.description = description;
    return this;
  }

  addParameter(param: ToolParameter): this {
    this.parameters.push(param);
    return this;
  }

  addStringParam(
    name: string,
    options?: { description?: string; required?: boolean; enum?: string[] }
  ): this {
    return this.addParameter({
      name,
      type: 'string',
      description: options?.description,
      required: options?.required ?? false,
      enum: options?.enum,
    });
  }

  addNumberParam(
    name: string,
    options?: { description?: string; required?: boolean }
  ): this {
    return this.addParameter({
      name,
      type: 'number',
      description: options?.description,
      required: options?.required ?? false,
    });
  }

  addBooleanParam(
    name: string,
    options?: { description?: string; required?: boolean; default?: boolean }
  ): this {
    return this.addParameter({
      name,
      type: 'boolean',
      description: options?.description,
      required: options?.required ?? false,
      default: options?.default,
    });
  }

  setWebhookUrl(url: string): this {
    this.webhookUrl = url;
    return this;
  }

  build(): Tool {
    if (!this.name) throw new Error('Tool name is required');
    if (!this.description) throw new Error('Tool description is required');

    return {
      name: this.name,
      description: this.description,
      parameters: this.parameters,
      webhookUrl: this.webhookUrl,
    };
  }
}
