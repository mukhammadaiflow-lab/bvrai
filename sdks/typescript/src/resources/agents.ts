/**
 * Builder Engine TypeScript SDK - Agents Resource
 *
 * Methods for managing AI voice agents.
 */

import type { BuilderEngine } from '../client';
import type {
  Agent,
  AgentStatus,
  CreateAgentRequest,
  UpdateAgentRequest,
  Call,
  Analytics,
  PaginatedResponse,
  ListParams,
} from '../models';

export interface ListAgentsParams extends ListParams {
  status?: AgentStatus;
  search?: string;
}

/**
 * Resource for managing AI voice agents.
 *
 * Agents are the core building blocks of Builder Engine. Each agent
 * represents a configured AI assistant that can handle voice calls.
 *
 * @example
 * ```typescript
 * const client = new BuilderEngine({ apiKey: '...' });
 *
 * // Create an agent
 * const agent = await client.agents.create({
 *   name: 'Sales Agent',
 *   voiceId: 'voice_abc123',
 *   llmConfig: { model: 'gpt-4-turbo', temperature: 0.7 }
 * });
 *
 * // List all agents
 * const agents = await client.agents.list();
 * ```
 */
export class AgentsResource {
  constructor(private readonly client: BuilderEngine) {}

  /**
   * List all agents.
   *
   * @param params - Pagination and filter parameters
   * @returns Paginated list of agents
   */
  async list(params: ListAgentsParams = {}): Promise<PaginatedResponse<Agent>> {
    return this.client.request({
      method: 'GET',
      path: '/api/v1/agents',
      params: {
        page: params.page || 1,
        page_size: params.pageSize || 20,
        status: params.status,
        search: params.search,
        sort_by: params.sortBy,
        sort_order: params.sortOrder,
      },
    });
  }

  /**
   * Get an agent by ID.
   *
   * @param agentId - The agent's unique identifier
   * @returns Agent object
   */
  async get(agentId: string): Promise<Agent> {
    return this.client.request({
      method: 'GET',
      path: `/api/v1/agents/${agentId}`,
    });
  }

  /**
   * Create a new agent.
   *
   * @param data - Agent creation data
   * @returns Created agent
   *
   * @example
   * ```typescript
   * const agent = await client.agents.create({
   *   name: 'Customer Support Agent',
   *   systemPrompt: 'You are a helpful customer support agent...',
   *   voiceId: 'voice_abc123',
   *   firstMessage: 'Hello! How can I help you today?',
   *   llmConfig: {
   *     model: 'gpt-4-turbo',
   *     temperature: 0.7,
   *     maxTokens: 500
   *   }
   * });
   * ```
   */
  async create(data: CreateAgentRequest): Promise<Agent> {
    const body: Record<string, any> = { name: data.name };

    if (data.description) body.description = data.description;
    if (data.voiceId) body.voice_id = data.voiceId;
    if (data.phoneNumberId) body.phone_number_id = data.phoneNumberId;
    if (data.knowledgeBaseIds) body.knowledge_base_ids = data.knowledgeBaseIds;
    if (data.workflowIds) body.workflow_ids = data.workflowIds;
    if (data.metadata) body.metadata = data.metadata;

    // Build config
    const config: Record<string, any> = {};

    if (data.config) Object.assign(config, data.config);

    if (data.voiceConfig) config.voice = data.voiceConfig;
    if (data.sttConfig) config.stt = data.sttConfig;
    if (data.llmConfig) config.llm = data.llmConfig;
    if (data.firstMessage) config.first_message = data.firstMessage;
    if (data.systemPrompt) {
      config.llm = config.llm || {};
      config.llm.system_prompt = data.systemPrompt;
    }
    if (data.functions) config.functions = data.functions;

    if (Object.keys(config).length > 0) {
      body.config = config;
    }

    return this.client.request({
      method: 'POST',
      path: '/api/v1/agents',
      body,
    });
  }

  /**
   * Update an existing agent.
   *
   * @param agentId - The agent's unique identifier
   * @param data - Update data
   * @returns Updated agent
   */
  async update(agentId: string, data: UpdateAgentRequest): Promise<Agent> {
    const body: Record<string, any> = {};

    if (data.name !== undefined) body.name = data.name;
    if (data.description !== undefined) body.description = data.description;
    if (data.status !== undefined) body.status = data.status;
    if (data.voiceId !== undefined) body.voice_id = data.voiceId;
    if (data.phoneNumberId !== undefined) body.phone_number_id = data.phoneNumberId;
    if (data.knowledgeBaseIds !== undefined) body.knowledge_base_ids = data.knowledgeBaseIds;
    if (data.workflowIds !== undefined) body.workflow_ids = data.workflowIds;
    if (data.config !== undefined) body.config = data.config;
    if (data.metadata !== undefined) body.metadata = data.metadata;

    return this.client.request({
      method: 'PATCH',
      path: `/api/v1/agents/${agentId}`,
      body,
    });
  }

  /**
   * Delete an agent.
   *
   * @param agentId - The agent's unique identifier
   */
  async delete(agentId: string): Promise<void> {
    await this.client.request({
      method: 'DELETE',
      path: `/api/v1/agents/${agentId}`,
    });
  }

  /**
   * Duplicate an existing agent.
   *
   * @param agentId - The agent's unique identifier
   * @param name - Name for the duplicated agent
   * @returns New agent
   */
  async duplicate(agentId: string, name?: string): Promise<Agent> {
    return this.client.request({
      method: 'POST',
      path: `/api/v1/agents/${agentId}/duplicate`,
      body: name ? { name } : undefined,
    });
  }

  /**
   * Get calls for an agent.
   *
   * @param agentId - The agent's unique identifier
   * @param params - Pagination parameters
   * @returns Paginated list of calls
   */
  async getCalls(
    agentId: string,
    params: ListParams & { status?: string; startDate?: string; endDate?: string } = {}
  ): Promise<PaginatedResponse<Call>> {
    return this.client.request({
      method: 'GET',
      path: `/api/v1/agents/${agentId}/calls`,
      params: {
        page: params.page || 1,
        page_size: params.pageSize || 20,
        status: params.status,
        start_date: params.startDate,
        end_date: params.endDate,
      },
    });
  }

  /**
   * Get analytics for an agent.
   *
   * @param agentId - The agent's unique identifier
   * @param params - Analytics parameters
   * @returns Analytics data
   */
  async getAnalytics(
    agentId: string,
    params: { period?: string; startDate?: string; endDate?: string } = {}
  ): Promise<Analytics> {
    return this.client.request({
      method: 'GET',
      path: `/api/v1/agents/${agentId}/analytics`,
      params: {
        period: params.period || 'week',
        start_date: params.startDate,
        end_date: params.endDate,
      },
    });
  }

  /**
   * Test an agent with a simulated call.
   *
   * @param agentId - The agent's unique identifier
   * @param params - Test parameters
   * @returns Test results
   */
  async test(
    agentId: string,
    params: { phoneNumber?: string; simulateUserMessages?: string[] } = {}
  ): Promise<Record<string, any>> {
    return this.client.request({
      method: 'POST',
      path: `/api/v1/agents/${agentId}/test`,
      body: {
        phone_number: params.phoneNumber,
        simulate_user_messages: params.simulateUserMessages,
      },
    });
  }

  /**
   * Archive an agent.
   *
   * @param agentId - The agent's unique identifier
   * @returns Updated agent
   */
  async archive(agentId: string): Promise<Agent> {
    return this.update(agentId, { status: 'archived' as AgentStatus });
  }

  /**
   * Activate an agent.
   *
   * @param agentId - The agent's unique identifier
   * @returns Updated agent
   */
  async activate(agentId: string): Promise<Agent> {
    return this.update(agentId, { status: 'active' as AgentStatus });
  }

  /**
   * Deactivate an agent.
   *
   * @param agentId - The agent's unique identifier
   * @returns Updated agent
   */
  async deactivate(agentId: string): Promise<Agent> {
    return this.update(agentId, { status: 'inactive' as AgentStatus });
  }
}
