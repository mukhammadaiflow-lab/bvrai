import type { BuilderEngine } from '../client';
import type { Workflow, WorkflowTrigger, PaginatedResponse, ListParams } from '../models';

export class WorkflowsResource {
  constructor(private readonly client: BuilderEngine) {}

  async list(params: ListParams = {}): Promise<PaginatedResponse<Workflow>> {
    return this.client.request({ method: 'GET', path: '/api/v1/workflows', params });
  }

  async get(workflowId: string): Promise<Workflow> {
    return this.client.request({ method: 'GET', path: `/api/v1/workflows/${workflowId}` });
  }

  async create(data: { name: string; trigger: WorkflowTrigger; actions: any[]; enabled?: boolean }): Promise<Workflow> {
    return this.client.request({ method: 'POST', path: '/api/v1/workflows', body: { ...data, enabled: data.enabled ?? true } });
  }

  async delete(workflowId: string): Promise<void> {
    await this.client.request({ method: 'DELETE', path: `/api/v1/workflows/${workflowId}` });
  }

  async enable(workflowId: string): Promise<Workflow> {
    return this.client.request({ method: 'POST', path: `/api/v1/workflows/${workflowId}/enable` });
  }

  async disable(workflowId: string): Promise<Workflow> {
    return this.client.request({ method: 'POST', path: `/api/v1/workflows/${workflowId}/disable` });
  }

  async execute(workflowId: string, context?: Record<string, any>): Promise<Record<string, any>> {
    return this.client.request({ method: 'POST', path: `/api/v1/workflows/${workflowId}/execute`, body: context ? { context } : undefined });
  }
}
