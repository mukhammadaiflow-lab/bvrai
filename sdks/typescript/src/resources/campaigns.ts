import type { BuilderEngine } from '../client';
import type { Campaign, CampaignContact, CreateCampaignRequest, PaginatedResponse, ListParams } from '../models';

export class CampaignsResource {
  constructor(private readonly client: BuilderEngine) {}

  async list(params: ListParams = {}): Promise<PaginatedResponse<Campaign>> {
    return this.client.request({ method: 'GET', path: '/api/v1/campaigns', params });
  }

  async get(campaignId: string): Promise<Campaign> {
    return this.client.request({ method: 'GET', path: `/api/v1/campaigns/${campaignId}` });
  }

  async create(data: CreateCampaignRequest): Promise<Campaign> {
    return this.client.request({ method: 'POST', path: '/api/v1/campaigns', body: { name: data.name, agent_id: data.agentId, contacts: data.contacts } });
  }

  async start(campaignId: string): Promise<Campaign> {
    return this.client.request({ method: 'POST', path: `/api/v1/campaigns/${campaignId}/start` });
  }

  async pause(campaignId: string): Promise<Campaign> {
    return this.client.request({ method: 'POST', path: `/api/v1/campaigns/${campaignId}/pause` });
  }

  async cancel(campaignId: string): Promise<Campaign> {
    return this.client.request({ method: 'POST', path: `/api/v1/campaigns/${campaignId}/cancel` });
  }

  async addContacts(campaignId: string, contacts: any[]): Promise<CampaignContact[]> {
    const response = await this.client.request<{ contacts: CampaignContact[] }>({ method: 'POST', path: `/api/v1/campaigns/${campaignId}/contacts`, body: { contacts } });
    return response.contacts;
  }
}
