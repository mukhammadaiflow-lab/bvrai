import type { BuilderEngine } from '../client';
import type { Organization, User, PaginatedResponse, ListParams } from '../models';

export class OrganizationsResource {
  constructor(private readonly client: BuilderEngine) {}

  async getCurrent(): Promise<Organization> {
    return this.client.request({ method: 'GET', path: '/api/v1/organizations/current' });
  }

  async get(organizationId: string): Promise<Organization> {
    return this.client.request({ method: 'GET', path: `/api/v1/organizations/${organizationId}` });
  }

  async update(organizationId: string, data: Partial<Organization>): Promise<Organization> {
    return this.client.request({ method: 'PATCH', path: `/api/v1/organizations/${organizationId}`, body: data });
  }

  async listMembers(organizationId: string, params: ListParams = {}): Promise<PaginatedResponse<User>> {
    return this.client.request({ method: 'GET', path: `/api/v1/organizations/${organizationId}/members`, params });
  }

  async inviteMember(organizationId: string, email: string, role: string = 'member'): Promise<Record<string, any>> {
    return this.client.request({ method: 'POST', path: `/api/v1/organizations/${organizationId}/invite`, body: { email, role } });
  }
}
