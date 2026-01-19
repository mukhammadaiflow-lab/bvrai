import type { BuilderEngine } from '../client';
import type { PhoneNumber, PhoneNumberType, PaginatedResponse, ListParams } from '../models';

export class PhoneNumbersResource {
  constructor(private readonly client: BuilderEngine) {}

  async list(params: ListParams = {}): Promise<PaginatedResponse<PhoneNumber>> {
    return this.client.request({ method: 'GET', path: '/api/v1/phone-numbers', params });
  }

  async get(phoneNumberId: string): Promise<PhoneNumber> {
    return this.client.request({ method: 'GET', path: `/api/v1/phone-numbers/${phoneNumberId}` });
  }

  async searchAvailable(params: { country?: string; areaCode?: string; type?: PhoneNumberType } = {}): Promise<any[]> {
    return this.client.request({ method: 'GET', path: '/api/v1/phone-numbers/available', params });
  }

  async purchase(number: string, options: { friendlyName?: string; agentId?: string } = {}): Promise<PhoneNumber> {
    return this.client.request({ method: 'POST', path: '/api/v1/phone-numbers/purchase', body: { number, ...options } });
  }

  async release(phoneNumberId: string): Promise<void> {
    await this.client.request({ method: 'POST', path: `/api/v1/phone-numbers/${phoneNumberId}/release` });
  }
}
