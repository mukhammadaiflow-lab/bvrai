import type { BuilderEngine } from '../client';
import type { Voice, VoiceProvider, VoiceConfig, PaginatedResponse, ListParams } from '../models';

export class VoicesResource {
  constructor(private readonly client: BuilderEngine) {}

  async list(params: ListParams = {}): Promise<PaginatedResponse<Voice>> {
    return this.client.request({ method: 'GET', path: '/api/v1/voices', params });
  }

  async get(voiceId: string): Promise<Voice> {
    return this.client.request({ method: 'GET', path: `/api/v1/voices/${voiceId}` });
  }

  async listLibrary(params: { provider?: VoiceProvider; language?: string } = {}): Promise<Voice[]> {
    const response = await this.client.request<{ voices: Voice[] }>({ method: 'GET', path: '/api/v1/voices/library', params });
    return response.voices;
  }

  async preview(voiceId: string, text: string): Promise<{ audioUrl: string }> {
    return this.client.request({ method: 'POST', path: `/api/v1/voices/${voiceId}/preview`, body: { text } });
  }

  async create(data: { name: string; provider: VoiceProvider; providerVoiceId: string; config?: VoiceConfig }): Promise<Voice> {
    return this.client.request({ method: 'POST', path: '/api/v1/voices', body: data });
  }
}
