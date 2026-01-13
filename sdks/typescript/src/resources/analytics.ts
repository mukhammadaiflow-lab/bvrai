import type { BuilderEngine } from '../client';
import type { Analytics, Usage } from '../models';

export class AnalyticsResource {
  constructor(private readonly client: BuilderEngine) {}

  async getOverview(params: { period?: string; startDate?: string; endDate?: string } = {}): Promise<Analytics> {
    return this.client.request({ method: 'GET', path: '/api/v1/analytics/overview', params: { period: params.period || 'week', start_date: params.startDate, end_date: params.endDate } });
  }

  async getUsage(period: string = 'month'): Promise<Usage> {
    return this.client.request({ method: 'GET', path: '/api/v1/analytics/usage', params: { period } });
  }

  async getCostBreakdown(params: { period?: string; groupBy?: string } = {}): Promise<Record<string, any>> {
    return this.client.request({ method: 'GET', path: '/api/v1/analytics/costs', params });
  }

  async export(params: { reportType: string; period?: string; format?: string }): Promise<{ downloadUrl: string }> {
    return this.client.request({ method: 'POST', path: '/api/v1/analytics/export', body: params });
  }
}
