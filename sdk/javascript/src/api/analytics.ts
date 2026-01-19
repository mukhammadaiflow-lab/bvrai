/**
 * Analytics API for Builder Engine SDK
 */

import type { HttpClient } from '../utils/http';
import type {
  CallMetrics,
  AgentMetrics,
  UsageMetrics,
  RealtimeMetrics,
  TimeSeries,
  MetricType,
  AggregationType,
  TimeGranularity,
  RequestOptions,
  PaginatedResponse,
} from '../types';

export interface Report {
  id: string;
  name: string;
  type: string;
  status: 'pending' | 'processing' | 'ready' | 'failed';
  createdAt: string;
  completedAt?: string;
  downloadUrl?: string;
  parameters: Record<string, unknown>;
}

export class AnalyticsAPI {
  constructor(private readonly http: HttpClient) {}

  // Call Metrics

  /**
   * Get aggregated call metrics
   */
  async getCallMetrics(
    params?: {
      startTime?: string;
      endTime?: string;
      agentId?: string;
    },
    options?: RequestOptions
  ): Promise<CallMetrics> {
    const response = await this.http.get<Record<string, unknown>>(
      '/v1/analytics/calls/metrics',
      {
        start_time: params?.startTime,
        end_time: params?.endTime,
        agent_id: params?.agentId,
      },
      options
    );
    return this.transformCallMetrics(response);
  }

  /**
   * Get call metrics as time series
   */
  async getCallTimeSeries(
    params: {
      metric: MetricType;
      granularity?: TimeGranularity;
      startTime?: string;
      endTime?: string;
      agentId?: string;
      aggregation?: AggregationType;
    },
    options?: RequestOptions
  ): Promise<TimeSeries> {
    const response = await this.http.get<Record<string, unknown>>(
      '/v1/analytics/calls/timeseries',
      {
        metric: params.metric,
        granularity: params.granularity || 'hour',
        start_time: params.startTime,
        end_time: params.endTime,
        agent_id: params.agentId,
        aggregation: params.aggregation || 'sum',
      },
      options
    );
    return this.transformTimeSeries(response);
  }

  // Agent Metrics

  /**
   * Get metrics for a specific agent
   */
  async getAgentMetrics(
    agentId: string,
    params?: {
      startTime?: string;
      endTime?: string;
    },
    options?: RequestOptions
  ): Promise<AgentMetrics> {
    const response = await this.http.get<Record<string, unknown>>(
      `/v1/analytics/agents/${agentId}/metrics`,
      {
        start_time: params?.startTime,
        end_time: params?.endTime,
      },
      options
    );
    return this.transformAgentMetrics(response);
  }

  /**
   * Get metrics for all agents
   */
  async getAllAgentsMetrics(
    params?: {
      startTime?: string;
      endTime?: string;
      sortBy?: string;
      limit?: number;
    },
    options?: RequestOptions
  ): Promise<AgentMetrics[]> {
    const response = await this.http.get<{ agents: Record<string, unknown>[] }>(
      '/v1/analytics/agents/metrics',
      {
        start_time: params?.startTime,
        end_time: params?.endTime,
        sort_by: params?.sortBy || 'total_calls',
        limit: params?.limit || 10,
      },
      options
    );
    return response.agents.map((a) => this.transformAgentMetrics(a));
  }

  // Usage & Billing

  /**
   * Get usage metrics for billing
   */
  async getUsage(
    params?: {
      startTime?: string;
      endTime?: string;
    },
    options?: RequestOptions
  ): Promise<UsageMetrics> {
    const response = await this.http.get<Record<string, unknown>>(
      '/v1/analytics/usage',
      {
        start_time: params?.startTime,
        end_time: params?.endTime,
      },
      options
    );
    return this.transformUsageMetrics(response);
  }

  /**
   * Get usage for current billing period
   */
  async getCurrentBillingPeriod(options?: RequestOptions): Promise<UsageMetrics> {
    const response = await this.http.get<Record<string, unknown>>(
      '/v1/analytics/usage/current',
      undefined,
      options
    );
    return this.transformUsageMetrics(response);
  }

  /**
   * Get cost breakdown
   */
  async getCostBreakdown(
    params?: {
      startTime?: string;
      endTime?: string;
      groupBy?: 'service' | 'agent' | 'day';
    },
    options?: RequestOptions
  ): Promise<Record<string, { usage: number; cost: number }>> {
    return this.http.get(
      '/v1/analytics/costs/breakdown',
      {
        start_time: params?.startTime,
        end_time: params?.endTime,
        group_by: params?.groupBy || 'service',
      },
      options
    );
  }

  // Real-time Metrics

  /**
   * Get current real-time metrics
   */
  async getRealtimeMetrics(options?: RequestOptions): Promise<RealtimeMetrics> {
    const response = await this.http.get<Record<string, unknown>>(
      '/v1/analytics/realtime',
      undefined,
      options
    );
    return this.transformRealtimeMetrics(response);
  }

  /**
   * Get count of currently active calls
   */
  async getActiveCallsCount(options?: RequestOptions): Promise<number> {
    const response = await this.http.get<{ count: number }>(
      '/v1/analytics/realtime/active-calls',
      undefined,
      options
    );
    return response.count;
  }

  // Reports

  /**
   * Create an analytics report
   */
  async createReport(
    params: {
      name: string;
      type: string;
      startTime: string;
      endTime: string;
      parameters?: Record<string, unknown>;
      format?: 'csv' | 'json' | 'pdf';
    },
    options?: RequestOptions
  ): Promise<Report> {
    const response = await this.http.post<Record<string, unknown>>(
      '/v1/analytics/reports',
      {
        name: params.name,
        type: params.type,
        start_time: params.startTime,
        end_time: params.endTime,
        parameters: params.parameters || {},
        format: params.format || 'csv',
      },
      options
    );
    return this.transformReport(response);
  }

  /**
   * Get a report by ID
   */
  async getReport(reportId: string, options?: RequestOptions): Promise<Report> {
    const response = await this.http.get<Record<string, unknown>>(
      `/v1/analytics/reports/${reportId}`,
      undefined,
      options
    );
    return this.transformReport(response);
  }

  /**
   * List all reports
   */
  async listReports(
    params?: {
      limit?: number;
      offset?: number;
    },
    options?: RequestOptions
  ): Promise<PaginatedResponse<Report>> {
    const response = await this.http.get<{
      reports: Record<string, unknown>[];
      total: number;
      limit: number;
      offset: number;
    }>('/v1/analytics/reports', params, options);

    return {
      items: response.reports.map((r) => this.transformReport(r)),
      total: response.total,
      limit: response.limit,
      offset: response.offset,
      hasMore: response.offset + response.reports.length < response.total,
    };
  }

  /**
   * Delete a report
   */
  async deleteReport(reportId: string, options?: RequestOptions): Promise<void> {
    await this.http.delete(`/v1/analytics/reports/${reportId}`, options);
  }

  /**
   * Download a report file
   */
  async downloadReport(reportId: string, options?: RequestOptions): Promise<ArrayBuffer> {
    return this.http.getRaw(`/v1/analytics/reports/${reportId}/download`, undefined, options);
  }

  // Convenience Methods

  /**
   * Get metrics for today
   */
  async getTodayMetrics(options?: RequestOptions): Promise<CallMetrics> {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return this.getCallMetrics({ startTime: today.toISOString() }, options);
  }

  /**
   * Get metrics for last 24 hours
   */
  async getLast24hMetrics(options?: RequestOptions): Promise<CallMetrics> {
    const end = new Date();
    const start = new Date(end.getTime() - 24 * 60 * 60 * 1000);
    return this.getCallMetrics(
      { startTime: start.toISOString(), endTime: end.toISOString() },
      options
    );
  }

  /**
   * Get metrics for last 7 days
   */
  async getLast7DaysMetrics(options?: RequestOptions): Promise<CallMetrics> {
    const end = new Date();
    const start = new Date(end.getTime() - 7 * 24 * 60 * 60 * 1000);
    return this.getCallMetrics(
      { startTime: start.toISOString(), endTime: end.toISOString() },
      options
    );
  }

  /**
   * Get metrics for last 30 days
   */
  async getLast30DaysMetrics(options?: RequestOptions): Promise<CallMetrics> {
    const end = new Date();
    const start = new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);
    return this.getCallMetrics(
      { startTime: start.toISOString(), endTime: end.toISOString() },
      options
    );
  }

  // Private helpers

  private transformCallMetrics(data: Record<string, unknown>): CallMetrics {
    return {
      totalCalls: (data.totalCalls || data.total_calls) as number,
      successfulCalls: (data.successfulCalls || data.successful_calls) as number,
      failedCalls: (data.failedCalls || data.failed_calls) as number,
      averageDurationSeconds: (data.averageDurationSeconds || data.average_duration_seconds) as number,
      totalDurationSeconds: (data.totalDurationSeconds || data.total_duration_seconds) as number,
      averageLatencyMs: (data.averageLatencyMs || data.average_latency_ms) as number,
      totalCost: (data.totalCost || data.total_cost) as number,
      totalTokens: (data.totalTokens || data.total_tokens) as number,
      successRate: (data.successRate || data.success_rate) as number,
    };
  }

  private transformAgentMetrics(data: Record<string, unknown>): AgentMetrics {
    return {
      agentId: (data.agentId || data.agent_id) as string,
      agentName: (data.agentName || data.agent_name) as string,
      totalCalls: (data.totalCalls || data.total_calls) as number,
      averageDurationSeconds: (data.averageDurationSeconds || data.average_duration_seconds) as number,
      successRate: (data.successRate || data.success_rate) as number,
      averageSentimentScore: (data.averageSentimentScore || data.average_sentiment_score) as number,
      averageResponseTimeMs: (data.averageResponseTimeMs || data.average_response_time_ms) as number,
      topIntents: (data.topIntents || data.top_intents) as Array<{ intent: string; count: number }>,
      totalCost: (data.totalCost || data.total_cost) as number,
    };
  }

  private transformUsageMetrics(data: Record<string, unknown>): UsageMetrics {
    return {
      periodStart: (data.periodStart || data.period_start) as string,
      periodEnd: (data.periodEnd || data.period_end) as string,
      totalMinutes: (data.totalMinutes || data.total_minutes) as number,
      totalCalls: (data.totalCalls || data.total_calls) as number,
      totalTokens: (data.totalTokens || data.total_tokens) as number,
      totalCost: (data.totalCost || data.total_cost) as number,
      breakdownByAgent: (data.breakdownByAgent || data.breakdown_by_agent) as Record<
        string,
        { calls: number; minutes: number; cost: number }
      >,
      breakdownByService: (data.breakdownByService || data.breakdown_by_service) as Record<
        string,
        { usage: number; cost: number }
      >,
    };
  }

  private transformRealtimeMetrics(data: Record<string, unknown>): RealtimeMetrics {
    return {
      timestamp: data.timestamp as string,
      activeCalls: (data.activeCalls || data.active_calls) as number,
      callsPerMinute: (data.callsPerMinute || data.calls_per_minute) as number,
      averageQueueTimeMs: (data.averageQueueTimeMs || data.average_queue_time_ms) as number,
      errorRate: (data.errorRate || data.error_rate) as number,
      activeAgents: (data.activeAgents || data.active_agents) as number,
      concurrentConnections: (data.concurrentConnections || data.concurrent_connections) as number,
    };
  }

  private transformTimeSeries(data: Record<string, unknown>): TimeSeries {
    return {
      metric: data.metric as string,
      granularity: data.granularity as TimeGranularity,
      points: ((data.points || []) as Array<Record<string, unknown>>).map((p) => ({
        timestamp: p.timestamp as string,
        value: p.value as number,
        metadata: p.metadata as Record<string, string | number | boolean | null>,
      })),
      aggregation: data.aggregation as AggregationType,
    };
  }

  private transformReport(data: Record<string, unknown>): Report {
    return {
      id: data.id as string,
      name: data.name as string,
      type: data.type as string,
      status: data.status as Report['status'],
      createdAt: (data.createdAt || data.created_at) as string,
      completedAt: (data.completedAt || data.completed_at) as string | undefined,
      downloadUrl: (data.downloadUrl || data.download_url) as string | undefined,
      parameters: data.parameters as Record<string, unknown>,
    };
  }
}

/**
 * Builder for analytics queries
 */
export class AnalyticsQueryBuilder {
  private _metric?: MetricType;
  private _aggregation: AggregationType = 'sum';
  private _granularity: TimeGranularity = 'hour';
  private _startTime?: string;
  private _endTime?: string;
  private _filters: Record<string, string> = {};
  private _groupBy: string[] = [];

  metric(metric: MetricType): this {
    this._metric = metric;
    return this;
  }

  aggregation(aggregation: AggregationType): this {
    this._aggregation = aggregation;
    return this;
  }

  granularity(granularity: TimeGranularity): this {
    this._granularity = granularity;
    return this;
  }

  timeRange(start: Date | string, end?: Date | string): this {
    this._startTime = typeof start === 'string' ? start : start.toISOString();
    this._endTime = end ? (typeof end === 'string' ? end : end.toISOString()) : undefined;
    return this;
  }

  lastHours(hours: number): this {
    const end = new Date();
    const start = new Date(end.getTime() - hours * 60 * 60 * 1000);
    return this.timeRange(start, end);
  }

  lastDays(days: number): this {
    const end = new Date();
    const start = new Date(end.getTime() - days * 24 * 60 * 60 * 1000);
    return this.timeRange(start, end);
  }

  filterAgent(agentId: string): this {
    this._filters.agent_id = agentId;
    return this;
  }

  filterStatus(status: string): this {
    this._filters.status = status;
    return this;
  }

  filterDirection(direction: 'inbound' | 'outbound'): this {
    this._filters.direction = direction;
    return this;
  }

  groupBy(...fields: string[]): this {
    this._groupBy.push(...fields);
    return this;
  }

  build(): Record<string, string | number | boolean | undefined> {
    if (!this._metric) {
      throw new Error('Metric is required');
    }

    return {
      metric: this._metric,
      aggregation: this._aggregation,
      granularity: this._granularity,
      start_time: this._startTime,
      end_time: this._endTime,
      ...this._filters,
      group_by: this._groupBy.length > 0 ? this._groupBy.join(',') : undefined,
    };
  }
}
