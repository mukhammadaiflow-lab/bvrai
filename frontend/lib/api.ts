/**
 * API client for the Builder Voice AI Platform.
 *
 * Provides typed methods for all API endpoints with error handling.
 */

import axios, { AxiosInstance, AxiosError } from "axios";
import type {
  Agent,
  AgentVersion,
  CreateAgentRequest,
  UpdateAgentRequest,
  Call,
  CallEvent,
  Conversation,
  Message,
  VoiceConfiguration,
  Voice,
  Webhook,
  WebhookDelivery,
  AnalyticsSummary,
  DashboardStats,
  UsageSummary,
  User,
  Organization,
  ApiKey,
  PaginatedResponse,
  ApiError,
} from "@/types";

// API Configuration
// Use proxy route for secure cookie-based auth, or direct backend URL
const USE_PROXY = process.env.NEXT_PUBLIC_USE_API_PROXY !== 'false';
const DIRECT_API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8086/api/v1";
const PROXY_API_URL = "/api/proxy";
const API_BASE_URL = USE_PROXY ? PROXY_API_URL : DIRECT_API_URL;

// Error class for API errors
export class APIError extends Error {
  public code: string;
  public status: number;
  public details?: Record<string, unknown>;

  constructor(message: string, code: string, status: number, details?: Record<string, unknown>) {
    super(message);
    this.name = "APIError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

// Create axios instance with defaults
const createAPIClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
      "Content-Type": "application/json",
    },
    withCredentials: true, // Include cookies for proxy auth
  });

  // Request interceptor - add auth token (only for direct API calls)
  client.interceptors.request.use(
    (config) => {
      // When using proxy, cookies are handled by the proxy route
      // When using direct API, use localStorage token (backward compatibility)
      if (!USE_PROXY) {
        const token = typeof window !== "undefined" ? localStorage.getItem("auth_token") : null;
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
      }
      return config;
    },
    (error) => Promise.reject(error)
  );

  // Response interceptor - handle errors
  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError<ApiError>) => {
      if (error.response) {
        const { data, status } = error.response;
        throw new APIError(
          data?.message || "An error occurred",
          data?.code || "UNKNOWN_ERROR",
          status,
          data?.details
        );
      }
      throw new APIError(
        error.message || "Network error",
        "NETWORK_ERROR",
        0
      );
    }
  );

  return client;
};

const api = createAPIClient();

// ============================================================================
// Authentication
// Uses Next.js API routes for secure httpOnly cookie-based auth
// ============================================================================

export const auth = {
  async login(email: string, password: string): Promise<{ access_token: string; token: string; user: User }> {
    // Use Next.js API route for secure cookie handling
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
      credentials: "include",
    });

    if (!response.ok) {
      const error = await response.json();
      throw new APIError(error.error || "Login failed", "AUTH_ERROR", response.status);
    }

    const data = await response.json();
    // Return compatible format for existing code
    return { access_token: "httponly", token: "httponly", user: data.user };
  },

  async register(email: string, password: string, name?: string, organizationName?: string): Promise<{ access_token: string; user: User }> {
    const response = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name, organization_name: organizationName }),
      credentials: "include",
    });

    if (!response.ok) {
      const error = await response.json();
      throw new APIError(error.error || "Registration failed", "AUTH_ERROR", response.status);
    }

    const data = await response.json();
    return { access_token: "httponly", user: data.user };
  },

  async logout(): Promise<void> {
    await fetch("/api/auth/logout", {
      method: "POST",
      credentials: "include",
    });
  },

  async getCurrentUser(): Promise<User> {
    const response = await fetch("/api/auth/me", {
      credentials: "include",
    });

    if (!response.ok) {
      throw new APIError("Not authenticated", "AUTH_ERROR", response.status);
    }

    return response.json();
  },

  async refreshToken(): Promise<{ token: string }> {
    // Token refresh is handled automatically via httpOnly cookies
    return { token: "httponly" };
  },
};

// ============================================================================
// Agents
// ============================================================================

export const agents = {
  async list(params?: {
    page?: number;
    page_size?: number;
    status?: string;
    search?: string;
  }): Promise<PaginatedResponse<Agent>> {
    const { data } = await api.get("/agents", { params });
    return data;
  },

  async get(agentId: string): Promise<Agent> {
    const { data } = await api.get(`/agents/${agentId}`);
    return data;
  },

  async create(agent: CreateAgentRequest): Promise<Agent> {
    const { data } = await api.post("/agents", agent);
    return data;
  },

  async update(agentId: string, updates: UpdateAgentRequest): Promise<Agent> {
    const { data } = await api.put(`/agents/${agentId}`, updates);
    return data;
  },

  async delete(agentId: string): Promise<void> {
    await api.delete(`/agents/${agentId}`);
  },

  async test(agentId: string, message: string): Promise<{ response: string; latency_ms: number }> {
    const { data } = await api.post(`/agents/${agentId}/test`, { message });
    return data;
  },

  async duplicate(agentId: string, name?: string): Promise<Agent> {
    const { data } = await api.post(`/agents/${agentId}/duplicate`, { name });
    return data;
  },

  async getVersions(agentId: string): Promise<AgentVersion[]> {
    const { data } = await api.get(`/agents/${agentId}/versions`);
    return data;
  },

  async rollback(agentId: string, version: number): Promise<Agent> {
    const { data } = await api.post(`/agents/${agentId}/rollback`, { version });
    return data;
  },

  async deploy(agentId: string): Promise<Agent> {
    const { data } = await api.post(`/agents/${agentId}/deploy`);
    return data;
  },
};

// ============================================================================
// Calls
// ============================================================================

export const calls = {
  async list(params?: {
    page?: number;
    page_size?: number;
    agent_id?: string;
    status?: string;
    direction?: string;
    from_date?: string;
    to_date?: string;
  }): Promise<PaginatedResponse<Call>> {
    const { data } = await api.get("/calls", { params });
    return data;
  },

  async get(callId: string): Promise<Call> {
    const { data } = await api.get(`/calls/${callId}`);
    return data;
  },

  async initiateOutbound(
    agentIdOrParams: string | { agent_id: string; to_number: string; from_number?: string; metadata?: Record<string, unknown> },
    toNumber?: string,
    metadata?: Record<string, unknown>
  ): Promise<Call> {
    const params = typeof agentIdOrParams === 'string'
      ? { agent_id: agentIdOrParams, to_number: toNumber!, metadata }
      : agentIdOrParams;
    const { data } = await api.post("/calls/outbound", params);
    return data;
  },

  async hangup(callId: string): Promise<void> {
    await api.post(`/calls/${callId}/hangup`);
  },

  async getConversation(callId: string): Promise<Conversation> {
    const { data } = await api.get(`/calls/${callId}/conversation`);
    return data;
  },

  async getMessages(callId: string): Promise<Message[]> {
    const { data } = await api.get(`/calls/${callId}/messages`);
    return data;
  },

  async getRecording(callId: string): Promise<{ url: string }> {
    const { data } = await api.get(`/calls/${callId}/recording`);
    return data;
  },

  async getEvents(callId: string): Promise<CallEvent[]> {
    const { data } = await api.get(`/calls/${callId}/events`);
    return data;
  },

  async end(callId: string): Promise<void> {
    await api.post(`/calls/${callId}/hangup`);
  },

  async getTranscript(callId: string): Promise<{ transcript: string; messages: Message[] }> {
    const { data } = await api.get(`/calls/${callId}/transcript`);
    return data;
  },

  async getAnalytics(params?: {
    startDate?: string;
    endDate?: string;
    agentId?: string;
  }): Promise<AnalyticsSummary> {
    const { data } = await api.get("/analytics/summary", {
      params: {
        from_date: params?.startDate,
        to_date: params?.endDate,
        agent_id: params?.agentId,
      }
    });
    return data;
  },
};

// ============================================================================
// Voice Configuration
// ============================================================================

export const voiceConfig = {
  async list(): Promise<VoiceConfiguration[]> {
    const { data } = await api.get("/voice-configs");
    return data;
  },

  async get(configId: string): Promise<VoiceConfiguration> {
    const { data } = await api.get(`/voice-configs/${configId}`);
    return data;
  },

  async create(config: Partial<VoiceConfiguration>): Promise<VoiceConfiguration> {
    const { data } = await api.post("/voice-configs", config);
    return data;
  },

  async update(configId: string, updates: Partial<VoiceConfiguration>): Promise<VoiceConfiguration> {
    const { data } = await api.put(`/voice-configs/${configId}`, updates);
    return data;
  },

  async delete(configId: string): Promise<void> {
    await api.delete(`/voice-configs/${configId}`);
  },

  async listVoices(
    providerOrParams?: string | { provider?: string; language?: string; gender?: string }
  ): Promise<Voice[]> {
    const params = typeof providerOrParams === 'string'
      ? { provider: providerOrParams }
      : providerOrParams;
    const { data } = await api.get("/voices", { params });
    return data;
  },

  async previewVoice(voiceId: string, text: string): Promise<Blob> {
    const { data } = await api.post(
      `/voices/${voiceId}/preview`,
      { text },
      { responseType: "blob" }
    );
    return data;
  },
};

// ============================================================================
// Analytics
// ============================================================================

export const analytics = {
  async getSummary(params?: {
    from_date?: string;
    to_date?: string;
    agent_id?: string;
  }): Promise<AnalyticsSummary> {
    const { data } = await api.get("/analytics/summary", { params });
    return data;
  },

  async getDashboard(): Promise<DashboardStats> {
    const { data } = await api.get("/analytics/dashboard");
    return data;
  },

  async getDashboardStats(): Promise<DashboardStats> {
    const { data } = await api.get("/analytics/dashboard");
    return data;
  },

  async getCallsByDay(params?: {
    from_date?: string;
    to_date?: string;
    agent_id?: string;
  }): Promise<Array<{ date: string; count: number; duration: number }>> {
    const { data } = await api.get("/analytics/calls-by-day", { params });
    return data;
  },

  async getCallsByHour(params?: {
    from_date?: string;
    to_date?: string;
  }): Promise<Array<{ hour: number; count: number }>> {
    const { data } = await api.get("/analytics/calls-by-hour", { params });
    return data;
  },

  async getAgentPerformance(): Promise<Array<{
    agent_id: string;
    agent_name: string;
    total_calls: number;
    success_rate: number;
    avg_duration: number;
  }>> {
    const { data } = await api.get("/analytics/agent-performance");
    return data;
  },
};

// ============================================================================
// Webhooks
// ============================================================================

export const webhooks = {
  async list(): Promise<Webhook[]> {
    const { data } = await api.get("/webhooks");
    return data;
  },

  async get(webhookId: string): Promise<Webhook> {
    const { data } = await api.get(`/webhooks/${webhookId}`);
    return data;
  },

  async create(webhook: Partial<Webhook>): Promise<Webhook> {
    const { data } = await api.post("/webhooks", webhook);
    return data;
  },

  async update(webhookId: string, updates: Partial<Webhook>): Promise<Webhook> {
    const { data } = await api.put(`/webhooks/${webhookId}`, updates);
    return data;
  },

  async delete(webhookId: string): Promise<void> {
    await api.delete(`/webhooks/${webhookId}`);
  },

  async test(webhookId: string): Promise<{ success: boolean; response_status: number }> {
    const { data } = await api.post(`/webhooks/${webhookId}/test`);
    return data;
  },

  async getDeliveries(webhookId: string, params?: { page?: number; page_size?: number }): Promise<PaginatedResponse<WebhookDelivery>> {
    const { data } = await api.get(`/webhooks/${webhookId}/deliveries`, { params });
    return data;
  },
};

// ============================================================================
// Billing
// ============================================================================

export const billing = {
  async getCurrentPlan(): Promise<{
    plan: string;
    usage: UsageSummary;
  }> {
    const { data } = await api.get("/billing/plan");
    return data;
  },

  async getUsage(params?: {
    from_date?: string;
    to_date?: string;
  }): Promise<UsageSummary[]> {
    const { data } = await api.get("/billing/usage", { params });
    return data;
  },

  async getInvoices(): Promise<Array<{
    id: string;
    amount_cents: number;
    status: string;
    created_at: string;
    pdf_url?: string;
  }>> {
    const { data } = await api.get("/billing/invoices");
    return data;
  },

  async updatePaymentMethod(paymentMethodId: string): Promise<void> {
    await api.post("/billing/payment-method", { payment_method_id: paymentMethodId });
  },

  async createCheckoutSession(planId: string): Promise<{ checkout_url: string; session_id: string }> {
    const { data } = await api.post("/billing/checkout", { plan_id: planId });
    return { checkout_url: data.url || data.checkout_url, session_id: data.session_id };
  },

  async cancelSubscription(): Promise<void> {
    await api.post("/billing/cancel");
  },
};

// ============================================================================
// Organization
// ============================================================================

export const organization = {
  async get(): Promise<Organization> {
    const { data } = await api.get("/organization");
    return data;
  },

  async getCurrent(): Promise<Organization> {
    const { data } = await api.get("/organization");
    return data;
  },

  async update(updates: Partial<Organization>): Promise<Organization> {
    const { data } = await api.put("/organization", updates);
    return data;
  },

  async getMembers(): Promise<User[]> {
    const { data } = await api.get("/organization/members");
    return data;
  },

  async inviteMember(email: string, role: string): Promise<void> {
    await api.post("/organization/invitations", { email, role });
  },

  async removeMember(userId: string): Promise<void> {
    await api.delete(`/organization/members/${userId}`);
  },

  async updateMemberRole(userId: string, role: string): Promise<User> {
    const { data } = await api.put(`/organization/members/${userId}`, { role });
    return data;
  },
};

// ============================================================================
// API Keys
// ============================================================================

export const apiKeys = {
  async list(): Promise<ApiKey[]> {
    const { data } = await api.get("/api-keys");
    return data;
  },

  async create(
    nameOrParams: string | { name: string; scopes?: string[]; expires_at?: string },
    scopes?: string[]
  ): Promise<{ key: string; api_key: ApiKey }> {
    const params = typeof nameOrParams === 'string'
      ? { name: nameOrParams, scopes }
      : nameOrParams;
    const { data } = await api.post("/api-keys", params);
    return data;
  },

  async revoke(keyId: string): Promise<void> {
    await api.delete(`/api-keys/${keyId}`);
  },
};

// ============================================================================
// Named exports for hooks (aliased)
// ============================================================================

export const authApi = auth;
export const agentsApi = agents;
export const callsApi = calls;
export const voiceConfigApi = voiceConfig;
export const analyticsApi = analytics;
export const webhooksApi = webhooks;
export const billingApi = billing;
export const organizationApi = organization;
export const apiKeysApi = apiKeys;

// ============================================================================
// Export default API object
// ============================================================================

export default {
  auth,
  agents,
  calls,
  voiceConfig,
  analytics,
  webhooks,
  billing,
  organization,
  apiKeys,
};
