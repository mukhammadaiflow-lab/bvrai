"use client";

import React, { useState, useMemo } from "react";
import DashboardLayout from "../components/DashboardLayout";

// Types
type ApiKeyStatus = "active" | "revoked" | "expired";
type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
type ParameterType = "string" | "number" | "boolean" | "object" | "array";
type ParameterLocation = "path" | "query" | "body" | "header";

interface ApiKey {
  id: string;
  name: string;
  key: string;
  status: ApiKeyStatus;
  permissions: string[];
  createdAt: string;
  lastUsed?: string;
  expiresAt?: string;
  requestCount: number;
  rateLimit: number;
  environment: "production" | "development" | "sandbox";
}

interface Webhook {
  id: string;
  name: string;
  url: string;
  events: string[];
  enabled: boolean;
  createdAt: string;
  lastTriggered?: string;
  successRate: number;
  secret?: string;
}

interface ApiParameter {
  name: string;
  type: ParameterType;
  location: ParameterLocation;
  required: boolean;
  description: string;
  example?: string;
  enum?: string[];
}

interface ApiEndpoint {
  id: string;
  method: HttpMethod;
  path: string;
  summary: string;
  description: string;
  category: string;
  parameters: ApiParameter[];
  requestBody?: {
    contentType: string;
    schema: object;
    example: object;
  };
  responses: {
    status: number;
    description: string;
    example?: object;
  }[];
  rateLimit?: string;
  authentication: boolean;
}

interface SdkLanguage {
  id: string;
  name: string;
  icon: string;
  version: string;
  installCommand: string;
  repository: string;
  documentation: string;
}

interface UsageMetric {
  date: string;
  requests: number;
  errors: number;
  latency: number;
}

interface ChangelogEntry {
  version: string;
  date: string;
  type: "feature" | "improvement" | "fix" | "breaking";
  title: string;
  description: string;
}

// Mock Data
const mockApiKeys: ApiKey[] = [
  {
    id: "key-1",
    name: "Production API Key",
    key: "bvr_live_sk_a1b2c3d4e5f6g7h8i9j0...",
    status: "active",
    permissions: ["calls:read", "calls:write", "agents:read", "agents:write", "recordings:read"],
    createdAt: "2024-01-01T00:00:00Z",
    lastUsed: "2024-01-20T10:30:00Z",
    requestCount: 125400,
    rateLimit: 1000,
    environment: "production",
  },
  {
    id: "key-2",
    name: "Development API Key",
    key: "bvr_test_sk_x1y2z3a4b5c6d7e8f9g0...",
    status: "active",
    permissions: ["calls:read", "calls:write", "agents:read", "agents:write"],
    createdAt: "2024-01-10T00:00:00Z",
    lastUsed: "2024-01-20T09:15:00Z",
    requestCount: 8560,
    rateLimit: 100,
    environment: "development",
  },
  {
    id: "key-3",
    name: "Legacy API Key",
    key: "bvr_live_sk_old_k1l2m3n4o5p6...",
    status: "revoked",
    permissions: ["calls:read"],
    createdAt: "2023-06-15T00:00:00Z",
    expiresAt: "2024-01-01T00:00:00Z",
    requestCount: 45000,
    rateLimit: 500,
    environment: "production",
  },
];

const mockWebhooks: Webhook[] = [
  {
    id: "wh-1",
    name: "Call Events",
    url: "https://api.myapp.com/webhooks/bvrai/calls",
    events: ["call.started", "call.ended", "call.transferred"],
    enabled: true,
    createdAt: "2024-01-05T00:00:00Z",
    lastTriggered: "2024-01-20T10:25:00Z",
    successRate: 99.8,
    secret: "whsec_a1b2c3d4...",
  },
  {
    id: "wh-2",
    name: "Transcription Complete",
    url: "https://api.myapp.com/webhooks/bvrai/transcriptions",
    events: ["transcription.completed"],
    enabled: true,
    createdAt: "2024-01-08T00:00:00Z",
    lastTriggered: "2024-01-20T10:20:00Z",
    successRate: 98.5,
  },
  {
    id: "wh-3",
    name: "Agent Status",
    url: "https://api.myapp.com/webhooks/bvrai/agents",
    events: ["agent.online", "agent.offline", "agent.busy"],
    enabled: false,
    createdAt: "2024-01-12T00:00:00Z",
    successRate: 95.2,
  },
];

const mockEndpoints: ApiEndpoint[] = [
  {
    id: "ep-1",
    method: "POST",
    path: "/v1/calls",
    summary: "Initiate a call",
    description: "Initiates a new outbound call using the specified agent and phone number. The call will be connected to the customer once they answer.",
    category: "Calls",
    parameters: [],
    requestBody: {
      contentType: "application/json",
      schema: {
        type: "object",
        properties: {
          agent_id: { type: "string" },
          to_number: { type: "string" },
          from_number: { type: "string" },
          metadata: { type: "object" },
        },
      },
      example: {
        agent_id: "agent_123",
        to_number: "+15551234567",
        from_number: "+15559876543",
        metadata: { campaign_id: "camp_abc" },
      },
    },
    responses: [
      { status: 200, description: "Call initiated successfully", example: { id: "call_xyz", status: "initiating" } },
      { status: 400, description: "Invalid request parameters" },
      { status: 401, description: "Unauthorized" },
    ],
    rateLimit: "100 requests/minute",
    authentication: true,
  },
  {
    id: "ep-2",
    method: "GET",
    path: "/v1/calls/{call_id}",
    summary: "Get call details",
    description: "Retrieves detailed information about a specific call including status, duration, recording URL, and transcription.",
    category: "Calls",
    parameters: [
      { name: "call_id", type: "string", location: "path", required: true, description: "The unique identifier of the call", example: "call_xyz123" },
      { name: "include", type: "string", location: "query", required: false, description: "Additional data to include", enum: ["recording", "transcription", "analytics"] },
    ],
    responses: [
      { status: 200, description: "Call details retrieved", example: { id: "call_xyz", status: "completed", duration: 245 } },
      { status: 404, description: "Call not found" },
    ],
    authentication: true,
  },
  {
    id: "ep-3",
    method: "GET",
    path: "/v1/calls",
    summary: "List all calls",
    description: "Returns a paginated list of all calls. Supports filtering by status, date range, and agent.",
    category: "Calls",
    parameters: [
      { name: "status", type: "string", location: "query", required: false, description: "Filter by call status", enum: ["active", "completed", "failed"] },
      { name: "agent_id", type: "string", location: "query", required: false, description: "Filter by agent ID" },
      { name: "start_date", type: "string", location: "query", required: false, description: "Filter by start date (ISO 8601)" },
      { name: "end_date", type: "string", location: "query", required: false, description: "Filter by end date (ISO 8601)" },
      { name: "limit", type: "number", location: "query", required: false, description: "Number of results per page", example: "20" },
      { name: "offset", type: "number", location: "query", required: false, description: "Pagination offset", example: "0" },
    ],
    responses: [
      { status: 200, description: "List of calls", example: { data: [], total: 0, has_more: false } },
    ],
    rateLimit: "500 requests/minute",
    authentication: true,
  },
  {
    id: "ep-4",
    method: "DELETE",
    path: "/v1/calls/{call_id}",
    summary: "End a call",
    description: "Immediately terminates an active call. Cannot be used on calls that have already ended.",
    category: "Calls",
    parameters: [
      { name: "call_id", type: "string", location: "path", required: true, description: "The unique identifier of the call" },
    ],
    responses: [
      { status: 200, description: "Call ended successfully" },
      { status: 400, description: "Call already ended" },
      { status: 404, description: "Call not found" },
    ],
    authentication: true,
  },
  {
    id: "ep-5",
    method: "GET",
    path: "/v1/agents",
    summary: "List all agents",
    description: "Returns a list of all voice AI agents configured for your account.",
    category: "Agents",
    parameters: [
      { name: "status", type: "string", location: "query", required: false, description: "Filter by agent status", enum: ["active", "inactive", "training"] },
    ],
    responses: [
      { status: 200, description: "List of agents", example: { data: [], total: 0 } },
    ],
    authentication: true,
  },
  {
    id: "ep-6",
    method: "POST",
    path: "/v1/agents",
    summary: "Create an agent",
    description: "Creates a new voice AI agent with the specified configuration.",
    category: "Agents",
    parameters: [],
    requestBody: {
      contentType: "application/json",
      schema: {
        type: "object",
        properties: {
          name: { type: "string" },
          voice_id: { type: "string" },
          prompt: { type: "string" },
          language: { type: "string" },
        },
      },
      example: {
        name: "Sales Agent",
        voice_id: "voice_en_us_female_1",
        prompt: "You are a helpful sales assistant...",
        language: "en-US",
      },
    },
    responses: [
      { status: 201, description: "Agent created successfully", example: { id: "agent_new", name: "Sales Agent" } },
      { status: 400, description: "Invalid configuration" },
    ],
    authentication: true,
  },
  {
    id: "ep-7",
    method: "GET",
    path: "/v1/recordings/{recording_id}",
    summary: "Get recording",
    description: "Retrieves a call recording file or metadata.",
    category: "Recordings",
    parameters: [
      { name: "recording_id", type: "string", location: "path", required: true, description: "The recording identifier" },
      { name: "format", type: "string", location: "query", required: false, description: "Audio format", enum: ["mp3", "wav", "ogg"] },
    ],
    responses: [
      { status: 200, description: "Recording retrieved" },
      { status: 404, description: "Recording not found" },
    ],
    authentication: true,
  },
  {
    id: "ep-8",
    method: "GET",
    path: "/v1/transcriptions/{transcription_id}",
    summary: "Get transcription",
    description: "Retrieves the transcription for a call including speaker diarization and timestamps.",
    category: "Transcriptions",
    parameters: [
      { name: "transcription_id", type: "string", location: "path", required: true, description: "The transcription identifier" },
      { name: "format", type: "string", location: "query", required: false, description: "Output format", enum: ["json", "srt", "vtt", "txt"] },
    ],
    responses: [
      { status: 200, description: "Transcription retrieved", example: { segments: [] } },
      { status: 404, description: "Transcription not found" },
    ],
    authentication: true,
  },
];

const mockSdks: SdkLanguage[] = [
  { id: "js", name: "JavaScript / Node.js", icon: "🟨", version: "2.1.0", installCommand: "npm install @bvrai/sdk", repository: "https://github.com/bvrai/sdk-js", documentation: "/docs/sdk/javascript" },
  { id: "python", name: "Python", icon: "🐍", version: "2.0.3", installCommand: "pip install bvrai", repository: "https://github.com/bvrai/sdk-python", documentation: "/docs/sdk/python" },
  { id: "ruby", name: "Ruby", icon: "💎", version: "1.8.0", installCommand: "gem install bvrai", repository: "https://github.com/bvrai/sdk-ruby", documentation: "/docs/sdk/ruby" },
  { id: "go", name: "Go", icon: "🔵", version: "1.5.2", installCommand: "go get github.com/bvrai/sdk-go", repository: "https://github.com/bvrai/sdk-go", documentation: "/docs/sdk/go" },
  { id: "java", name: "Java", icon: "☕", version: "1.4.0", installCommand: "implementation 'com.bvrai:sdk:1.4.0'", repository: "https://github.com/bvrai/sdk-java", documentation: "/docs/sdk/java" },
  { id: "php", name: "PHP", icon: "🐘", version: "1.3.1", installCommand: "composer require bvrai/sdk", repository: "https://github.com/bvrai/sdk-php", documentation: "/docs/sdk/php" },
];

const mockUsageMetrics: UsageMetric[] = [
  { date: "Jan 14", requests: 15200, errors: 45, latency: 125 },
  { date: "Jan 15", requests: 18400, errors: 52, latency: 132 },
  { date: "Jan 16", requests: 16800, errors: 38, latency: 118 },
  { date: "Jan 17", requests: 21500, errors: 67, latency: 145 },
  { date: "Jan 18", requests: 19800, errors: 41, latency: 128 },
  { date: "Jan 19", requests: 14600, errors: 29, latency: 112 },
  { date: "Jan 20", requests: 17200, errors: 35, latency: 120 },
];

const mockChangelog: ChangelogEntry[] = [
  { version: "2.5.0", date: "2024-01-18", type: "feature", title: "Real-time Transcription API", description: "Added support for real-time transcription streaming via WebSocket connection." },
  { version: "2.4.2", date: "2024-01-15", type: "fix", title: "Webhook Retry Logic", description: "Fixed an issue where webhook retries were not respecting the exponential backoff policy." },
  { version: "2.4.1", date: "2024-01-12", type: "improvement", title: "Reduced API Latency", description: "Optimized response times for the /calls endpoint by 25%." },
  { version: "2.4.0", date: "2024-01-08", type: "feature", title: "Batch Call API", description: "New endpoint to initiate multiple calls in a single API request." },
  { version: "2.3.5", date: "2024-01-05", type: "breaking", title: "Deprecated v0 Endpoints", description: "All v0 API endpoints have been removed. Please migrate to v1." },
];

const webhookEvents = [
  { event: "call.started", description: "Triggered when a call is initiated" },
  { event: "call.answered", description: "Triggered when the call is answered" },
  { event: "call.ended", description: "Triggered when a call ends" },
  { event: "call.transferred", description: "Triggered when a call is transferred" },
  { event: "call.failed", description: "Triggered when a call fails" },
  { event: "transcription.completed", description: "Triggered when transcription is ready" },
  { event: "recording.ready", description: "Triggered when recording is available" },
  { event: "agent.online", description: "Triggered when an agent comes online" },
  { event: "agent.offline", description: "Triggered when an agent goes offline" },
];

// Components
const MethodBadge: React.FC<{ method: HttpMethod }> = ({ method }) => {
  const colors: Record<HttpMethod, string> = {
    GET: "bg-green-500/20 text-green-400",
    POST: "bg-blue-500/20 text-blue-400",
    PUT: "bg-yellow-500/20 text-yellow-400",
    PATCH: "bg-orange-500/20 text-orange-400",
    DELETE: "bg-red-500/20 text-red-400",
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-mono font-bold rounded ${colors[method]}`}>
      {method}
    </span>
  );
};

const ApiKeyStatusBadge: React.FC<{ status: ApiKeyStatus }> = ({ status }) => {
  const config: Record<ApiKeyStatus, { color: string; label: string }> = {
    active: { color: "bg-green-500/20 text-green-400 border-green-500/30", label: "Active" },
    revoked: { color: "bg-red-500/20 text-red-400 border-red-500/30", label: "Revoked" },
    expired: { color: "bg-gray-500/20 text-gray-400 border-gray-500/30", label: "Expired" },
  };

  const { color, label } = config[status];

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${color}`}>{label}</span>
  );
};

const EnvironmentBadge: React.FC<{ env: ApiKey["environment"] }> = ({ env }) => {
  const colors: Record<string, string> = {
    production: "bg-purple-500/20 text-purple-400",
    development: "bg-blue-500/20 text-blue-400",
    sandbox: "bg-yellow-500/20 text-yellow-400",
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${colors[env]} capitalize`}>
      {env}
    </span>
  );
};

const CodeBlock: React.FC<{ code: string; language?: string }> = ({ code, language = "bash" }) => (
  <div className="bg-gray-950 rounded-lg p-4 font-mono text-sm overflow-x-auto">
    <pre className="text-gray-300">{code}</pre>
  </div>
);

const ApiKeyCard: React.FC<{
  apiKey: ApiKey;
  onRevoke: (id: string) => void;
  onCopy: (key: string) => void;
}> = ({ apiKey, onRevoke, onCopy }) => {
  const [showKey, setShowKey] = useState(false);

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <h3 className="font-semibold text-white">{apiKey.name}</h3>
            <ApiKeyStatusBadge status={apiKey.status} />
            <EnvironmentBadge env={apiKey.environment} />
          </div>
          <p className="text-gray-400 text-sm">Created on {new Date(apiKey.createdAt).toLocaleDateString()}</p>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-400">{apiKey.requestCount.toLocaleString()} requests</p>
          <p className="text-xs text-gray-500">{apiKey.rateLimit}/min limit</p>
        </div>
      </div>

      <div className="bg-gray-900/50 rounded-lg p-3 mb-4 flex items-center justify-between">
        <code className="text-sm text-gray-300 font-mono">
          {showKey ? apiKey.key : apiKey.key.replace(/[a-z0-9]/gi, "•")}
        </code>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowKey(!showKey)}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title={showKey ? "Hide" : "Show"}
          >
            {showKey ? "👁️" : "👁️‍🗨️"}
          </button>
          <button
            onClick={() => onCopy(apiKey.key)}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Copy"
          >
            📋
          </button>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex flex-wrap gap-1">
          {apiKey.permissions.slice(0, 3).map((perm) => (
            <span key={perm} className="px-2 py-0.5 bg-gray-700/50 text-gray-400 text-xs rounded-full">
              {perm}
            </span>
          ))}
          {apiKey.permissions.length > 3 && (
            <span className="text-gray-500 text-xs">+{apiKey.permissions.length - 3} more</span>
          )}
        </div>
        {apiKey.status === "active" && (
          <button
            onClick={() => onRevoke(apiKey.id)}
            className="px-3 py-1.5 text-sm text-red-400 hover:text-red-300 transition-colors"
          >
            Revoke
          </button>
        )}
      </div>
    </div>
  );
};

const WebhookCard: React.FC<{
  webhook: Webhook;
  onToggle: (id: string) => void;
  onEdit: (webhook: Webhook) => void;
}> = ({ webhook, onToggle, onEdit }) => (
  <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5">
    <div className="flex items-start justify-between mb-4">
      <div>
        <div className="flex items-center gap-2 mb-2">
          <h3 className="font-semibold text-white">{webhook.name}</h3>
          <span className={`w-2 h-2 rounded-full ${webhook.enabled ? "bg-green-500" : "bg-gray-500"}`} />
        </div>
        <p className="text-gray-400 text-sm font-mono truncate max-w-md">{webhook.url}</p>
      </div>
      <button
        onClick={() => onToggle(webhook.id)}
        className={`w-12 h-6 rounded-full relative transition-colors ${
          webhook.enabled ? "bg-purple-500" : "bg-gray-700"
        }`}
      >
        <div className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
          webhook.enabled ? "translate-x-6" : "translate-x-0.5"
        }`} />
      </button>
    </div>

    <div className="flex flex-wrap gap-1 mb-4">
      {webhook.events.map((event) => (
        <span key={event} className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded-full">
          {event}
        </span>
      ))}
    </div>

    <div className="flex items-center justify-between text-sm">
      <div className="flex items-center gap-4 text-gray-500">
        <span>Success rate: {webhook.successRate}%</span>
        {webhook.lastTriggered && (
          <span>Last triggered: {new Date(webhook.lastTriggered).toLocaleTimeString()}</span>
        )}
      </div>
      <button
        onClick={() => onEdit(webhook)}
        className="text-purple-400 hover:text-purple-300 transition-colors"
      >
        Edit
      </button>
    </div>
  </div>
);

const EndpointCard: React.FC<{
  endpoint: ApiEndpoint;
  onSelect: (endpoint: ApiEndpoint) => void;
}> = ({ endpoint, onSelect }) => (
  <div
    className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-4 hover:border-purple-500/50 transition-all duration-300 cursor-pointer"
    onClick={() => onSelect(endpoint)}
  >
    <div className="flex items-center gap-3 mb-2">
      <MethodBadge method={endpoint.method} />
      <code className="text-white font-mono text-sm">{endpoint.path}</code>
    </div>
    <p className="text-gray-400 text-sm">{endpoint.summary}</p>
    {endpoint.rateLimit && (
      <p className="text-xs text-gray-500 mt-2">Rate limit: {endpoint.rateLimit}</p>
    )}
  </div>
);

const SdkCard: React.FC<{ sdk: SdkLanguage }> = ({ sdk }) => (
  <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5">
    <div className="flex items-center gap-3 mb-4">
      <span className="text-3xl">{sdk.icon}</span>
      <div>
        <h3 className="font-semibold text-white">{sdk.name}</h3>
        <p className="text-sm text-gray-400">v{sdk.version}</p>
      </div>
    </div>
    <CodeBlock code={sdk.installCommand} />
    <div className="flex items-center gap-4 mt-4">
      <a href={sdk.documentation} className="text-sm text-purple-400 hover:text-purple-300">
        Documentation →
      </a>
      <a href={sdk.repository} className="text-sm text-gray-400 hover:text-white">
        GitHub
      </a>
    </div>
  </div>
);

const UsageChart: React.FC<{ data: UsageMetric[] }> = ({ data }) => {
  const maxRequests = Math.max(...data.map((d) => d.requests));

  return (
    <div className="h-32 flex items-end gap-1">
      {data.map((point, index) => (
        <div key={index} className="flex-1 flex flex-col items-center gap-1">
          <div
            className="w-full bg-gradient-to-t from-purple-500 to-pink-500 rounded-t-sm"
            style={{ height: `${(point.requests / maxRequests) * 100}%` }}
            title={`${point.date}: ${point.requests.toLocaleString()} requests`}
          />
          <span className="text-[10px] text-gray-500">{point.date.split(" ")[1]}</span>
        </div>
      ))}
    </div>
  );
};

const EndpointDetailDialog: React.FC<{
  endpoint: ApiEndpoint | null;
  onClose: () => void;
}> = ({ endpoint, onClose }) => {
  const [activeTab, setActiveTab] = useState<"details" | "try_it">("details");

  if (!endpoint) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <MethodBadge method={endpoint.method} />
                <code className="text-lg text-white font-mono">{endpoint.path}</code>
              </div>
              <p className="text-gray-400">{endpoint.summary}</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="flex items-center gap-2 mt-4">
            {(["details", "try_it"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  activeTab === tab
                    ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                    : "text-gray-400 hover:text-white hover:bg-gray-800"
                }`}
              >
                {tab === "details" ? "Details" : "Try It"}
              </button>
            ))}
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {activeTab === "details" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-2">Description</h3>
                <p className="text-gray-300">{endpoint.description}</p>
              </div>

              {endpoint.parameters.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-3">Parameters</h3>
                  <div className="bg-gray-800/50 rounded-lg overflow-hidden">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-700">
                          <th className="text-left text-xs text-gray-500 p-3">Name</th>
                          <th className="text-left text-xs text-gray-500 p-3">Type</th>
                          <th className="text-left text-xs text-gray-500 p-3">Location</th>
                          <th className="text-left text-xs text-gray-500 p-3">Required</th>
                          <th className="text-left text-xs text-gray-500 p-3">Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        {endpoint.parameters.map((param) => (
                          <tr key={param.name} className="border-b border-gray-700/50 last:border-0">
                            <td className="p-3 font-mono text-sm text-white">{param.name}</td>
                            <td className="p-3 text-sm text-purple-400">{param.type}</td>
                            <td className="p-3 text-sm text-gray-400">{param.location}</td>
                            <td className="p-3">
                              {param.required ? (
                                <span className="text-xs text-red-400">Required</span>
                              ) : (
                                <span className="text-xs text-gray-500">Optional</span>
                              )}
                            </td>
                            <td className="p-3 text-sm text-gray-400">{param.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {endpoint.requestBody && (
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-3">Request Body</h3>
                  <CodeBlock code={JSON.stringify(endpoint.requestBody.example, null, 2)} language="json" />
                </div>
              )}

              <div>
                <h3 className="text-sm font-medium text-gray-400 mb-3">Responses</h3>
                <div className="space-y-3">
                  {endpoint.responses.map((response) => (
                    <div key={response.status} className="bg-gray-800/50 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`px-2 py-0.5 text-xs font-bold rounded ${
                          response.status < 300 ? "bg-green-500/20 text-green-400" :
                          response.status < 400 ? "bg-yellow-500/20 text-yellow-400" :
                          "bg-red-500/20 text-red-400"
                        }`}>
                          {response.status}
                        </span>
                        <span className="text-gray-400">{response.description}</span>
                      </div>
                      {response.example && (
                        <CodeBlock code={JSON.stringify(response.example, null, 2)} language="json" />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === "try_it" && (
            <div className="space-y-6">
              <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4">
                <p className="text-yellow-400 text-sm">
                  Use your API key to test this endpoint. Requests will be made to the actual API.
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">API Key</label>
                <select className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500">
                  <option>Select an API key...</option>
                  {mockApiKeys.filter((k) => k.status === "active").map((key) => (
                    <option key={key.id} value={key.id}>{key.name}</option>
                  ))}
                </select>
              </div>

              {endpoint.parameters.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Parameters</label>
                  <div className="space-y-3">
                    {endpoint.parameters.map((param) => (
                      <div key={param.name}>
                        <label className="block text-xs text-gray-500 mb-1">
                          {param.name} {param.required && <span className="text-red-400">*</span>}
                        </label>
                        <input
                          type="text"
                          placeholder={param.example || param.description}
                          className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {endpoint.requestBody && (
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Request Body</label>
                  <textarea
                    rows={8}
                    defaultValue={JSON.stringify(endpoint.requestBody.example, null, 2)}
                    className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white font-mono text-sm focus:outline-none focus:border-purple-500 resize-none"
                  />
                </div>
              )}

              <button className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
                Send Request
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Main Component
export default function DeveloperPortalPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "api_keys" | "webhooks" | "api_reference" | "sdks" | "changelog">("overview");
  const [selectedEndpoint, setSelectedEndpoint] = useState<ApiEndpoint | null>(null);
  const [apiCategory, setApiCategory] = useState<string>("all");

  const categories = useMemo(() => {
    const cats = new Set(mockEndpoints.map((e) => e.category));
    return ["all", ...Array.from(cats)];
  }, []);

  const filteredEndpoints = useMemo(() => {
    return mockEndpoints.filter((e) => apiCategory === "all" || e.category === apiCategory);
  }, [apiCategory]);

  const totalRequests = mockUsageMetrics.reduce((sum, m) => sum + m.requests, 0);
  const totalErrors = mockUsageMetrics.reduce((sum, m) => sum + m.errors, 0);
  const avgLatency = Math.round(mockUsageMetrics.reduce((sum, m) => sum + m.latency, 0) / mockUsageMetrics.length);

  const tabs = [
    { id: "overview", label: "Overview", icon: "📊" },
    { id: "api_keys", label: "API Keys", icon: "🔑" },
    { id: "webhooks", label: "Webhooks", icon: "🔔" },
    { id: "api_reference", label: "API Reference", icon: "📚" },
    { id: "sdks", label: "SDKs", icon: "📦" },
    { id: "changelog", label: "Changelog", icon: "📋" },
  ];

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-900">
        {/* Header */}
        <div className="border-b border-gray-800 bg-gray-900/95 backdrop-blur-sm sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-2xl font-bold text-white">Developer Portal</h1>
                <p className="text-gray-400 mt-1">API documentation, keys, and developer tools</p>
              </div>
              <div className="flex items-center gap-3">
                <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
                  API Status: <span className="text-green-400">Operational</span>
                </button>
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
                  Create API Key
                </button>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex items-center gap-2 overflow-x-auto pb-2">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg whitespace-nowrap transition-colors ${
                    activeTab === tab.id
                      ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                      : "text-gray-400 hover:text-white hover:bg-gray-800"
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Overview Tab */}
          {activeTab === "overview" && (
            <div className="space-y-8">
              {/* Quick Stats */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <p className="text-gray-400 text-sm">API Requests (7 days)</p>
                  <p className="text-2xl font-bold text-white mt-1">{totalRequests.toLocaleString()}</p>
                </div>
                <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <p className="text-gray-400 text-sm">Error Rate</p>
                  <p className="text-2xl font-bold text-white mt-1">{((totalErrors / totalRequests) * 100).toFixed(2)}%</p>
                </div>
                <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <p className="text-gray-400 text-sm">Avg Latency</p>
                  <p className="text-2xl font-bold text-white mt-1">{avgLatency}ms</p>
                </div>
                <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <p className="text-gray-400 text-sm">Active API Keys</p>
                  <p className="text-2xl font-bold text-white mt-1">{mockApiKeys.filter((k) => k.status === "active").length}</p>
                </div>
              </div>

              {/* Usage Chart */}
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">API Usage (Last 7 Days)</h3>
                <UsageChart data={mockUsageMetrics} />
              </div>

              {/* Quick Start */}
              <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Quick Start</h3>
                <p className="text-gray-400 mb-4">Get started with a simple API call using cURL:</p>
                <CodeBlock code={`curl -X POST https://api.bvrai.com/v1/calls \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"agent_id": "agent_123", "to_number": "+15551234567"}'`} />
              </div>

              {/* Recent Changelog */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Recent Updates</h3>
                  <button
                    onClick={() => setActiveTab("changelog")}
                    className="text-sm text-purple-400 hover:text-purple-300"
                  >
                    View All →
                  </button>
                </div>
                <div className="space-y-3">
                  {mockChangelog.slice(0, 3).map((entry) => (
                    <div key={entry.version} className="bg-gray-800/50 rounded-lg p-4">
                      <div className="flex items-center gap-3">
                        <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                          entry.type === "feature" ? "bg-green-500/20 text-green-400" :
                          entry.type === "improvement" ? "bg-blue-500/20 text-blue-400" :
                          entry.type === "fix" ? "bg-yellow-500/20 text-yellow-400" :
                          "bg-red-500/20 text-red-400"
                        }`}>
                          {entry.type}
                        </span>
                        <span className="text-white font-medium">{entry.title}</span>
                        <span className="text-gray-500 text-sm">v{entry.version}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* API Keys Tab */}
          {activeTab === "api_keys" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-400">{mockApiKeys.length} API keys</p>
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
                  Create New Key
                </button>
              </div>
              <div className="space-y-4">
                {mockApiKeys.map((key) => (
                  <ApiKeyCard
                    key={key.id}
                    apiKey={key}
                    onRevoke={(id) => console.log("Revoke:", id)}
                    onCopy={(k) => console.log("Copy:", k)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Webhooks Tab */}
          {activeTab === "webhooks" && (
            <div className="space-y-8">
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-400">{mockWebhooks.length} webhooks configured</p>
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
                  Add Webhook
                </button>
              </div>

              <div className="space-y-4">
                {mockWebhooks.map((webhook) => (
                  <WebhookCard
                    key={webhook.id}
                    webhook={webhook}
                    onToggle={(id) => console.log("Toggle:", id)}
                    onEdit={(wh) => console.log("Edit:", wh)}
                  />
                ))}
              </div>

              {/* Available Events */}
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Available Events</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {webhookEvents.map((event) => (
                    <div key={event.event} className="p-3 bg-gray-700/30 rounded-lg">
                      <code className="text-purple-400 text-sm">{event.event}</code>
                      <p className="text-gray-400 text-xs mt-1">{event.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* API Reference Tab */}
          {activeTab === "api_reference" && (
            <div className="space-y-6">
              {/* Category Filter */}
              <div className="flex items-center gap-2">
                {categories.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setApiCategory(cat)}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      apiCategory === cat
                        ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                        : "text-gray-400 hover:text-white hover:bg-gray-800"
                    }`}
                  >
                    {cat === "all" ? "All Endpoints" : cat}
                  </button>
                ))}
              </div>

              {/* Endpoints */}
              <div className="space-y-4">
                {filteredEndpoints.map((endpoint) => (
                  <EndpointCard key={endpoint.id} endpoint={endpoint} onSelect={setSelectedEndpoint} />
                ))}
              </div>
            </div>
          )}

          {/* SDKs Tab */}
          {activeTab === "sdks" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {mockSdks.map((sdk) => (
                  <SdkCard key={sdk.id} sdk={sdk} />
                ))}
              </div>
            </div>
          )}

          {/* Changelog Tab */}
          {activeTab === "changelog" && (
            <div className="space-y-4">
              {mockChangelog.map((entry) => (
                <div key={entry.version} className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <span className={`px-3 py-1 text-sm font-medium rounded-full ${
                        entry.type === "feature" ? "bg-green-500/20 text-green-400" :
                        entry.type === "improvement" ? "bg-blue-500/20 text-blue-400" :
                        entry.type === "fix" ? "bg-yellow-500/20 text-yellow-400" :
                        "bg-red-500/20 text-red-400"
                      }`}>
                        {entry.type}
                      </span>
                      <h3 className="text-lg font-semibold text-white">{entry.title}</h3>
                    </div>
                    <div className="text-right">
                      <p className="text-white font-mono">v{entry.version}</p>
                      <p className="text-sm text-gray-500">{new Date(entry.date).toLocaleDateString()}</p>
                    </div>
                  </div>
                  <p className="text-gray-400">{entry.description}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Endpoint Detail Dialog */}
        <EndpointDetailDialog endpoint={selectedEndpoint} onClose={() => setSelectedEndpoint(null)} />
      </div>
    </DashboardLayout>
  );
}
