"use client";

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import {
  Code,
  Play,
  Copy,
  Check,
  ChevronRight,
  ChevronDown,
  Search,
  Filter,
  BookOpen,
  Zap,
  Key,
  Lock,
  Unlock,
  Globe,
  Server,
  Database,
  Terminal,
  FileJson,
  Send,
  RefreshCw,
  Settings,
  Eye,
  EyeOff,
  Plus,
  Minus,
  Trash2,
  Download,
  Upload,
  History,
  Star,
  StarOff,
  Clock,
  AlertCircle,
  CheckCircle,
  XCircle,
  Info,
  ExternalLink,
  Clipboard,
  MoreVertical,
  Tag,
  Layers,
  Box,
  Phone,
  MessageSquare,
  Users,
  Bot,
  Mic,
  Volume2,
  FileText,
  Webhook,
  Link,
  ArrowRight,
  ArrowUpRight,
  ArrowDownRight,
  Hash,
  Braces,
  List,
  ToggleLeft,
  Calendar,
  BarChart3,
  Activity,
  Shield,
  CreditCard,
  Building2,
} from "lucide-react";

// Types
type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
type ParamType = "string" | "number" | "boolean" | "array" | "object";
type AuthType = "bearer" | "api_key" | "basic" | "oauth2" | "none";

interface APIParameter {
  name: string;
  type: ParamType;
  description: string;
  required: boolean;
  default?: any;
  enum?: string[];
  example?: any;
  children?: APIParameter[];
}

interface APIEndpoint {
  id: string;
  name: string;
  description: string;
  method: HttpMethod;
  path: string;
  category: string;
  auth: AuthType;
  params?: {
    path?: APIParameter[];
    query?: APIParameter[];
    body?: APIParameter[];
    headers?: APIParameter[];
  };
  response: {
    success: {
      status: number;
      example: any;
    };
    errors: {
      status: number;
      code: string;
      message: string;
    }[];
  };
  rateLimit?: {
    requests: number;
    window: string;
  };
  tags: string[];
  deprecated?: boolean;
}

interface APICategory {
  id: string;
  name: string;
  description: string;
  icon: any;
  endpoints: string[];
}

interface RequestHistoryItem {
  id: string;
  endpoint: string;
  method: HttpMethod;
  path: string;
  timestamp: string;
  status: number;
  duration: number;
  favorite?: boolean;
}

// Sample API Endpoints
const apiEndpoints: APIEndpoint[] = [
  // Agents
  {
    id: "agents_list",
    name: "List Agents",
    description: "Retrieve a paginated list of all AI agents in your account",
    method: "GET",
    path: "/v1/agents",
    category: "agents",
    auth: "bearer",
    params: {
      query: [
        { name: "page", type: "number", description: "Page number for pagination", required: false, default: 1, example: 1 },
        { name: "limit", type: "number", description: "Number of items per page (max 100)", required: false, default: 20, example: 20 },
        { name: "status", type: "string", description: "Filter by agent status", required: false, enum: ["active", "inactive", "draft"], example: "active" },
        { name: "search", type: "string", description: "Search agents by name", required: false, example: "sales" },
      ],
    },
    response: {
      success: {
        status: 200,
        example: {
          data: [
            {
              id: "agent_abc123",
              name: "Sales Assistant",
              status: "active",
              voice_id: "voice_xyz",
              language: "en-US",
              created_at: "2024-01-15T10:00:00Z",
            },
          ],
          meta: { total: 45, page: 1, limit: 20, pages: 3 },
        },
      },
      errors: [
        { status: 401, code: "unauthorized", message: "Invalid or missing API key" },
        { status: 429, code: "rate_limited", message: "Too many requests" },
      ],
    },
    rateLimit: { requests: 100, window: "1 minute" },
    tags: ["agents", "list"],
  },
  {
    id: "agents_create",
    name: "Create Agent",
    description: "Create a new AI agent with specified configuration",
    method: "POST",
    path: "/v1/agents",
    category: "agents",
    auth: "bearer",
    params: {
      body: [
        { name: "name", type: "string", description: "Display name for the agent", required: true, example: "Customer Support AI" },
        { name: "voice_id", type: "string", description: "Voice ID to use for the agent", required: true, example: "voice_xyz789" },
        { name: "language", type: "string", description: "Primary language (ISO code)", required: false, default: "en-US", example: "en-US" },
        { name: "greeting", type: "string", description: "Initial greeting message", required: false, example: "Hello! How can I help you today?" },
        {
          name: "personality",
          type: "object",
          description: "Personality configuration",
          required: false,
          children: [
            { name: "tone", type: "string", description: "Communication tone", required: false, enum: ["professional", "friendly", "casual", "formal"], example: "friendly" },
            { name: "verbosity", type: "string", description: "Response verbosity", required: false, enum: ["concise", "balanced", "detailed"], example: "balanced" },
          ],
        },
        { name: "knowledge_base_ids", type: "array", description: "Array of knowledge base IDs to attach", required: false, example: ["kb_123", "kb_456"] },
        { name: "webhook_url", type: "string", description: "Webhook URL for call events", required: false, example: "https://example.com/webhooks/calls" },
        { name: "metadata", type: "object", description: "Custom metadata key-value pairs", required: false, example: { team: "sales", region: "us-west" } },
      ],
    },
    response: {
      success: {
        status: 201,
        example: {
          id: "agent_new456",
          name: "Customer Support AI",
          status: "draft",
          voice_id: "voice_xyz789",
          language: "en-US",
          greeting: "Hello! How can I help you today?",
          created_at: "2024-01-20T15:30:00Z",
        },
      },
      errors: [
        { status: 400, code: "validation_error", message: "Invalid request body" },
        { status: 401, code: "unauthorized", message: "Invalid or missing API key" },
        { status: 422, code: "invalid_voice", message: "Specified voice_id does not exist" },
      ],
    },
    rateLimit: { requests: 50, window: "1 minute" },
    tags: ["agents", "create"],
  },
  {
    id: "agents_get",
    name: "Get Agent",
    description: "Retrieve details of a specific agent by ID",
    method: "GET",
    path: "/v1/agents/{agent_id}",
    category: "agents",
    auth: "bearer",
    params: {
      path: [
        { name: "agent_id", type: "string", description: "Unique agent identifier", required: true, example: "agent_abc123" },
      ],
    },
    response: {
      success: {
        status: 200,
        example: {
          id: "agent_abc123",
          name: "Sales Assistant",
          status: "active",
          voice_id: "voice_xyz",
          language: "en-US",
          greeting: "Hi there! I'm your sales assistant.",
          personality: { tone: "friendly", verbosity: "balanced" },
          knowledge_base_ids: ["kb_123"],
          call_count: 1250,
          avg_duration: 185,
          created_at: "2024-01-15T10:00:00Z",
          updated_at: "2024-01-18T14:30:00Z",
        },
      },
      errors: [
        { status: 404, code: "not_found", message: "Agent not found" },
        { status: 401, code: "unauthorized", message: "Invalid or missing API key" },
      ],
    },
    rateLimit: { requests: 100, window: "1 minute" },
    tags: ["agents", "read"],
  },
  {
    id: "agents_update",
    name: "Update Agent",
    description: "Update an existing agent's configuration",
    method: "PATCH",
    path: "/v1/agents/{agent_id}",
    category: "agents",
    auth: "bearer",
    params: {
      path: [
        { name: "agent_id", type: "string", description: "Unique agent identifier", required: true, example: "agent_abc123" },
      ],
      body: [
        { name: "name", type: "string", description: "Display name for the agent", required: false, example: "Updated Agent Name" },
        { name: "status", type: "string", description: "Agent status", required: false, enum: ["active", "inactive", "draft"], example: "active" },
        { name: "voice_id", type: "string", description: "Voice ID to use", required: false, example: "voice_new123" },
        { name: "greeting", type: "string", description: "Initial greeting message", required: false, example: "Welcome! How may I assist?" },
      ],
    },
    response: {
      success: {
        status: 200,
        example: {
          id: "agent_abc123",
          name: "Updated Agent Name",
          status: "active",
          updated_at: "2024-01-20T16:00:00Z",
        },
      },
      errors: [
        { status: 404, code: "not_found", message: "Agent not found" },
        { status: 400, code: "validation_error", message: "Invalid request body" },
      ],
    },
    rateLimit: { requests: 50, window: "1 minute" },
    tags: ["agents", "update"],
  },
  {
    id: "agents_delete",
    name: "Delete Agent",
    description: "Permanently delete an agent and all associated data",
    method: "DELETE",
    path: "/v1/agents/{agent_id}",
    category: "agents",
    auth: "bearer",
    params: {
      path: [
        { name: "agent_id", type: "string", description: "Unique agent identifier", required: true, example: "agent_abc123" },
      ],
    },
    response: {
      success: {
        status: 204,
        example: null,
      },
      errors: [
        { status: 404, code: "not_found", message: "Agent not found" },
        { status: 409, code: "agent_in_use", message: "Agent is currently in an active call" },
      ],
    },
    rateLimit: { requests: 20, window: "1 minute" },
    tags: ["agents", "delete"],
  },
  // Calls
  {
    id: "calls_initiate",
    name: "Initiate Call",
    description: "Start an outbound call to a phone number",
    method: "POST",
    path: "/v1/calls",
    category: "calls",
    auth: "bearer",
    params: {
      body: [
        { name: "agent_id", type: "string", description: "Agent to use for the call", required: true, example: "agent_abc123" },
        { name: "to", type: "string", description: "Phone number to call (E.164 format)", required: true, example: "+15551234567" },
        { name: "from", type: "string", description: "Caller ID phone number", required: false, example: "+15559876543" },
        { name: "max_duration", type: "number", description: "Maximum call duration in seconds", required: false, default: 3600, example: 1800 },
        { name: "record", type: "boolean", description: "Enable call recording", required: false, default: true, example: true },
        { name: "webhook_url", type: "string", description: "Override agent webhook for this call", required: false, example: "https://example.com/call-webhook" },
        {
          name: "context",
          type: "object",
          description: "Additional context for the agent",
          required: false,
          example: { customer_name: "John Smith", account_id: "acc_123" },
        },
      ],
    },
    response: {
      success: {
        status: 201,
        example: {
          id: "call_xyz789",
          status: "initiating",
          agent_id: "agent_abc123",
          to: "+15551234567",
          from: "+15559876543",
          direction: "outbound",
          created_at: "2024-01-20T16:30:00Z",
        },
      },
      errors: [
        { status: 400, code: "invalid_phone", message: "Invalid phone number format" },
        { status: 402, code: "insufficient_credits", message: "Not enough credits to make call" },
        { status: 404, code: "agent_not_found", message: "Specified agent does not exist" },
      ],
    },
    rateLimit: { requests: 30, window: "1 minute" },
    tags: ["calls", "outbound"],
  },
  {
    id: "calls_get",
    name: "Get Call",
    description: "Retrieve details of a specific call",
    method: "GET",
    path: "/v1/calls/{call_id}",
    category: "calls",
    auth: "bearer",
    params: {
      path: [
        { name: "call_id", type: "string", description: "Unique call identifier", required: true, example: "call_xyz789" },
      ],
    },
    response: {
      success: {
        status: 200,
        example: {
          id: "call_xyz789",
          status: "completed",
          agent_id: "agent_abc123",
          to: "+15551234567",
          from: "+15559876543",
          direction: "outbound",
          duration: 245,
          recording_url: "https://storage.bvrai.com/recordings/call_xyz789.mp3",
          transcript_url: "https://storage.bvrai.com/transcripts/call_xyz789.json",
          cost: 0.25,
          sentiment: "positive",
          started_at: "2024-01-20T16:30:15Z",
          ended_at: "2024-01-20T16:34:20Z",
        },
      },
      errors: [
        { status: 404, code: "not_found", message: "Call not found" },
      ],
    },
    rateLimit: { requests: 100, window: "1 minute" },
    tags: ["calls", "read"],
  },
  {
    id: "calls_list",
    name: "List Calls",
    description: "Retrieve a paginated list of calls",
    method: "GET",
    path: "/v1/calls",
    category: "calls",
    auth: "bearer",
    params: {
      query: [
        { name: "page", type: "number", description: "Page number", required: false, default: 1 },
        { name: "limit", type: "number", description: "Items per page", required: false, default: 20 },
        { name: "agent_id", type: "string", description: "Filter by agent", required: false },
        { name: "status", type: "string", description: "Filter by status", required: false, enum: ["queued", "ringing", "in_progress", "completed", "failed", "no_answer"] },
        { name: "direction", type: "string", description: "Filter by direction", required: false, enum: ["inbound", "outbound"] },
        { name: "from_date", type: "string", description: "Start date (ISO 8601)", required: false },
        { name: "to_date", type: "string", description: "End date (ISO 8601)", required: false },
      ],
    },
    response: {
      success: {
        status: 200,
        example: {
          data: [
            {
              id: "call_xyz789",
              status: "completed",
              agent_id: "agent_abc123",
              direction: "outbound",
              duration: 245,
              created_at: "2024-01-20T16:30:00Z",
            },
          ],
          meta: { total: 1250, page: 1, limit: 20, pages: 63 },
        },
      },
      errors: [
        { status: 401, code: "unauthorized", message: "Invalid or missing API key" },
      ],
    },
    rateLimit: { requests: 100, window: "1 minute" },
    tags: ["calls", "list"],
  },
  {
    id: "calls_end",
    name: "End Call",
    description: "Terminate an active call",
    method: "POST",
    path: "/v1/calls/{call_id}/end",
    category: "calls",
    auth: "bearer",
    params: {
      path: [
        { name: "call_id", type: "string", description: "Unique call identifier", required: true, example: "call_xyz789" },
      ],
      body: [
        { name: "reason", type: "string", description: "Reason for ending call", required: false, example: "agent_request" },
      ],
    },
    response: {
      success: {
        status: 200,
        example: {
          id: "call_xyz789",
          status: "ended",
          ended_at: "2024-01-20T16:40:00Z",
        },
      },
      errors: [
        { status: 404, code: "not_found", message: "Call not found" },
        { status: 409, code: "call_not_active", message: "Call is not currently active" },
      ],
    },
    rateLimit: { requests: 50, window: "1 minute" },
    tags: ["calls", "control"],
  },
  // Knowledge Bases
  {
    id: "kb_list",
    name: "List Knowledge Bases",
    description: "Retrieve all knowledge bases in your account",
    method: "GET",
    path: "/v1/knowledge-bases",
    category: "knowledge",
    auth: "bearer",
    params: {
      query: [
        { name: "page", type: "number", description: "Page number", required: false, default: 1 },
        { name: "limit", type: "number", description: "Items per page", required: false, default: 20 },
      ],
    },
    response: {
      success: {
        status: 200,
        example: {
          data: [
            {
              id: "kb_123",
              name: "Product Documentation",
              status: "ready",
              source_count: 15,
              chunk_count: 450,
              created_at: "2024-01-10T10:00:00Z",
            },
          ],
          meta: { total: 8, page: 1, limit: 20, pages: 1 },
        },
      },
      errors: [
        { status: 401, code: "unauthorized", message: "Invalid or missing API key" },
      ],
    },
    rateLimit: { requests: 100, window: "1 minute" },
    tags: ["knowledge", "list"],
  },
  {
    id: "kb_create",
    name: "Create Knowledge Base",
    description: "Create a new knowledge base",
    method: "POST",
    path: "/v1/knowledge-bases",
    category: "knowledge",
    auth: "bearer",
    params: {
      body: [
        { name: "name", type: "string", description: "Knowledge base name", required: true, example: "FAQ Database" },
        { name: "description", type: "string", description: "Description", required: false, example: "Frequently asked questions" },
      ],
    },
    response: {
      success: {
        status: 201,
        example: {
          id: "kb_new789",
          name: "FAQ Database",
          status: "empty",
          created_at: "2024-01-20T17:00:00Z",
        },
      },
      errors: [
        { status: 400, code: "validation_error", message: "Name is required" },
      ],
    },
    rateLimit: { requests: 20, window: "1 minute" },
    tags: ["knowledge", "create"],
  },
  {
    id: "kb_add_source",
    name: "Add Source",
    description: "Add a content source to a knowledge base",
    method: "POST",
    path: "/v1/knowledge-bases/{kb_id}/sources",
    category: "knowledge",
    auth: "bearer",
    params: {
      path: [
        { name: "kb_id", type: "string", description: "Knowledge base ID", required: true, example: "kb_123" },
      ],
      body: [
        { name: "type", type: "string", description: "Source type", required: true, enum: ["url", "document", "text", "sitemap"], example: "url" },
        { name: "url", type: "string", description: "URL to crawl (for url/sitemap types)", required: false, example: "https://docs.example.com" },
        { name: "content", type: "string", description: "Text content (for text type)", required: false },
        { name: "file", type: "string", description: "Base64 encoded file (for document type)", required: false },
        { name: "name", type: "string", description: "Source name", required: false, example: "Documentation Site" },
      ],
    },
    response: {
      success: {
        status: 202,
        example: {
          id: "source_xyz",
          kb_id: "kb_123",
          type: "url",
          url: "https://docs.example.com",
          status: "processing",
          created_at: "2024-01-20T17:15:00Z",
        },
      },
      errors: [
        { status: 400, code: "invalid_source", message: "Invalid source configuration" },
        { status: 404, code: "kb_not_found", message: "Knowledge base not found" },
      ],
    },
    rateLimit: { requests: 30, window: "1 minute" },
    tags: ["knowledge", "sources"],
  },
  // Phone Numbers
  {
    id: "phone_list",
    name: "List Phone Numbers",
    description: "Retrieve all phone numbers in your account",
    method: "GET",
    path: "/v1/phone-numbers",
    category: "phone",
    auth: "bearer",
    params: {
      query: [
        { name: "page", type: "number", description: "Page number", required: false, default: 1 },
        { name: "limit", type: "number", description: "Items per page", required: false, default: 20 },
        { name: "country", type: "string", description: "Filter by country code", required: false },
      ],
    },
    response: {
      success: {
        status: 200,
        example: {
          data: [
            {
              id: "phone_123",
              number: "+15551234567",
              country: "US",
              type: "local",
              capabilities: ["voice", "sms"],
              status: "active",
              assigned_agent_id: "agent_abc123",
            },
          ],
          meta: { total: 12, page: 1, limit: 20, pages: 1 },
        },
      },
      errors: [
        { status: 401, code: "unauthorized", message: "Invalid or missing API key" },
      ],
    },
    rateLimit: { requests: 100, window: "1 minute" },
    tags: ["phone", "list"],
  },
  {
    id: "phone_purchase",
    name: "Purchase Number",
    description: "Purchase a new phone number",
    method: "POST",
    path: "/v1/phone-numbers/purchase",
    category: "phone",
    auth: "bearer",
    params: {
      body: [
        { name: "country", type: "string", description: "Country code (ISO 3166-1 alpha-2)", required: true, example: "US" },
        { name: "area_code", type: "string", description: "Preferred area code", required: false, example: "415" },
        { name: "type", type: "string", description: "Number type", required: false, enum: ["local", "toll_free", "mobile"], default: "local" },
        { name: "capabilities", type: "array", description: "Required capabilities", required: false, example: ["voice", "sms"] },
      ],
    },
    response: {
      success: {
        status: 201,
        example: {
          id: "phone_new456",
          number: "+14155559999",
          country: "US",
          type: "local",
          capabilities: ["voice", "sms"],
          status: "active",
          monthly_cost: 1.0,
          created_at: "2024-01-20T18:00:00Z",
        },
      },
      errors: [
        { status: 400, code: "no_numbers_available", message: "No numbers available in specified area" },
        { status: 402, code: "insufficient_credits", message: "Not enough credits" },
      ],
    },
    rateLimit: { requests: 10, window: "1 minute" },
    tags: ["phone", "purchase"],
  },
  // Webhooks
  {
    id: "webhooks_list",
    name: "List Webhooks",
    description: "Retrieve all webhook endpoints",
    method: "GET",
    path: "/v1/webhooks",
    category: "webhooks",
    auth: "bearer",
    params: {
      query: [
        { name: "page", type: "number", description: "Page number", required: false, default: 1 },
        { name: "limit", type: "number", description: "Items per page", required: false, default: 20 },
      ],
    },
    response: {
      success: {
        status: 200,
        example: {
          data: [
            {
              id: "webhook_123",
              url: "https://example.com/webhooks",
              events: ["call.started", "call.ended", "call.transcription"],
              status: "active",
              created_at: "2024-01-15T10:00:00Z",
            },
          ],
          meta: { total: 3, page: 1, limit: 20, pages: 1 },
        },
      },
      errors: [
        { status: 401, code: "unauthorized", message: "Invalid or missing API key" },
      ],
    },
    rateLimit: { requests: 100, window: "1 minute" },
    tags: ["webhooks", "list"],
  },
  {
    id: "webhooks_create",
    name: "Create Webhook",
    description: "Create a new webhook endpoint",
    method: "POST",
    path: "/v1/webhooks",
    category: "webhooks",
    auth: "bearer",
    params: {
      body: [
        { name: "url", type: "string", description: "Webhook endpoint URL", required: true, example: "https://example.com/webhook" },
        { name: "events", type: "array", description: "Events to subscribe to", required: true, example: ["call.started", "call.ended"] },
        { name: "secret", type: "string", description: "Webhook signing secret", required: false, example: "whsec_abc123xyz" },
      ],
    },
    response: {
      success: {
        status: 201,
        example: {
          id: "webhook_new789",
          url: "https://example.com/webhook",
          events: ["call.started", "call.ended"],
          status: "active",
          secret: "whsec_abc123xyz",
          created_at: "2024-01-20T19:00:00Z",
        },
      },
      errors: [
        { status: 400, code: "invalid_url", message: "Invalid webhook URL" },
        { status: 400, code: "invalid_events", message: "Unknown event types specified" },
      ],
    },
    rateLimit: { requests: 20, window: "1 minute" },
    tags: ["webhooks", "create"],
  },
  // Analytics
  {
    id: "analytics_overview",
    name: "Get Analytics Overview",
    description: "Retrieve analytics overview for your account",
    method: "GET",
    path: "/v1/analytics/overview",
    category: "analytics",
    auth: "bearer",
    params: {
      query: [
        { name: "from_date", type: "string", description: "Start date (ISO 8601)", required: true, example: "2024-01-01" },
        { name: "to_date", type: "string", description: "End date (ISO 8601)", required: true, example: "2024-01-31" },
        { name: "agent_id", type: "string", description: "Filter by agent", required: false },
        { name: "granularity", type: "string", description: "Data granularity", required: false, enum: ["hour", "day", "week", "month"], default: "day" },
      ],
    },
    response: {
      success: {
        status: 200,
        example: {
          period: { from: "2024-01-01", to: "2024-01-31" },
          summary: {
            total_calls: 5420,
            total_duration: 845000,
            avg_duration: 156,
            success_rate: 0.87,
            total_cost: 542.0,
          },
          by_date: [
            { date: "2024-01-01", calls: 180, duration: 28000, cost: 18.0 },
            { date: "2024-01-02", calls: 195, duration: 31000, cost: 19.5 },
          ],
        },
      },
      errors: [
        { status: 400, code: "invalid_date_range", message: "Invalid date range" },
      ],
    },
    rateLimit: { requests: 30, window: "1 minute" },
    tags: ["analytics", "overview"],
  },
];

// API Categories
const apiCategories: APICategory[] = [
  {
    id: "agents",
    name: "Agents",
    description: "Manage AI voice agents",
    icon: Bot,
    endpoints: ["agents_list", "agents_create", "agents_get", "agents_update", "agents_delete"],
  },
  {
    id: "calls",
    name: "Calls",
    description: "Initiate and manage phone calls",
    icon: Phone,
    endpoints: ["calls_initiate", "calls_get", "calls_list", "calls_end"],
  },
  {
    id: "knowledge",
    name: "Knowledge Bases",
    description: "Manage knowledge for agents",
    icon: BookOpen,
    endpoints: ["kb_list", "kb_create", "kb_add_source"],
  },
  {
    id: "phone",
    name: "Phone Numbers",
    description: "Manage phone numbers",
    icon: Phone,
    endpoints: ["phone_list", "phone_purchase"],
  },
  {
    id: "webhooks",
    name: "Webhooks",
    description: "Configure webhook endpoints",
    icon: Webhook,
    endpoints: ["webhooks_list", "webhooks_create"],
  },
  {
    id: "analytics",
    name: "Analytics",
    description: "Access usage analytics",
    icon: BarChart3,
    endpoints: ["analytics_overview"],
  },
];

// Utility functions
const getMethodColor = (method: HttpMethod) => {
  switch (method) {
    case "GET":
      return "bg-green-500/20 text-green-400 border-green-500/30";
    case "POST":
      return "bg-blue-500/20 text-blue-400 border-blue-500/30";
    case "PUT":
      return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
    case "PATCH":
      return "bg-orange-500/20 text-orange-400 border-orange-500/30";
    case "DELETE":
      return "bg-red-500/20 text-red-400 border-red-500/30";
    default:
      return "bg-gray-500/20 text-gray-400 border-gray-500/30";
  }
};

// Components
function CodeBlock({
  code,
  language = "json",
  title,
}: {
  code: string;
  language?: string;
  title?: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-[#0d0d1a] rounded-xl border border-white/5 overflow-hidden">
      {title && (
        <div className="px-4 py-2 border-b border-white/5 flex items-center justify-between">
          <span className="text-xs font-medium text-gray-400">{title}</span>
          <button
            onClick={handleCopy}
            className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-green-400" /> : <Copy className="h-3.5 w-3.5" />}
          </button>
        </div>
      )}
      <div className="relative">
        <pre className="p-4 text-sm text-gray-300 overflow-x-auto font-mono">
          <code>{code}</code>
        </pre>
        {!title && (
          <button
            onClick={handleCopy}
            className="absolute top-2 right-2 p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-green-400" /> : <Copy className="h-3.5 w-3.5" />}
          </button>
        )}
      </div>
    </div>
  );
}

function ParameterTable({ params, title }: { params: APIParameter[]; title: string }) {
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-gray-300">{title}</h4>
      <div className="bg-white/5 rounded-xl overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-white/5">
              <th className="text-left px-4 py-2 text-xs font-medium text-gray-400 uppercase tracking-wider">
                Name
              </th>
              <th className="text-left px-4 py-2 text-xs font-medium text-gray-400 uppercase tracking-wider">
                Type
              </th>
              <th className="text-left px-4 py-2 text-xs font-medium text-gray-400 uppercase tracking-wider">
                Required
              </th>
              <th className="text-left px-4 py-2 text-xs font-medium text-gray-400 uppercase tracking-wider">
                Description
              </th>
            </tr>
          </thead>
          <tbody>
            {params.map((param, index) => (
              <tr key={index} className="border-b border-white/5 last:border-0">
                <td className="px-4 py-3">
                  <code className="text-sm text-purple-400 font-mono">{param.name}</code>
                </td>
                <td className="px-4 py-3">
                  <span className="text-xs px-2 py-1 rounded-full bg-white/10 text-gray-300">
                    {param.type}
                  </span>
                </td>
                <td className="px-4 py-3">
                  {param.required ? (
                    <span className="text-xs px-2 py-1 rounded-full bg-red-500/20 text-red-400">
                      required
                    </span>
                  ) : (
                    <span className="text-xs px-2 py-1 rounded-full bg-gray-500/20 text-gray-400">
                      optional
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-sm text-gray-400">
                  {param.description}
                  {param.enum && (
                    <div className="mt-1 text-xs text-gray-500">
                      Values: {param.enum.map((v) => `"${v}"`).join(", ")}
                    </div>
                  )}
                  {param.default !== undefined && (
                    <div className="mt-1 text-xs text-gray-500">Default: {JSON.stringify(param.default)}</div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EndpointCard({
  endpoint,
  isSelected,
  onClick,
}: {
  endpoint: APIEndpoint;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-3 rounded-xl border transition-all ${
        isSelected
          ? "bg-purple-500/10 border-purple-500/30"
          : "bg-white/[0.02] border-white/5 hover:bg-white/5 hover:border-white/10"
      }`}
    >
      <div className="flex items-center gap-3">
        <span
          className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${getMethodColor(endpoint.method)}`}
        >
          {endpoint.method}
        </span>
        <span className="text-sm font-mono text-gray-300 truncate">{endpoint.path}</span>
      </div>
      <div className="mt-1.5 text-sm text-gray-400 line-clamp-1">{endpoint.name}</div>
    </button>
  );
}

function APIPlayground({
  endpoint,
  apiKey,
}: {
  endpoint: APIEndpoint;
  apiKey: string;
}) {
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<any>(null);
  const [responseStatus, setResponseStatus] = useState<number | null>(null);
  const [responseDuration, setResponseDuration] = useState<number | null>(null);
  const [pathParams, setPathParams] = useState<Record<string, string>>({});
  const [queryParams, setQueryParams] = useState<Record<string, string>>({});
  const [bodyParams, setBodyParams] = useState<string>(
    endpoint.params?.body
      ? JSON.stringify(
          Object.fromEntries(
            endpoint.params.body.map((p) => [p.name, p.example ?? p.default ?? ""])
          ),
          null,
          2
        )
      : "{}"
  );
  const [activeTab, setActiveTab] = useState<"params" | "headers" | "body">("params");

  useEffect(() => {
    // Initialize path params
    if (endpoint.params?.path) {
      const initial: Record<string, string> = {};
      endpoint.params.path.forEach((p) => {
        initial[p.name] = String(p.example ?? "");
      });
      setPathParams(initial);
    }
    // Initialize query params
    if (endpoint.params?.query) {
      const initial: Record<string, string> = {};
      endpoint.params.query.forEach((p) => {
        if (p.required || p.example) {
          initial[p.name] = String(p.example ?? p.default ?? "");
        }
      });
      setQueryParams(initial);
    }
    // Initialize body
    if (endpoint.params?.body) {
      setBodyParams(
        JSON.stringify(
          Object.fromEntries(
            endpoint.params.body.map((p) => [p.name, p.example ?? p.default ?? ""])
          ),
          null,
          2
        )
      );
    }
  }, [endpoint]);

  const buildUrl = () => {
    let url = `https://api.bvrai.com${endpoint.path}`;
    // Replace path params
    Object.entries(pathParams).forEach(([key, value]) => {
      url = url.replace(`{${key}}`, value);
    });
    // Add query params
    const queryString = Object.entries(queryParams)
      .filter(([, v]) => v)
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
      .join("&");
    if (queryString) url += `?${queryString}`;
    return url;
  };

  const handleSend = async () => {
    setIsLoading(true);
    const startTime = Date.now();

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 500 + Math.random() * 1000));

    const duration = Date.now() - startTime;
    setResponseDuration(duration);
    setResponseStatus(endpoint.response.success.status);
    setResponse(endpoint.response.success.example);
    setIsLoading(false);
  };

  const generateCurlCommand = () => {
    let curl = `curl -X ${endpoint.method} "${buildUrl()}"`;
    curl += `\n  -H "Authorization: Bearer ${apiKey || "YOUR_API_KEY"}"`;
    curl += `\n  -H "Content-Type: application/json"`;
    if (endpoint.method !== "GET" && endpoint.method !== "DELETE" && endpoint.params?.body) {
      curl += `\n  -d '${bodyParams}'`;
    }
    return curl;
  };

  const generatePythonCode = () => {
    let code = `import requests\n\n`;
    code += `url = "${buildUrl()}"\n`;
    code += `headers = {\n`;
    code += `    "Authorization": "Bearer ${apiKey || "YOUR_API_KEY"}",\n`;
    code += `    "Content-Type": "application/json"\n`;
    code += `}\n`;
    if (endpoint.method !== "GET" && endpoint.method !== "DELETE" && endpoint.params?.body) {
      code += `data = ${bodyParams}\n\n`;
      code += `response = requests.${endpoint.method.toLowerCase()}(url, headers=headers, json=data)\n`;
    } else {
      code += `\nresponse = requests.${endpoint.method.toLowerCase()}(url, headers=headers)\n`;
    }
    code += `print(response.json())`;
    return code;
  };

  const generateNodeCode = () => {
    let code = `const response = await fetch("${buildUrl()}", {\n`;
    code += `  method: "${endpoint.method}",\n`;
    code += `  headers: {\n`;
    code += `    "Authorization": "Bearer ${apiKey || "YOUR_API_KEY"}",\n`;
    code += `    "Content-Type": "application/json"\n`;
    code += `  }`;
    if (endpoint.method !== "GET" && endpoint.method !== "DELETE" && endpoint.params?.body) {
      code += `,\n  body: JSON.stringify(${bodyParams})`;
    }
    code += `\n});\n\n`;
    code += `const data = await response.json();\nconsole.log(data);`;
    return code;
  };

  const [codeLanguage, setCodeLanguage] = useState<"curl" | "python" | "node">("curl");

  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 overflow-hidden">
      {/* URL Bar */}
      <div className="p-4 border-b border-white/5">
        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1.5 rounded-lg text-sm font-bold uppercase ${getMethodColor(
              endpoint.method
            )}`}
          >
            {endpoint.method}
          </span>
          <div className="flex-1 font-mono text-sm text-gray-300 bg-white/5 rounded-lg px-4 py-2.5 overflow-x-auto">
            {buildUrl()}
          </div>
          <button
            onClick={handleSend}
            disabled={isLoading}
            className="px-4 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90 transition-opacity flex items-center gap-2 disabled:opacity-50"
          >
            {isLoading ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            Send
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/5">
        {endpoint.params?.path && endpoint.params.path.length > 0 && (
          <button
            onClick={() => setActiveTab("params")}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "params"
                ? "text-purple-400 border-purple-500"
                : "text-gray-400 border-transparent hover:text-white"
            }`}
          >
            Path Params
          </button>
        )}
        {endpoint.params?.query && endpoint.params.query.length > 0 && (
          <button
            onClick={() => setActiveTab("params")}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "params"
                ? "text-purple-400 border-purple-500"
                : "text-gray-400 border-transparent hover:text-white"
            }`}
          >
            Query Params
          </button>
        )}
        <button
          onClick={() => setActiveTab("headers")}
          className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
            activeTab === "headers"
              ? "text-purple-400 border-purple-500"
              : "text-gray-400 border-transparent hover:text-white"
          }`}
        >
          Headers
        </button>
        {endpoint.params?.body && endpoint.params.body.length > 0 && (
          <button
            onClick={() => setActiveTab("body")}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "body"
                ? "text-purple-400 border-purple-500"
                : "text-gray-400 border-transparent hover:text-white"
            }`}
          >
            Body
          </button>
        )}
      </div>

      {/* Tab Content */}
      <div className="p-4 border-b border-white/5 max-h-64 overflow-y-auto">
        {activeTab === "params" && (
          <div className="space-y-4">
            {endpoint.params?.path && endpoint.params.path.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Path Parameters
                </h4>
                {endpoint.params.path.map((param) => (
                  <div key={param.name} className="flex items-center gap-3">
                    <label className="w-32 text-sm text-gray-400 font-mono">{param.name}</label>
                    <input
                      type="text"
                      value={pathParams[param.name] || ""}
                      onChange={(e) =>
                        setPathParams({ ...pathParams, [param.name]: e.target.value })
                      }
                      placeholder={String(param.example || "")}
                      className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm font-mono focus:outline-none focus:border-purple-500"
                    />
                  </div>
                ))}
              </div>
            )}
            {endpoint.params?.query && endpoint.params.query.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Query Parameters
                </h4>
                {endpoint.params.query.map((param) => (
                  <div key={param.name} className="flex items-center gap-3">
                    <label className="w-32 text-sm text-gray-400 font-mono">{param.name}</label>
                    {param.enum ? (
                      <select
                        value={queryParams[param.name] || ""}
                        onChange={(e) =>
                          setQueryParams({ ...queryParams, [param.name]: e.target.value })
                        }
                        className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:border-purple-500"
                      >
                        <option value="">Select...</option>
                        {param.enum.map((v) => (
                          <option key={v} value={v}>
                            {v}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="text"
                        value={queryParams[param.name] || ""}
                        onChange={(e) =>
                          setQueryParams({ ...queryParams, [param.name]: e.target.value })
                        }
                        placeholder={String(param.example || param.default || "")}
                        className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm font-mono focus:outline-none focus:border-purple-500"
                      />
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        {activeTab === "headers" && (
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <label className="w-32 text-sm text-gray-400 font-mono">Authorization</label>
              <input
                type="text"
                value={`Bearer ${apiKey || "YOUR_API_KEY"}`}
                readOnly
                className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-gray-400 text-sm font-mono"
              />
            </div>
            <div className="flex items-center gap-3">
              <label className="w-32 text-sm text-gray-400 font-mono">Content-Type</label>
              <input
                type="text"
                value="application/json"
                readOnly
                className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-gray-400 text-sm font-mono"
              />
            </div>
          </div>
        )}
        {activeTab === "body" && endpoint.params?.body && (
          <div>
            <textarea
              value={bodyParams}
              onChange={(e) => setBodyParams(e.target.value)}
              rows={10}
              className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white font-mono text-sm focus:outline-none focus:border-purple-500 resize-none"
            />
          </div>
        )}
      </div>

      {/* Response */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium text-gray-300">Response</h4>
          {responseStatus && (
            <div className="flex items-center gap-3 text-sm">
              <span
                className={`px-2 py-1 rounded ${
                  responseStatus >= 200 && responseStatus < 300
                    ? "bg-green-500/20 text-green-400"
                    : "bg-red-500/20 text-red-400"
                }`}
              >
                {responseStatus}
              </span>
              {responseDuration && <span className="text-gray-400">{responseDuration}ms</span>}
            </div>
          )}
        </div>
        <CodeBlock
          code={response ? JSON.stringify(response, null, 2) : "// Send a request to see the response"}
          language="json"
        />
      </div>

      {/* Code Examples */}
      <div className="p-4 border-t border-white/5">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium text-gray-300">Code Examples</h4>
          <div className="flex items-center bg-white/5 rounded-lg p-1">
            {(["curl", "python", "node"] as const).map((lang) => (
              <button
                key={lang}
                onClick={() => setCodeLanguage(lang)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                  codeLanguage === lang
                    ? "bg-purple-500/20 text-purple-400"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                {lang === "curl" ? "cURL" : lang === "python" ? "Python" : "Node.js"}
              </button>
            ))}
          </div>
        </div>
        <CodeBlock
          code={
            codeLanguage === "curl"
              ? generateCurlCommand()
              : codeLanguage === "python"
              ? generatePythonCode()
              : generateNodeCode()
          }
          language={codeLanguage}
        />
      </div>
    </div>
  );
}

function EndpointDocumentation({ endpoint }: { endpoint: APIEndpoint }) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <span
            className={`px-3 py-1.5 rounded-lg text-sm font-bold uppercase ${getMethodColor(
              endpoint.method
            )}`}
          >
            {endpoint.method}
          </span>
          <code className="text-lg font-mono text-gray-300">{endpoint.path}</code>
          {endpoint.deprecated && (
            <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs font-medium">
              Deprecated
            </span>
          )}
        </div>
        <h2 className="text-xl font-semibold text-white mb-2">{endpoint.name}</h2>
        <p className="text-gray-400">{endpoint.description}</p>
      </div>

      {/* Authentication */}
      <div className="bg-white/5 rounded-xl p-4">
        <div className="flex items-center gap-2 text-sm">
          <Lock className="h-4 w-4 text-yellow-400" />
          <span className="text-gray-300">Authentication required:</span>
          <span className="text-white font-medium capitalize">{endpoint.auth} Token</span>
        </div>
      </div>

      {/* Rate Limit */}
      {endpoint.rateLimit && (
        <div className="bg-blue-500/10 rounded-xl p-4 border border-blue-500/20">
          <div className="flex items-center gap-2 text-sm">
            <Clock className="h-4 w-4 text-blue-400" />
            <span className="text-gray-300">Rate Limit:</span>
            <span className="text-white font-medium">
              {endpoint.rateLimit.requests} requests per {endpoint.rateLimit.window}
            </span>
          </div>
        </div>
      )}

      {/* Parameters */}
      {endpoint.params?.path && endpoint.params.path.length > 0 && (
        <ParameterTable params={endpoint.params.path} title="Path Parameters" />
      )}
      {endpoint.params?.query && endpoint.params.query.length > 0 && (
        <ParameterTable params={endpoint.params.query} title="Query Parameters" />
      )}
      {endpoint.params?.body && endpoint.params.body.length > 0 && (
        <ParameterTable params={endpoint.params.body} title="Request Body" />
      )}

      {/* Response */}
      <div className="space-y-4">
        <h4 className="text-sm font-medium text-gray-300">Success Response ({endpoint.response.success.status})</h4>
        <CodeBlock
          code={JSON.stringify(endpoint.response.success.example, null, 2)}
          language="json"
          title="Response Example"
        />
      </div>

      {/* Errors */}
      {endpoint.response.errors.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300">Error Responses</h4>
          <div className="space-y-2">
            {endpoint.response.errors.map((error, index) => (
              <div
                key={index}
                className="flex items-center gap-3 p-3 bg-white/5 rounded-lg"
              >
                <span className="px-2 py-1 bg-red-500/20 text-red-400 rounded text-xs font-medium">
                  {error.status}
                </span>
                <code className="text-sm text-purple-400">{error.code}</code>
                <span className="text-sm text-gray-400">{error.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Main Page Component
export default function APIDocsPage() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>("agents");
  const [selectedEndpoint, setSelectedEndpoint] = useState<APIEndpoint | null>(
    apiEndpoints.find((e) => e.id === "agents_list") || null
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [viewMode, setViewMode] = useState<"docs" | "playground">("docs");

  // Filter endpoints
  const filteredEndpoints = useMemo(() => {
    if (!searchQuery) return apiEndpoints;
    const query = searchQuery.toLowerCase();
    return apiEndpoints.filter(
      (e) =>
        e.name.toLowerCase().includes(query) ||
        e.path.toLowerCase().includes(query) ||
        e.description.toLowerCase().includes(query)
    );
  }, [searchQuery]);

  // Get endpoints for selected category
  const categoryEndpoints = useMemo(() => {
    if (!selectedCategory) return filteredEndpoints;
    const category = apiCategories.find((c) => c.id === selectedCategory);
    if (!category) return filteredEndpoints;
    return filteredEndpoints.filter((e) => category.endpoints.includes(e.id));
  }, [selectedCategory, filteredEndpoints]);

  return (
    <DashboardLayout>
      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar */}
        <div className="w-80 border-r border-white/5 bg-[#0d0d1a]/50 flex flex-col">
          {/* Search */}
          <div className="p-4 border-b border-white/5">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
              <input
                type="text"
                placeholder="Search endpoints..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>

          {/* Categories */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {apiCategories.map((category) => {
              const Icon = category.icon;
              const isSelected = selectedCategory === category.id;
              const endpointCount = category.endpoints.length;

              return (
                <div key={category.id}>
                  <button
                    onClick={() =>
                      setSelectedCategory(isSelected ? null : category.id)
                    }
                    className={`w-full flex items-center justify-between p-3 rounded-xl transition-all ${
                      isSelected
                        ? "bg-purple-500/10 text-purple-400"
                        : "text-gray-300 hover:bg-white/5"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <Icon className="h-5 w-5" />
                      <span className="font-medium">{category.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500">{endpointCount}</span>
                      {isSelected ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronRight className="h-4 w-4" />
                      )}
                    </div>
                  </button>
                  {isSelected && (
                    <div className="mt-2 ml-4 space-y-1">
                      {categoryEndpoints.map((endpoint) => (
                        <EndpointCard
                          key={endpoint.id}
                          endpoint={endpoint}
                          isSelected={selectedEndpoint?.id === endpoint.id}
                          onClick={() => setSelectedEndpoint(endpoint)}
                        />
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* API Key */}
          <div className="p-4 border-t border-white/5">
            <label className="block text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">
              API Key
            </label>
            <div className="relative">
              <Key className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
              <input
                type={showApiKey ? "text" : "password"}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your API key"
                className="w-full pl-10 pr-10 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 text-sm"
              />
              <button
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
              >
                {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Toolbar */}
          <div className="p-4 border-b border-white/5 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h1 className="text-xl font-semibold text-white">API Reference</h1>
              <span className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-xs font-medium">
                v1.0
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center bg-white/5 rounded-xl p-1">
                <button
                  onClick={() => setViewMode("docs")}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors flex items-center gap-2 ${
                    viewMode === "docs"
                      ? "bg-purple-500/20 text-purple-400"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  <BookOpen className="h-4 w-4" />
                  Documentation
                </button>
                <button
                  onClick={() => setViewMode("playground")}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors flex items-center gap-2 ${
                    viewMode === "playground"
                      ? "bg-purple-500/20 text-purple-400"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  <Terminal className="h-4 w-4" />
                  Playground
                </button>
              </div>
              <button className="px-4 py-2 bg-white/5 border border-white/10 rounded-xl text-gray-300 text-sm font-medium hover:bg-white/10 transition-colors flex items-center gap-2">
                <Download className="h-4 w-4" />
                OpenAPI Spec
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6">
            {selectedEndpoint ? (
              viewMode === "docs" ? (
                <EndpointDocumentation endpoint={selectedEndpoint} />
              ) : (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-xl font-semibold text-white mb-2">
                      {selectedEndpoint.name}
                    </h2>
                    <p className="text-gray-400">{selectedEndpoint.description}</p>
                  </div>
                  <APIPlayground endpoint={selectedEndpoint} apiKey={apiKey} />
                </div>
              )
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mb-4">
                  <Code className="h-8 w-8 text-purple-400" />
                </div>
                <h2 className="text-xl font-semibold text-white mb-2">
                  Welcome to the API Reference
                </h2>
                <p className="text-gray-400 max-w-md mb-6">
                  Select an endpoint from the sidebar to view its documentation or test it in the
                  playground.
                </p>
                <div className="flex items-center gap-3">
                  <button className="px-4 py-2 bg-purple-500/20 text-purple-400 rounded-xl text-sm font-medium hover:bg-purple-500/30 transition-colors flex items-center gap-2">
                    <BookOpen className="h-4 w-4" />
                    Getting Started
                  </button>
                  <button className="px-4 py-2 bg-white/5 text-gray-300 rounded-xl text-sm font-medium hover:bg-white/10 transition-colors flex items-center gap-2">
                    <ExternalLink className="h-4 w-4" />
                    View Full Docs
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
