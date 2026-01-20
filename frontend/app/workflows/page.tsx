"use client";

import React, { useState, useMemo, useCallback, useRef } from "react";
import { DashboardLayout } from "@/components/dashboard-layout";
import {
  Plus,
  Search,
  Filter,
  MoreVertical,
  Play,
  Pause,
  Copy,
  Trash2,
  Edit,
  Eye,
  Settings,
  Zap,
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  PhoneMissed,
  MessageSquare,
  Mail,
  Calendar,
  Clock,
  Webhook,
  Bot,
  Users,
  Database,
  GitBranch,
  ArrowRight,
  ArrowDown,
  ArrowLeft,
  CheckCircle,
  XCircle,
  AlertCircle,
  Loader2,
  ChevronRight,
  ChevronDown,
  ChevronLeft,
  X,
  Check,
  Sparkles,
  Target,
  Repeat,
  Timer,
  Bell,
  Send,
  FileText,
  Tag,
  DollarSign,
  Globe,
  Link,
  Code,
  PlusCircle,
  MinusCircle,
  Move,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  Undo,
  Redo,
  Save,
  Download,
  Upload,
  Share2,
  History,
  TestTube,
  Rocket,
  Lightbulb,
  Workflow,
  Box,
  Circle,
  Square,
  Diamond,
  Triangle,
  Hexagon,
  GripVertical,
  LayoutGrid,
  List,
  Star,
  StarOff,
  Folder,
  FolderOpen,
  Activity,
  TrendingUp,
  BarChart3,
  PieChart,
  RefreshCw,
  ExternalLink,
  Info,
  HelpCircle,
  Layers,
  MousePointer2,
  Hand,
  Crosshair,
} from "lucide-react";

// Types
interface WorkflowNode {
  id: string;
  type: "trigger" | "action" | "condition" | "delay" | "loop";
  subtype: string;
  name: string;
  description: string;
  icon: string;
  config: Record<string, any>;
  position: { x: number; y: number };
  connections: string[];
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  status: "active" | "inactive" | "draft";
  trigger: string;
  nodes: WorkflowNode[];
  createdAt: string;
  updatedAt: string;
  lastRunAt: string | null;
  runCount: number;
  successCount: number;
  failureCount: number;
  isFavorite: boolean;
  tags: string[];
  folder: string | null;
}

interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
  complexity: "simple" | "medium" | "advanced";
  popularity: number;
}

interface NodeCategory {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  nodes: NodeType[];
}

interface NodeType {
  id: string;
  name: string;
  description: string;
  icon: string;
  type: "trigger" | "action" | "condition" | "delay" | "loop";
  category: string;
  configFields: ConfigField[];
}

interface ConfigField {
  id: string;
  name: string;
  type: "text" | "select" | "number" | "boolean" | "textarea" | "json";
  required: boolean;
  options?: { value: string; label: string }[];
  defaultValue?: any;
  placeholder?: string;
}

// Node type icons mapping
const nodeIcons: Record<string, React.ReactNode> = {
  call_started: <PhoneIncoming className="w-5 h-5" />,
  call_ended: <Phone className="w-5 h-5" />,
  call_missed: <PhoneMissed className="w-5 h-5" />,
  webhook: <Webhook className="w-5 h-5" />,
  schedule: <Clock className="w-5 h-5" />,
  manual: <MousePointer2 className="w-5 h-5" />,
  send_sms: <MessageSquare className="w-5 h-5" />,
  send_email: <Mail className="w-5 h-5" />,
  update_crm: <Database className="w-5 h-5" />,
  create_task: <CheckCircle className="w-5 h-5" />,
  schedule_meeting: <Calendar className="w-5 h-5" />,
  transfer_call: <PhoneOutgoing className="w-5 h-5" />,
  send_webhook: <Webhook className="w-5 h-5" />,
  notify_slack: <Bell className="w-5 h-5" />,
  add_tag: <Tag className="w-5 h-5" />,
  if_else: <GitBranch className="w-5 h-5" />,
  switch: <GitBranch className="w-5 h-5" />,
  wait: <Timer className="w-5 h-5" />,
  delay: <Clock className="w-5 h-5" />,
  for_each: <Repeat className="w-5 h-5" />,
  while: <Repeat className="w-5 h-5" />,
  ai_response: <Bot className="w-5 h-5" />,
  http_request: <Globe className="w-5 h-5" />,
  run_script: <Code className="w-5 h-5" />,
};

// Node categories
const nodeCategories: NodeCategory[] = [
  {
    id: "triggers",
    name: "Triggers",
    description: "Events that start your workflow",
    icon: <Zap className="w-5 h-5" />,
    color: "yellow",
    nodes: [
      {
        id: "call_started",
        name: "Call Started",
        description: "Triggers when a call begins",
        icon: "call_started",
        type: "trigger",
        category: "triggers",
        configFields: [
          {
            id: "direction",
            name: "Call Direction",
            type: "select",
            required: false,
            options: [
              { value: "any", label: "Any" },
              { value: "inbound", label: "Inbound Only" },
              { value: "outbound", label: "Outbound Only" },
            ],
            defaultValue: "any",
          },
          {
            id: "agent",
            name: "Specific Agent",
            type: "select",
            required: false,
            options: [
              { value: "any", label: "Any Agent" },
              { value: "agent_1", label: "Sales Assistant" },
              { value: "agent_2", label: "Support Bot" },
            ],
            defaultValue: "any",
          },
        ],
      },
      {
        id: "call_ended",
        name: "Call Ended",
        description: "Triggers when a call ends",
        icon: "call_ended",
        type: "trigger",
        category: "triggers",
        configFields: [
          {
            id: "outcome",
            name: "Call Outcome",
            type: "select",
            required: false,
            options: [
              { value: "any", label: "Any Outcome" },
              { value: "completed", label: "Completed" },
              { value: "voicemail", label: "Voicemail" },
              { value: "no_answer", label: "No Answer" },
            ],
            defaultValue: "any",
          },
        ],
      },
      {
        id: "webhook",
        name: "Webhook",
        description: "Triggers from external webhook",
        icon: "webhook",
        type: "trigger",
        category: "triggers",
        configFields: [
          {
            id: "method",
            name: "HTTP Method",
            type: "select",
            required: true,
            options: [
              { value: "POST", label: "POST" },
              { value: "GET", label: "GET" },
              { value: "PUT", label: "PUT" },
            ],
            defaultValue: "POST",
          },
        ],
      },
      {
        id: "schedule",
        name: "Schedule",
        description: "Triggers on a schedule",
        icon: "schedule",
        type: "trigger",
        category: "triggers",
        configFields: [
          {
            id: "frequency",
            name: "Frequency",
            type: "select",
            required: true,
            options: [
              { value: "hourly", label: "Every Hour" },
              { value: "daily", label: "Daily" },
              { value: "weekly", label: "Weekly" },
              { value: "monthly", label: "Monthly" },
              { value: "cron", label: "Custom (Cron)" },
            ],
            defaultValue: "daily",
          },
          {
            id: "time",
            name: "Time",
            type: "text",
            required: false,
            placeholder: "09:00",
          },
        ],
      },
      {
        id: "manual",
        name: "Manual Trigger",
        description: "Run manually or via API",
        icon: "manual",
        type: "trigger",
        category: "triggers",
        configFields: [],
      },
    ],
  },
  {
    id: "actions",
    name: "Actions",
    description: "Actions to perform",
    icon: <Play className="w-5 h-5" />,
    color: "blue",
    nodes: [
      {
        id: "send_sms",
        name: "Send SMS",
        description: "Send an SMS message",
        icon: "send_sms",
        type: "action",
        category: "actions",
        configFields: [
          {
            id: "to",
            name: "To Number",
            type: "text",
            required: true,
            placeholder: "{{caller_phone}}",
          },
          {
            id: "message",
            name: "Message",
            type: "textarea",
            required: true,
            placeholder: "Enter your message...",
          },
        ],
      },
      {
        id: "send_email",
        name: "Send Email",
        description: "Send an email notification",
        icon: "send_email",
        type: "action",
        category: "actions",
        configFields: [
          {
            id: "to",
            name: "To Email",
            type: "text",
            required: true,
            placeholder: "{{caller_email}}",
          },
          {
            id: "subject",
            name: "Subject",
            type: "text",
            required: true,
            placeholder: "Email subject...",
          },
          {
            id: "body",
            name: "Body",
            type: "textarea",
            required: true,
            placeholder: "Email content...",
          },
        ],
      },
      {
        id: "update_crm",
        name: "Update CRM",
        description: "Update a record in your CRM",
        icon: "update_crm",
        type: "action",
        category: "actions",
        configFields: [
          {
            id: "crm",
            name: "CRM",
            type: "select",
            required: true,
            options: [
              { value: "salesforce", label: "Salesforce" },
              { value: "hubspot", label: "HubSpot" },
              { value: "pipedrive", label: "Pipedrive" },
            ],
          },
          {
            id: "action",
            name: "Action",
            type: "select",
            required: true,
            options: [
              { value: "create", label: "Create Record" },
              { value: "update", label: "Update Record" },
              { value: "log", label: "Log Activity" },
            ],
          },
        ],
      },
      {
        id: "schedule_meeting",
        name: "Schedule Meeting",
        description: "Create a calendar event",
        icon: "schedule_meeting",
        type: "action",
        category: "actions",
        configFields: [
          {
            id: "calendar",
            name: "Calendar",
            type: "select",
            required: true,
            options: [
              { value: "google", label: "Google Calendar" },
              { value: "outlook", label: "Outlook" },
              { value: "calendly", label: "Calendly" },
            ],
          },
          {
            id: "title",
            name: "Event Title",
            type: "text",
            required: true,
            placeholder: "Meeting title...",
          },
          {
            id: "duration",
            name: "Duration (minutes)",
            type: "number",
            required: true,
            defaultValue: 30,
          },
        ],
      },
      {
        id: "transfer_call",
        name: "Transfer Call",
        description: "Transfer to another number or agent",
        icon: "transfer_call",
        type: "action",
        category: "actions",
        configFields: [
          {
            id: "target",
            name: "Transfer To",
            type: "select",
            required: true,
            options: [
              { value: "agent", label: "Another Agent" },
              { value: "number", label: "Phone Number" },
              { value: "queue", label: "Call Queue" },
            ],
          },
          {
            id: "value",
            name: "Target Value",
            type: "text",
            required: true,
            placeholder: "Agent ID or phone number...",
          },
        ],
      },
      {
        id: "send_webhook",
        name: "Send Webhook",
        description: "Send data to external URL",
        icon: "send_webhook",
        type: "action",
        category: "actions",
        configFields: [
          {
            id: "url",
            name: "Webhook URL",
            type: "text",
            required: true,
            placeholder: "https://...",
          },
          {
            id: "method",
            name: "Method",
            type: "select",
            required: true,
            options: [
              { value: "POST", label: "POST" },
              { value: "PUT", label: "PUT" },
              { value: "PATCH", label: "PATCH" },
            ],
            defaultValue: "POST",
          },
          {
            id: "payload",
            name: "Payload",
            type: "json",
            required: false,
            placeholder: '{"key": "value"}',
          },
        ],
      },
      {
        id: "notify_slack",
        name: "Notify Slack",
        description: "Send a Slack notification",
        icon: "notify_slack",
        type: "action",
        category: "actions",
        configFields: [
          {
            id: "channel",
            name: "Channel",
            type: "text",
            required: true,
            placeholder: "#sales",
          },
          {
            id: "message",
            name: "Message",
            type: "textarea",
            required: true,
            placeholder: "Slack message...",
          },
        ],
      },
      {
        id: "add_tag",
        name: "Add Tag",
        description: "Add a tag to the contact/call",
        icon: "add_tag",
        type: "action",
        category: "actions",
        configFields: [
          {
            id: "tag",
            name: "Tag Name",
            type: "text",
            required: true,
            placeholder: "hot-lead",
          },
        ],
      },
      {
        id: "ai_response",
        name: "AI Response",
        description: "Generate AI response",
        icon: "ai_response",
        type: "action",
        category: "actions",
        configFields: [
          {
            id: "prompt",
            name: "Prompt",
            type: "textarea",
            required: true,
            placeholder: "Analyze the call and...",
          },
          {
            id: "model",
            name: "Model",
            type: "select",
            required: true,
            options: [
              { value: "gpt-4", label: "GPT-4" },
              { value: "claude", label: "Claude" },
              { value: "gemini", label: "Gemini" },
            ],
            defaultValue: "gpt-4",
          },
        ],
      },
      {
        id: "http_request",
        name: "HTTP Request",
        description: "Make an HTTP API call",
        icon: "http_request",
        type: "action",
        category: "actions",
        configFields: [
          {
            id: "url",
            name: "URL",
            type: "text",
            required: true,
            placeholder: "https://api.example.com/...",
          },
          {
            id: "method",
            name: "Method",
            type: "select",
            required: true,
            options: [
              { value: "GET", label: "GET" },
              { value: "POST", label: "POST" },
              { value: "PUT", label: "PUT" },
              { value: "DELETE", label: "DELETE" },
            ],
          },
          {
            id: "headers",
            name: "Headers",
            type: "json",
            required: false,
          },
          {
            id: "body",
            name: "Body",
            type: "json",
            required: false,
          },
        ],
      },
    ],
  },
  {
    id: "conditions",
    name: "Conditions",
    description: "Add logic to your workflow",
    icon: <GitBranch className="w-5 h-5" />,
    color: "purple",
    nodes: [
      {
        id: "if_else",
        name: "If / Else",
        description: "Branch based on conditions",
        icon: "if_else",
        type: "condition",
        category: "conditions",
        configFields: [
          {
            id: "field",
            name: "Field",
            type: "text",
            required: true,
            placeholder: "{{call_duration}}",
          },
          {
            id: "operator",
            name: "Operator",
            type: "select",
            required: true,
            options: [
              { value: "equals", label: "Equals" },
              { value: "not_equals", label: "Not Equals" },
              { value: "greater_than", label: "Greater Than" },
              { value: "less_than", label: "Less Than" },
              { value: "contains", label: "Contains" },
              { value: "not_contains", label: "Does Not Contain" },
              { value: "is_empty", label: "Is Empty" },
              { value: "is_not_empty", label: "Is Not Empty" },
            ],
          },
          {
            id: "value",
            name: "Value",
            type: "text",
            required: false,
            placeholder: "Compare value...",
          },
        ],
      },
      {
        id: "switch",
        name: "Switch",
        description: "Multiple condition branches",
        icon: "switch",
        type: "condition",
        category: "conditions",
        configFields: [
          {
            id: "field",
            name: "Field",
            type: "text",
            required: true,
            placeholder: "{{call_outcome}}",
          },
          {
            id: "cases",
            name: "Cases",
            type: "json",
            required: true,
            placeholder: '["completed", "voicemail", "missed"]',
          },
        ],
      },
    ],
  },
  {
    id: "timing",
    name: "Timing",
    description: "Control workflow timing",
    icon: <Timer className="w-5 h-5" />,
    color: "green",
    nodes: [
      {
        id: "wait",
        name: "Wait",
        description: "Wait for a specified time",
        icon: "wait",
        type: "delay",
        category: "timing",
        configFields: [
          {
            id: "duration",
            name: "Duration",
            type: "number",
            required: true,
            defaultValue: 5,
          },
          {
            id: "unit",
            name: "Unit",
            type: "select",
            required: true,
            options: [
              { value: "seconds", label: "Seconds" },
              { value: "minutes", label: "Minutes" },
              { value: "hours", label: "Hours" },
              { value: "days", label: "Days" },
            ],
            defaultValue: "minutes",
          },
        ],
      },
      {
        id: "delay",
        name: "Delay Until",
        description: "Wait until a specific time",
        icon: "delay",
        type: "delay",
        category: "timing",
        configFields: [
          {
            id: "time",
            name: "Time",
            type: "text",
            required: true,
            placeholder: "09:00",
          },
          {
            id: "timezone",
            name: "Timezone",
            type: "select",
            required: true,
            options: [
              { value: "UTC", label: "UTC" },
              { value: "America/New_York", label: "Eastern" },
              { value: "America/Los_Angeles", label: "Pacific" },
              { value: "Europe/London", label: "London" },
            ],
            defaultValue: "UTC",
          },
        ],
      },
    ],
  },
  {
    id: "loops",
    name: "Loops",
    description: "Iterate over data",
    icon: <Repeat className="w-5 h-5" />,
    color: "orange",
    nodes: [
      {
        id: "for_each",
        name: "For Each",
        description: "Loop through a list",
        icon: "for_each",
        type: "loop",
        category: "loops",
        configFields: [
          {
            id: "items",
            name: "Items",
            type: "text",
            required: true,
            placeholder: "{{participants}}",
          },
        ],
      },
      {
        id: "while",
        name: "While",
        description: "Loop while condition is true",
        icon: "while",
        type: "loop",
        category: "loops",
        configFields: [
          {
            id: "condition",
            name: "Condition",
            type: "text",
            required: true,
            placeholder: "{{retry_count}} < 3",
          },
        ],
      },
    ],
  },
];

// Workflow templates
const workflowTemplates: WorkflowTemplate[] = [
  {
    id: "tmpl_1",
    name: "Post-Call Summary Email",
    description: "Send an email summary after each call ends",
    category: "Communication",
    icon: "send_email",
    complexity: "simple",
    popularity: 95,
  },
  {
    id: "tmpl_2",
    name: "CRM Auto-Update",
    description: "Update CRM records automatically after calls",
    category: "CRM",
    icon: "update_crm",
    complexity: "simple",
    popularity: 92,
  },
  {
    id: "tmpl_3",
    name: "Lead Qualification Flow",
    description: "Qualify leads based on call outcomes",
    category: "Sales",
    icon: "if_else",
    complexity: "medium",
    popularity: 88,
  },
  {
    id: "tmpl_4",
    name: "Appointment Confirmation",
    description: "Send SMS confirmations for scheduled appointments",
    category: "Scheduling",
    icon: "send_sms",
    complexity: "simple",
    popularity: 85,
  },
  {
    id: "tmpl_5",
    name: "Escalation Workflow",
    description: "Escalate calls based on sentiment or keywords",
    category: "Support",
    icon: "transfer_call",
    complexity: "medium",
    popularity: 82,
  },
  {
    id: "tmpl_6",
    name: "Multi-Channel Follow-up",
    description: "Follow up via email, SMS, and Slack",
    category: "Communication",
    icon: "notify_slack",
    complexity: "advanced",
    popularity: 78,
  },
  {
    id: "tmpl_7",
    name: "Scheduled Report",
    description: "Generate and send daily/weekly reports",
    category: "Analytics",
    icon: "schedule",
    complexity: "medium",
    popularity: 75,
  },
  {
    id: "tmpl_8",
    name: "AI Call Analysis",
    description: "Analyze calls with AI and categorize",
    category: "AI",
    icon: "ai_response",
    complexity: "advanced",
    popularity: 72,
  },
];

// Mock workflows
const mockWorkflows: Workflow[] = [
  {
    id: "wf_1",
    name: "Post-Call Email Summary",
    description: "Send email summary after each completed call",
    status: "active",
    trigger: "call_ended",
    nodes: [],
    createdAt: "2024-01-10T10:00:00Z",
    updatedAt: "2024-01-18T14:30:00Z",
    lastRunAt: "2024-01-20T09:15:00Z",
    runCount: 1456,
    successCount: 1432,
    failureCount: 24,
    isFavorite: true,
    tags: ["email", "summary", "post-call"],
    folder: "Communication",
  },
  {
    id: "wf_2",
    name: "CRM Lead Update",
    description: "Update Salesforce with call data",
    status: "active",
    trigger: "call_ended",
    nodes: [],
    createdAt: "2024-01-08T09:00:00Z",
    updatedAt: "2024-01-17T11:20:00Z",
    lastRunAt: "2024-01-20T09:12:00Z",
    runCount: 2341,
    successCount: 2298,
    failureCount: 43,
    isFavorite: true,
    tags: ["crm", "salesforce", "leads"],
    folder: "CRM",
  },
  {
    id: "wf_3",
    name: "Hot Lead Notification",
    description: "Slack notification for qualified leads",
    status: "active",
    trigger: "call_ended",
    nodes: [],
    createdAt: "2024-01-12T08:00:00Z",
    updatedAt: "2024-01-19T16:45:00Z",
    lastRunAt: "2024-01-20T08:45:00Z",
    runCount: 234,
    successCount: 231,
    failureCount: 3,
    isFavorite: false,
    tags: ["slack", "leads", "notification"],
    folder: "Sales",
  },
  {
    id: "wf_4",
    name: "Missed Call Follow-up",
    description: "Send SMS when calls are missed",
    status: "active",
    trigger: "call_missed",
    nodes: [],
    createdAt: "2024-01-14T11:00:00Z",
    updatedAt: "2024-01-18T09:30:00Z",
    lastRunAt: "2024-01-19T22:15:00Z",
    runCount: 89,
    successCount: 87,
    failureCount: 2,
    isFavorite: false,
    tags: ["sms", "follow-up", "missed"],
    folder: "Communication",
  },
  {
    id: "wf_5",
    name: "Daily Analytics Report",
    description: "Generate and email daily call analytics",
    status: "active",
    trigger: "schedule",
    nodes: [],
    createdAt: "2024-01-05T14:00:00Z",
    updatedAt: "2024-01-15T10:15:00Z",
    lastRunAt: "2024-01-20T06:00:00Z",
    runCount: 15,
    successCount: 15,
    failureCount: 0,
    isFavorite: false,
    tags: ["analytics", "report", "daily"],
    folder: "Analytics",
  },
  {
    id: "wf_6",
    name: "Appointment Reminder",
    description: "Send reminder 24h before appointments",
    status: "inactive",
    trigger: "schedule",
    nodes: [],
    createdAt: "2024-01-11T13:00:00Z",
    updatedAt: "2024-01-16T08:45:00Z",
    lastRunAt: null,
    runCount: 0,
    successCount: 0,
    failureCount: 0,
    isFavorite: false,
    tags: ["appointment", "reminder"],
    folder: "Scheduling",
  },
  {
    id: "wf_7",
    name: "Support Escalation",
    description: "Escalate negative sentiment calls",
    status: "draft",
    trigger: "call_ended",
    nodes: [],
    createdAt: "2024-01-18T15:00:00Z",
    updatedAt: "2024-01-19T17:30:00Z",
    lastRunAt: null,
    runCount: 0,
    successCount: 0,
    failureCount: 0,
    isFavorite: false,
    tags: ["support", "escalation", "sentiment"],
    folder: "Support",
  },
];

// Status badge component
function StatusBadge({ status }: { status: Workflow["status"] }) {
  const styles = {
    active: "bg-green-500/10 text-green-400 border-green-500/20",
    inactive: "bg-gray-500/10 text-gray-400 border-gray-500/20",
    draft: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  };

  const icons = {
    active: <CheckCircle className="w-3 h-3" />,
    inactive: <Pause className="w-3 h-3" />,
    draft: <Edit className="w-3 h-3" />,
  };

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${styles[status]}`}
    >
      {icons[status]}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

// Trigger badge component
function TriggerBadge({ trigger }: { trigger: string }) {
  const triggerConfig: Record<
    string,
    { label: string; icon: React.ReactNode; color: string }
  > = {
    call_started: {
      label: "Call Started",
      icon: <PhoneIncoming className="w-3 h-3" />,
      color: "blue",
    },
    call_ended: {
      label: "Call Ended",
      icon: <Phone className="w-3 h-3" />,
      color: "purple",
    },
    call_missed: {
      label: "Call Missed",
      icon: <PhoneMissed className="w-3 h-3" />,
      color: "red",
    },
    webhook: {
      label: "Webhook",
      icon: <Webhook className="w-3 h-3" />,
      color: "cyan",
    },
    schedule: {
      label: "Schedule",
      icon: <Clock className="w-3 h-3" />,
      color: "green",
    },
    manual: {
      label: "Manual",
      icon: <MousePointer2 className="w-3 h-3" />,
      color: "gray",
    },
  };

  const config = triggerConfig[trigger] || {
    label: trigger,
    icon: <Zap className="w-3 h-3" />,
    color: "gray",
  };

  const colorStyles: Record<string, string> = {
    blue: "bg-blue-500/10 text-blue-400",
    purple: "bg-purple-500/10 text-purple-400",
    red: "bg-red-500/10 text-red-400",
    cyan: "bg-cyan-500/10 text-cyan-400",
    green: "bg-green-500/10 text-green-400",
    gray: "bg-gray-500/10 text-gray-400",
  };

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded ${colorStyles[config.color]}`}
    >
      {config.icon}
      {config.label}
    </span>
  );
}

// Stats card component
function StatsCard({
  title,
  value,
  icon,
  color,
  change,
}: {
  title: string;
  value: string;
  icon: React.ReactNode;
  color: string;
  change?: { value: string; type: "up" | "down" };
}) {
  const colorStyles: Record<string, string> = {
    purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    blue: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    green: "bg-green-500/10 text-green-400 border-green-500/20",
    red: "bg-red-500/10 text-red-400 border-red-500/20",
  };

  return (
    <div className="bg-[#1a1a2e]/50 backdrop-blur-sm rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-gray-400">{title}</span>
        <div className={`p-2 rounded-lg border ${colorStyles[color]}`}>
          {icon}
        </div>
      </div>
      <div className="flex items-end justify-between">
        <span className="text-2xl font-bold text-white">{value}</span>
        {change && (
          <span
            className={`text-xs ${
              change.type === "up" ? "text-green-400" : "text-red-400"
            }`}
          >
            {change.type === "up" ? "+" : "-"}
            {change.value}
          </span>
        )}
      </div>
    </div>
  );
}

// Workflow card component
function WorkflowCard({
  workflow,
  onEdit,
  onDuplicate,
  onDelete,
  onToggle,
  onToggleFavorite,
}: {
  workflow: Workflow;
  onEdit: (wf: Workflow) => void;
  onDuplicate: (wf: Workflow) => void;
  onDelete: (id: string) => void;
  onToggle: (id: string) => void;
  onToggleFavorite: (id: string) => void;
}) {
  const [showMenu, setShowMenu] = useState(false);
  const successRate =
    workflow.runCount > 0
      ? Math.round((workflow.successCount / workflow.runCount) * 100)
      : 0;

  return (
    <div className="group bg-[#1a1a2e]/30 hover:bg-[#1a1a2e]/60 rounded-xl border border-white/5 hover:border-white/10 p-5 transition-all">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center flex-shrink-0">
            <Workflow className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-white group-hover:text-purple-400 transition-colors">
                {workflow.name}
              </h3>
              <button
                onClick={() => onToggleFavorite(workflow.id)}
                className="opacity-0 group-hover:opacity-100 transition-opacity"
              >
                {workflow.isFavorite ? (
                  <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                ) : (
                  <StarOff className="w-4 h-4 text-gray-500 hover:text-yellow-400" />
                )}
              </button>
            </div>
            <p className="text-sm text-gray-400 mt-0.5">{workflow.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={workflow.status} />
          <div className="relative">
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="p-1.5 hover:bg-white/10 rounded-lg transition-colors"
            >
              <MoreVertical className="w-4 h-4 text-gray-400" />
            </button>
            {showMenu && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setShowMenu(false)}
                />
                <div className="absolute right-0 top-full mt-1 w-44 bg-[#1a1a2e] border border-white/10 rounded-xl shadow-xl z-20 py-1 overflow-hidden">
                  <button
                    onClick={() => {
                      onEdit(workflow);
                      setShowMenu(false);
                    }}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10"
                  >
                    <Edit className="w-4 h-4" />
                    Edit
                  </button>
                  <button
                    onClick={() => {
                      onDuplicate(workflow);
                      setShowMenu(false);
                    }}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10"
                  >
                    <Copy className="w-4 h-4" />
                    Duplicate
                  </button>
                  <button
                    onClick={() => {
                      onToggle(workflow.id);
                      setShowMenu(false);
                    }}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10"
                  >
                    {workflow.status === "active" ? (
                      <>
                        <Pause className="w-4 h-4" />
                        Deactivate
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Activate
                      </>
                    )}
                  </button>
                  <button className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10">
                    <History className="w-4 h-4" />
                    View History
                  </button>
                  <button className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10">
                    <TestTube className="w-4 h-4" />
                    Test Run
                  </button>
                  <div className="border-t border-white/10 my-1" />
                  <button
                    onClick={() => {
                      onDelete(workflow.id);
                      setShowMenu(false);
                    }}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-red-400 hover:bg-red-500/10"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4 mb-4">
        <TriggerBadge trigger={workflow.trigger} />
        {workflow.folder && (
          <span className="flex items-center gap-1 text-xs text-gray-500">
            <Folder className="w-3 h-3" />
            {workflow.folder}
          </span>
        )}
      </div>

      {workflow.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-4">
          {workflow.tags.map((tag) => (
            <span
              key={tag}
              className="px-2 py-0.5 text-xs bg-white/5 text-gray-400 rounded"
            >
              #{tag}
            </span>
          ))}
        </div>
      )}

      <div className="grid grid-cols-4 gap-4 pt-4 border-t border-white/5">
        <div>
          <span className="block text-xs text-gray-500 mb-1">Total Runs</span>
          <span className="text-sm font-medium text-white">
            {workflow.runCount.toLocaleString()}
          </span>
        </div>
        <div>
          <span className="block text-xs text-gray-500 mb-1">Success</span>
          <span className="text-sm font-medium text-green-400">
            {workflow.successCount.toLocaleString()}
          </span>
        </div>
        <div>
          <span className="block text-xs text-gray-500 mb-1">Failed</span>
          <span className="text-sm font-medium text-red-400">
            {workflow.failureCount.toLocaleString()}
          </span>
        </div>
        <div>
          <span className="block text-xs text-gray-500 mb-1">Success Rate</span>
          <span
            className={`text-sm font-medium ${
              successRate >= 95
                ? "text-green-400"
                : successRate >= 80
                ? "text-yellow-400"
                : "text-red-400"
            }`}
          >
            {successRate}%
          </span>
        </div>
      </div>

      {workflow.lastRunAt && (
        <div className="mt-3 text-xs text-gray-500">
          Last run: {new Date(workflow.lastRunAt).toLocaleString()}
        </div>
      )}
    </div>
  );
}

// Template card component
function TemplateCard({
  template,
  onUse,
}: {
  template: WorkflowTemplate;
  onUse: (template: WorkflowTemplate) => void;
}) {
  const complexityColors = {
    simple: "bg-green-500/10 text-green-400",
    medium: "bg-yellow-500/10 text-yellow-400",
    advanced: "bg-red-500/10 text-red-400",
  };

  return (
    <div className="group bg-[#1a1a2e]/30 hover:bg-[#1a1a2e]/60 rounded-xl border border-white/5 hover:border-white/10 p-4 transition-all">
      <div className="flex items-start gap-3 mb-3">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/20 flex items-center justify-center">
          {nodeIcons[template.icon] || <Workflow className="w-5 h-5 text-purple-400" />}
        </div>
        <div className="flex-1">
          <h4 className="font-medium text-white group-hover:text-purple-400 transition-colors">
            {template.name}
          </h4>
          <p className="text-xs text-gray-400 mt-0.5">{template.description}</p>
        </div>
      </div>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 text-xs bg-white/5 text-gray-400 rounded">
            {template.category}
          </span>
          <span
            className={`px-2 py-0.5 text-xs rounded ${complexityColors[template.complexity]}`}
          >
            {template.complexity}
          </span>
        </div>
        <button
          onClick={() => onUse(template)}
          className="px-3 py-1.5 text-xs bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded-lg transition-colors"
        >
          Use Template
        </button>
      </div>
    </div>
  );
}

// Create workflow dialog
function CreateWorkflowDialog({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const [step, setStep] = useState(1);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [selectedTrigger, setSelectedTrigger] = useState<string | null>(null);
  const [selectedTemplate, setSelectedTemplate] =
    useState<WorkflowTemplate | null>(null);
  const [createMode, setCreateMode] = useState<"scratch" | "template">("scratch");

  if (!isOpen) return null;

  const triggers = nodeCategories.find((c) => c.id === "triggers")?.nodes || [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-[#0f0f1a] border border-white/10 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <div>
            <h2 className="text-xl font-semibold text-white">
              Create New Workflow
            </h2>
            <p className="text-sm text-gray-400">
              {step === 1
                ? "Choose how to start"
                : step === 2
                ? "Configure your workflow"
                : "Select a trigger"}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Progress */}
        <div className="px-6 py-3 bg-white/5 border-b border-white/5">
          <div className="flex items-center gap-4">
            {[1, 2, 3].map((s) => (
              <div key={s} className="flex items-center gap-2">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                    s === step
                      ? "bg-purple-500 text-white"
                      : s < step
                      ? "bg-green-500 text-white"
                      : "bg-white/10 text-gray-400"
                  }`}
                >
                  {s < step ? <Check className="w-4 h-4" /> : s}
                </div>
                <span
                  className={`text-sm ${
                    s === step ? "text-white" : "text-gray-400"
                  }`}
                >
                  {s === 1 ? "Start" : s === 2 ? "Details" : "Trigger"}
                </span>
                {s < 3 && (
                  <ChevronRight className="w-4 h-4 text-gray-600 mx-2" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {step === 1 && (
            <div className="space-y-4">
              <div
                onClick={() => {
                  setCreateMode("scratch");
                  setStep(2);
                }}
                className={`flex items-start gap-4 p-4 rounded-xl border cursor-pointer transition-all ${
                  createMode === "scratch"
                    ? "border-purple-500 bg-purple-500/10"
                    : "border-white/10 hover:border-white/20"
                }`}
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                  <PlusCircle className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">Start from Scratch</h3>
                  <p className="text-sm text-gray-400 mt-1">
                    Create a custom workflow with full control over triggers and
                    actions
                  </p>
                </div>
              </div>

              <div
                onClick={() => {
                  setCreateMode("template");
                  setStep(2);
                }}
                className={`flex items-start gap-4 p-4 rounded-xl border cursor-pointer transition-all ${
                  createMode === "template"
                    ? "border-purple-500 bg-purple-500/10"
                    : "border-white/10 hover:border-white/20"
                }`}
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">Use a Template</h3>
                  <p className="text-sm text-gray-400 mt-1">
                    Start with a pre-built workflow and customize it to your
                    needs
                  </p>
                </div>
              </div>
            </div>
          )}

          {step === 2 && createMode === "scratch" && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Workflow Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., Post-Call Follow-up"
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe what this workflow does..."
                  rows={3}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                />
              </div>
            </div>
          )}

          {step === 2 && createMode === "template" && (
            <div className="grid grid-cols-2 gap-3">
              {workflowTemplates.map((template) => (
                <div
                  key={template.id}
                  onClick={() => setSelectedTemplate(template)}
                  className={`p-4 rounded-xl border cursor-pointer transition-all ${
                    selectedTemplate?.id === template.id
                      ? "border-purple-500 bg-purple-500/10"
                      : "border-white/10 hover:border-white/20"
                  }`}
                >
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                      {nodeIcons[template.icon] || (
                        <Workflow className="w-4 h-4 text-gray-400" />
                      )}
                    </div>
                    <h4 className="font-medium text-white text-sm">
                      {template.name}
                    </h4>
                  </div>
                  <p className="text-xs text-gray-400">{template.description}</p>
                </div>
              ))}
            </div>
          )}

          {step === 3 && (
            <div className="space-y-3">
              <p className="text-sm text-gray-400 mb-4">
                Select what will trigger this workflow to run:
              </p>
              {triggers.map((trigger) => (
                <div
                  key={trigger.id}
                  onClick={() => setSelectedTrigger(trigger.id)}
                  className={`flex items-center gap-4 p-4 rounded-xl border cursor-pointer transition-all ${
                    selectedTrigger === trigger.id
                      ? "border-purple-500 bg-purple-500/10"
                      : "border-white/10 hover:border-white/20"
                  }`}
                >
                  <div className="w-10 h-10 rounded-lg bg-yellow-500/10 border border-yellow-500/20 flex items-center justify-center">
                    {nodeIcons[trigger.icon] || <Zap className="w-5 h-5 text-yellow-400" />}
                  </div>
                  <div>
                    <h4 className="font-medium text-white">{trigger.name}</h4>
                    <p className="text-sm text-gray-400">{trigger.description}</p>
                  </div>
                  {selectedTrigger === trigger.id && (
                    <CheckCircle className="w-5 h-5 text-purple-400 ml-auto" />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-white/10 bg-white/5">
          {step > 1 ? (
            <button
              onClick={() => setStep(step - 1)}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Back
            </button>
          ) : (
            <div />
          )}
          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            {step < 3 ? (
              <button
                onClick={() => setStep(step + 1)}
                disabled={
                  (step === 2 && createMode === "scratch" && !name) ||
                  (step === 2 && createMode === "template" && !selectedTemplate)
                }
                className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Continue
                <ChevronRight className="w-4 h-4" />
              </button>
            ) : (
              <button
                onClick={onClose}
                disabled={!selectedTrigger}
                className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Rocket className="w-4 h-4" />
                Create Workflow
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Node palette (for workflow editor)
function NodePalette({
  onDragStart,
}: {
  onDragStart: (nodeType: NodeType) => void;
}) {
  const [expandedCategories, setExpandedCategories] = useState<string[]>([
    "triggers",
    "actions",
  ]);

  const toggleCategory = (id: string) => {
    setExpandedCategories((prev) =>
      prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id]
    );
  };

  const categoryColors: Record<string, string> = {
    yellow: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
    blue: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    green: "bg-green-500/10 text-green-400 border-green-500/20",
    orange: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  };

  return (
    <div className="w-64 bg-[#1a1a2e]/50 border-r border-white/5 overflow-y-auto">
      <div className="p-4 border-b border-white/5">
        <h3 className="font-semibold text-white">Nodes</h3>
        <p className="text-xs text-gray-500 mt-1">Drag to add to workflow</p>
      </div>
      <div className="p-2">
        {nodeCategories.map((category) => (
          <div key={category.id} className="mb-2">
            <button
              onClick={() => toggleCategory(category.id)}
              className="flex items-center justify-between w-full px-3 py-2 rounded-lg hover:bg-white/5 transition-colors"
            >
              <div className="flex items-center gap-2">
                <div
                  className={`p-1 rounded border ${categoryColors[category.color]}`}
                >
                  {category.icon}
                </div>
                <span className="text-sm font-medium text-white">
                  {category.name}
                </span>
              </div>
              <ChevronDown
                className={`w-4 h-4 text-gray-400 transition-transform ${
                  expandedCategories.includes(category.id) ? "rotate-180" : ""
                }`}
              />
            </button>
            {expandedCategories.includes(category.id) && (
              <div className="mt-1 ml-2 space-y-1">
                {category.nodes.map((node) => (
                  <div
                    key={node.id}
                    draggable
                    onDragStart={() => onDragStart(node)}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-white/10 cursor-grab active:cursor-grabbing transition-colors"
                  >
                    <div className="w-6 h-6 rounded flex items-center justify-center bg-white/5">
                      {nodeIcons[node.icon] || (
                        <Box className="w-4 h-4 text-gray-400" />
                      )}
                    </div>
                    <span className="text-sm text-gray-300">{node.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// Workflow canvas (simplified)
function WorkflowCanvas() {
  const [zoom, setZoom] = useState(100);
  const [tool, setTool] = useState<"select" | "pan">("select");

  return (
    <div className="flex-1 bg-[#0a0a14] relative overflow-hidden">
      {/* Toolbar */}
      <div className="absolute top-4 left-4 right-4 z-10 flex items-center justify-between">
        <div className="flex items-center gap-2 bg-[#1a1a2e]/80 backdrop-blur-sm rounded-lg p-1">
          <button
            onClick={() => setTool("select")}
            className={`p-2 rounded-lg transition-colors ${
              tool === "select" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:bg-white/10"
            }`}
            title="Select"
          >
            <MousePointer2 className="w-4 h-4" />
          </button>
          <button
            onClick={() => setTool("pan")}
            className={`p-2 rounded-lg transition-colors ${
              tool === "pan" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:bg-white/10"
            }`}
            title="Pan"
          >
            <Hand className="w-4 h-4" />
          </button>
          <div className="w-px h-6 bg-white/10 mx-1" />
          <button className="p-2 text-gray-400 hover:bg-white/10 rounded-lg transition-colors" title="Undo">
            <Undo className="w-4 h-4" />
          </button>
          <button className="p-2 text-gray-400 hover:bg-white/10 rounded-lg transition-colors" title="Redo">
            <Redo className="w-4 h-4" />
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button className="flex items-center gap-2 px-3 py-1.5 bg-[#1a1a2e]/80 backdrop-blur-sm text-gray-400 hover:text-white rounded-lg transition-colors">
            <TestTube className="w-4 h-4" />
            <span className="text-sm">Test</span>
          </button>
          <button className="flex items-center gap-2 px-4 py-1.5 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors">
            <Save className="w-4 h-4" />
            <span className="text-sm">Save</span>
          </button>
        </div>
      </div>

      {/* Zoom controls */}
      <div className="absolute bottom-4 left-4 z-10 flex items-center gap-2 bg-[#1a1a2e]/80 backdrop-blur-sm rounded-lg p-1">
        <button
          onClick={() => setZoom(Math.max(25, zoom - 25))}
          className="p-2 text-gray-400 hover:bg-white/10 rounded-lg transition-colors"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <span className="text-sm text-gray-400 w-12 text-center">{zoom}%</span>
        <button
          onClick={() => setZoom(Math.min(200, zoom + 25))}
          className="p-2 text-gray-400 hover:bg-white/10 rounded-lg transition-colors"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <div className="w-px h-6 bg-white/10 mx-1" />
        <button className="p-2 text-gray-400 hover:bg-white/10 rounded-lg transition-colors">
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Grid background */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: `radial-gradient(circle, rgba(255,255,255,0.05) 1px, transparent 1px)`,
          backgroundSize: "20px 20px",
          transform: `scale(${zoom / 100})`,
          transformOrigin: "center center",
        }}
      />

      {/* Empty state */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-4">
            <Workflow className="w-8 h-8 text-gray-600" />
          </div>
          <h3 className="text-lg font-medium text-white mb-2">
            Start Building Your Workflow
          </h3>
          <p className="text-gray-500 mb-4">
            Drag nodes from the left panel to get started
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
            <span className="flex items-center gap-1 px-3 py-1 bg-white/5 rounded">
              <Zap className="w-3 h-3 text-yellow-400" />
              Add a trigger
            </span>
            <ChevronRight className="w-4 h-4" />
            <span className="flex items-center gap-1 px-3 py-1 bg-white/5 rounded">
              <Play className="w-3 h-3 text-blue-400" />
              Add actions
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Main component
export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState(mockWorkflows);
  const [view, setView] = useState<"list" | "editor">("list");
  const [editingWorkflow, setEditingWorkflow] = useState<Workflow | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterTrigger, setFilterTrigger] = useState<string>("all");
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);

  // Stats
  const stats = useMemo(() => {
    const active = workflows.filter((w) => w.status === "active").length;
    const totalRuns = workflows.reduce((sum, w) => sum + w.runCount, 0);
    const totalSuccess = workflows.reduce((sum, w) => sum + w.successCount, 0);
    const totalFailures = workflows.reduce((sum, w) => sum + w.failureCount, 0);
    const successRate =
      totalRuns > 0 ? Math.round((totalSuccess / totalRuns) * 100) : 0;

    return { active, totalRuns, successRate, totalFailures };
  }, [workflows]);

  // Filtered workflows
  const filteredWorkflows = useMemo(() => {
    return workflows.filter((wf) => {
      if (
        searchQuery &&
        !wf.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !wf.description.toLowerCase().includes(searchQuery.toLowerCase())
      ) {
        return false;
      }
      if (filterStatus !== "all" && wf.status !== filterStatus) {
        return false;
      }
      if (filterTrigger !== "all" && wf.trigger !== filterTrigger) {
        return false;
      }
      return true;
    });
  }, [workflows, searchQuery, filterStatus, filterTrigger]);

  const handleEdit = (workflow: Workflow) => {
    setEditingWorkflow(workflow);
    setView("editor");
  };

  const handleDuplicate = (workflow: Workflow) => {
    const newWorkflow: Workflow = {
      ...workflow,
      id: `wf_${Date.now()}`,
      name: `${workflow.name} (Copy)`,
      status: "draft",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      lastRunAt: null,
      runCount: 0,
      successCount: 0,
      failureCount: 0,
    };
    setWorkflows([newWorkflow, ...workflows]);
  };

  const handleDelete = (id: string) => {
    setWorkflows(workflows.filter((w) => w.id !== id));
  };

  const handleToggle = (id: string) => {
    setWorkflows(
      workflows.map((w) =>
        w.id === id
          ? { ...w, status: w.status === "active" ? "inactive" : "active" }
          : w
      )
    );
  };

  const handleToggleFavorite = (id: string) => {
    setWorkflows(
      workflows.map((w) =>
        w.id === id ? { ...w, isFavorite: !w.isFavorite } : w
      )
    );
  };

  // Editor view
  if (view === "editor") {
    return (
      <DashboardLayout>
        <div className="h-screen flex flex-col">
          {/* Editor header */}
          <div className="flex items-center justify-between px-6 py-3 bg-[#1a1a2e]/50 border-b border-white/5">
            <div className="flex items-center gap-4">
              <button
                onClick={() => {
                  setView("list");
                  setEditingWorkflow(null);
                }}
                className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                Back
              </button>
              <div className="w-px h-6 bg-white/10" />
              <div>
                <h1 className="text-lg font-semibold text-white">
                  {editingWorkflow?.name || "New Workflow"}
                </h1>
                <p className="text-xs text-gray-500">
                  {editingWorkflow?.description || "Configure your workflow"}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <StatusBadge status={editingWorkflow?.status || "draft"} />
            </div>
          </div>

          {/* Editor content */}
          <div className="flex-1 flex overflow-hidden">
            <NodePalette onDragStart={() => {}} />
            <WorkflowCanvas />
          </div>
        </div>
      </DashboardLayout>
    );
  }

  // List view
  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Workflows</h1>
            <p className="text-gray-400">
              Automate actions based on call events and triggers
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowTemplates(!showTemplates)}
              className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg border border-white/10 transition-colors"
            >
              <Sparkles className="w-4 h-4" />
              Templates
            </button>
            <button
              onClick={() => setShowCreateDialog(true)}
              className="flex items-center gap-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              Create Workflow
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-4">
          <StatsCard
            title="Active Workflows"
            value={stats.active.toString()}
            icon={<Activity className="w-4 h-4" />}
            color="purple"
          />
          <StatsCard
            title="Total Runs"
            value={stats.totalRuns.toLocaleString()}
            icon={<Play className="w-4 h-4" />}
            color="blue"
            change={{ value: "12%", type: "up" }}
          />
          <StatsCard
            title="Success Rate"
            value={`${stats.successRate}%`}
            icon={<CheckCircle className="w-4 h-4" />}
            color="green"
          />
          <StatsCard
            title="Failed Runs"
            value={stats.totalFailures.toLocaleString()}
            icon={<XCircle className="w-4 h-4" />}
            color="red"
          />
        </div>

        {/* Templates section */}
        {showTemplates && (
          <div className="bg-[#1a1a2e]/30 rounded-xl border border-white/5 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-white">
                  Workflow Templates
                </h2>
                <p className="text-sm text-gray-400">
                  Start with a pre-built workflow
                </p>
              </div>
              <button
                onClick={() => setShowTemplates(false)}
                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              >
                <X className="w-4 h-4 text-gray-400" />
              </button>
            </div>
            <div className="grid grid-cols-4 gap-4">
              {workflowTemplates.slice(0, 4).map((template) => (
                <TemplateCard
                  key={template.id}
                  template={template}
                  onUse={() => {}}
                />
              ))}
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="flex items-center gap-4">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search workflows..."
              className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
            />
          </div>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
          >
            <option value="all">All Status</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
            <option value="draft">Draft</option>
          </select>
          <select
            value={filterTrigger}
            onChange={(e) => setFilterTrigger(e.target.value)}
            className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
          >
            <option value="all">All Triggers</option>
            <option value="call_started">Call Started</option>
            <option value="call_ended">Call Ended</option>
            <option value="call_missed">Call Missed</option>
            <option value="webhook">Webhook</option>
            <option value="schedule">Schedule</option>
          </select>
        </div>

        {/* Workflows grid */}
        <div className="grid grid-cols-2 gap-4">
          {filteredWorkflows.map((workflow) => (
            <WorkflowCard
              key={workflow.id}
              workflow={workflow}
              onEdit={handleEdit}
              onDuplicate={handleDuplicate}
              onDelete={handleDelete}
              onToggle={handleToggle}
              onToggleFavorite={handleToggleFavorite}
            />
          ))}
        </div>

        {/* Empty state */}
        {filteredWorkflows.length === 0 && (
          <div className="text-center py-12">
            <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-4">
              <Workflow className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-white mb-2">
              No workflows found
            </h3>
            <p className="text-gray-400 mb-4">
              {searchQuery || filterStatus !== "all" || filterTrigger !== "all"
                ? "Try adjusting your filters"
                : "Create your first workflow to automate your voice AI"}
            </p>
            <button
              onClick={() => setShowCreateDialog(true)}
              className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              Create Workflow
            </button>
          </div>
        )}

        {/* Create dialog */}
        <CreateWorkflowDialog
          isOpen={showCreateDialog}
          onClose={() => setShowCreateDialog(false)}
        />
      </div>
    </DashboardLayout>
  );
}
