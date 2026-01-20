"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import {
  Calendar,
  Clock,
  Play,
  Pause,
  Trash2,
  Edit,
  Plus,
  Search,
  Filter,
  MoreVertical,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Zap,
  Settings,
  History,
  Target,
  Bell,
  Mail,
  MessageSquare,
  Phone,
  Database,
  Cloud,
  FileText,
  Users,
  TrendingUp,
  ArrowRight,
  Copy,
  ExternalLink,
  RotateCcw,
  Timer,
  Repeat,
  CalendarDays,
  Sun,
  Moon,
  Sunrise,
  Code,
  Activity,
  Archive,
  Eye,
  EyeOff,
  Globe,
  Lock,
  Unlock,
  ChevronUp,
  Loader2,
  Download,
  BarChart3,
  Layers,
  GitBranch,
  Webhook,
  Bot,
  Sparkles,
  CircleDot,
  PlayCircle,
  StopCircle,
  FastForward,
  Rewind,
  SkipForward,
  Info,
} from "lucide-react";

// Types
type JobStatus = "active" | "paused" | "completed" | "failed" | "pending";
type JobType = "campaign" | "report" | "notification" | "backup" | "sync" | "cleanup" | "webhook" | "custom";
type FrequencyType = "once" | "hourly" | "daily" | "weekly" | "monthly" | "custom";
type DayOfWeek = "monday" | "tuesday" | "wednesday" | "thursday" | "friday" | "saturday" | "sunday";

interface JobExecution {
  id: string;
  startedAt: Date;
  completedAt?: Date;
  status: "running" | "success" | "failed" | "cancelled";
  duration?: number;
  output?: string;
  error?: string;
  metrics?: {
    itemsProcessed?: number;
    itemsFailed?: number;
    dataSize?: string;
  };
}

interface ScheduledJob {
  id: string;
  name: string;
  description: string;
  type: JobType;
  status: JobStatus;
  frequency: FrequencyType;
  cronExpression?: string;
  timezone: string;
  schedule: {
    time?: string;
    daysOfWeek?: DayOfWeek[];
    dayOfMonth?: number;
    startDate?: Date;
    endDate?: Date;
  };
  action: {
    type: string;
    config: Record<string, any>;
  };
  retryPolicy: {
    maxRetries: number;
    retryDelay: number;
    exponentialBackoff: boolean;
  };
  notifications: {
    onSuccess: boolean;
    onFailure: boolean;
    channels: ("email" | "sms" | "webhook" | "slack")[];
  };
  createdAt: Date;
  updatedAt: Date;
  nextRunAt?: Date;
  lastRunAt?: Date;
  lastExecution?: JobExecution;
  executions: JobExecution[];
  tags: string[];
  priority: "low" | "medium" | "high" | "critical";
  timeout: number;
  isLocked: boolean;
  createdBy: string;
}

// Mock data
const mockJobs: ScheduledJob[] = [
  {
    id: "job-1",
    name: "Daily Campaign Report",
    description: "Generate and send daily performance reports for all active campaigns",
    type: "report",
    status: "active",
    frequency: "daily",
    timezone: "America/New_York",
    schedule: {
      time: "08:00",
      daysOfWeek: ["monday", "tuesday", "wednesday", "thursday", "friday"],
    },
    action: {
      type: "generate_report",
      config: {
        reportType: "campaign_performance",
        format: "pdf",
        recipients: ["team@company.com"],
      },
    },
    retryPolicy: {
      maxRetries: 3,
      retryDelay: 300,
      exponentialBackoff: true,
    },
    notifications: {
      onSuccess: false,
      onFailure: true,
      channels: ["email", "slack"],
    },
    createdAt: new Date("2024-01-15"),
    updatedAt: new Date("2024-03-10"),
    nextRunAt: new Date(Date.now() + 8 * 60 * 60 * 1000),
    lastRunAt: new Date(Date.now() - 16 * 60 * 60 * 1000),
    lastExecution: {
      id: "exec-1",
      startedAt: new Date(Date.now() - 16 * 60 * 60 * 1000),
      completedAt: new Date(Date.now() - 16 * 60 * 60 * 1000 + 45000),
      status: "success",
      duration: 45,
      metrics: {
        itemsProcessed: 12,
        dataSize: "2.4 MB",
      },
    },
    executions: [],
    tags: ["reports", "daily", "campaigns"],
    priority: "high",
    timeout: 300,
    isLocked: false,
    createdBy: "John Doe",
  },
  {
    id: "job-2",
    name: "Database Backup",
    description: "Full database backup to cloud storage with compression",
    type: "backup",
    status: "active",
    frequency: "daily",
    cronExpression: "0 2 * * *",
    timezone: "UTC",
    schedule: {
      time: "02:00",
    },
    action: {
      type: "database_backup",
      config: {
        destination: "s3://backups/db/",
        compression: "gzip",
        encryption: true,
      },
    },
    retryPolicy: {
      maxRetries: 5,
      retryDelay: 600,
      exponentialBackoff: true,
    },
    notifications: {
      onSuccess: true,
      onFailure: true,
      channels: ["email", "slack", "webhook"],
    },
    createdAt: new Date("2024-01-01"),
    updatedAt: new Date("2024-03-15"),
    nextRunAt: new Date(Date.now() + 12 * 60 * 60 * 1000),
    lastRunAt: new Date(Date.now() - 12 * 60 * 60 * 1000),
    lastExecution: {
      id: "exec-2",
      startedAt: new Date(Date.now() - 12 * 60 * 60 * 1000),
      completedAt: new Date(Date.now() - 12 * 60 * 60 * 1000 + 180000),
      status: "success",
      duration: 180,
      metrics: {
        dataSize: "1.2 GB",
      },
    },
    executions: [],
    tags: ["backup", "critical", "database"],
    priority: "critical",
    timeout: 1800,
    isLocked: true,
    createdBy: "System",
  },
  {
    id: "job-3",
    name: "CRM Contact Sync",
    description: "Synchronize contacts with external CRM system",
    type: "sync",
    status: "active",
    frequency: "hourly",
    timezone: "America/Los_Angeles",
    schedule: {},
    action: {
      type: "sync_contacts",
      config: {
        source: "internal",
        destination: "salesforce",
        mode: "incremental",
      },
    },
    retryPolicy: {
      maxRetries: 3,
      retryDelay: 120,
      exponentialBackoff: false,
    },
    notifications: {
      onSuccess: false,
      onFailure: true,
      channels: ["email"],
    },
    createdAt: new Date("2024-02-01"),
    updatedAt: new Date("2024-03-18"),
    nextRunAt: new Date(Date.now() + 45 * 60 * 1000),
    lastRunAt: new Date(Date.now() - 15 * 60 * 1000),
    lastExecution: {
      id: "exec-3",
      startedAt: new Date(Date.now() - 15 * 60 * 1000),
      completedAt: new Date(Date.now() - 15 * 60 * 1000 + 25000),
      status: "success",
      duration: 25,
      metrics: {
        itemsProcessed: 156,
        itemsFailed: 0,
      },
    },
    executions: [],
    tags: ["sync", "crm", "contacts"],
    priority: "medium",
    timeout: 600,
    isLocked: false,
    createdBy: "Jane Smith",
  },
  {
    id: "job-4",
    name: "Weekly Agent Performance Digest",
    description: "Send weekly summary of agent performance to managers",
    type: "notification",
    status: "active",
    frequency: "weekly",
    timezone: "America/New_York",
    schedule: {
      time: "09:00",
      daysOfWeek: ["monday"],
    },
    action: {
      type: "send_notification",
      config: {
        template: "agent_performance_weekly",
        recipients: ["managers@company.com"],
        includeCharts: true,
      },
    },
    retryPolicy: {
      maxRetries: 2,
      retryDelay: 300,
      exponentialBackoff: false,
    },
    notifications: {
      onSuccess: false,
      onFailure: true,
      channels: ["email"],
    },
    createdAt: new Date("2024-01-20"),
    updatedAt: new Date("2024-03-01"),
    nextRunAt: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000),
    lastRunAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
    lastExecution: {
      id: "exec-4",
      startedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      completedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000 + 60000),
      status: "success",
      duration: 60,
      metrics: {
        itemsProcessed: 24,
      },
    },
    executions: [],
    tags: ["notifications", "weekly", "agents"],
    priority: "medium",
    timeout: 300,
    isLocked: false,
    createdBy: "Admin",
  },
  {
    id: "job-5",
    name: "Outbound Campaign Runner",
    description: "Execute scheduled outbound calling campaign",
    type: "campaign",
    status: "paused",
    frequency: "daily",
    timezone: "America/Chicago",
    schedule: {
      time: "10:00",
      daysOfWeek: ["monday", "tuesday", "wednesday", "thursday", "friday"],
    },
    action: {
      type: "run_campaign",
      config: {
        campaignId: "camp-123",
        maxCalls: 1000,
        pacing: "progressive",
      },
    },
    retryPolicy: {
      maxRetries: 1,
      retryDelay: 600,
      exponentialBackoff: false,
    },
    notifications: {
      onSuccess: true,
      onFailure: true,
      channels: ["email", "sms"],
    },
    createdAt: new Date("2024-02-15"),
    updatedAt: new Date("2024-03-19"),
    nextRunAt: undefined,
    lastRunAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
    lastExecution: {
      id: "exec-5",
      startedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
      completedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000 + 3600000),
      status: "success",
      duration: 3600,
      metrics: {
        itemsProcessed: 847,
        itemsFailed: 23,
      },
    },
    executions: [],
    tags: ["campaign", "outbound", "sales"],
    priority: "high",
    timeout: 7200,
    isLocked: false,
    createdBy: "Sales Team",
  },
  {
    id: "job-6",
    name: "Log Cleanup",
    description: "Clean up old logs and temporary files older than 30 days",
    type: "cleanup",
    status: "active",
    frequency: "weekly",
    cronExpression: "0 3 * * 0",
    timezone: "UTC",
    schedule: {
      time: "03:00",
      daysOfWeek: ["sunday"],
    },
    action: {
      type: "cleanup_logs",
      config: {
        retentionDays: 30,
        targets: ["logs", "temp", "cache"],
        dryRun: false,
      },
    },
    retryPolicy: {
      maxRetries: 2,
      retryDelay: 300,
      exponentialBackoff: false,
    },
    notifications: {
      onSuccess: false,
      onFailure: true,
      channels: ["email"],
    },
    createdAt: new Date("2024-01-05"),
    updatedAt: new Date("2024-02-20"),
    nextRunAt: new Date(Date.now() + 4 * 24 * 60 * 60 * 1000),
    lastRunAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
    lastExecution: {
      id: "exec-6",
      startedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
      completedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000 + 120000),
      status: "success",
      duration: 120,
      metrics: {
        itemsProcessed: 15420,
        dataSize: "4.8 GB",
      },
    },
    executions: [],
    tags: ["maintenance", "cleanup", "automated"],
    priority: "low",
    timeout: 900,
    isLocked: false,
    createdBy: "System",
  },
  {
    id: "job-7",
    name: "Webhook Event Processor",
    description: "Process queued webhook events and retry failed deliveries",
    type: "webhook",
    status: "active",
    frequency: "custom",
    cronExpression: "*/5 * * * *",
    timezone: "UTC",
    schedule: {},
    action: {
      type: "process_webhooks",
      config: {
        batchSize: 100,
        maxRetries: 5,
        timeout: 30,
      },
    },
    retryPolicy: {
      maxRetries: 3,
      retryDelay: 60,
      exponentialBackoff: true,
    },
    notifications: {
      onSuccess: false,
      onFailure: true,
      channels: ["slack"],
    },
    createdAt: new Date("2024-02-10"),
    updatedAt: new Date("2024-03-18"),
    nextRunAt: new Date(Date.now() + 3 * 60 * 1000),
    lastRunAt: new Date(Date.now() - 2 * 60 * 1000),
    lastExecution: {
      id: "exec-7",
      startedAt: new Date(Date.now() - 2 * 60 * 1000),
      completedAt: new Date(Date.now() - 2 * 60 * 1000 + 8000),
      status: "success",
      duration: 8,
      metrics: {
        itemsProcessed: 47,
        itemsFailed: 2,
      },
    },
    executions: [],
    tags: ["webhooks", "events", "processing"],
    priority: "high",
    timeout: 120,
    isLocked: false,
    createdBy: "API Team",
  },
  {
    id: "job-8",
    name: "Monthly Billing Report",
    description: "Generate comprehensive billing report for finance team",
    type: "report",
    status: "active",
    frequency: "monthly",
    timezone: "America/New_York",
    schedule: {
      time: "06:00",
      dayOfMonth: 1,
    },
    action: {
      type: "generate_report",
      config: {
        reportType: "billing_summary",
        format: "xlsx",
        recipients: ["finance@company.com"],
        includeBreakdown: true,
      },
    },
    retryPolicy: {
      maxRetries: 3,
      retryDelay: 600,
      exponentialBackoff: true,
    },
    notifications: {
      onSuccess: true,
      onFailure: true,
      channels: ["email"],
    },
    createdAt: new Date("2024-01-01"),
    updatedAt: new Date("2024-03-01"),
    nextRunAt: new Date("2024-04-01T06:00:00"),
    lastRunAt: new Date("2024-03-01T06:00:00"),
    lastExecution: {
      id: "exec-8",
      startedAt: new Date("2024-03-01T06:00:00"),
      completedAt: new Date("2024-03-01T06:02:30"),
      status: "success",
      duration: 150,
      metrics: {
        dataSize: "856 KB",
      },
    },
    executions: [],
    tags: ["billing", "finance", "monthly"],
    priority: "high",
    timeout: 600,
    isLocked: true,
    createdBy: "Finance Team",
  },
  {
    id: "job-9",
    name: "Failed Call Retry",
    description: "Retry failed outbound calls from the previous day",
    type: "campaign",
    status: "failed",
    frequency: "daily",
    timezone: "America/New_York",
    schedule: {
      time: "14:00",
      daysOfWeek: ["monday", "tuesday", "wednesday", "thursday", "friday"],
    },
    action: {
      type: "retry_failed_calls",
      config: {
        maxAge: 24,
        maxRetries: 2,
        priority: "normal",
      },
    },
    retryPolicy: {
      maxRetries: 2,
      retryDelay: 300,
      exponentialBackoff: false,
    },
    notifications: {
      onSuccess: false,
      onFailure: true,
      channels: ["email", "slack"],
    },
    createdAt: new Date("2024-02-20"),
    updatedAt: new Date("2024-03-19"),
    nextRunAt: new Date(Date.now() + 20 * 60 * 60 * 1000),
    lastRunAt: new Date(Date.now() - 4 * 60 * 60 * 1000),
    lastExecution: {
      id: "exec-9",
      startedAt: new Date(Date.now() - 4 * 60 * 60 * 1000),
      status: "failed",
      error: "Connection timeout to dialer service",
    },
    executions: [],
    tags: ["calls", "retry", "recovery"],
    priority: "medium",
    timeout: 1800,
    isLocked: false,
    createdBy: "Operations",
  },
  {
    id: "job-10",
    name: "AI Model Training Data Export",
    description: "Export anonymized call data for AI model training",
    type: "custom",
    status: "pending",
    frequency: "once",
    timezone: "UTC",
    schedule: {
      startDate: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000),
      time: "00:00",
    },
    action: {
      type: "export_training_data",
      config: {
        dateRange: "last_90_days",
        anonymize: true,
        format: "jsonl",
        destination: "s3://ml-data/training/",
      },
    },
    retryPolicy: {
      maxRetries: 3,
      retryDelay: 900,
      exponentialBackoff: true,
    },
    notifications: {
      onSuccess: true,
      onFailure: true,
      channels: ["email", "slack"],
    },
    createdAt: new Date("2024-03-18"),
    updatedAt: new Date("2024-03-18"),
    nextRunAt: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000),
    executions: [],
    tags: ["ai", "training", "export"],
    priority: "medium",
    timeout: 3600,
    isLocked: false,
    createdBy: "ML Team",
  },
];

// Helper functions
const getStatusColor = (status: JobStatus) => {
  switch (status) {
    case "active":
      return "text-green-400 bg-green-500/20";
    case "paused":
      return "text-yellow-400 bg-yellow-500/20";
    case "completed":
      return "text-blue-400 bg-blue-500/20";
    case "failed":
      return "text-red-400 bg-red-500/20";
    case "pending":
      return "text-purple-400 bg-purple-500/20";
  }
};

const getStatusIcon = (status: JobStatus) => {
  switch (status) {
    case "active":
      return <PlayCircle className="w-4 h-4" />;
    case "paused":
      return <Pause className="w-4 h-4" />;
    case "completed":
      return <CheckCircle className="w-4 h-4" />;
    case "failed":
      return <XCircle className="w-4 h-4" />;
    case "pending":
      return <Clock className="w-4 h-4" />;
  }
};

const getTypeIcon = (type: JobType) => {
  switch (type) {
    case "campaign":
      return <Phone className="w-4 h-4" />;
    case "report":
      return <FileText className="w-4 h-4" />;
    case "notification":
      return <Bell className="w-4 h-4" />;
    case "backup":
      return <Database className="w-4 h-4" />;
    case "sync":
      return <RefreshCw className="w-4 h-4" />;
    case "cleanup":
      return <Trash2 className="w-4 h-4" />;
    case "webhook":
      return <Webhook className="w-4 h-4" />;
    case "custom":
      return <Code className="w-4 h-4" />;
  }
};

const getTypeColor = (type: JobType) => {
  switch (type) {
    case "campaign":
      return "text-blue-400 bg-blue-500/20";
    case "report":
      return "text-purple-400 bg-purple-500/20";
    case "notification":
      return "text-yellow-400 bg-yellow-500/20";
    case "backup":
      return "text-green-400 bg-green-500/20";
    case "sync":
      return "text-cyan-400 bg-cyan-500/20";
    case "cleanup":
      return "text-orange-400 bg-orange-500/20";
    case "webhook":
      return "text-pink-400 bg-pink-500/20";
    case "custom":
      return "text-gray-400 bg-gray-500/20";
  }
};

const getPriorityColor = (priority: string) => {
  switch (priority) {
    case "critical":
      return "text-red-400 bg-red-500/20";
    case "high":
      return "text-orange-400 bg-orange-500/20";
    case "medium":
      return "text-yellow-400 bg-yellow-500/20";
    case "low":
      return "text-gray-400 bg-gray-500/20";
    default:
      return "text-gray-400 bg-gray-500/20";
  }
};

const getFrequencyLabel = (frequency: FrequencyType) => {
  switch (frequency) {
    case "once":
      return "One-time";
    case "hourly":
      return "Hourly";
    case "daily":
      return "Daily";
    case "weekly":
      return "Weekly";
    case "monthly":
      return "Monthly";
    case "custom":
      return "Custom";
  }
};

const formatRelativeTime = (date: Date) => {
  const now = new Date();
  const diff = date.getTime() - now.getTime();
  const absDiff = Math.abs(diff);

  const minutes = Math.floor(absDiff / (1000 * 60));
  const hours = Math.floor(absDiff / (1000 * 60 * 60));
  const days = Math.floor(absDiff / (1000 * 60 * 60 * 24));

  if (diff > 0) {
    if (minutes < 60) return `in ${minutes}m`;
    if (hours < 24) return `in ${hours}h`;
    return `in ${days}d`;
  } else {
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  }
};

const formatDuration = (seconds: number) => {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
};

// Job Card Component
const JobCard: React.FC<{
  job: ScheduledJob;
  onEdit: (job: ScheduledJob) => void;
  onDelete: (job: ScheduledJob) => void;
  onToggle: (job: ScheduledJob) => void;
  onRunNow: (job: ScheduledJob) => void;
  onViewHistory: (job: ScheduledJob) => void;
  isExpanded: boolean;
  onToggleExpand: () => void;
}> = ({ job, onEdit, onDelete, onToggle, onRunNow, onViewHistory, isExpanded, onToggleExpand }) => {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden hover:border-purple-500/30 transition-all duration-300">
      {/* Header */}
      <div className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3 flex-1 min-w-0">
            <button
              onClick={onToggleExpand}
              className="p-1 rounded hover:bg-white/10 transition-colors mt-0.5"
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4 text-gray-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-gray-400" />
              )}
            </button>

            <div
              className={`p-2 rounded-lg ${getTypeColor(job.type)}`}
            >
              {getTypeIcon(job.type)}
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-white truncate">{job.name}</h3>
                {job.isLocked && (
                  <Lock className="w-3.5 h-3.5 text-yellow-400 flex-shrink-0" />
                )}
              </div>
              <p className="text-sm text-gray-400 line-clamp-1 mt-0.5">{job.description}</p>

              <div className="flex items-center gap-2 mt-2 flex-wrap">
                <span className={`px-2 py-0.5 rounded-full text-xs font-medium flex items-center gap-1 ${getStatusColor(job.status)}`}>
                  {getStatusIcon(job.status)}
                  {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                </span>

                <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getPriorityColor(job.priority)}`}>
                  {job.priority.toUpperCase()}
                </span>

                <span className="px-2 py-0.5 rounded-full text-xs font-medium text-gray-400 bg-gray-500/20 flex items-center gap-1">
                  <Repeat className="w-3 h-3" />
                  {getFrequencyLabel(job.frequency)}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {job.status === "active" && (
              <button
                onClick={() => onRunNow(job)}
                className="p-2 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 transition-colors"
                title="Run Now"
              >
                <Play className="w-4 h-4" />
              </button>
            )}

            <button
              onClick={() => onToggle(job)}
              className={`p-2 rounded-lg transition-colors ${
                job.status === "active"
                  ? "bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400"
                  : "bg-green-500/20 hover:bg-green-500/30 text-green-400"
              }`}
              title={job.status === "active" ? "Pause" : "Resume"}
            >
              {job.status === "active" ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
            </button>

            <div className="relative">
              <button
                onClick={() => setShowMenu(!showMenu)}
                className="p-2 rounded-lg hover:bg-white/10 text-gray-400 transition-colors"
              >
                <MoreVertical className="w-4 h-4" />
              </button>

              {showMenu && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setShowMenu(false)}
                  />
                  <div className="absolute right-0 top-full mt-1 w-48 bg-[#252540] border border-white/10 rounded-lg shadow-xl z-20 py-1">
                    <button
                      onClick={() => {
                        onEdit(job);
                        setShowMenu(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                    >
                      <Edit className="w-4 h-4" />
                      Edit Job
                    </button>
                    <button
                      onClick={() => {
                        onViewHistory(job);
                        setShowMenu(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                    >
                      <History className="w-4 h-4" />
                      View History
                    </button>
                    <button
                      onClick={() => setShowMenu(false)}
                      className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                    >
                      <Copy className="w-4 h-4" />
                      Duplicate
                    </button>
                    <hr className="my-1 border-white/10" />
                    <button
                      onClick={() => {
                        onDelete(job);
                        setShowMenu(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-red-400 hover:bg-red-500/10 flex items-center gap-2"
                      disabled={job.isLocked}
                    >
                      <Trash2 className="w-4 h-4" />
                      Delete Job
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Schedule Info */}
        <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wide">Next Run</p>
            <p className="text-sm text-white mt-1 flex items-center gap-1">
              <Clock className="w-3.5 h-3.5 text-purple-400" />
              {job.nextRunAt ? formatRelativeTime(job.nextRunAt) : "Not scheduled"}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wide">Last Run</p>
            <p className="text-sm text-white mt-1 flex items-center gap-1">
              <History className="w-3.5 h-3.5 text-gray-400" />
              {job.lastRunAt ? formatRelativeTime(job.lastRunAt) : "Never"}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wide">Last Duration</p>
            <p className="text-sm text-white mt-1 flex items-center gap-1">
              <Timer className="w-3.5 h-3.5 text-blue-400" />
              {job.lastExecution?.duration ? formatDuration(job.lastExecution.duration) : "-"}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wide">Timezone</p>
            <p className="text-sm text-white mt-1 flex items-center gap-1">
              <Globe className="w-3.5 h-3.5 text-green-400" />
              {job.timezone.split("/")[1]?.replace("_", " ") || job.timezone}
            </p>
          </div>
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="border-t border-white/10 p-4 bg-black/20">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Schedule Details */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Calendar className="w-4 h-4 text-purple-400" />
                Schedule Configuration
              </h4>
              <div className="space-y-2 text-sm">
                {job.schedule.time && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Time:</span>
                    <span className="text-white">{job.schedule.time}</span>
                  </div>
                )}
                {job.schedule.daysOfWeek && job.schedule.daysOfWeek.length > 0 && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Days:</span>
                    <span className="text-white">{job.schedule.daysOfWeek.map(d => d.charAt(0).toUpperCase() + d.slice(1, 3)).join(", ")}</span>
                  </div>
                )}
                {job.schedule.dayOfMonth && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Day of Month:</span>
                    <span className="text-white">{job.schedule.dayOfMonth}</span>
                  </div>
                )}
                {job.cronExpression && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Cron:</span>
                    <code className="text-purple-400 font-mono text-xs bg-purple-500/10 px-2 py-0.5 rounded">{job.cronExpression}</code>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-gray-400">Timeout:</span>
                  <span className="text-white">{formatDuration(job.timeout)}</span>
                </div>
              </div>
            </div>

            {/* Retry Policy */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <RotateCcw className="w-4 h-4 text-blue-400" />
                Retry Policy
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Max Retries:</span>
                  <span className="text-white">{job.retryPolicy.maxRetries}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Retry Delay:</span>
                  <span className="text-white">{formatDuration(job.retryPolicy.retryDelay)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Exponential Backoff:</span>
                  <span className={job.retryPolicy.exponentialBackoff ? "text-green-400" : "text-gray-500"}>
                    {job.retryPolicy.exponentialBackoff ? "Enabled" : "Disabled"}
                  </span>
                </div>
              </div>
            </div>

            {/* Notifications */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Bell className="w-4 h-4 text-yellow-400" />
                Notifications
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">On Success:</span>
                  <span className={job.notifications.onSuccess ? "text-green-400" : "text-gray-500"}>
                    {job.notifications.onSuccess ? "Yes" : "No"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">On Failure:</span>
                  <span className={job.notifications.onFailure ? "text-green-400" : "text-gray-500"}>
                    {job.notifications.onFailure ? "Yes" : "No"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Channels:</span>
                  <span className="text-white">{job.notifications.channels.join(", ") || "None"}</span>
                </div>
              </div>
            </div>

            {/* Action Config */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Settings className="w-4 h-4 text-gray-400" />
                Action Configuration
              </h4>
              <div className="bg-black/30 rounded-lg p-3 font-mono text-xs text-gray-300 overflow-auto max-h-32">
                <pre>{JSON.stringify(job.action.config, null, 2)}</pre>
              </div>
            </div>
          </div>

          {/* Last Execution */}
          {job.lastExecution && (
            <div className="mt-4 pt-4 border-t border-white/10">
              <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4 text-green-400" />
                Last Execution Details
              </h4>
              <div className="bg-black/30 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      job.lastExecution.status === "success" ? "bg-green-500/20 text-green-400" :
                      job.lastExecution.status === "failed" ? "bg-red-500/20 text-red-400" :
                      job.lastExecution.status === "running" ? "bg-blue-500/20 text-blue-400" :
                      "bg-gray-500/20 text-gray-400"
                    }`}>
                      {job.lastExecution.status.toUpperCase()}
                    </span>
                    <span className="text-sm text-gray-400">
                      {job.lastExecution.startedAt.toLocaleString()}
                    </span>
                  </div>
                  {job.lastExecution.duration && (
                    <span className="text-sm text-gray-400">
                      Duration: {formatDuration(job.lastExecution.duration)}
                    </span>
                  )}
                </div>

                {job.lastExecution.metrics && (
                  <div className="flex gap-4 mb-3">
                    {job.lastExecution.metrics.itemsProcessed !== undefined && (
                      <div className="text-sm">
                        <span className="text-gray-400">Processed:</span>{" "}
                        <span className="text-white">{job.lastExecution.metrics.itemsProcessed}</span>
                      </div>
                    )}
                    {job.lastExecution.metrics.itemsFailed !== undefined && (
                      <div className="text-sm">
                        <span className="text-gray-400">Failed:</span>{" "}
                        <span className="text-red-400">{job.lastExecution.metrics.itemsFailed}</span>
                      </div>
                    )}
                    {job.lastExecution.metrics.dataSize && (
                      <div className="text-sm">
                        <span className="text-gray-400">Data Size:</span>{" "}
                        <span className="text-white">{job.lastExecution.metrics.dataSize}</span>
                      </div>
                    )}
                  </div>
                )}

                {job.lastExecution.error && (
                  <div className="mt-2 p-2 bg-red-500/10 border border-red-500/20 rounded text-sm text-red-400">
                    <strong>Error:</strong> {job.lastExecution.error}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Tags */}
          {job.tags.length > 0 && (
            <div className="mt-4 flex items-center gap-2 flex-wrap">
              <span className="text-xs text-gray-500">Tags:</span>
              {job.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-0.5 rounded-full text-xs bg-white/5 text-gray-400 border border-white/10"
                >
                  #{tag}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Create Job Dialog
const CreateJobDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onSave: (job: Partial<ScheduledJob>) => void;
  editingJob?: ScheduledJob | null;
}> = ({ isOpen, onClose, onSave, editingJob }) => {
  const [step, setStep] = useState(1);
  const [jobData, setJobData] = useState<Partial<ScheduledJob>>({
    name: "",
    description: "",
    type: "report",
    frequency: "daily",
    timezone: "America/New_York",
    schedule: {
      time: "09:00",
      daysOfWeek: [],
    },
    retryPolicy: {
      maxRetries: 3,
      retryDelay: 300,
      exponentialBackoff: true,
    },
    notifications: {
      onSuccess: false,
      onFailure: true,
      channels: ["email"],
    },
    priority: "medium",
    timeout: 300,
    tags: [],
  });

  useEffect(() => {
    if (editingJob) {
      setJobData(editingJob);
    } else {
      setJobData({
        name: "",
        description: "",
        type: "report",
        frequency: "daily",
        timezone: "America/New_York",
        schedule: {
          time: "09:00",
          daysOfWeek: [],
        },
        retryPolicy: {
          maxRetries: 3,
          retryDelay: 300,
          exponentialBackoff: true,
        },
        notifications: {
          onSuccess: false,
          onFailure: true,
          channels: ["email"],
        },
        priority: "medium",
        timeout: 300,
        tags: [],
      });
    }
    setStep(1);
  }, [editingJob, isOpen]);

  if (!isOpen) return null;

  const jobTypes: { value: JobType; label: string; description: string }[] = [
    { value: "campaign", label: "Campaign", description: "Run outbound calling campaigns" },
    { value: "report", label: "Report", description: "Generate and send reports" },
    { value: "notification", label: "Notification", description: "Send scheduled notifications" },
    { value: "backup", label: "Backup", description: "Database and file backups" },
    { value: "sync", label: "Sync", description: "Synchronize data with external systems" },
    { value: "cleanup", label: "Cleanup", description: "Clean up old data and files" },
    { value: "webhook", label: "Webhook", description: "Process webhook events" },
    { value: "custom", label: "Custom", description: "Custom automation script" },
  ];

  const timezones = [
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Phoenix",
    "America/Anchorage",
    "Pacific/Honolulu",
    "UTC",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Asia/Tokyo",
    "Asia/Shanghai",
    "Asia/Singapore",
    "Australia/Sydney",
  ];

  const daysOfWeek: DayOfWeek[] = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];

  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Job Name</label>
              <input
                type="text"
                value={jobData.name}
                onChange={(e) => setJobData({ ...jobData, name: e.target.value })}
                className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                placeholder="e.g., Daily Sales Report"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Description</label>
              <textarea
                value={jobData.description}
                onChange={(e) => setJobData({ ...jobData, description: e.target.value })}
                className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                rows={3}
                placeholder="Describe what this job does..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-3">Job Type</label>
              <div className="grid grid-cols-2 gap-3">
                {jobTypes.map((type) => (
                  <button
                    key={type.value}
                    onClick={() => setJobData({ ...jobData, type: type.value })}
                    className={`p-4 rounded-lg border text-left transition-all ${
                      jobData.type === type.value
                        ? "border-purple-500 bg-purple-500/20"
                        : "border-white/10 bg-black/30 hover:border-white/20"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${getTypeColor(type.value)}`}>
                        {getTypeIcon(type.value)}
                      </div>
                      <div>
                        <p className="text-white font-medium">{type.label}</p>
                        <p className="text-xs text-gray-400">{type.description}</p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Priority</label>
              <div className="flex gap-2">
                {(["low", "medium", "high", "critical"] as const).map((priority) => (
                  <button
                    key={priority}
                    onClick={() => setJobData({ ...jobData, priority })}
                    className={`flex-1 py-2 rounded-lg border transition-all text-sm font-medium ${
                      jobData.priority === priority
                        ? `${getPriorityColor(priority)} border-current`
                        : "border-white/10 text-gray-400 hover:border-white/20"
                    }`}
                  >
                    {priority.charAt(0).toUpperCase() + priority.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-3">Frequency</label>
              <div className="grid grid-cols-3 gap-3">
                {(["once", "hourly", "daily", "weekly", "monthly", "custom"] as FrequencyType[]).map((freq) => (
                  <button
                    key={freq}
                    onClick={() => setJobData({ ...jobData, frequency: freq })}
                    className={`py-3 px-4 rounded-lg border transition-all ${
                      jobData.frequency === freq
                        ? "border-purple-500 bg-purple-500/20 text-white"
                        : "border-white/10 text-gray-400 hover:border-white/20"
                    }`}
                  >
                    {getFrequencyLabel(freq)}
                  </button>
                ))}
              </div>
            </div>

            {jobData.frequency === "custom" && (
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Cron Expression</label>
                <input
                  type="text"
                  value={jobData.cronExpression || ""}
                  onChange={(e) => setJobData({ ...jobData, cronExpression: e.target.value })}
                  className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white font-mono placeholder-gray-500 focus:outline-none focus:border-purple-500"
                  placeholder="*/5 * * * *"
                />
                <p className="mt-2 text-xs text-gray-500">
                  Format: minute hour day-of-month month day-of-week
                </p>
              </div>
            )}

            {(jobData.frequency === "daily" || jobData.frequency === "weekly" || jobData.frequency === "monthly" || jobData.frequency === "once") && (
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Time</label>
                <input
                  type="time"
                  value={jobData.schedule?.time || "09:00"}
                  onChange={(e) => setJobData({
                    ...jobData,
                    schedule: { ...jobData.schedule, time: e.target.value },
                  })}
                  className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                />
              </div>
            )}

            {jobData.frequency === "weekly" && (
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-3">Days of Week</label>
                <div className="flex gap-2">
                  {daysOfWeek.map((day) => (
                    <button
                      key={day}
                      onClick={() => {
                        const current = jobData.schedule?.daysOfWeek || [];
                        const updated = current.includes(day)
                          ? current.filter((d) => d !== day)
                          : [...current, day];
                        setJobData({
                          ...jobData,
                          schedule: { ...jobData.schedule, daysOfWeek: updated },
                        });
                      }}
                      className={`flex-1 py-2 rounded-lg border transition-all text-xs font-medium ${
                        jobData.schedule?.daysOfWeek?.includes(day)
                          ? "border-purple-500 bg-purple-500/20 text-white"
                          : "border-white/10 text-gray-400 hover:border-white/20"
                      }`}
                    >
                      {day.charAt(0).toUpperCase() + day.slice(1, 3)}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {jobData.frequency === "monthly" && (
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Day of Month</label>
                <select
                  value={jobData.schedule?.dayOfMonth || 1}
                  onChange={(e) => setJobData({
                    ...jobData,
                    schedule: { ...jobData.schedule, dayOfMonth: parseInt(e.target.value) },
                  })}
                  className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  {Array.from({ length: 31 }, (_, i) => (
                    <option key={i + 1} value={i + 1}>{i + 1}</option>
                  ))}
                </select>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Timezone</label>
              <select
                value={jobData.timezone}
                onChange={(e) => setJobData({ ...jobData, timezone: e.target.value })}
                className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
              >
                {timezones.map((tz) => (
                  <option key={tz} value={tz}>{tz}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Timeout (seconds)</label>
              <input
                type="number"
                value={jobData.timeout}
                onChange={(e) => setJobData({ ...jobData, timeout: parseInt(e.target.value) })}
                className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                min={60}
                max={86400}
              />
              <p className="mt-2 text-xs text-gray-500">
                Maximum execution time before the job is terminated
              </p>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-4">Retry Policy</h4>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs text-gray-400 mb-2">Maximum Retries</label>
                  <input
                    type="number"
                    value={jobData.retryPolicy?.maxRetries || 3}
                    onChange={(e) => setJobData({
                      ...jobData,
                      retryPolicy: { ...jobData.retryPolicy!, maxRetries: parseInt(e.target.value) },
                    })}
                    className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                    min={0}
                    max={10}
                  />
                </div>

                <div>
                  <label className="block text-xs text-gray-400 mb-2">Retry Delay (seconds)</label>
                  <input
                    type="number"
                    value={jobData.retryPolicy?.retryDelay || 300}
                    onChange={(e) => setJobData({
                      ...jobData,
                      retryPolicy: { ...jobData.retryPolicy!, retryDelay: parseInt(e.target.value) },
                    })}
                    className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                    min={30}
                    max={3600}
                  />
                </div>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={jobData.retryPolicy?.exponentialBackoff || false}
                    onChange={(e) => setJobData({
                      ...jobData,
                      retryPolicy: { ...jobData.retryPolicy!, exponentialBackoff: e.target.checked },
                    })}
                    className="w-4 h-4 rounded border-white/20 bg-black/30 text-purple-500 focus:ring-purple-500"
                  />
                  <div>
                    <p className="text-sm text-white">Exponential Backoff</p>
                    <p className="text-xs text-gray-400">Double the delay after each retry</p>
                  </div>
                </label>
              </div>
            </div>

            <div className="pt-4 border-t border-white/10">
              <h4 className="text-sm font-medium text-gray-300 mb-4">Notifications</h4>

              <div className="space-y-3">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={jobData.notifications?.onSuccess || false}
                    onChange={(e) => setJobData({
                      ...jobData,
                      notifications: { ...jobData.notifications!, onSuccess: e.target.checked },
                    })}
                    className="w-4 h-4 rounded border-white/20 bg-black/30 text-purple-500 focus:ring-purple-500"
                  />
                  <span className="text-sm text-white">Notify on success</span>
                </label>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={jobData.notifications?.onFailure || false}
                    onChange={(e) => setJobData({
                      ...jobData,
                      notifications: { ...jobData.notifications!, onFailure: e.target.checked },
                    })}
                    className="w-4 h-4 rounded border-white/20 bg-black/30 text-purple-500 focus:ring-purple-500"
                  />
                  <span className="text-sm text-white">Notify on failure</span>
                </label>
              </div>

              <div className="mt-4">
                <label className="block text-xs text-gray-400 mb-3">Notification Channels</label>
                <div className="flex gap-2 flex-wrap">
                  {(["email", "sms", "webhook", "slack"] as const).map((channel) => (
                    <button
                      key={channel}
                      onClick={() => {
                        const current = jobData.notifications?.channels || [];
                        const updated = current.includes(channel)
                          ? current.filter((c) => c !== channel)
                          : [...current, channel];
                        setJobData({
                          ...jobData,
                          notifications: { ...jobData.notifications!, channels: updated },
                        });
                      }}
                      className={`px-3 py-1.5 rounded-lg border text-sm transition-all ${
                        jobData.notifications?.channels?.includes(channel)
                          ? "border-purple-500 bg-purple-500/20 text-white"
                          : "border-white/10 text-gray-400 hover:border-white/20"
                      }`}
                    >
                      {channel.charAt(0).toUpperCase() + channel.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-white/10">
              <label className="block text-sm font-medium text-gray-300 mb-2">Tags</label>
              <input
                type="text"
                value={jobData.tags?.join(", ") || ""}
                onChange={(e) => setJobData({
                  ...jobData,
                  tags: e.target.value.split(",").map((t) => t.trim()).filter(Boolean),
                })}
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                placeholder="Enter tags separated by commas"
              />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-2xl bg-[#1a1a2e] border border-white/10 rounded-2xl overflow-hidden">
        {/* Header */}
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">
              {editingJob ? "Edit Scheduled Job" : "Create Scheduled Job"}
            </h2>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
            >
              <XCircle className="w-5 h-5" />
            </button>
          </div>

          {/* Step Indicator */}
          <div className="flex items-center gap-2 mt-4">
            {[1, 2, 3].map((s) => (
              <React.Fragment key={s}>
                <button
                  onClick={() => setStep(s)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
                    step === s
                      ? "bg-purple-500/20 text-purple-400"
                      : step > s
                      ? "text-green-400"
                      : "text-gray-500"
                  }`}
                >
                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${
                    step === s
                      ? "bg-purple-500 text-white"
                      : step > s
                      ? "bg-green-500 text-white"
                      : "bg-white/10 text-gray-500"
                  }`}>
                    {step > s ? <CheckCircle className="w-4 h-4" /> : s}
                  </span>
                  <span className="text-sm hidden sm:inline">
                    {s === 1 ? "Basic Info" : s === 2 ? "Schedule" : "Settings"}
                  </span>
                </button>
                {s < 3 && (
                  <div className={`flex-1 h-0.5 ${step > s ? "bg-green-500" : "bg-white/10"}`} />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {renderStep()}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-white/10 flex justify-between">
          <button
            onClick={() => step > 1 && setStep(step - 1)}
            className={`px-4 py-2 rounded-lg text-gray-400 hover:bg-white/10 transition-colors ${
              step === 1 ? "invisible" : ""
            }`}
          >
            Back
          </button>

          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-gray-400 hover:bg-white/10 transition-colors"
            >
              Cancel
            </button>
            {step < 3 ? (
              <button
                onClick={() => setStep(step + 1)}
                className="px-6 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium hover:opacity-90 transition-opacity"
              >
                Next
              </button>
            ) : (
              <button
                onClick={() => onSave(jobData)}
                className="px-6 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium hover:opacity-90 transition-opacity"
              >
                {editingJob ? "Save Changes" : "Create Job"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Execution History Dialog
const ExecutionHistoryDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  job: ScheduledJob | null;
}> = ({ isOpen, onClose, job }) => {
  if (!isOpen || !job) return null;

  const mockExecutions: JobExecution[] = [
    {
      id: "exec-h1",
      startedAt: new Date(Date.now() - 1 * 60 * 60 * 1000),
      completedAt: new Date(Date.now() - 1 * 60 * 60 * 1000 + 45000),
      status: "success",
      duration: 45,
      metrics: { itemsProcessed: 125, itemsFailed: 0 },
    },
    {
      id: "exec-h2",
      startedAt: new Date(Date.now() - 5 * 60 * 60 * 1000),
      completedAt: new Date(Date.now() - 5 * 60 * 60 * 1000 + 52000),
      status: "success",
      duration: 52,
      metrics: { itemsProcessed: 118, itemsFailed: 2 },
    },
    {
      id: "exec-h3",
      startedAt: new Date(Date.now() - 9 * 60 * 60 * 1000),
      status: "failed",
      error: "Database connection timeout",
    },
    {
      id: "exec-h4",
      startedAt: new Date(Date.now() - 13 * 60 * 60 * 1000),
      completedAt: new Date(Date.now() - 13 * 60 * 60 * 1000 + 38000),
      status: "success",
      duration: 38,
      metrics: { itemsProcessed: 97, itemsFailed: 0 },
    },
    {
      id: "exec-h5",
      startedAt: new Date(Date.now() - 17 * 60 * 60 * 1000),
      completedAt: new Date(Date.now() - 17 * 60 * 60 * 1000 + 41000),
      status: "success",
      duration: 41,
      metrics: { itemsProcessed: 112, itemsFailed: 1 },
    },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-3xl bg-[#1a1a2e] border border-white/10 rounded-2xl overflow-hidden">
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white">Execution History</h2>
              <p className="text-sm text-gray-400 mt-1">{job.name}</p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
            >
              <XCircle className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="p-6 max-h-[60vh] overflow-y-auto">
          <div className="space-y-3">
            {mockExecutions.map((exec) => (
              <div
                key={exec.id}
                className="p-4 bg-black/30 rounded-lg border border-white/5"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      exec.status === "success" ? "bg-green-500/20 text-green-400" :
                      exec.status === "failed" ? "bg-red-500/20 text-red-400" :
                      exec.status === "running" ? "bg-blue-500/20 text-blue-400" :
                      "bg-gray-500/20 text-gray-400"
                    }`}>
                      {exec.status.toUpperCase()}
                    </span>
                    <span className="text-sm text-gray-400">
                      {exec.startedAt.toLocaleString()}
                    </span>
                  </div>
                  {exec.duration && (
                    <span className="text-sm text-gray-400">
                      Duration: {formatDuration(exec.duration)}
                    </span>
                  )}
                </div>

                {exec.metrics && (
                  <div className="flex gap-4">
                    {exec.metrics.itemsProcessed !== undefined && (
                      <span className="text-sm text-gray-400">
                        Processed: <span className="text-white">{exec.metrics.itemsProcessed}</span>
                      </span>
                    )}
                    {exec.metrics.itemsFailed !== undefined && exec.metrics.itemsFailed > 0 && (
                      <span className="text-sm text-gray-400">
                        Failed: <span className="text-red-400">{exec.metrics.itemsFailed}</span>
                      </span>
                    )}
                  </div>
                )}

                {exec.error && (
                  <div className="mt-2 p-2 bg-red-500/10 border border-red-500/20 rounded text-sm text-red-400">
                    {exec.error}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="p-6 border-t border-white/10 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-gray-400 hover:bg-white/10 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

// Main Component
export default function ScheduledJobsPage() {
  const [jobs, setJobs] = useState<ScheduledJob[]>(mockJobs);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<JobStatus | "all">("all");
  const [typeFilter, setTypeFilter] = useState<JobType | "all">("all");
  const [expandedJobs, setExpandedJobs] = useState<Set<string>>(new Set());
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [editingJob, setEditingJob] = useState<ScheduledJob | null>(null);
  const [showHistoryDialog, setShowHistoryDialog] = useState(false);
  const [selectedJobForHistory, setSelectedJobForHistory] = useState<ScheduledJob | null>(null);

  // Filter jobs
  const filteredJobs = useMemo(() => {
    return jobs.filter((job) => {
      const matchesSearch =
        job.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        job.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        job.tags.some((t) => t.toLowerCase().includes(searchQuery.toLowerCase()));

      const matchesStatus = statusFilter === "all" || job.status === statusFilter;
      const matchesType = typeFilter === "all" || job.type === typeFilter;

      return matchesSearch && matchesStatus && matchesType;
    });
  }, [jobs, searchQuery, statusFilter, typeFilter]);

  // Stats
  const stats = useMemo(() => {
    const total = jobs.length;
    const active = jobs.filter((j) => j.status === "active").length;
    const paused = jobs.filter((j) => j.status === "paused").length;
    const failed = jobs.filter((j) => j.status === "failed").length;

    return { total, active, paused, failed };
  }, [jobs]);

  // Handlers
  const handleToggleExpand = (jobId: string) => {
    setExpandedJobs((prev) => {
      const next = new Set(prev);
      if (next.has(jobId)) {
        next.delete(jobId);
      } else {
        next.add(jobId);
      }
      return next;
    });
  };

  const handleEdit = (job: ScheduledJob) => {
    setEditingJob(job);
    setShowCreateDialog(true);
  };

  const handleDelete = (job: ScheduledJob) => {
    if (confirm(`Are you sure you want to delete "${job.name}"?`)) {
      setJobs((prev) => prev.filter((j) => j.id !== job.id));
    }
  };

  const handleToggle = (job: ScheduledJob) => {
    setJobs((prev) =>
      prev.map((j) =>
        j.id === job.id
          ? { ...j, status: j.status === "active" ? "paused" : "active" }
          : j
      )
    );
  };

  const handleRunNow = (job: ScheduledJob) => {
    console.log("Running job now:", job.name);
    // In real implementation, this would trigger the job
  };

  const handleViewHistory = (job: ScheduledJob) => {
    setSelectedJobForHistory(job);
    setShowHistoryDialog(true);
  };

  const handleSaveJob = (jobData: Partial<ScheduledJob>) => {
    if (editingJob) {
      setJobs((prev) =>
        prev.map((j) =>
          j.id === editingJob.id ? { ...j, ...jobData, updatedAt: new Date() } : j
        )
      );
    } else {
      const newJob: ScheduledJob = {
        ...jobData,
        id: `job-${Date.now()}`,
        status: "pending",
        createdAt: new Date(),
        updatedAt: new Date(),
        executions: [],
        isLocked: false,
        createdBy: "Current User",
        action: { type: jobData.type || "custom", config: {} },
      } as ScheduledJob;
      setJobs((prev) => [newJob, ...prev]);
    }
    setShowCreateDialog(false);
    setEditingJob(null);
  };

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Scheduled Jobs</h1>
            <p className="text-gray-400 mt-1">Manage and monitor your scheduled tasks and automations</p>
          </div>

          <button
            onClick={() => {
              setEditingJob(null);
              setShowCreateDialog(true);
            }}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90 transition-opacity"
          >
            <Plus className="w-4 h-4" />
            Create Job
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400">
                <Layers className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.total}</p>
                <p className="text-sm text-gray-400">Total Jobs</p>
              </div>
            </div>
          </div>

          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-500/20 text-green-400">
                <PlayCircle className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.active}</p>
                <p className="text-sm text-gray-400">Active</p>
              </div>
            </div>
          </div>

          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-yellow-500/20 text-yellow-400">
                <Pause className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.paused}</p>
                <p className="text-sm text-gray-400">Paused</p>
              </div>
            </div>
          </div>

          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-red-500/20 text-red-400">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.failed}</p>
                <p className="text-sm text-gray-400">Failed</p>
              </div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-col md:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search jobs by name, description, or tags..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
            />
          </div>

          <div className="flex gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as JobStatus | "all")}
              className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Statuses</option>
              <option value="active">Active</option>
              <option value="paused">Paused</option>
              <option value="pending">Pending</option>
              <option value="failed">Failed</option>
              <option value="completed">Completed</option>
            </select>

            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value as JobType | "all")}
              className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Types</option>
              <option value="campaign">Campaign</option>
              <option value="report">Report</option>
              <option value="notification">Notification</option>
              <option value="backup">Backup</option>
              <option value="sync">Sync</option>
              <option value="cleanup">Cleanup</option>
              <option value="webhook">Webhook</option>
              <option value="custom">Custom</option>
            </select>
          </div>
        </div>

        {/* Jobs List */}
        <div className="space-y-4">
          {filteredJobs.length > 0 ? (
            filteredJobs.map((job) => (
              <JobCard
                key={job.id}
                job={job}
                onEdit={handleEdit}
                onDelete={handleDelete}
                onToggle={handleToggle}
                onRunNow={handleRunNow}
                onViewHistory={handleViewHistory}
                isExpanded={expandedJobs.has(job.id)}
                onToggleExpand={() => handleToggleExpand(job.id)}
              />
            ))
          ) : (
            <div className="text-center py-16 bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl">
              <Calendar className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">No jobs found</h3>
              <p className="text-gray-400 mb-6">
                {searchQuery || statusFilter !== "all" || typeFilter !== "all"
                  ? "No jobs match your current filters"
                  : "Create your first scheduled job to automate tasks"}
              </p>
              <button
                onClick={() => {
                  setEditingJob(null);
                  setShowCreateDialog(true);
                }}
                className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors"
              >
                <Plus className="w-4 h-4" />
                Create Job
              </button>
            </div>
          )}
        </div>

        {/* Create/Edit Dialog */}
        <CreateJobDialog
          isOpen={showCreateDialog}
          onClose={() => {
            setShowCreateDialog(false);
            setEditingJob(null);
          }}
          onSave={handleSaveJob}
          editingJob={editingJob}
        />

        {/* Execution History Dialog */}
        <ExecutionHistoryDialog
          isOpen={showHistoryDialog}
          onClose={() => {
            setShowHistoryDialog(false);
            setSelectedJobForHistory(null);
          }}
          job={selectedJobForHistory}
        />
      </div>
    </DashboardLayout>
  );
}
