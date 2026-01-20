"use client";

import { useState, useMemo } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import {
  Shield,
  Search,
  Filter,
  Download,
  RefreshCw,
  Calendar,
  Clock,
  User,
  Users,
  Bot,
  Settings,
  Key,
  Lock,
  Unlock,
  Eye,
  Edit,
  Trash2,
  Plus,
  Copy,
  ExternalLink,
  AlertTriangle,
  AlertCircle,
  CheckCircle,
  XCircle,
  Info,
  ChevronRight,
  ChevronDown,
  ChevronLeft,
  MoreVertical,
  Activity,
  FileText,
  Phone,
  PhoneCall,
  MessageSquare,
  Mail,
  CreditCard,
  Building2,
  Globe,
  Server,
  Database,
  Webhook,
  Zap,
  PlayCircle,
  PauseCircle,
  StopCircle,
  LogIn,
  LogOut,
  UserPlus,
  UserMinus,
  UserCheck,
  UserX,
  Link,
  Unlink,
  Upload,
  Send,
  X,
  LayoutGrid,
  List,
  SlidersHorizontal,
  Tag,
  Hash,
  ArrowUpRight,
  ArrowDownRight,
  Laptop,
  Smartphone,
  Map,
  MapPin,
  Fingerprint,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  History,
  Timer,
} from "lucide-react";

// Types
type AuditEventType =
  | "auth"
  | "user"
  | "agent"
  | "call"
  | "campaign"
  | "integration"
  | "billing"
  | "settings"
  | "api"
  | "security";

type AuditSeverity = "info" | "warning" | "critical" | "success";

type AuditAction =
  | "created"
  | "updated"
  | "deleted"
  | "viewed"
  | "exported"
  | "imported"
  | "enabled"
  | "disabled"
  | "login"
  | "logout"
  | "failed"
  | "started"
  | "stopped"
  | "connected"
  | "disconnected";

interface AuditLogEntry {
  id: string;
  timestamp: string;
  eventType: AuditEventType;
  action: AuditAction;
  severity: AuditSeverity;
  actor: {
    id: string;
    name: string;
    email: string;
    avatar?: string;
    type: "user" | "system" | "api";
  };
  resource: {
    type: string;
    id: string;
    name: string;
  };
  description: string;
  changes?: {
    field: string;
    oldValue: any;
    newValue: any;
  }[];
  metadata: {
    ipAddress: string;
    userAgent: string;
    location?: string;
    device?: string;
    sessionId?: string;
  };
  tags: string[];
}

interface AuditFilter {
  eventTypes: AuditEventType[];
  actions: AuditAction[];
  severities: AuditSeverity[];
  actors: string[];
  dateRange: {
    start: string;
    end: string;
  };
  searchQuery: string;
}

// Sample Data
const sampleAuditLogs: AuditLogEntry[] = [
  {
    id: "audit_001",
    timestamp: "2024-01-20T16:45:32Z",
    eventType: "auth",
    action: "login",
    severity: "success",
    actor: {
      id: "user_1",
      name: "John Smith",
      email: "john@company.com",
      type: "user",
    },
    resource: { type: "session", id: "sess_abc123", name: "Web Session" },
    description: "User logged in successfully",
    metadata: {
      ipAddress: "192.168.1.100",
      userAgent: "Chrome 120 / macOS",
      location: "New York, NY, US",
      device: "Desktop",
      sessionId: "sess_abc123",
    },
    tags: ["auth", "login"],
  },
  {
    id: "audit_002",
    timestamp: "2024-01-20T16:30:15Z",
    eventType: "agent",
    action: "created",
    severity: "info",
    actor: {
      id: "user_2",
      name: "Sarah Johnson",
      email: "sarah@company.com",
      type: "user",
    },
    resource: { type: "agent", id: "agent_xyz", name: "Sales Assistant AI" },
    description: 'Created new AI agent "Sales Assistant AI"',
    changes: [
      { field: "name", oldValue: null, newValue: "Sales Assistant AI" },
      { field: "voice_id", oldValue: null, newValue: "voice_123" },
      { field: "language", oldValue: null, newValue: "en-US" },
    ],
    metadata: {
      ipAddress: "192.168.1.101",
      userAgent: "Firefox 121 / Windows",
      location: "Los Angeles, CA, US",
      device: "Desktop",
    },
    tags: ["agent", "create"],
  },
  {
    id: "audit_003",
    timestamp: "2024-01-20T16:15:00Z",
    eventType: "security",
    action: "failed",
    severity: "warning",
    actor: {
      id: "unknown",
      name: "Unknown",
      email: "attacker@suspicious.com",
      type: "user",
    },
    resource: { type: "auth", id: "auth_attempt", name: "Login Attempt" },
    description: "Failed login attempt - invalid password (3rd attempt)",
    metadata: {
      ipAddress: "203.0.113.42",
      userAgent: "curl/7.68.0",
      location: "Unknown",
      device: "Unknown",
    },
    tags: ["security", "failed-login", "suspicious"],
  },
  {
    id: "audit_004",
    timestamp: "2024-01-20T15:45:22Z",
    eventType: "campaign",
    action: "started",
    severity: "info",
    actor: {
      id: "user_3",
      name: "Mike Davis",
      email: "mike@company.com",
      type: "user",
    },
    resource: { type: "campaign", id: "camp_q1", name: "Q1 Sales Outreach" },
    description: 'Started campaign "Q1 Sales Outreach" with 5,000 contacts',
    metadata: {
      ipAddress: "192.168.1.102",
      userAgent: "Chrome 120 / Windows",
      location: "Chicago, IL, US",
      device: "Desktop",
    },
    tags: ["campaign", "started"],
  },
  {
    id: "audit_005",
    timestamp: "2024-01-20T15:30:10Z",
    eventType: "billing",
    action: "updated",
    severity: "info",
    actor: {
      id: "user_1",
      name: "John Smith",
      email: "john@company.com",
      type: "user",
    },
    resource: { type: "subscription", id: "sub_pro", name: "Pro Plan" },
    description: "Upgraded subscription from Starter to Pro Plan",
    changes: [
      { field: "plan", oldValue: "Starter", newValue: "Pro" },
      { field: "price", oldValue: "$49/mo", newValue: "$149/mo" },
    ],
    metadata: {
      ipAddress: "192.168.1.100",
      userAgent: "Chrome 120 / macOS",
      location: "New York, NY, US",
      device: "Desktop",
    },
    tags: ["billing", "upgrade"],
  },
  {
    id: "audit_006",
    timestamp: "2024-01-20T15:00:00Z",
    eventType: "integration",
    action: "connected",
    severity: "success",
    actor: {
      id: "user_2",
      name: "Sarah Johnson",
      email: "sarah@company.com",
      type: "user",
    },
    resource: { type: "integration", id: "int_sf", name: "Salesforce" },
    description: "Connected Salesforce integration",
    metadata: {
      ipAddress: "192.168.1.101",
      userAgent: "Firefox 121 / Windows",
      location: "Los Angeles, CA, US",
      device: "Desktop",
    },
    tags: ["integration", "salesforce"],
  },
  {
    id: "audit_007",
    timestamp: "2024-01-20T14:30:45Z",
    eventType: "user",
    action: "created",
    severity: "info",
    actor: {
      id: "user_1",
      name: "John Smith",
      email: "john@company.com",
      type: "user",
    },
    resource: { type: "user", id: "user_new", name: "Emily Davis" },
    description: 'Invited new team member "Emily Davis" with Admin role',
    changes: [
      { field: "email", oldValue: null, newValue: "emily@company.com" },
      { field: "role", oldValue: null, newValue: "Admin" },
    ],
    metadata: {
      ipAddress: "192.168.1.100",
      userAgent: "Chrome 120 / macOS",
      location: "New York, NY, US",
      device: "Desktop",
    },
    tags: ["user", "invite"],
  },
  {
    id: "audit_008",
    timestamp: "2024-01-20T14:00:00Z",
    eventType: "api",
    action: "created",
    severity: "info",
    actor: {
      id: "user_3",
      name: "Mike Davis",
      email: "mike@company.com",
      type: "user",
    },
    resource: { type: "api_key", id: "key_prod", name: "Production API Key" },
    description: "Generated new API key for production environment",
    metadata: {
      ipAddress: "192.168.1.102",
      userAgent: "Chrome 120 / Windows",
      location: "Chicago, IL, US",
      device: "Desktop",
    },
    tags: ["api", "key"],
  },
  {
    id: "audit_009",
    timestamp: "2024-01-20T13:30:22Z",
    eventType: "settings",
    action: "updated",
    severity: "warning",
    actor: {
      id: "user_1",
      name: "John Smith",
      email: "john@company.com",
      type: "user",
    },
    resource: { type: "security_settings", id: "sec_2fa", name: "Two-Factor Authentication" },
    description: "Disabled two-factor authentication",
    changes: [{ field: "2fa_enabled", oldValue: true, newValue: false }],
    metadata: {
      ipAddress: "192.168.1.100",
      userAgent: "Chrome 120 / macOS",
      location: "New York, NY, US",
      device: "Desktop",
    },
    tags: ["settings", "security", "2fa"],
  },
  {
    id: "audit_010",
    timestamp: "2024-01-20T13:00:00Z",
    eventType: "call",
    action: "exported",
    severity: "info",
    actor: {
      id: "system",
      name: "System",
      email: "system@bvrai.com",
      type: "system",
    },
    resource: { type: "calls", id: "export_123", name: "Call Export" },
    description: "Exported 1,250 call records to CSV",
    metadata: {
      ipAddress: "10.0.0.1",
      userAgent: "BVRAI System",
      location: "Cloud",
      device: "Server",
    },
    tags: ["call", "export", "automated"],
  },
  {
    id: "audit_011",
    timestamp: "2024-01-20T12:45:00Z",
    eventType: "agent",
    action: "deleted",
    severity: "critical",
    actor: {
      id: "user_2",
      name: "Sarah Johnson",
      email: "sarah@company.com",
      type: "user",
    },
    resource: { type: "agent", id: "agent_old", name: "Legacy Support Bot" },
    description: 'Permanently deleted agent "Legacy Support Bot"',
    metadata: {
      ipAddress: "192.168.1.101",
      userAgent: "Firefox 121 / Windows",
      location: "Los Angeles, CA, US",
      device: "Desktop",
    },
    tags: ["agent", "delete"],
  },
  {
    id: "audit_012",
    timestamp: "2024-01-20T12:00:00Z",
    eventType: "security",
    action: "login",
    severity: "warning",
    actor: {
      id: "user_4",
      name: "Robert Wilson",
      email: "robert@company.com",
      type: "user",
    },
    resource: { type: "session", id: "sess_def456", name: "Mobile Session" },
    description: "Login from new device detected",
    metadata: {
      ipAddress: "74.125.224.72",
      userAgent: "Safari / iOS 17",
      location: "San Francisco, CA, US",
      device: "iPhone 15 Pro",
    },
    tags: ["auth", "new-device"],
  },
];

// Utility functions
const getEventTypeIcon = (type: AuditEventType) => {
  switch (type) {
    case "auth":
      return LogIn;
    case "user":
      return User;
    case "agent":
      return Bot;
    case "call":
      return PhoneCall;
    case "campaign":
      return Send;
    case "integration":
      return Link;
    case "billing":
      return CreditCard;
    case "settings":
      return Settings;
    case "api":
      return Key;
    case "security":
      return Shield;
    default:
      return Activity;
  }
};

const getEventTypeColor = (type: AuditEventType) => {
  switch (type) {
    case "auth":
      return "bg-blue-500/20 text-blue-400";
    case "user":
      return "bg-purple-500/20 text-purple-400";
    case "agent":
      return "bg-pink-500/20 text-pink-400";
    case "call":
      return "bg-green-500/20 text-green-400";
    case "campaign":
      return "bg-orange-500/20 text-orange-400";
    case "integration":
      return "bg-cyan-500/20 text-cyan-400";
    case "billing":
      return "bg-yellow-500/20 text-yellow-400";
    case "settings":
      return "bg-gray-500/20 text-gray-400";
    case "api":
      return "bg-indigo-500/20 text-indigo-400";
    case "security":
      return "bg-red-500/20 text-red-400";
    default:
      return "bg-gray-500/20 text-gray-400";
  }
};

const getSeverityColor = (severity: AuditSeverity) => {
  switch (severity) {
    case "info":
      return "bg-blue-500/20 text-blue-400 border-blue-500/30";
    case "warning":
      return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
    case "critical":
      return "bg-red-500/20 text-red-400 border-red-500/30";
    case "success":
      return "bg-green-500/20 text-green-400 border-green-500/30";
    default:
      return "bg-gray-500/20 text-gray-400 border-gray-500/30";
  }
};

const getSeverityIcon = (severity: AuditSeverity) => {
  switch (severity) {
    case "info":
      return <Info className="h-4 w-4" />;
    case "warning":
      return <AlertTriangle className="h-4 w-4" />;
    case "critical":
      return <AlertCircle className="h-4 w-4" />;
    case "success":
      return <CheckCircle className="h-4 w-4" />;
    default:
      return <Info className="h-4 w-4" />;
  }
};

const getActionIcon = (action: AuditAction) => {
  switch (action) {
    case "created":
      return <Plus className="h-3.5 w-3.5" />;
    case "updated":
      return <Edit className="h-3.5 w-3.5" />;
    case "deleted":
      return <Trash2 className="h-3.5 w-3.5" />;
    case "viewed":
      return <Eye className="h-3.5 w-3.5" />;
    case "exported":
      return <Download className="h-3.5 w-3.5" />;
    case "imported":
      return <Upload className="h-3.5 w-3.5" />;
    case "enabled":
      return <CheckCircle className="h-3.5 w-3.5" />;
    case "disabled":
      return <XCircle className="h-3.5 w-3.5" />;
    case "login":
      return <LogIn className="h-3.5 w-3.5" />;
    case "logout":
      return <LogOut className="h-3.5 w-3.5" />;
    case "failed":
      return <XCircle className="h-3.5 w-3.5" />;
    case "started":
      return <PlayCircle className="h-3.5 w-3.5" />;
    case "stopped":
      return <StopCircle className="h-3.5 w-3.5" />;
    case "connected":
      return <Link className="h-3.5 w-3.5" />;
    case "disconnected":
      return <Unlink className="h-3.5 w-3.5" />;
    default:
      return <Activity className="h-3.5 w-3.5" />;
  }
};

const formatDateTime = (dateString: string) => {
  return new Date(dateString).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
};

const formatRelativeTime = (dateString: string) => {
  const now = new Date();
  const date = new Date(dateString);
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return formatDateTime(dateString);
};

// Components
function StatsCard({
  title,
  value,
  change,
  changeType,
  icon: Icon,
  color,
}: {
  title: string;
  value: string | number;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: any;
  color: string;
}) {
  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4 hover:border-white/10 transition-all">
      <div className="flex items-center justify-between">
        <div className="p-2 rounded-lg" style={{ backgroundColor: `${color}20` }}>
          <Icon className="h-4 w-4" style={{ color }} />
        </div>
        {change && (
          <div
            className={`flex items-center gap-1 text-xs font-medium ${
              changeType === "positive"
                ? "text-green-400"
                : changeType === "negative"
                ? "text-red-400"
                : "text-gray-400"
            }`}
          >
            {changeType === "positive" ? (
              <ArrowUpRight className="h-3 w-3" />
            ) : changeType === "negative" ? (
              <ArrowDownRight className="h-3 w-3" />
            ) : null}
            {change}
          </div>
        )}
      </div>
      <div className="mt-2">
        <div className="text-xl font-bold text-white">{value}</div>
        <div className="text-xs text-gray-400">{title}</div>
      </div>
    </div>
  );
}

function AuditLogRow({
  entry,
  isExpanded,
  onToggleExpand,
}: {
  entry: AuditLogEntry;
  isExpanded: boolean;
  onToggleExpand: () => void;
}) {
  const EventIcon = getEventTypeIcon(entry.eventType);

  return (
    <div
      className={`border-b border-white/5 last:border-0 transition-colors ${
        entry.severity === "critical" || entry.severity === "warning"
          ? "bg-" + (entry.severity === "critical" ? "red" : "yellow") + "-500/5"
          : ""
      }`}
    >
      <div
        className="px-4 py-3 cursor-pointer hover:bg-white/[0.02] transition-colors"
        onClick={onToggleExpand}
      >
        <div className="flex items-center gap-4">
          {/* Severity Indicator */}
          <div className={`w-1 h-10 rounded-full ${getSeverityColor(entry.severity).split(" ")[0]}`} />

          {/* Event Type Icon */}
          <div className={`p-2 rounded-lg ${getEventTypeColor(entry.eventType)}`}>
            <EventIcon className="h-4 w-4" />
          </div>

          {/* Main Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-0.5">
              <span className="font-medium text-white">{entry.description}</span>
              <span className={`px-1.5 py-0.5 rounded text-xs font-medium border ${getSeverityColor(entry.severity)}`}>
                {entry.severity}
              </span>
            </div>
            <div className="flex items-center gap-3 text-sm text-gray-400">
              <span className="flex items-center gap-1">
                {entry.actor.type === "user" ? (
                  <User className="h-3 w-3" />
                ) : entry.actor.type === "system" ? (
                  <Server className="h-3 w-3" />
                ) : (
                  <Key className="h-3 w-3" />
                )}
                {entry.actor.name}
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatRelativeTime(entry.timestamp)}
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <MapPin className="h-3 w-3" />
                {entry.metadata.location || entry.metadata.ipAddress}
              </span>
            </div>
          </div>

          {/* Action Badge */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-white/5 rounded-lg text-sm text-gray-400">
            {getActionIcon(entry.action)}
            <span className="capitalize">{entry.action}</span>
          </div>

          {/* Expand Toggle */}
          <button className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400">
            {isExpanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-4 pb-4 pl-[72px]">
          <div className="bg-[#0d0d1a] rounded-xl p-4 space-y-4">
            {/* Metadata */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Event ID</div>
                <div className="text-sm text-gray-300 font-mono">{entry.id}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Timestamp</div>
                <div className="text-sm text-gray-300">{formatDateTime(entry.timestamp)}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">IP Address</div>
                <div className="text-sm text-gray-300 font-mono">{entry.metadata.ipAddress}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Device</div>
                <div className="text-sm text-gray-300">{entry.metadata.device || "Unknown"}</div>
              </div>
            </div>

            {/* Actor Info */}
            <div className="flex items-center gap-4 p-3 bg-white/5 rounded-lg">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white font-bold">
                {entry.actor.name.charAt(0)}
              </div>
              <div>
                <div className="font-medium text-white">{entry.actor.name}</div>
                <div className="text-sm text-gray-400">{entry.actor.email}</div>
              </div>
              <div className="ml-auto text-xs px-2 py-1 bg-white/5 rounded-lg text-gray-400 capitalize">
                {entry.actor.type}
              </div>
            </div>

            {/* Resource */}
            <div>
              <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Resource</div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-400 capitalize">{entry.resource.type}:</span>
                <span className="text-sm text-white">{entry.resource.name}</span>
                <span className="text-xs text-gray-500 font-mono">({entry.resource.id})</span>
              </div>
            </div>

            {/* Changes */}
            {entry.changes && entry.changes.length > 0 && (
              <div>
                <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Changes</div>
                <div className="space-y-2">
                  {entry.changes.map((change, index) => (
                    <div key={index} className="flex items-center gap-3 text-sm">
                      <span className="text-gray-400 font-medium">{change.field}:</span>
                      {change.oldValue !== null && (
                        <>
                          <span className="px-2 py-0.5 bg-red-500/20 text-red-400 rounded line-through">
                            {JSON.stringify(change.oldValue)}
                          </span>
                          <ChevronRight className="h-3 w-3 text-gray-500" />
                        </>
                      )}
                      <span className="px-2 py-0.5 bg-green-500/20 text-green-400 rounded">
                        {JSON.stringify(change.newValue)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* User Agent */}
            <div>
              <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">User Agent</div>
              <div className="text-sm text-gray-400">{entry.metadata.userAgent}</div>
            </div>

            {/* Tags */}
            {entry.tags.length > 0 && (
              <div className="flex items-center gap-2">
                {entry.tags.map((tag) => (
                  <span key={tag} className="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs">
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function FilterPanel({
  filters,
  onFiltersChange,
  onClose,
}: {
  filters: AuditFilter;
  onFiltersChange: (filters: AuditFilter) => void;
  onClose: () => void;
}) {
  const eventTypes: AuditEventType[] = [
    "auth",
    "user",
    "agent",
    "call",
    "campaign",
    "integration",
    "billing",
    "settings",
    "api",
    "security",
  ];

  const severities: AuditSeverity[] = ["info", "success", "warning", "critical"];

  const toggleEventType = (type: AuditEventType) => {
    const newTypes = filters.eventTypes.includes(type)
      ? filters.eventTypes.filter((t) => t !== type)
      : [...filters.eventTypes, type];
    onFiltersChange({ ...filters, eventTypes: newTypes });
  };

  const toggleSeverity = (severity: AuditSeverity) => {
    const newSeverities = filters.severities.includes(severity)
      ? filters.severities.filter((s) => s !== severity)
      : [...filters.severities, severity];
    onFiltersChange({ ...filters, severities: newSeverities });
  };

  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4 mb-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <SlidersHorizontal className="h-4 w-4 text-purple-400" />
          Filters
        </h3>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="space-y-4">
        {/* Event Types */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Event Types</label>
          <div className="flex flex-wrap gap-2">
            {eventTypes.map((type) => {
              const Icon = getEventTypeIcon(type);
              const isSelected = filters.eventTypes.includes(type);
              return (
                <button
                  key={type}
                  onClick={() => toggleEventType(type)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1.5 transition-colors ${
                    isSelected
                      ? getEventTypeColor(type)
                      : "bg-white/5 text-gray-400 hover:bg-white/10"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  <span className="capitalize">{type}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Severity */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Severity</label>
          <div className="flex flex-wrap gap-2">
            {severities.map((severity) => {
              const isSelected = filters.severities.includes(severity);
              return (
                <button
                  key={severity}
                  onClick={() => toggleSeverity(severity)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1.5 transition-colors ${
                    isSelected
                      ? getSeverityColor(severity)
                      : "bg-white/5 text-gray-400 hover:bg-white/10"
                  }`}
                >
                  {getSeverityIcon(severity)}
                  <span className="capitalize">{severity}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Date Range */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Date Range</label>
          <div className="grid grid-cols-2 gap-3">
            <input
              type="date"
              value={filters.dateRange.start}
              onChange={(e) =>
                onFiltersChange({
                  ...filters,
                  dateRange: { ...filters.dateRange, start: e.target.value },
                })
              }
              className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            />
            <input
              type="date"
              value={filters.dateRange.end}
              onChange={(e) =>
                onFiltersChange({
                  ...filters,
                  dateRange: { ...filters.dateRange, end: e.target.value },
                })
              }
              className="px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={() =>
              onFiltersChange({
                eventTypes: [],
                actions: [],
                severities: [],
                actors: [],
                dateRange: { start: "", end: "" },
                searchQuery: "",
              })
            }
            className="px-4 py-2 bg-white/5 text-gray-300 rounded-lg text-sm font-medium hover:bg-white/10 transition-colors"
          >
            Clear All
          </button>
        </div>
      </div>
    </div>
  );
}

// Main Page Component
export default function AuditLogPage() {
  const [auditLogs] = useState<AuditLogEntry[]>(sampleAuditLogs);
  const [searchQuery, setSearchQuery] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [filters, setFilters] = useState<AuditFilter>({
    eventTypes: [],
    actions: [],
    severities: [],
    actors: [],
    dateRange: { start: "", end: "" },
    searchQuery: "",
  });
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  // Filter logs
  const filteredLogs = useMemo(() => {
    let filtered = auditLogs;

    if (filters.eventTypes.length > 0) {
      filtered = filtered.filter((log) => filters.eventTypes.includes(log.eventType));
    }

    if (filters.severities.length > 0) {
      filtered = filtered.filter((log) => filters.severities.includes(log.severity));
    }

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (log) =>
          log.description.toLowerCase().includes(query) ||
          log.actor.name.toLowerCase().includes(query) ||
          log.actor.email.toLowerCase().includes(query) ||
          log.resource.name.toLowerCase().includes(query)
      );
    }

    return filtered;
  }, [auditLogs, filters, searchQuery]);

  // Calculate stats
  const stats = useMemo(() => {
    const today = new Date().toDateString();
    const todayLogs = auditLogs.filter((log) => new Date(log.timestamp).toDateString() === today);
    const securityEvents = auditLogs.filter((log) => log.eventType === "security").length;
    const criticalEvents = auditLogs.filter((log) => log.severity === "critical").length;
    const warningEvents = auditLogs.filter((log) => log.severity === "warning").length;

    return {
      total: auditLogs.length,
      today: todayLogs.length,
      security: securityEvents,
      critical: criticalEvents,
      warning: warningEvents,
    };
  }, [auditLogs]);

  // Pagination
  const totalPages = Math.ceil(filteredLogs.length / itemsPerPage);
  const paginatedLogs = filteredLogs.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  return (
    <DashboardLayout>
      <div className="p-6 lg:p-8 max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
          <div>
            <h1 className="text-2xl lg:text-3xl font-bold text-white flex items-center gap-3">
              <Shield className="h-7 w-7 text-purple-400" />
              Audit Log
            </h1>
            <p className="text-gray-400 mt-1">Track all activities and changes in your account</p>
          </div>
          <div className="flex items-center gap-3">
            <button className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white font-medium hover:bg-white/10 transition-colors flex items-center gap-2">
              <Download className="h-4 w-4" />
              Export
            </button>
            <button className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white font-medium hover:bg-white/10 transition-colors flex items-center gap-2">
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
          <StatsCard title="Total Events" value={stats.total} icon={Activity} color="#8B5CF6" />
          <StatsCard title="Today" value={stats.today} icon={Calendar} color="#3B82F6" />
          <StatsCard title="Security Events" value={stats.security} icon={Shield} color="#10B981" />
          <StatsCard
            title="Critical"
            value={stats.critical}
            icon={AlertCircle}
            color="#EF4444"
          />
          <StatsCard
            title="Warnings"
            value={stats.warning}
            icon={AlertTriangle}
            color="#F59E0B"
          />
        </div>

        {/* Search & Filters */}
        <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4 mb-4">
          <div className="flex items-center gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
              <input
                type="text"
                placeholder="Search events, users, resources..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`px-4 py-2.5 rounded-xl font-medium flex items-center gap-2 transition-colors ${
                showFilters
                  ? "bg-purple-500/20 text-purple-400"
                  : "bg-white/5 text-gray-300 hover:bg-white/10"
              }`}
            >
              <Filter className="h-4 w-4" />
              Filters
              {(filters.eventTypes.length > 0 || filters.severities.length > 0) && (
                <span className="ml-1 w-5 h-5 rounded-full bg-purple-500 text-white text-xs flex items-center justify-center">
                  {filters.eventTypes.length + filters.severities.length}
                </span>
              )}
            </button>
          </div>
        </div>

        {/* Filters Panel */}
        {showFilters && (
          <FilterPanel
            filters={filters}
            onFiltersChange={setFilters}
            onClose={() => setShowFilters(false)}
          />
        )}

        {/* Audit Log List */}
        <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 overflow-hidden">
          {/* Header */}
          <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
            <div className="text-sm text-gray-400">
              Showing {paginatedLogs.length} of {filteredLogs.length} events
            </div>
            <div className="text-sm text-gray-400">
              Page {currentPage} of {totalPages}
            </div>
          </div>

          {/* Log Entries */}
          <div className="divide-y divide-white/5">
            {paginatedLogs.length === 0 ? (
              <div className="p-12 text-center">
                <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mx-auto mb-4">
                  <History className="h-8 w-8 text-purple-400" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">No events found</h3>
                <p className="text-gray-400">
                  {searchQuery || filters.eventTypes.length > 0 || filters.severities.length > 0
                    ? "Try adjusting your search or filters."
                    : "No audit events have been recorded yet."}
                </p>
              </div>
            ) : (
              paginatedLogs.map((entry) => (
                <AuditLogRow
                  key={entry.id}
                  entry={entry}
                  isExpanded={expandedId === entry.id}
                  onToggleExpand={() => setExpandedId(expandedId === entry.id ? null : entry.id)}
                />
              ))
            )}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="px-4 py-3 border-t border-white/5 flex items-center justify-between">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-4 py-2 bg-white/5 text-gray-300 rounded-lg text-sm font-medium hover:bg-white/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                <ChevronLeft className="h-4 w-4" />
                Previous
              </button>
              <div className="flex items-center gap-2">
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  const page = i + 1;
                  return (
                    <button
                      key={page}
                      onClick={() => setCurrentPage(page)}
                      className={`w-8 h-8 rounded-lg text-sm font-medium transition-colors ${
                        currentPage === page
                          ? "bg-purple-500 text-white"
                          : "bg-white/5 text-gray-400 hover:bg-white/10"
                      }`}
                    >
                      {page}
                    </button>
                  );
                })}
              </div>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-4 py-2 bg-white/5 text-gray-300 rounded-lg text-sm font-medium hover:bg-white/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
