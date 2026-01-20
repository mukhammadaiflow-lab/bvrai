"use client";

import React, { useState, useMemo, useEffect, useRef } from "react";
import {
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  PhoneMissed,
  Bot,
  Users,
  User,
  Settings,
  Zap,
  Workflow,
  CreditCard,
  Shield,
  AlertCircle,
  CheckCircle,
  XCircle,
  Clock,
  Calendar,
  MessageSquare,
  Mail,
  Webhook,
  Database,
  Upload,
  Download,
  Edit,
  Trash2,
  Plus,
  Copy,
  Eye,
  Play,
  Pause,
  RefreshCw,
  Link,
  Unlink,
  Star,
  Flag,
  Tag,
  Filter,
  Search,
  ChevronDown,
  ChevronRight,
  MoreHorizontal,
  ExternalLink,
  ArrowRight,
  Activity,
  TrendingUp,
  Sparkles,
  Bell,
  Volume2,
  Mic,
  FileText,
  Globe,
  Key,
  Lock,
  Unlock,
  Server,
  Code,
  GitBranch,
  Terminal,
  Send,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";

// Types
export interface ActivityItem {
  id: string;
  type: ActivityType;
  category: ActivityCategory;
  title: string;
  description?: string;
  timestamp: string;
  user?: {
    id: string;
    name: string;
    avatar?: string;
    email?: string;
  };
  metadata?: Record<string, any>;
  status?: "success" | "error" | "warning" | "info" | "pending";
  actionUrl?: string;
  children?: ActivityItem[];
  isNew?: boolean;
}

export type ActivityType =
  | "call_started"
  | "call_ended"
  | "call_missed"
  | "call_transferred"
  | "agent_created"
  | "agent_updated"
  | "agent_deleted"
  | "agent_activated"
  | "agent_deactivated"
  | "workflow_created"
  | "workflow_run"
  | "workflow_error"
  | "integration_connected"
  | "integration_disconnected"
  | "integration_error"
  | "team_member_added"
  | "team_member_removed"
  | "team_member_role_changed"
  | "billing_payment"
  | "billing_subscription"
  | "billing_invoice"
  | "security_login"
  | "security_logout"
  | "security_password_changed"
  | "security_2fa_enabled"
  | "settings_updated"
  | "api_key_created"
  | "api_key_deleted"
  | "webhook_created"
  | "webhook_triggered"
  | "phone_number_added"
  | "phone_number_removed"
  | "recording_created"
  | "transcript_ready"
  | "export_completed"
  | "import_completed"
  | "system_notification"
  | "custom";

export type ActivityCategory =
  | "calls"
  | "agents"
  | "workflows"
  | "integrations"
  | "team"
  | "billing"
  | "security"
  | "settings"
  | "system";

export interface ActivityFeedProps {
  activities: ActivityItem[];
  loading?: boolean;
  showFilters?: boolean;
  showSearch?: boolean;
  showLoadMore?: boolean;
  maxItems?: number;
  groupByDate?: boolean;
  realtime?: boolean;
  onLoadMore?: () => void;
  onActivityClick?: (activity: ActivityItem) => void;
  onFilterChange?: (filters: ActivityFilters) => void;
  className?: string;
  compact?: boolean;
  emptyMessage?: string;
}

export interface ActivityFilters {
  categories: ActivityCategory[];
  dateRange?: { start: Date; end: Date };
  search?: string;
  status?: string[];
}

// Activity type configuration
const activityTypeConfig: Record<
  ActivityType,
  {
    icon: React.ReactNode;
    color: string;
    bgColor: string;
    label: string;
  }
> = {
  call_started: {
    icon: <PhoneIncoming className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Call Started",
  },
  call_ended: {
    icon: <Phone className="w-4 h-4" />,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    label: "Call Ended",
  },
  call_missed: {
    icon: <PhoneMissed className="w-4 h-4" />,
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    label: "Call Missed",
  },
  call_transferred: {
    icon: <PhoneOutgoing className="w-4 h-4" />,
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
    label: "Call Transferred",
  },
  agent_created: {
    icon: <Plus className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Agent Created",
  },
  agent_updated: {
    icon: <Edit className="w-4 h-4" />,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    label: "Agent Updated",
  },
  agent_deleted: {
    icon: <Trash2 className="w-4 h-4" />,
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    label: "Agent Deleted",
  },
  agent_activated: {
    icon: <Play className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Agent Activated",
  },
  agent_deactivated: {
    icon: <Pause className="w-4 h-4" />,
    color: "text-yellow-400",
    bgColor: "bg-yellow-500/10",
    label: "Agent Deactivated",
  },
  workflow_created: {
    icon: <Workflow className="w-4 h-4" />,
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
    label: "Workflow Created",
  },
  workflow_run: {
    icon: <Play className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Workflow Run",
  },
  workflow_error: {
    icon: <AlertCircle className="w-4 h-4" />,
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    label: "Workflow Error",
  },
  integration_connected: {
    icon: <Link className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Integration Connected",
  },
  integration_disconnected: {
    icon: <Unlink className="w-4 h-4" />,
    color: "text-yellow-400",
    bgColor: "bg-yellow-500/10",
    label: "Integration Disconnected",
  },
  integration_error: {
    icon: <AlertCircle className="w-4 h-4" />,
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    label: "Integration Error",
  },
  team_member_added: {
    icon: <Users className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Team Member Added",
  },
  team_member_removed: {
    icon: <Users className="w-4 h-4" />,
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    label: "Team Member Removed",
  },
  team_member_role_changed: {
    icon: <Users className="w-4 h-4" />,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    label: "Role Changed",
  },
  billing_payment: {
    icon: <CreditCard className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Payment Received",
  },
  billing_subscription: {
    icon: <CreditCard className="w-4 h-4" />,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    label: "Subscription Updated",
  },
  billing_invoice: {
    icon: <FileText className="w-4 h-4" />,
    color: "text-gray-400",
    bgColor: "bg-gray-500/10",
    label: "Invoice Generated",
  },
  security_login: {
    icon: <Lock className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Login",
  },
  security_logout: {
    icon: <Unlock className="w-4 h-4" />,
    color: "text-gray-400",
    bgColor: "bg-gray-500/10",
    label: "Logout",
  },
  security_password_changed: {
    icon: <Key className="w-4 h-4" />,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    label: "Password Changed",
  },
  security_2fa_enabled: {
    icon: <Shield className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "2FA Enabled",
  },
  settings_updated: {
    icon: <Settings className="w-4 h-4" />,
    color: "text-gray-400",
    bgColor: "bg-gray-500/10",
    label: "Settings Updated",
  },
  api_key_created: {
    icon: <Key className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "API Key Created",
  },
  api_key_deleted: {
    icon: <Key className="w-4 h-4" />,
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    label: "API Key Deleted",
  },
  webhook_created: {
    icon: <Webhook className="w-4 h-4" />,
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
    label: "Webhook Created",
  },
  webhook_triggered: {
    icon: <Webhook className="w-4 h-4" />,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    label: "Webhook Triggered",
  },
  phone_number_added: {
    icon: <Phone className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Phone Number Added",
  },
  phone_number_removed: {
    icon: <Phone className="w-4 h-4" />,
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    label: "Phone Number Removed",
  },
  recording_created: {
    icon: <Mic className="w-4 h-4" />,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    label: "Recording Created",
  },
  transcript_ready: {
    icon: <FileText className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Transcript Ready",
  },
  export_completed: {
    icon: <Download className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Export Completed",
  },
  import_completed: {
    icon: <Upload className="w-4 h-4" />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    label: "Import Completed",
  },
  system_notification: {
    icon: <Bell className="w-4 h-4" />,
    color: "text-gray-400",
    bgColor: "bg-gray-500/10",
    label: "System Notification",
  },
  custom: {
    icon: <Activity className="w-4 h-4" />,
    color: "text-gray-400",
    bgColor: "bg-gray-500/10",
    label: "Activity",
  },
};

// Category configuration
const categoryConfig: Record<
  ActivityCategory,
  { icon: React.ReactNode; label: string; color: string }
> = {
  calls: {
    icon: <Phone className="w-4 h-4" />,
    label: "Calls",
    color: "blue",
  },
  agents: {
    icon: <Bot className="w-4 h-4" />,
    label: "Agents",
    color: "purple",
  },
  workflows: {
    icon: <Workflow className="w-4 h-4" />,
    label: "Workflows",
    color: "orange",
  },
  integrations: {
    icon: <Zap className="w-4 h-4" />,
    label: "Integrations",
    color: "cyan",
  },
  team: {
    icon: <Users className="w-4 h-4" />,
    label: "Team",
    color: "green",
  },
  billing: {
    icon: <CreditCard className="w-4 h-4" />,
    label: "Billing",
    color: "pink",
  },
  security: {
    icon: <Shield className="w-4 h-4" />,
    label: "Security",
    color: "red",
  },
  settings: {
    icon: <Settings className="w-4 h-4" />,
    label: "Settings",
    color: "gray",
  },
  system: {
    icon: <Server className="w-4 h-4" />,
    label: "System",
    color: "gray",
  },
};

// Utility functions
function getRelativeTime(timestamp: string): string {
  const now = new Date();
  const date = new Date(timestamp);
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function getDateGroup(timestamp: string): string {
  const date = new Date(timestamp);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  if (date.toDateString() === today.toDateString()) {
    return "Today";
  }
  if (date.toDateString() === yesterday.toDateString()) {
    return "Yesterday";
  }
  if (date > new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000)) {
    return date.toLocaleDateString("en-US", { weekday: "long" });
  }
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: date.getFullYear() !== today.getFullYear() ? "numeric" : undefined,
  });
}

// User Avatar component
function UserAvatar({
  user,
  size = "sm",
}: {
  user?: ActivityItem["user"];
  size?: "sm" | "md" | "lg";
}) {
  const sizeClasses = {
    sm: "w-8 h-8 text-xs",
    md: "w-10 h-10 text-sm",
    lg: "w-12 h-12 text-base",
  };

  if (!user) {
    return (
      <div
        className={`${sizeClasses[size]} rounded-full bg-gray-600 flex items-center justify-center`}
      >
        <User className="w-1/2 h-1/2 text-gray-400" />
      </div>
    );
  }

  if (user.avatar) {
    return (
      <img
        src={user.avatar}
        alt={user.name}
        className={`${sizeClasses[size]} rounded-full object-cover`}
      />
    );
  }

  const initials = user.name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  return (
    <div
      className={`${sizeClasses[size]} rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center font-medium text-white`}
    >
      {initials}
    </div>
  );
}

// Activity Item Component
function ActivityItemComponent({
  activity,
  onClick,
  compact,
  isLast,
}: {
  activity: ActivityItem;
  onClick?: (activity: ActivityItem) => void;
  compact?: boolean;
  isLast?: boolean;
}) {
  const [showMenu, setShowMenu] = useState(false);
  const typeConfig = activityTypeConfig[activity.type] || activityTypeConfig.custom;

  const statusColors = {
    success: "text-green-400",
    error: "text-red-400",
    warning: "text-yellow-400",
    info: "text-blue-400",
    pending: "text-gray-400",
  };

  return (
    <div className="relative group">
      {/* Timeline line */}
      {!isLast && (
        <div className="absolute left-4 top-10 bottom-0 w-0.5 bg-white/5" />
      )}

      <div
        className={`flex gap-4 p-4 transition-colors ${
          onClick ? "cursor-pointer hover:bg-white/5" : ""
        } ${activity.isNew ? "bg-purple-500/5" : ""}`}
        onClick={() => onClick?.(activity)}
      >
        {/* Icon */}
        <div
          className={`relative flex-shrink-0 w-8 h-8 rounded-full ${typeConfig.bgColor} ${typeConfig.color} flex items-center justify-center z-10`}
        >
          {typeConfig.icon}
          {activity.isNew && (
            <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-purple-500 rounded-full border-2 border-[#0f0f1a]" />
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <p className="text-sm text-white">
                <span className="font-medium">{activity.title}</span>
                {activity.user && (
                  <span className="text-gray-400">
                    {" "}
                    by {activity.user.name}
                  </span>
                )}
              </p>
              {activity.description && !compact && (
                <p className="text-sm text-gray-500 mt-0.5 line-clamp-2">
                  {activity.description}
                </p>
              )}
            </div>

            {/* Time and menu */}
            <div className="flex items-center gap-2 flex-shrink-0">
              <span
                className={`text-xs ${
                  activity.status
                    ? statusColors[activity.status]
                    : "text-gray-500"
                }`}
              >
                {getRelativeTime(activity.timestamp)}
              </span>
              {activity.actionUrl && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    window.location.href = activity.actionUrl!;
                  }}
                  className="p-1 opacity-0 group-hover:opacity-100 hover:bg-white/10 rounded transition-all"
                >
                  <ExternalLink className="w-3 h-3 text-gray-400" />
                </button>
              )}
            </div>
          </div>

          {/* Metadata */}
          {activity.metadata && !compact && (
            <div className="flex flex-wrap gap-2 mt-2">
              {Object.entries(activity.metadata).map(([key, value]) => (
                <span
                  key={key}
                  className="inline-flex items-center px-2 py-0.5 text-xs bg-white/5 text-gray-400 rounded"
                >
                  <span className="text-gray-500 mr-1">{key}:</span>
                  {String(value)}
                </span>
              ))}
            </div>
          )}

          {/* Nested activities */}
          {activity.children && activity.children.length > 0 && (
            <div className="mt-3 pl-4 border-l border-white/10 space-y-2">
              {activity.children.map((child) => (
                <div key={child.id} className="text-sm">
                  <span className="text-gray-400">{child.title}</span>
                  <span className="text-gray-600 ml-2">
                    {getRelativeTime(child.timestamp)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Filter bar component
function FilterBar({
  filters,
  onChange,
}: {
  filters: ActivityFilters;
  onChange: (filters: ActivityFilters) => void;
}) {
  const [showCategoryDropdown, setShowCategoryDropdown] = useState(false);

  const toggleCategory = (category: ActivityCategory) => {
    const newCategories = filters.categories.includes(category)
      ? filters.categories.filter((c) => c !== category)
      : [...filters.categories, category];
    onChange({ ...filters, categories: newCategories });
  };

  return (
    <div className="flex items-center gap-3 p-4 border-b border-white/10">
      {/* Search */}
      <div className="relative flex-1 max-w-sm">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          value={filters.search || ""}
          onChange={(e) => onChange({ ...filters, search: e.target.value })}
          placeholder="Search activity..."
          className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
        />
      </div>

      {/* Category filter */}
      <div className="relative">
        <button
          onClick={() => setShowCategoryDropdown(!showCategoryDropdown)}
          className="flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-gray-400 transition-colors"
        >
          <Filter className="w-4 h-4" />
          <span>
            {filters.categories.length === 0
              ? "All Categories"
              : `${filters.categories.length} selected`}
          </span>
          <ChevronDown className="w-4 h-4" />
        </button>
        {showCategoryDropdown && (
          <>
            <div
              className="fixed inset-0 z-10"
              onClick={() => setShowCategoryDropdown(false)}
            />
            <div className="absolute right-0 top-full mt-2 w-56 bg-[#1a1a2e] border border-white/10 rounded-xl shadow-xl z-20 py-2">
              {(Object.keys(categoryConfig) as ActivityCategory[]).map(
                (category) => {
                  const config = categoryConfig[category];
                  const isSelected = filters.categories.includes(category);
                  return (
                    <button
                      key={category}
                      onClick={() => toggleCategory(category)}
                      className={`flex items-center gap-3 w-full px-4 py-2 text-sm hover:bg-white/5 transition-colors ${
                        isSelected ? "text-white" : "text-gray-400"
                      }`}
                    >
                      <div
                        className={`w-4 h-4 rounded border flex items-center justify-center ${
                          isSelected
                            ? "bg-purple-500 border-purple-500"
                            : "border-gray-600"
                        }`}
                      >
                        {isSelected && (
                          <CheckCircle className="w-3 h-3 text-white" />
                        )}
                      </div>
                      {config.icon}
                      <span>{config.label}</span>
                    </button>
                  );
                }
              )}
              {filters.categories.length > 0 && (
                <>
                  <div className="border-t border-white/10 my-2" />
                  <button
                    onClick={() => onChange({ ...filters, categories: [] })}
                    className="w-full px-4 py-2 text-sm text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
                  >
                    Clear all
                  </button>
                </>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// Main Activity Feed Component
export function ActivityFeed({
  activities,
  loading = false,
  showFilters = true,
  showSearch = true,
  showLoadMore = true,
  maxItems,
  groupByDate = true,
  realtime = false,
  onLoadMore,
  onActivityClick,
  onFilterChange,
  className = "",
  compact = false,
  emptyMessage = "No activity to show",
}: ActivityFeedProps) {
  const [filters, setFilters] = useState<ActivityFilters>({
    categories: [],
    search: "",
  });
  const feedRef = useRef<HTMLDivElement>(null);
  const [newActivityCount, setNewActivityCount] = useState(0);

  // Filter activities
  const filteredActivities = useMemo(() => {
    let result = activities;

    // Filter by category
    if (filters.categories.length > 0) {
      result = result.filter((a) => filters.categories.includes(a.category));
    }

    // Filter by search
    if (filters.search) {
      const query = filters.search.toLowerCase();
      result = result.filter(
        (a) =>
          a.title.toLowerCase().includes(query) ||
          a.description?.toLowerCase().includes(query) ||
          a.user?.name.toLowerCase().includes(query)
      );
    }

    // Limit results
    if (maxItems) {
      result = result.slice(0, maxItems);
    }

    return result;
  }, [activities, filters, maxItems]);

  // Group by date
  const groupedActivities = useMemo(() => {
    if (!groupByDate) return { All: filteredActivities };

    const groups: Record<string, ActivityItem[]> = {};
    filteredActivities.forEach((activity) => {
      const group = getDateGroup(activity.timestamp);
      if (!groups[group]) {
        groups[group] = [];
      }
      groups[group].push(activity);
    });
    return groups;
  }, [filteredActivities, groupByDate]);

  // Handle filter change
  const handleFilterChange = (newFilters: ActivityFilters) => {
    setFilters(newFilters);
    onFilterChange?.(newFilters);
  };

  // Real-time indicator
  useEffect(() => {
    if (realtime && activities.length > 0) {
      const newItems = activities.filter((a) => a.isNew).length;
      setNewActivityCount(newItems);
    }
  }, [activities, realtime]);

  // Loading skeleton
  if (loading && activities.length === 0) {
    return (
      <div
        className={`bg-[#1a1a2e]/30 rounded-xl border border-white/5 ${className}`}
      >
        {showFilters && (
          <div className="p-4 border-b border-white/10">
            <div className="h-10 bg-white/5 rounded-lg animate-pulse" />
          </div>
        )}
        <div className="divide-y divide-white/5">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex gap-4 p-4 animate-pulse">
              <div className="w-8 h-8 rounded-full bg-white/5" />
              <div className="flex-1 space-y-2">
                <div className="h-4 bg-white/5 rounded w-3/4" />
                <div className="h-3 bg-white/5 rounded w-1/2" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div
      ref={feedRef}
      className={`bg-[#1a1a2e]/30 rounded-xl border border-white/5 overflow-hidden ${className}`}
    >
      {/* Realtime indicator */}
      {realtime && newActivityCount > 0 && (
        <div className="sticky top-0 z-20 px-4 py-2 bg-purple-500/20 border-b border-purple-500/20">
          <button
            onClick={() => feedRef.current?.scrollTo({ top: 0, behavior: "smooth" })}
            className="flex items-center gap-2 text-sm text-purple-400"
          >
            <Activity className="w-4 h-4 animate-pulse" />
            {newActivityCount} new activit{newActivityCount > 1 ? "ies" : "y"}
            <ArrowUpRight className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* Filters */}
      {showFilters && (
        <FilterBar filters={filters} onChange={handleFilterChange} />
      )}

      {/* Activity list */}
      {filteredActivities.length > 0 ? (
        <div className="divide-y divide-white/5">
          {Object.entries(groupedActivities).map(([date, items]) => (
            <div key={date}>
              {/* Date header */}
              {groupByDate && (
                <div className="sticky top-0 z-10 px-4 py-2 bg-[#1a1a2e]/80 backdrop-blur-sm">
                  <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {date}
                  </span>
                </div>
              )}

              {/* Items */}
              {items.map((activity, index) => (
                <ActivityItemComponent
                  key={activity.id}
                  activity={activity}
                  onClick={onActivityClick}
                  compact={compact}
                  isLast={index === items.length - 1}
                />
              ))}
            </div>
          ))}
        </div>
      ) : (
        <div className="py-12 text-center">
          <Activity className="w-10 h-10 text-gray-600 mx-auto mb-3" />
          <p className="text-gray-500">{emptyMessage}</p>
        </div>
      )}

      {/* Load more */}
      {showLoadMore && onLoadMore && filteredActivities.length > 0 && (
        <div className="p-4 border-t border-white/10">
          <button
            onClick={onLoadMore}
            disabled={loading}
            className="w-full py-2 text-sm text-gray-400 hover:text-white hover:bg-white/5 rounded-lg transition-colors disabled:opacity-50"
          >
            {loading ? (
              <RefreshCw className="w-4 h-4 animate-spin mx-auto" />
            ) : (
              "Load more"
            )}
          </button>
        </div>
      )}
    </div>
  );
}

// Mini activity feed for dashboard widgets
export function MiniActivityFeed({
  activities,
  maxItems = 5,
  onViewAll,
  className = "",
}: {
  activities: ActivityItem[];
  maxItems?: number;
  onViewAll?: () => void;
  className?: string;
}) {
  const displayActivities = activities.slice(0, maxItems);

  return (
    <div
      className={`bg-[#1a1a2e]/30 rounded-xl border border-white/5 ${className}`}
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <h3 className="font-semibold text-white">Recent Activity</h3>
        {onViewAll && (
          <button
            onClick={onViewAll}
            className="text-sm text-purple-400 hover:text-purple-300"
          >
            View all
          </button>
        )}
      </div>
      <div className="divide-y divide-white/5">
        {displayActivities.map((activity) => {
          const typeConfig =
            activityTypeConfig[activity.type] || activityTypeConfig.custom;
          return (
            <div
              key={activity.id}
              className="flex items-center gap-3 px-4 py-3 hover:bg-white/5 transition-colors"
            >
              <div
                className={`w-6 h-6 rounded-full ${typeConfig.bgColor} ${typeConfig.color} flex items-center justify-center flex-shrink-0`}
              >
                {React.cloneElement(typeConfig.icon as React.ReactElement, {
                  className: "w-3 h-3",
                })}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-white truncate">{activity.title}</p>
              </div>
              <span className="text-xs text-gray-500 flex-shrink-0">
                {getRelativeTime(activity.timestamp)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default ActivityFeed;
