"use client";

import React, { useState, useMemo, useEffect, useRef } from "react";
import {
  Bell,
  X,
  Check,
  CheckCheck,
  Trash2,
  Settings,
  Filter,
  Search,
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  PhoneMissed,
  Bot,
  Users,
  CreditCard,
  Shield,
  AlertCircle,
  AlertTriangle,
  Info,
  CheckCircle,
  XCircle,
  Clock,
  Calendar,
  Zap,
  MessageSquare,
  Mail,
  Webhook,
  Activity,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Server,
  Database,
  Globe,
  Link,
  ExternalLink,
  ChevronRight,
  ChevronDown,
  MoreHorizontal,
  Volume2,
  VolumeX,
  Eye,
  EyeOff,
  Sparkles,
  Gift,
  Star,
  Heart,
  Flame,
  RefreshCw,
  Download,
  Upload,
  FileText,
  Archive,
  Inbox,
  BellOff,
  BellRing,
  ToggleLeft,
  ToggleRight,
} from "lucide-react";

// Types
export interface Notification {
  id: string;
  type: "info" | "success" | "warning" | "error";
  category: NotificationCategory;
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionUrl?: string;
  actionLabel?: string;
  metadata?: Record<string, any>;
  persistent?: boolean;
}

export type NotificationCategory =
  | "calls"
  | "agents"
  | "system"
  | "billing"
  | "security"
  | "integrations"
  | "team"
  | "updates";

export interface NotificationPreferences {
  enabled: boolean;
  sound: boolean;
  desktop: boolean;
  email: boolean;
  categories: Record<NotificationCategory, boolean>;
}

// Category icons and colors
const categoryConfig: Record<
  NotificationCategory,
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
  system: {
    icon: <Server className="w-4 h-4" />,
    label: "System",
    color: "gray",
  },
  billing: {
    icon: <CreditCard className="w-4 h-4" />,
    label: "Billing",
    color: "green",
  },
  security: {
    icon: <Shield className="w-4 h-4" />,
    label: "Security",
    color: "red",
  },
  integrations: {
    icon: <Zap className="w-4 h-4" />,
    label: "Integrations",
    color: "orange",
  },
  team: {
    icon: <Users className="w-4 h-4" />,
    label: "Team",
    color: "cyan",
  },
  updates: {
    icon: <Sparkles className="w-4 h-4" />,
    label: "Updates",
    color: "pink",
  },
};

// Type icons and colors
const typeConfig: Record<
  Notification["type"],
  { icon: React.ReactNode; bgColor: string; textColor: string }
> = {
  info: {
    icon: <Info className="w-4 h-4" />,
    bgColor: "bg-blue-500/10",
    textColor: "text-blue-400",
  },
  success: {
    icon: <CheckCircle className="w-4 h-4" />,
    bgColor: "bg-green-500/10",
    textColor: "text-green-400",
  },
  warning: {
    icon: <AlertTriangle className="w-4 h-4" />,
    bgColor: "bg-yellow-500/10",
    textColor: "text-yellow-400",
  },
  error: {
    icon: <XCircle className="w-4 h-4" />,
    bgColor: "bg-red-500/10",
    textColor: "text-red-400",
  },
};

// Mock notifications
const mockNotifications: Notification[] = [
  {
    id: "n_1",
    type: "success",
    category: "calls",
    title: "Call Completed",
    message:
      "Sales Assistant completed a 5-minute call with John Smith. Lead qualified.",
    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    read: false,
    actionUrl: "/calls/call_123",
    actionLabel: "View Call",
    metadata: { callId: "call_123", duration: 300 },
  },
  {
    id: "n_2",
    type: "warning",
    category: "agents",
    title: "Agent Error Rate High",
    message:
      "Support Bot has experienced 5 errors in the last hour. Consider reviewing.",
    timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    read: false,
    actionUrl: "/agents/agent_2",
    actionLabel: "View Agent",
  },
  {
    id: "n_3",
    type: "info",
    category: "billing",
    title: "Invoice Generated",
    message: "Your January invoice for $243.00 is ready for download.",
    timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    read: false,
    actionUrl: "/billing/invoices",
    actionLabel: "View Invoice",
  },
  {
    id: "n_4",
    type: "success",
    category: "integrations",
    title: "Integration Connected",
    message: "Salesforce integration has been successfully connected.",
    timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
    read: true,
    actionUrl: "/integrations",
    actionLabel: "Manage Integrations",
  },
  {
    id: "n_5",
    type: "error",
    category: "security",
    title: "Failed Login Attempt",
    message:
      "Someone tried to access your account from an unknown location. Password was incorrect.",
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    read: true,
    actionUrl: "/settings/security",
    actionLabel: "Review Security",
  },
  {
    id: "n_6",
    type: "info",
    category: "team",
    title: "New Team Member",
    message: "Sarah Johnson has joined your team as an Admin.",
    timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
    read: true,
    actionUrl: "/settings/team",
    actionLabel: "View Team",
  },
  {
    id: "n_7",
    type: "success",
    category: "updates",
    title: "New Feature: Workflows",
    message:
      "Introducing Workflows! Automate actions based on call events and triggers.",
    timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    read: true,
    actionUrl: "/workflows",
    actionLabel: "Try Workflows",
    persistent: true,
  },
  {
    id: "n_8",
    type: "warning",
    category: "billing",
    title: "Usage Alert",
    message: "You've used 80% of your monthly call minutes.",
    timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    read: true,
    actionUrl: "/billing",
    actionLabel: "Upgrade Plan",
  },
  {
    id: "n_9",
    type: "info",
    category: "calls",
    title: "Missed Call",
    message: "Missed call from +1 (555) 123-4567 at 2:30 PM.",
    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
    read: true,
    actionUrl: "/calls",
    actionLabel: "View Calls",
  },
  {
    id: "n_10",
    type: "success",
    category: "system",
    title: "Backup Completed",
    message: "Your data has been successfully backed up.",
    timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    read: true,
  },
];

// Utility function for relative time
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

// Notification item component
function NotificationItem({
  notification,
  onRead,
  onDelete,
  onAction,
  compact = false,
}: {
  notification: Notification;
  onRead: (id: string) => void;
  onDelete: (id: string) => void;
  onAction?: (notification: Notification) => void;
  compact?: boolean;
}) {
  const [showMenu, setShowMenu] = useState(false);
  const typeStyle = typeConfig[notification.type];
  const categoryStyle = categoryConfig[notification.category];

  return (
    <div
      className={`group relative p-4 transition-all ${
        notification.read
          ? "bg-transparent hover:bg-white/5"
          : "bg-purple-500/5 hover:bg-purple-500/10"
      } ${compact ? "p-3" : ""}`}
    >
      <div className="flex items-start gap-3">
        {/* Type indicator */}
        <div
          className={`flex-shrink-0 w-8 h-8 rounded-lg ${typeStyle.bgColor} ${typeStyle.textColor} flex items-center justify-center`}
        >
          {typeStyle.icon}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h4
                  className={`text-sm font-medium ${
                    notification.read ? "text-gray-300" : "text-white"
                  }`}
                >
                  {notification.title}
                </h4>
                {!notification.read && (
                  <span className="w-2 h-2 rounded-full bg-purple-500" />
                )}
              </div>
              <p
                className={`text-sm mt-0.5 ${
                  notification.read ? "text-gray-500" : "text-gray-400"
                } ${compact ? "line-clamp-1" : "line-clamp-2"}`}
              >
                {notification.message}
              </p>
            </div>

            {/* Menu */}
            <div className="relative flex-shrink-0">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowMenu(!showMenu);
                }}
                className="p-1 opacity-0 group-hover:opacity-100 hover:bg-white/10 rounded transition-all"
              >
                <MoreHorizontal className="w-4 h-4 text-gray-400" />
              </button>
              {showMenu && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setShowMenu(false)}
                  />
                  <div className="absolute right-0 top-full mt-1 w-40 bg-[#1a1a2e] border border-white/10 rounded-lg shadow-xl z-20 py-1 overflow-hidden">
                    {!notification.read && (
                      <button
                        onClick={() => {
                          onRead(notification.id);
                          setShowMenu(false);
                        }}
                        className="flex items-center gap-2 w-full px-3 py-2 text-sm text-gray-300 hover:bg-white/10"
                      >
                        <Check className="w-4 h-4" />
                        Mark as read
                      </button>
                    )}
                    <button
                      onClick={() => {
                        onDelete(notification.id);
                        setShowMenu(false);
                      }}
                      className="flex items-center gap-2 w-full px-3 py-2 text-sm text-red-400 hover:bg-red-500/10"
                    >
                      <Trash2 className="w-4 h-4" />
                      Delete
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Meta info */}
          <div className="flex items-center gap-3 mt-2">
            <span className="flex items-center gap-1 text-xs text-gray-500">
              <Clock className="w-3 h-3" />
              {getRelativeTime(notification.timestamp)}
            </span>
            <span
              className={`flex items-center gap-1 text-xs px-1.5 py-0.5 rounded ${
                categoryStyle.color === "blue"
                  ? "bg-blue-500/10 text-blue-400"
                  : categoryStyle.color === "purple"
                  ? "bg-purple-500/10 text-purple-400"
                  : categoryStyle.color === "green"
                  ? "bg-green-500/10 text-green-400"
                  : categoryStyle.color === "red"
                  ? "bg-red-500/10 text-red-400"
                  : categoryStyle.color === "orange"
                  ? "bg-orange-500/10 text-orange-400"
                  : categoryStyle.color === "cyan"
                  ? "bg-cyan-500/10 text-cyan-400"
                  : categoryStyle.color === "pink"
                  ? "bg-pink-500/10 text-pink-400"
                  : "bg-gray-500/10 text-gray-400"
              }`}
            >
              {categoryStyle.icon}
              {categoryStyle.label}
            </span>
          </div>

          {/* Action button */}
          {notification.actionUrl && notification.actionLabel && (
            <button
              onClick={() => onAction?.(notification)}
              className="flex items-center gap-1 mt-2 text-xs text-purple-400 hover:text-purple-300 transition-colors"
            >
              {notification.actionLabel}
              <ChevronRight className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// Notification Bell with dropdown
export function NotificationBell({
  notifications = mockNotifications,
  onNotificationClick,
  onViewAll,
  onMarkAllRead,
}: {
  notifications?: Notification[];
  onNotificationClick?: (notification: Notification) => void;
  onViewAll?: () => void;
  onMarkAllRead?: () => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [localNotifications, setLocalNotifications] = useState(notifications);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const unreadCount = localNotifications.filter((n) => !n.read).length;

  // Close on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleRead = (id: string) => {
    setLocalNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  };

  const handleDelete = (id: string) => {
    setLocalNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const handleMarkAllRead = () => {
    setLocalNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
    onMarkAllRead?.();
  };

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Bell button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 hover:bg-white/10 rounded-lg transition-colors"
      >
        <Bell className="w-5 h-5 text-gray-400" />
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 w-5 h-5 flex items-center justify-center text-xs font-medium bg-purple-500 text-white rounded-full">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-96 bg-[#1a1a2e] border border-white/10 rounded-xl shadow-2xl z-50 overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
            <h3 className="font-semibold text-white">Notifications</h3>
            <div className="flex items-center gap-2">
              {unreadCount > 0 && (
                <button
                  onClick={handleMarkAllRead}
                  className="text-xs text-purple-400 hover:text-purple-300"
                >
                  Mark all read
                </button>
              )}
              <button className="p-1 hover:bg-white/10 rounded">
                <Settings className="w-4 h-4 text-gray-400" />
              </button>
            </div>
          </div>

          {/* Notification list */}
          <div className="max-h-[400px] overflow-y-auto divide-y divide-white/5">
            {localNotifications.length > 0 ? (
              localNotifications.slice(0, 5).map((notification) => (
                <NotificationItem
                  key={notification.id}
                  notification={notification}
                  onRead={handleRead}
                  onDelete={handleDelete}
                  onAction={(n) => {
                    onNotificationClick?.(n);
                    handleRead(n.id);
                    setIsOpen(false);
                  }}
                  compact
                />
              ))
            ) : (
              <div className="py-12 text-center">
                <Bell className="w-10 h-10 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-500">No notifications</p>
              </div>
            )}
          </div>

          {/* Footer */}
          {localNotifications.length > 0 && (
            <div className="px-4 py-3 border-t border-white/10 bg-white/5">
              <button
                onClick={() => {
                  onViewAll?.();
                  setIsOpen(false);
                }}
                className="w-full text-center text-sm text-purple-400 hover:text-purple-300"
              >
                View all notifications
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Notification preferences component
export function NotificationPreferences({
  preferences,
  onChange,
}: {
  preferences: NotificationPreferences;
  onChange: (preferences: NotificationPreferences) => void;
}) {
  const [localPrefs, setLocalPrefs] = useState(preferences);

  const handleToggle = (key: keyof NotificationPreferences) => {
    if (typeof localPrefs[key] === "boolean") {
      const newPrefs = { ...localPrefs, [key]: !localPrefs[key] };
      setLocalPrefs(newPrefs);
      onChange(newPrefs);
    }
  };

  const handleCategoryToggle = (category: NotificationCategory) => {
    const newCategories = {
      ...localPrefs.categories,
      [category]: !localPrefs.categories[category],
    };
    const newPrefs = { ...localPrefs, categories: newCategories };
    setLocalPrefs(newPrefs);
    onChange(newPrefs);
  };

  return (
    <div className="space-y-6">
      {/* Master toggle */}
      <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
            <Bell className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h4 className="font-medium text-white">Enable Notifications</h4>
            <p className="text-sm text-gray-500">
              Receive notifications about your account
            </p>
          </div>
        </div>
        <button
          onClick={() => handleToggle("enabled")}
          className={`relative w-12 h-6 rounded-full transition-colors ${
            localPrefs.enabled ? "bg-purple-500" : "bg-gray-600"
          }`}
        >
          <span
            className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
              localPrefs.enabled ? "translate-x-7" : "translate-x-1"
            }`}
          />
        </button>
      </div>

      {/* Delivery methods */}
      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-3">
          Delivery Methods
        </h4>
        <div className="space-y-2">
          {[
            {
              key: "sound" as const,
              icon: <Volume2 className="w-4 h-4" />,
              label: "Sound",
              description: "Play a sound for new notifications",
            },
            {
              key: "desktop" as const,
              icon: <Bell className="w-4 h-4" />,
              label: "Desktop",
              description: "Show desktop push notifications",
            },
            {
              key: "email" as const,
              icon: <Mail className="w-4 h-4" />,
              label: "Email",
              description: "Send email for important notifications",
            },
          ].map((item) => (
            <div
              key={item.key}
              className="flex items-center justify-between p-3 bg-white/5 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <span className="text-gray-400">{item.icon}</span>
                <div>
                  <span className="text-sm text-white">{item.label}</span>
                  <p className="text-xs text-gray-500">{item.description}</p>
                </div>
              </div>
              <button
                onClick={() => handleToggle(item.key)}
                disabled={!localPrefs.enabled}
                className={`relative w-10 h-5 rounded-full transition-colors ${
                  localPrefs[item.key] && localPrefs.enabled
                    ? "bg-purple-500"
                    : "bg-gray-600"
                } ${!localPrefs.enabled ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                <span
                  className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                    localPrefs[item.key] ? "translate-x-5" : "translate-x-0.5"
                  }`}
                />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Categories */}
      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-3">Categories</h4>
        <div className="grid grid-cols-2 gap-2">
          {(Object.keys(categoryConfig) as NotificationCategory[]).map(
            (category) => {
              const config = categoryConfig[category];
              return (
                <div
                  key={category}
                  className="flex items-center justify-between p-3 bg-white/5 rounded-lg"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`${
                        config.color === "blue"
                          ? "text-blue-400"
                          : config.color === "purple"
                          ? "text-purple-400"
                          : config.color === "green"
                          ? "text-green-400"
                          : config.color === "red"
                          ? "text-red-400"
                          : config.color === "orange"
                          ? "text-orange-400"
                          : config.color === "cyan"
                          ? "text-cyan-400"
                          : config.color === "pink"
                          ? "text-pink-400"
                          : "text-gray-400"
                      }`}
                    >
                      {config.icon}
                    </span>
                    <span className="text-sm text-white">{config.label}</span>
                  </div>
                  <button
                    onClick={() => handleCategoryToggle(category)}
                    disabled={!localPrefs.enabled}
                    className={`relative w-10 h-5 rounded-full transition-colors ${
                      localPrefs.categories[category] && localPrefs.enabled
                        ? "bg-purple-500"
                        : "bg-gray-600"
                    } ${!localPrefs.enabled ? "opacity-50 cursor-not-allowed" : ""}`}
                  >
                    <span
                      className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                        localPrefs.categories[category]
                          ? "translate-x-5"
                          : "translate-x-0.5"
                      }`}
                    />
                  </button>
                </div>
              );
            }
          )}
        </div>
      </div>
    </div>
  );
}

// Full notifications page component
export function NotificationsPage() {
  const [notifications, setNotifications] = useState(mockNotifications);
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [filterRead, setFilterRead] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [showPreferences, setShowPreferences] = useState(false);

  const [preferences, setPreferences] = useState<NotificationPreferences>({
    enabled: true,
    sound: true,
    desktop: true,
    email: false,
    categories: {
      calls: true,
      agents: true,
      system: true,
      billing: true,
      security: true,
      integrations: true,
      team: true,
      updates: true,
    },
  });

  const filteredNotifications = useMemo(() => {
    return notifications.filter((n) => {
      if (filterCategory !== "all" && n.category !== filterCategory) {
        return false;
      }
      if (filterRead === "unread" && n.read) {
        return false;
      }
      if (filterRead === "read" && !n.read) {
        return false;
      }
      if (
        searchQuery &&
        !n.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !n.message.toLowerCase().includes(searchQuery.toLowerCase())
      ) {
        return false;
      }
      return true;
    });
  }, [notifications, filterCategory, filterRead, searchQuery]);

  const unreadCount = notifications.filter((n) => !n.read).length;

  const handleRead = (id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  };

  const handleDelete = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const handleMarkAllRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  const handleClearAll = () => {
    setNotifications([]);
  };

  // Group notifications by date
  const groupedNotifications = useMemo(() => {
    const groups: Record<string, Notification[]> = {};
    filteredNotifications.forEach((n) => {
      const date = new Date(n.timestamp);
      const today = new Date();
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);

      let key: string;
      if (date.toDateString() === today.toDateString()) {
        key = "Today";
      } else if (date.toDateString() === yesterday.toDateString()) {
        key = "Yesterday";
      } else {
        key = date.toLocaleDateString("en-US", {
          weekday: "long",
          month: "short",
          day: "numeric",
        });
      }

      if (!groups[key]) {
        groups[key] = [];
      }
      groups[key].push(n);
    });
    return groups;
  }, [filteredNotifications]);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Notifications</h1>
          <p className="text-gray-400">
            {unreadCount > 0
              ? `You have ${unreadCount} unread notification${
                  unreadCount > 1 ? "s" : ""
                }`
              : "All caught up!"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {unreadCount > 0 && (
            <button
              onClick={handleMarkAllRead}
              className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 text-gray-300 rounded-lg transition-colors"
            >
              <CheckCheck className="w-4 h-4" />
              Mark all read
            </button>
          )}
          <button
            onClick={() => setShowPreferences(!showPreferences)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
              showPreferences
                ? "bg-purple-500/20 text-purple-400"
                : "bg-white/5 hover:bg-white/10 text-gray-300"
            }`}
          >
            <Settings className="w-4 h-4" />
            Preferences
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex gap-6">
        {/* Notifications list */}
        <div className="flex-1 space-y-4">
          {/* Filters */}
          <div className="flex items-center gap-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search notifications..."
                className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Categories</option>
              {(Object.keys(categoryConfig) as NotificationCategory[]).map(
                (category) => (
                  <option key={category} value={category}>
                    {categoryConfig[category].label}
                  </option>
                )
              )}
            </select>
            <select
              value={filterRead}
              onChange={(e) => setFilterRead(e.target.value)}
              className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All</option>
              <option value="unread">Unread</option>
              <option value="read">Read</option>
            </select>
          </div>

          {/* Grouped notifications */}
          {Object.keys(groupedNotifications).length > 0 ? (
            <div className="space-y-6">
              {Object.entries(groupedNotifications).map(([date, notifs]) => (
                <div key={date}>
                  <h3 className="text-sm font-medium text-gray-500 mb-2">
                    {date}
                  </h3>
                  <div className="bg-[#1a1a2e]/30 rounded-xl border border-white/5 divide-y divide-white/5 overflow-hidden">
                    {notifs.map((notification) => (
                      <NotificationItem
                        key={notification.id}
                        notification={notification}
                        onRead={handleRead}
                        onDelete={handleDelete}
                        onAction={() => {
                          handleRead(notification.id);
                          if (notification.actionUrl) {
                            window.location.href = notification.actionUrl;
                          }
                        }}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-16">
              <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-4">
                <Inbox className="w-8 h-8 text-gray-600" />
              </div>
              <h3 className="text-lg font-medium text-white mb-2">
                No notifications
              </h3>
              <p className="text-gray-500">
                {searchQuery ||
                filterCategory !== "all" ||
                filterRead !== "all"
                  ? "Try adjusting your filters"
                  : "You're all caught up!"}
              </p>
            </div>
          )}

          {/* Clear all button */}
          {notifications.length > 0 && (
            <div className="text-center pt-4">
              <button
                onClick={handleClearAll}
                className="text-sm text-gray-500 hover:text-gray-400 transition-colors"
              >
                Clear all notifications
              </button>
            </div>
          )}
        </div>

        {/* Preferences sidebar */}
        {showPreferences && (
          <div className="w-80 bg-[#1a1a2e]/30 rounded-xl border border-white/5 p-4 h-fit">
            <h3 className="font-semibold text-white mb-4">
              Notification Preferences
            </h3>
            <NotificationPreferences
              preferences={preferences}
              onChange={setPreferences}
            />
          </div>
        )}
      </div>
    </div>
  );
}

// Toast notification component
export function NotificationToast({
  notification,
  onClose,
  onAction,
}: {
  notification: Notification;
  onClose: () => void;
  onAction?: () => void;
}) {
  const typeStyle = typeConfig[notification.type];

  useEffect(() => {
    if (!notification.persistent) {
      const timer = setTimeout(onClose, 5000);
      return () => clearTimeout(timer);
    }
  }, [notification.persistent, onClose]);

  return (
    <div className="flex items-start gap-3 p-4 bg-[#1a1a2e] border border-white/10 rounded-xl shadow-2xl max-w-sm animate-slide-in-right">
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-lg ${typeStyle.bgColor} ${typeStyle.textColor} flex items-center justify-center`}
      >
        {typeStyle.icon}
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-medium text-white">{notification.title}</h4>
        <p className="text-sm text-gray-400 mt-0.5 line-clamp-2">
          {notification.message}
        </p>
        {notification.actionLabel && (
          <button
            onClick={onAction}
            className="text-xs text-purple-400 hover:text-purple-300 mt-2"
          >
            {notification.actionLabel}
          </button>
        )}
      </div>
      <button
        onClick={onClose}
        className="flex-shrink-0 p-1 hover:bg-white/10 rounded transition-colors"
      >
        <X className="w-4 h-4 text-gray-400" />
      </button>
    </div>
  );
}

// Notification provider/context (simplified)
export function useNotifications() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [toasts, setToasts] = useState<Notification[]>([]);

  const addNotification = (
    notification: Omit<Notification, "id" | "timestamp" | "read">
  ) => {
    const newNotification: Notification = {
      ...notification,
      id: `n_${Date.now()}`,
      timestamp: new Date().toISOString(),
      read: false,
    };
    setNotifications((prev) => [newNotification, ...prev]);
    setToasts((prev) => [newNotification, ...prev]);
  };

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const markAsRead = (id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  };

  const markAllAsRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  const deleteNotification = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const clearAll = () => {
    setNotifications([]);
  };

  return {
    notifications,
    toasts,
    unreadCount: notifications.filter((n) => !n.read).length,
    addNotification,
    removeToast,
    markAsRead,
    markAllAsRead,
    deleteNotification,
    clearAll,
  };
}

export default NotificationBell;
