"use client";

import React, {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
  createContext,
  useContext,
} from "react";
import {
  Search,
  Command,
  ArrowUp,
  ArrowDown,
  CornerDownLeft,
  X,
  Home,
  Bot,
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  BarChart3,
  Settings,
  Users,
  CreditCard,
  HelpCircle,
  FileText,
  Plus,
  Zap,
  Clock,
  Star,
  Sparkles,
  MessageSquare,
  Calendar,
  Mail,
  Globe,
  Code,
  Shield,
  Bell,
  Workflow,
  PlayCircle,
  PauseCircle,
  RefreshCw,
  Download,
  Upload,
  Copy,
  Trash2,
  Edit,
  Eye,
  ExternalLink,
  ChevronRight,
  Hash,
  Tag,
  Folder,
  Filter,
  ArrowRight,
  Mic,
  Volume2,
  Headphones,
  Activity,
  TrendingUp,
  DollarSign,
  Send,
  BookOpen,
  Lightbulb,
  Target,
  Layers,
  Database,
  Server,
  Cloud,
  Keyboard,
  Moon,
  Sun,
  Monitor,
  LogOut,
  User,
  Key,
  Lock,
  Unlock,
  Link,
  Unlink,
  Check,
  AlertCircle,
} from "lucide-react";

// Types
export interface CommandItem {
  id: string;
  title: string;
  description?: string;
  icon?: React.ReactNode;
  category: CommandCategory;
  keywords?: string[];
  shortcut?: string[];
  action?: () => void;
  href?: string;
  disabled?: boolean;
  badge?: string;
  badgeColor?: string;
}

export type CommandCategory =
  | "navigation"
  | "actions"
  | "agents"
  | "calls"
  | "settings"
  | "recent"
  | "help";

interface CommandGroup {
  category: CommandCategory;
  label: string;
  items: CommandItem[];
}

// Category config
const categoryConfig: Record<
  CommandCategory,
  { label: string; icon: React.ReactNode }
> = {
  navigation: { label: "Navigation", icon: <ArrowRight className="w-4 h-4" /> },
  actions: { label: "Actions", icon: <Zap className="w-4 h-4" /> },
  agents: { label: "Agents", icon: <Bot className="w-4 h-4" /> },
  calls: { label: "Recent Calls", icon: <Phone className="w-4 h-4" /> },
  settings: { label: "Settings", icon: <Settings className="w-4 h-4" /> },
  recent: { label: "Recent", icon: <Clock className="w-4 h-4" /> },
  help: { label: "Help", icon: <HelpCircle className="w-4 h-4" /> },
};

// Default commands
const defaultCommands: CommandItem[] = [
  // Navigation
  {
    id: "nav-dashboard",
    title: "Go to Dashboard",
    description: "View your main dashboard",
    icon: <Home className="w-4 h-4" />,
    category: "navigation",
    keywords: ["home", "main", "overview"],
    shortcut: ["G", "D"],
    href: "/dashboard",
  },
  {
    id: "nav-agents",
    title: "Go to Agents",
    description: "Manage your AI voice agents",
    icon: <Bot className="w-4 h-4" />,
    category: "navigation",
    keywords: ["bots", "assistants", "ai"],
    shortcut: ["G", "A"],
    href: "/agents",
  },
  {
    id: "nav-calls",
    title: "Go to Calls",
    description: "View call history and analytics",
    icon: <Phone className="w-4 h-4" />,
    category: "navigation",
    keywords: ["history", "logs", "conversations"],
    shortcut: ["G", "C"],
    href: "/calls",
  },
  {
    id: "nav-analytics",
    title: "Go to Analytics",
    description: "View detailed analytics and reports",
    icon: <BarChart3 className="w-4 h-4" />,
    category: "navigation",
    keywords: ["stats", "metrics", "reports", "data"],
    shortcut: ["G", "N"],
    href: "/analytics",
  },
  {
    id: "nav-phone-numbers",
    title: "Go to Phone Numbers",
    description: "Manage your phone numbers",
    icon: <PhoneIncoming className="w-4 h-4" />,
    category: "navigation",
    keywords: ["numbers", "telephony", "sip"],
    href: "/phone-numbers",
  },
  {
    id: "nav-integrations",
    title: "Go to Integrations",
    description: "Connect external services",
    icon: <Zap className="w-4 h-4" />,
    category: "navigation",
    keywords: ["connect", "apps", "crm", "calendar"],
    href: "/integrations",
  },
  {
    id: "nav-workflows",
    title: "Go to Workflows",
    description: "Manage automation workflows",
    icon: <Workflow className="w-4 h-4" />,
    category: "navigation",
    keywords: ["automation", "triggers", "actions"],
    href: "/workflows",
  },
  {
    id: "nav-templates",
    title: "Go to Templates",
    description: "Browse agent templates",
    icon: <FileText className="w-4 h-4" />,
    category: "navigation",
    keywords: ["presets", "starter"],
    href: "/templates",
  },
  {
    id: "nav-team",
    title: "Go to Team",
    description: "Manage team members",
    icon: <Users className="w-4 h-4" />,
    category: "navigation",
    keywords: ["members", "users", "collaborate"],
    href: "/settings/team",
  },
  {
    id: "nav-billing",
    title: "Go to Billing",
    description: "Manage subscription and payments",
    icon: <CreditCard className="w-4 h-4" />,
    category: "navigation",
    keywords: ["payment", "invoice", "subscription", "plan"],
    href: "/settings/billing",
  },
  {
    id: "nav-settings",
    title: "Go to Settings",
    description: "Configure account settings",
    icon: <Settings className="w-4 h-4" />,
    category: "navigation",
    keywords: ["preferences", "account", "config"],
    shortcut: ["G", "S"],
    href: "/settings",
  },
  {
    id: "nav-help",
    title: "Go to Help Center",
    description: "Find help and documentation",
    icon: <HelpCircle className="w-4 h-4" />,
    category: "navigation",
    keywords: ["support", "docs", "faq"],
    shortcut: ["?"],
    href: "/help",
  },

  // Actions
  {
    id: "action-create-agent",
    title: "Create New Agent",
    description: "Create a new AI voice agent",
    icon: <Plus className="w-4 h-4" />,
    category: "actions",
    keywords: ["new", "add", "build"],
    shortcut: ["C", "A"],
    badge: "Quick Action",
    badgeColor: "purple",
  },
  {
    id: "action-make-call",
    title: "Make a Test Call",
    description: "Test your agent with a call",
    icon: <PhoneOutgoing className="w-4 h-4" />,
    category: "actions",
    keywords: ["test", "dial", "outbound"],
    shortcut: ["C", "C"],
  },
  {
    id: "action-buy-number",
    title: "Get New Phone Number",
    description: "Purchase a new phone number",
    icon: <Phone className="w-4 h-4" />,
    category: "actions",
    keywords: ["buy", "purchase", "number"],
  },
  {
    id: "action-create-workflow",
    title: "Create Workflow",
    description: "Create a new automation workflow",
    icon: <Workflow className="w-4 h-4" />,
    category: "actions",
    keywords: ["automation", "trigger"],
  },
  {
    id: "action-export-data",
    title: "Export Data",
    description: "Export calls, recordings, or analytics",
    icon: <Download className="w-4 h-4" />,
    category: "actions",
    keywords: ["download", "csv", "report"],
  },
  {
    id: "action-invite-member",
    title: "Invite Team Member",
    description: "Invite someone to your team",
    icon: <Users className="w-4 h-4" />,
    category: "actions",
    keywords: ["add", "user", "collaborate"],
  },
  {
    id: "action-refresh",
    title: "Refresh Data",
    description: "Refresh the current page data",
    icon: <RefreshCw className="w-4 h-4" />,
    category: "actions",
    keywords: ["reload", "sync", "update"],
    shortcut: ["R"],
  },

  // Agents (dynamic - these would come from API)
  {
    id: "agent-sales-assistant",
    title: "Sales Assistant",
    description: "Your sales qualification agent",
    icon: <Bot className="w-4 h-4" />,
    category: "agents",
    keywords: ["sales", "lead"],
    badge: "Active",
    badgeColor: "green",
    href: "/agents/agent_1",
  },
  {
    id: "agent-support-bot",
    title: "Support Bot",
    description: "Customer support agent",
    icon: <Bot className="w-4 h-4" />,
    category: "agents",
    keywords: ["support", "help", "customer"],
    badge: "Active",
    badgeColor: "green",
    href: "/agents/agent_2",
  },
  {
    id: "agent-appointment-scheduler",
    title: "Appointment Scheduler",
    description: "Scheduling and booking agent",
    icon: <Bot className="w-4 h-4" />,
    category: "agents",
    keywords: ["calendar", "booking", "schedule"],
    badge: "Draft",
    badgeColor: "yellow",
    href: "/agents/agent_3",
  },

  // Recent calls (dynamic)
  {
    id: "call-recent-1",
    title: "Call with John Smith",
    description: "5 minutes ago - 4:32 duration",
    icon: <PhoneIncoming className="w-4 h-4" />,
    category: "calls",
    keywords: ["john", "smith"],
    href: "/calls/call_123",
  },
  {
    id: "call-recent-2",
    title: "Call with Sarah Johnson",
    description: "2 hours ago - 8:15 duration",
    icon: <PhoneIncoming className="w-4 h-4" />,
    category: "calls",
    keywords: ["sarah", "johnson"],
    href: "/calls/call_124",
  },
  {
    id: "call-recent-3",
    title: "Call with Tech Solutions Inc",
    description: "Yesterday - 12:45 duration",
    icon: <PhoneOutgoing className="w-4 h-4" />,
    category: "calls",
    keywords: ["tech", "solutions"],
    href: "/calls/call_125",
  },

  // Settings
  {
    id: "settings-profile",
    title: "Edit Profile",
    description: "Update your profile information",
    icon: <User className="w-4 h-4" />,
    category: "settings",
    keywords: ["account", "name", "email"],
    href: "/settings/profile",
  },
  {
    id: "settings-api-keys",
    title: "API Keys",
    description: "Manage your API keys",
    icon: <Key className="w-4 h-4" />,
    category: "settings",
    keywords: ["token", "secret", "developer"],
    href: "/settings/api-keys",
  },
  {
    id: "settings-security",
    title: "Security Settings",
    description: "Password, 2FA, and sessions",
    icon: <Shield className="w-4 h-4" />,
    category: "settings",
    keywords: ["password", "2fa", "authentication"],
    href: "/settings/security",
  },
  {
    id: "settings-notifications",
    title: "Notification Settings",
    description: "Configure notification preferences",
    icon: <Bell className="w-4 h-4" />,
    category: "settings",
    keywords: ["alerts", "email", "push"],
    href: "/settings/notifications",
  },
  {
    id: "settings-theme",
    title: "Toggle Dark Mode",
    description: "Switch between light and dark theme",
    icon: <Moon className="w-4 h-4" />,
    category: "settings",
    keywords: ["theme", "light", "appearance"],
    shortcut: ["T"],
  },

  // Help
  {
    id: "help-docs",
    title: "Documentation",
    description: "Read the full documentation",
    icon: <BookOpen className="w-4 h-4" />,
    category: "help",
    keywords: ["docs", "guide", "manual"],
    href: "/help",
  },
  {
    id: "help-api-docs",
    title: "API Documentation",
    description: "REST API reference",
    icon: <Code className="w-4 h-4" />,
    category: "help",
    keywords: ["api", "developer", "reference"],
    href: "/help/api",
  },
  {
    id: "help-shortcuts",
    title: "Keyboard Shortcuts",
    description: "View all keyboard shortcuts",
    icon: <Keyboard className="w-4 h-4" />,
    category: "help",
    keywords: ["keys", "hotkeys"],
    shortcut: ["Ctrl", "/"],
  },
  {
    id: "help-contact",
    title: "Contact Support",
    description: "Get help from our team",
    icon: <MessageSquare className="w-4 h-4" />,
    category: "help",
    keywords: ["support", "chat", "email"],
    href: "/help/contact",
  },
  {
    id: "help-feedback",
    title: "Send Feedback",
    description: "Share your thoughts with us",
    icon: <Send className="w-4 h-4" />,
    category: "help",
    keywords: ["suggestion", "bug", "feature"],
  },
];

// Fuzzy search function
function fuzzyMatch(query: string, text: string): number {
  const q = query.toLowerCase();
  const t = text.toLowerCase();

  if (t.includes(q)) return 100;
  if (t.startsWith(q)) return 90;

  let score = 0;
  let lastIndex = -1;

  for (const char of q) {
    const index = t.indexOf(char, lastIndex + 1);
    if (index === -1) return 0;
    score += index === lastIndex + 1 ? 10 : 5;
    lastIndex = index;
  }

  return Math.min(score, 80);
}

function searchCommands(query: string, commands: CommandItem[]): CommandItem[] {
  if (!query.trim()) return [];

  const results = commands
    .map((cmd) => {
      const titleScore = fuzzyMatch(query, cmd.title);
      const descScore = cmd.description
        ? fuzzyMatch(query, cmd.description) * 0.5
        : 0;
      const keywordScore = cmd.keywords
        ? Math.max(
            ...cmd.keywords.map((k) => fuzzyMatch(query, k) * 0.7),
            0
          )
        : 0;

      const score = Math.max(titleScore, descScore, keywordScore);
      return { cmd, score };
    })
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .map(({ cmd }) => cmd);

  return results;
}

// Highlight matching text
function highlightMatch(text: string, query: string): React.ReactNode {
  if (!query.trim()) return text;

  const q = query.toLowerCase();
  const t = text.toLowerCase();
  const index = t.indexOf(q);

  if (index === -1) return text;

  return (
    <>
      {text.slice(0, index)}
      <span className="text-purple-400 font-medium">
        {text.slice(index, index + query.length)}
      </span>
      {text.slice(index + query.length)}
    </>
  );
}

// Shortcut badge component
function ShortcutBadge({ keys }: { keys: string[] }) {
  return (
    <div className="flex items-center gap-1">
      {keys.map((key, i) => (
        <span
          key={i}
          className="px-1.5 py-0.5 text-xs bg-white/10 text-gray-400 rounded font-mono"
        >
          {key === "Ctrl" ? "^" : key === "Cmd" ? "⌘" : key}
        </span>
      ))}
    </div>
  );
}

// Command item component
function CommandItemComponent({
  item,
  isSelected,
  query,
  onSelect,
}: {
  item: CommandItem;
  isSelected: boolean;
  query: string;
  onSelect: () => void;
}) {
  return (
    <div
      onClick={onSelect}
      className={`flex items-center gap-3 px-4 py-3 cursor-pointer transition-colors ${
        isSelected
          ? "bg-purple-500/20 text-white"
          : "text-gray-300 hover:bg-white/5"
      } ${item.disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      {/* Icon */}
      <span
        className={`flex-shrink-0 ${
          isSelected ? "text-purple-400" : "text-gray-500"
        }`}
      >
        {item.icon || <FileText className="w-4 h-4" />}
      </span>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium truncate">
            {highlightMatch(item.title, query)}
          </span>
          {item.badge && (
            <span
              className={`px-1.5 py-0.5 text-xs rounded ${
                item.badgeColor === "green"
                  ? "bg-green-500/10 text-green-400"
                  : item.badgeColor === "yellow"
                  ? "bg-yellow-500/10 text-yellow-400"
                  : item.badgeColor === "red"
                  ? "bg-red-500/10 text-red-400"
                  : "bg-purple-500/10 text-purple-400"
              }`}
            >
              {item.badge}
            </span>
          )}
        </div>
        {item.description && (
          <p className="text-sm text-gray-500 truncate">
            {highlightMatch(item.description, query)}
          </p>
        )}
      </div>

      {/* Shortcut or arrow */}
      {item.shortcut ? (
        <ShortcutBadge keys={item.shortcut} />
      ) : (
        <ChevronRight
          className={`w-4 h-4 flex-shrink-0 ${
            isSelected ? "text-purple-400" : "text-gray-600"
          }`}
        />
      )}
    </div>
  );
}

// Context
interface CommandPaletteContextType {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  registerCommand: (command: CommandItem) => void;
  unregisterCommand: (id: string) => void;
}

const CommandPaletteContext = createContext<CommandPaletteContextType | null>(
  null
);

export function useCommandPalette() {
  const context = useContext(CommandPaletteContext);
  if (!context) {
    throw new Error(
      "useCommandPalette must be used within CommandPaletteProvider"
    );
  }
  return context;
}

// Provider component
export function CommandPaletteProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [customCommands, setCustomCommands] = useState<CommandItem[]>([]);

  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);
  const toggle = useCallback(() => setIsOpen((prev) => !prev), []);

  const registerCommand = useCallback((command: CommandItem) => {
    setCustomCommands((prev) => {
      if (prev.some((c) => c.id === command.id)) {
        return prev.map((c) => (c.id === command.id ? command : c));
      }
      return [...prev, command];
    });
  }, []);

  const unregisterCommand = useCallback((id: string) => {
    setCustomCommands((prev) => prev.filter((c) => c.id !== id));
  }, []);

  // Global keyboard shortcut
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Cmd/Ctrl + K
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        toggle();
      }
      // Escape
      if (e.key === "Escape" && isOpen) {
        close();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [toggle, close, isOpen]);

  const allCommands = useMemo(
    () => [...defaultCommands, ...customCommands],
    [customCommands]
  );

  return (
    <CommandPaletteContext.Provider
      value={{
        isOpen,
        open,
        close,
        toggle,
        registerCommand,
        unregisterCommand,
      }}
    >
      {children}
      <CommandPaletteModal
        isOpen={isOpen}
        onClose={close}
        commands={allCommands}
      />
    </CommandPaletteContext.Provider>
  );
}

// Modal component
function CommandPaletteModal({
  isOpen,
  onClose,
  commands,
}: {
  isOpen: boolean;
  onClose: () => void;
  commands: CommandItem[];
}) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Get filtered results
  const results = useMemo(() => {
    if (!query.trim()) {
      // Show default groups when no query
      const groups: CommandGroup[] = [];

      // Recent (show first 3)
      const recentItems = commands
        .filter((c) => c.category === "recent" || c.category === "calls")
        .slice(0, 3);
      if (recentItems.length > 0) {
        groups.push({ category: "recent", label: "Recent", items: recentItems });
      }

      // Quick actions
      const actionItems = commands
        .filter((c) => c.category === "actions")
        .slice(0, 4);
      if (actionItems.length > 0) {
        groups.push({
          category: "actions",
          label: "Quick Actions",
          items: actionItems,
        });
      }

      // Navigation
      const navItems = commands
        .filter((c) => c.category === "navigation")
        .slice(0, 6);
      if (navItems.length > 0) {
        groups.push({
          category: "navigation",
          label: "Navigation",
          items: navItems,
        });
      }

      return groups;
    }

    // Search mode
    const searchResults = searchCommands(query, commands);
    if (searchResults.length === 0) return [];

    // Group by category
    const grouped: Record<CommandCategory, CommandItem[]> = {} as any;
    searchResults.forEach((item) => {
      if (!grouped[item.category]) {
        grouped[item.category] = [];
      }
      grouped[item.category].push(item);
    });

    return Object.entries(grouped).map(([category, items]) => ({
      category: category as CommandCategory,
      label: categoryConfig[category as CommandCategory].label,
      items,
    }));
  }, [query, commands]);

  // Flatten items for keyboard navigation
  const flatItems = useMemo(
    () => results.flatMap((group) => group.items),
    [results]
  );

  // Handle keyboard navigation
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (!isOpen) return;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev < flatItems.length - 1 ? prev + 1 : prev
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) => (prev > 0 ? prev - 1 : prev));
          break;
        case "Enter":
          e.preventDefault();
          if (flatItems[selectedIndex]) {
            executeCommand(flatItems[selectedIndex]);
          }
          break;
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, selectedIndex, flatItems]);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current && selectedIndex >= 0) {
      const selectedElement = listRef.current.querySelector(
        `[data-index="${selectedIndex}"]`
      );
      selectedElement?.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  // Reset selection when query changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  const executeCommand = (item: CommandItem) => {
    if (item.disabled) return;

    if (item.action) {
      item.action();
    } else if (item.href) {
      window.location.href = item.href;
    }

    onClose();
  };

  if (!isOpen) return null;

  let itemIndex = 0;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-2xl bg-[#0f0f1a] border border-white/10 rounded-2xl shadow-2xl overflow-hidden">
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-4 border-b border-white/10">
          <Search className="w-5 h-5 text-gray-500 flex-shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search commands, pages, agents..."
            className="flex-1 bg-transparent text-white text-lg placeholder:text-gray-500 focus:outline-none"
          />
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="px-1.5 py-0.5 bg-white/10 rounded">esc</span>
            <span>to close</span>
          </div>
        </div>

        {/* Results */}
        <div
          ref={listRef}
          className="max-h-[400px] overflow-y-auto scrollbar-thin scrollbar-thumb-white/10"
        >
          {results.length > 0 ? (
            results.map((group) => (
              <div key={group.category}>
                {/* Group header */}
                <div className="px-4 py-2 bg-white/5 sticky top-0">
                  <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                    {group.label}
                  </span>
                </div>

                {/* Group items */}
                {group.items.map((item) => {
                  const currentIndex = itemIndex++;
                  return (
                    <div key={item.id} data-index={currentIndex}>
                      <CommandItemComponent
                        item={item}
                        isSelected={currentIndex === selectedIndex}
                        query={query}
                        onSelect={() => executeCommand(item)}
                      />
                    </div>
                  );
                })}
              </div>
            ))
          ) : query.trim() ? (
            <div className="px-4 py-12 text-center">
              <AlertCircle className="w-10 h-10 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-500">
                No results found for "{query}"
              </p>
              <p className="text-sm text-gray-600 mt-1">
                Try different keywords or browse categories
              </p>
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-white/10 bg-white/5">
          <div className="flex items-center gap-4 text-xs text-gray-500">
            <div className="flex items-center gap-1">
              <ArrowUp className="w-3 h-3" />
              <ArrowDown className="w-3 h-3" />
              <span>Navigate</span>
            </div>
            <div className="flex items-center gap-1">
              <CornerDownLeft className="w-3 h-3" />
              <span>Select</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="px-1 py-0.5 bg-white/10 rounded text-[10px]">
                esc
              </span>
              <span>Close</span>
            </div>
          </div>
          <div className="flex items-center gap-1 text-xs text-gray-500">
            <Command className="w-3 h-3" />
            <span>K</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Keyboard shortcut trigger button
export function CommandPaletteTrigger() {
  const { open } = useCommandPalette();

  return (
    <button
      onClick={open}
      className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg transition-colors"
    >
      <Search className="w-4 h-4 text-gray-400" />
      <span className="text-sm text-gray-400">Search...</span>
      <div className="flex items-center gap-0.5 ml-2">
        <span className="px-1 py-0.5 text-[10px] bg-white/10 text-gray-500 rounded">
          ⌘
        </span>
        <span className="px-1 py-0.5 text-[10px] bg-white/10 text-gray-500 rounded">
          K
        </span>
      </div>
    </button>
  );
}

// Export default commands for customization
export { defaultCommands };
