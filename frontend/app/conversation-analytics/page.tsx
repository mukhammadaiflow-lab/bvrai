"use client";

import React, { useState, useEffect, useMemo } from "react";
import DashboardLayout from "../components/DashboardLayout";
import {
  MessageSquare,
  MessageCircle,
  BarChart3,
  PieChart,
  TrendingUp,
  TrendingDown,
  ArrowUp,
  ArrowDown,
  ArrowRight,
  Activity,
  Zap,
  Target,
  Award,
  Users,
  Clock,
  Calendar,
  Filter,
  Search,
  Download,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Eye,
  Play,
  Pause,
  Volume2,
  Mic,
  MicOff,
  Phone,
  PhoneCall,
  PhoneIncoming,
  PhoneOutgoing,
  CheckCircle,
  XCircle,
  AlertTriangle,
  AlertCircle,
  Info,
  ThumbsUp,
  ThumbsDown,
  Heart,
  Frown,
  Meh,
  Smile,
  Star,
  Bot,
  User,
  Brain,
  Sparkles,
  Hash,
  Tag,
  FileText,
  List,
  Grid,
  Layers,
  GitBranch,
  Lightbulb,
  HelpCircle,
  ExternalLink,
  Copy,
  MoreVertical,
  Settings,
  Globe,
  MapPin,
} from "lucide-react";

// Types
type TimeRange = "today" | "yesterday" | "7d" | "30d" | "90d" | "custom";
type SentimentType = "positive" | "neutral" | "negative" | "mixed";
type IntentCategory = "inquiry" | "complaint" | "purchase" | "support" | "feedback" | "other";

interface ConversationMetrics {
  totalConversations: number;
  avgDuration: number;
  avgTurns: number;
  resolutionRate: number;
  escalationRate: number;
  sentimentBreakdown: {
    positive: number;
    neutral: number;
    negative: number;
    mixed: number;
  };
  topIntents: { name: string; count: number; percentage: number }[];
  topTopics: { name: string; count: number; trend: number }[];
  hourlyDistribution: { hour: number; count: number }[];
  responseTimeDistribution: { range: string; count: number }[];
}

interface ConversationSummary {
  id: string;
  startTime: Date;
  duration: number;
  turns: number;
  sentiment: SentimentType;
  sentimentScore: number;
  intent: IntentCategory;
  topics: string[];
  resolved: boolean;
  escalated: boolean;
  agentId: string;
  agentName: string;
  customerId?: string;
  customerName?: string;
  summary: string;
  keyInsights: string[];
  actionItems: string[];
}

interface InsightItem {
  id: string;
  type: "trend" | "anomaly" | "opportunity" | "warning";
  title: string;
  description: string;
  metric: string;
  change: number;
  priority: "high" | "medium" | "low";
  timestamp: Date;
}

// Mock data
const mockMetrics: ConversationMetrics = {
  totalConversations: 15847,
  avgDuration: 5.8,
  avgTurns: 12.4,
  resolutionRate: 87.5,
  escalationRate: 8.2,
  sentimentBreakdown: {
    positive: 62,
    neutral: 25,
    negative: 10,
    mixed: 3,
  },
  topIntents: [
    { name: "Product Inquiry", count: 4520, percentage: 28.5 },
    { name: "Order Status", count: 3180, percentage: 20.1 },
    { name: "Technical Support", count: 2890, percentage: 18.2 },
    { name: "Billing Question", count: 2145, percentage: 13.5 },
    { name: "Returns/Refunds", count: 1680, percentage: 10.6 },
    { name: "Other", count: 1432, percentage: 9.0 },
  ],
  topTopics: [
    { name: "Pricing", count: 3245, trend: 12.5 },
    { name: "Delivery", count: 2890, trend: 8.3 },
    { name: "Product Features", count: 2456, trend: -3.2 },
    { name: "Account Issues", count: 1987, trend: 15.7 },
    { name: "Integration", count: 1654, trend: 22.1 },
    { name: "Cancellation", count: 1234, trend: -8.5 },
  ],
  hourlyDistribution: [
    { hour: 0, count: 120 }, { hour: 1, count: 85 }, { hour: 2, count: 45 }, { hour: 3, count: 32 },
    { hour: 4, count: 28 }, { hour: 5, count: 45 }, { hour: 6, count: 125 }, { hour: 7, count: 310 },
    { hour: 8, count: 580 }, { hour: 9, count: 890 }, { hour: 10, count: 1120 }, { hour: 11, count: 1250 },
    { hour: 12, count: 1100 }, { hour: 13, count: 1180 }, { hour: 14, count: 1320 }, { hour: 15, count: 1280 },
    { hour: 16, count: 1150 }, { hour: 17, count: 980 }, { hour: 18, count: 720 }, { hour: 19, count: 540 },
    { hour: 20, count: 380 }, { hour: 21, count: 290 }, { hour: 22, count: 210 }, { hour: 23, count: 167 },
  ],
  responseTimeDistribution: [
    { range: "< 1s", count: 4520 },
    { range: "1-2s", count: 5680 },
    { range: "2-3s", count: 3245 },
    { range: "3-5s", count: 1567 },
    { range: "> 5s", count: 835 },
  ],
};

const mockConversations: ConversationSummary[] = [
  {
    id: "conv-1",
    startTime: new Date(Date.now() - 30 * 60 * 1000),
    duration: 5.5,
    turns: 14,
    sentiment: "positive",
    sentimentScore: 0.85,
    intent: "purchase",
    topics: ["pricing", "enterprise plan", "features"],
    resolved: true,
    escalated: false,
    agentId: "agent-1",
    agentName: "Sales AI",
    customerName: "John Smith",
    summary: "Customer inquired about enterprise pricing and features. Successfully provided detailed information and scheduled a demo.",
    keyInsights: ["High purchase intent", "Interested in team features", "Budget approved"],
    actionItems: ["Send pricing proposal", "Schedule demo for next week"],
  },
  {
    id: "conv-2",
    startTime: new Date(Date.now() - 2 * 60 * 60 * 1000),
    duration: 8.2,
    turns: 22,
    sentiment: "negative",
    sentimentScore: 0.25,
    intent: "complaint",
    topics: ["billing", "overcharge", "refund"],
    resolved: false,
    escalated: true,
    agentId: "agent-2",
    agentName: "Support AI",
    customerName: "Sarah Johnson",
    summary: "Customer reported billing discrepancy. Issue requires manual review and escalated to billing team.",
    keyInsights: ["Recurring billing issue", "Customer frustrated", "Churn risk"],
    actionItems: ["Review billing history", "Process refund", "Follow up within 24 hours"],
  },
  {
    id: "conv-3",
    startTime: new Date(Date.now() - 4 * 60 * 60 * 1000),
    duration: 3.8,
    turns: 8,
    sentiment: "neutral",
    sentimentScore: 0.55,
    intent: "inquiry",
    topics: ["integration", "api", "documentation"],
    resolved: true,
    escalated: false,
    agentId: "agent-1",
    agentName: "Sales AI",
    customerName: "Mike Chen",
    summary: "Developer asking about API integration capabilities. Provided documentation links and code examples.",
    keyInsights: ["Technical user", "Self-service preferred", "API-first approach"],
    actionItems: ["Send API documentation", "Share sample code repository"],
  },
  {
    id: "conv-4",
    startTime: new Date(Date.now() - 6 * 60 * 60 * 1000),
    duration: 6.1,
    turns: 16,
    sentiment: "positive",
    sentimentScore: 0.78,
    intent: "support",
    topics: ["onboarding", "setup", "configuration"],
    resolved: true,
    escalated: false,
    agentId: "agent-3",
    agentName: "Onboarding AI",
    customerName: "Emily Davis",
    summary: "New customer completing initial setup. Successfully guided through configuration and activated account.",
    keyInsights: ["New customer", "Quick learner", "Positive first impression"],
    actionItems: ["Send welcome email", "Schedule check-in call"],
  },
  {
    id: "conv-5",
    startTime: new Date(Date.now() - 8 * 60 * 60 * 1000),
    duration: 2.4,
    turns: 6,
    sentiment: "positive",
    sentimentScore: 0.92,
    intent: "feedback",
    topics: ["feature request", "product feedback"],
    resolved: true,
    escalated: false,
    agentId: "agent-2",
    agentName: "Support AI",
    customerName: "David Wilson",
    summary: "Customer provided positive feedback about recent update and requested a new feature.",
    keyInsights: ["Power user", "Engaged customer", "Feature advocate"],
    actionItems: ["Log feature request", "Add to feedback board"],
  },
];

const mockInsights: InsightItem[] = [
  {
    id: "ins-1",
    type: "trend",
    title: "Positive sentiment trending up",
    description: "Customer satisfaction has increased by 15% over the past week",
    metric: "Sentiment Score",
    change: 15,
    priority: "high",
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
  },
  {
    id: "ins-2",
    type: "anomaly",
    title: "Spike in billing inquiries",
    description: "Billing-related conversations increased 45% in the last 24 hours",
    metric: "Topic Volume",
    change: 45,
    priority: "high",
    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
  },
  {
    id: "ins-3",
    type: "opportunity",
    title: "High purchase intent detected",
    description: "32 conversations showed strong buying signals in the last 6 hours",
    metric: "Intent Score",
    change: 12,
    priority: "medium",
    timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000),
  },
  {
    id: "ins-4",
    type: "warning",
    title: "Escalation rate increasing",
    description: "Agent escalations up 8% - consider reviewing conversation flows",
    metric: "Escalation Rate",
    change: 8,
    priority: "medium",
    timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000),
  },
];

// Helper functions
const getSentimentColor = (sentiment: SentimentType) => {
  switch (sentiment) {
    case "positive":
      return "text-green-400 bg-green-500/20";
    case "neutral":
      return "text-gray-400 bg-gray-500/20";
    case "negative":
      return "text-red-400 bg-red-500/20";
    case "mixed":
      return "text-yellow-400 bg-yellow-500/20";
  }
};

const getSentimentIcon = (sentiment: SentimentType) => {
  switch (sentiment) {
    case "positive":
      return <Smile className="w-4 h-4" />;
    case "neutral":
      return <Meh className="w-4 h-4" />;
    case "negative":
      return <Frown className="w-4 h-4" />;
    case "mixed":
      return <AlertCircle className="w-4 h-4" />;
  }
};

const getInsightColor = (type: string) => {
  switch (type) {
    case "trend":
      return "text-blue-400 bg-blue-500/20 border-blue-500/30";
    case "anomaly":
      return "text-orange-400 bg-orange-500/20 border-orange-500/30";
    case "opportunity":
      return "text-green-400 bg-green-500/20 border-green-500/30";
    case "warning":
      return "text-red-400 bg-red-500/20 border-red-500/30";
    default:
      return "text-gray-400 bg-gray-500/20 border-gray-500/30";
  }
};

const getInsightIcon = (type: string) => {
  switch (type) {
    case "trend":
      return <TrendingUp className="w-4 h-4" />;
    case "anomaly":
      return <AlertTriangle className="w-4 h-4" />;
    case "opportunity":
      return <Lightbulb className="w-4 h-4" />;
    case "warning":
      return <AlertCircle className="w-4 h-4" />;
    default:
      return <Info className="w-4 h-4" />;
  }
};

const formatDuration = (minutes: number) => {
  if (minutes < 1) return `${Math.round(minutes * 60)}s`;
  return `${minutes.toFixed(1)}m`;
};

const formatRelativeTime = (date: Date) => {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / (1000 * 60));
  const hours = Math.floor(diff / (1000 * 60 * 60));

  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return date.toLocaleDateString();
};

const formatNumber = (num: number) => {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
};

// Metric Card Component
const MetricCard: React.FC<{
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
  color: string;
  suffix?: string;
}> = ({ title, value, change, icon, color, suffix }) => (
  <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4 hover:border-purple-500/30 transition-all duration-300">
    <div className="flex items-start justify-between">
      <div className={`p-2 rounded-lg ${color}`}>
        {icon}
      </div>
      {change !== undefined && (
        <div className={`flex items-center gap-1 text-sm ${change >= 0 ? "text-green-400" : "text-red-400"}`}>
          {change >= 0 ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
          {Math.abs(change)}%
        </div>
      )}
    </div>
    <div className="mt-3">
      <p className="text-2xl font-bold text-white">
        {typeof value === "number" ? formatNumber(value) : value}
        {suffix && <span className="text-lg text-gray-400 ml-1">{suffix}</span>}
      </p>
      <p className="text-sm text-gray-400 mt-1">{title}</p>
    </div>
  </div>
);

// Conversation Card Component
const ConversationCard: React.FC<{
  conversation: ConversationSummary;
  onClick: () => void;
}> = ({ conversation, onClick }) => (
  <div
    onClick={onClick}
    className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4 hover:border-purple-500/30 transition-all duration-300 cursor-pointer"
  >
    <div className="flex items-start justify-between mb-3">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${getSentimentColor(conversation.sentiment)}`}>
          {getSentimentIcon(conversation.sentiment)}
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h4 className="font-medium text-white">{conversation.customerName || "Anonymous"}</h4>
            {conversation.escalated && (
              <span className="px-1.5 py-0.5 text-xs bg-red-500/20 text-red-400 rounded">Escalated</span>
            )}
            {conversation.resolved && (
              <span className="px-1.5 py-0.5 text-xs bg-green-500/20 text-green-400 rounded">Resolved</span>
            )}
          </div>
          <p className="text-xs text-gray-500 mt-0.5">
            {conversation.agentName} - {formatRelativeTime(conversation.startTime)}
          </p>
        </div>
      </div>
      <div className="text-right">
        <p className="text-sm text-white">{formatDuration(conversation.duration)}</p>
        <p className="text-xs text-gray-500">{conversation.turns} turns</p>
      </div>
    </div>

    <p className="text-sm text-gray-400 line-clamp-2 mb-3">{conversation.summary}</p>

    <div className="flex items-center gap-2 flex-wrap">
      <span className="px-2 py-0.5 rounded text-xs bg-purple-500/20 text-purple-400 capitalize">
        {conversation.intent}
      </span>
      {conversation.topics.slice(0, 3).map((topic) => (
        <span key={topic} className="px-2 py-0.5 rounded text-xs bg-white/5 text-gray-400">
          {topic}
        </span>
      ))}
    </div>
  </div>
);

// Insight Card Component
const InsightCard: React.FC<{ insight: InsightItem }> = ({ insight }) => (
  <div className={`p-4 rounded-xl border ${getInsightColor(insight.type)}`}>
    <div className="flex items-start gap-3">
      <div className="mt-0.5">
        {getInsightIcon(insight.type)}
      </div>
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1">
          <h4 className="font-medium text-white">{insight.title}</h4>
          <span className={`px-2 py-0.5 rounded text-xs capitalize ${
            insight.priority === "high" ? "bg-red-500/20 text-red-400" :
            insight.priority === "medium" ? "bg-yellow-500/20 text-yellow-400" :
            "bg-gray-500/20 text-gray-400"
          }`}>
            {insight.priority}
          </span>
        </div>
        <p className="text-sm text-gray-400">{insight.description}</p>
        <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
          <span>{insight.metric}</span>
          <span className={insight.change >= 0 ? "text-green-400" : "text-red-400"}>
            {insight.change >= 0 ? "+" : ""}{insight.change}%
          </span>
          <span>{formatRelativeTime(insight.timestamp)}</span>
        </div>
      </div>
    </div>
  </div>
);

// Conversation Detail Dialog
const ConversationDetailDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  conversation: ConversationSummary | null;
}> = ({ isOpen, onClose, conversation }) => {
  if (!isOpen || !conversation) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-2xl bg-[#1a1a2e] border border-white/10 rounded-2xl overflow-hidden">
        <div className="p-6 border-b border-white/10">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-xl ${getSentimentColor(conversation.sentiment)}`}>
                {getSentimentIcon(conversation.sentiment)}
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">{conversation.customerName || "Anonymous"}</h2>
                <p className="text-gray-400">{conversation.agentName} - {formatRelativeTime(conversation.startTime)}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
            >
              <XCircle className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="p-6 max-h-[60vh] overflow-y-auto space-y-6">
          {/* Stats */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-black/30 rounded-lg p-3 text-center">
              <p className="text-lg font-bold text-white">{formatDuration(conversation.duration)}</p>
              <p className="text-xs text-gray-500">Duration</p>
            </div>
            <div className="bg-black/30 rounded-lg p-3 text-center">
              <p className="text-lg font-bold text-white">{conversation.turns}</p>
              <p className="text-xs text-gray-500">Turns</p>
            </div>
            <div className="bg-black/30 rounded-lg p-3 text-center">
              <p className={`text-lg font-bold ${
                conversation.sentimentScore >= 0.7 ? "text-green-400" :
                conversation.sentimentScore >= 0.4 ? "text-yellow-400" :
                "text-red-400"
              }`}>
                {Math.round(conversation.sentimentScore * 100)}%
              </p>
              <p className="text-xs text-gray-500">Sentiment</p>
            </div>
            <div className="bg-black/30 rounded-lg p-3 text-center">
              <p className={`text-lg font-bold ${conversation.resolved ? "text-green-400" : "text-yellow-400"}`}>
                {conversation.resolved ? "Yes" : "No"}
              </p>
              <p className="text-xs text-gray-500">Resolved</p>
            </div>
          </div>

          {/* Summary */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-2">Summary</h4>
            <p className="text-gray-400">{conversation.summary}</p>
          </div>

          {/* Intent & Topics */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-semibold text-gray-300 mb-2">Intent</h4>
              <span className="px-3 py-1.5 rounded-lg bg-purple-500/20 text-purple-400 capitalize">
                {conversation.intent}
              </span>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-300 mb-2">Topics</h4>
              <div className="flex flex-wrap gap-2">
                {conversation.topics.map((topic) => (
                  <span key={topic} className="px-2 py-1 rounded bg-white/5 text-gray-400 text-sm">
                    {topic}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Key Insights */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-2">Key Insights</h4>
            <div className="space-y-2">
              {conversation.keyInsights.map((insight, i) => (
                <div key={i} className="flex items-start gap-2">
                  <Lightbulb className="w-4 h-4 text-yellow-400 mt-0.5" />
                  <span className="text-gray-400 text-sm">{insight}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Action Items */}
          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-2">Action Items</h4>
            <div className="space-y-2">
              {conversation.actionItems.map((item, i) => (
                <div key={i} className="flex items-start gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5" />
                  <span className="text-gray-400 text-sm">{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="p-6 border-t border-white/10 flex justify-end gap-3">
          <button className="px-4 py-2 rounded-lg bg-white/10 text-white hover:bg-white/20 transition-colors flex items-center gap-2">
            <Play className="w-4 h-4" />
            Play Recording
          </button>
          <button className="px-4 py-2 rounded-lg bg-white/10 text-white hover:bg-white/20 transition-colors flex items-center gap-2">
            <FileText className="w-4 h-4" />
            View Transcript
          </button>
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
export default function ConversationAnalyticsPage() {
  const [metrics, setMetrics] = useState<ConversationMetrics>(mockMetrics);
  const [conversations, setConversations] = useState<ConversationSummary[]>(mockConversations);
  const [insights, setInsights] = useState<InsightItem[]>(mockInsights);
  const [timeRange, setTimeRange] = useState<TimeRange>("7d");
  const [selectedConversation, setSelectedConversation] = useState<ConversationSummary | null>(null);
  const [showDetailDialog, setShowDetailDialog] = useState(false);
  const [activeView, setActiveView] = useState<"overview" | "conversations" | "insights">("overview");

  // Simulated chart data
  const sentimentData = [
    { name: "Positive", value: metrics.sentimentBreakdown.positive, color: "#22c55e" },
    { name: "Neutral", value: metrics.sentimentBreakdown.neutral, color: "#6b7280" },
    { name: "Negative", value: metrics.sentimentBreakdown.negative, color: "#ef4444" },
    { name: "Mixed", value: metrics.sentimentBreakdown.mixed, color: "#eab308" },
  ];

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Conversation Analytics</h1>
            <p className="text-gray-400 mt-1">Analyze and understand your AI conversations</p>
          </div>

          <div className="flex items-center gap-3">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as TimeRange)}
              className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="today">Today</option>
              <option value="yesterday">Yesterday</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
              <option value="90d">Last 90 Days</option>
            </select>

            <button className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors">
              <Download className="w-4 h-4" />
              Export
            </button>

            <button className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white transition-colors">
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* View Tabs */}
        <div className="flex gap-2 p-1 bg-[#1a1a2e]/80 rounded-lg w-fit">
          {(["overview", "conversations", "insights"] as const).map((view) => (
            <button
              key={view}
              onClick={() => setActiveView(view)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeView === view
                  ? "bg-purple-500/20 text-purple-400"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              {view.charAt(0).toUpperCase() + view.slice(1)}
            </button>
          ))}
        </div>

        {activeView === "overview" && (
          <>
            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <MetricCard
                title="Total Conversations"
                value={metrics.totalConversations}
                change={12}
                icon={<MessageSquare className="w-5 h-5" />}
                color="bg-blue-500/20 text-blue-400"
              />
              <MetricCard
                title="Avg Duration"
                value={metrics.avgDuration}
                suffix="min"
                change={-5}
                icon={<Clock className="w-5 h-5" />}
                color="bg-purple-500/20 text-purple-400"
              />
              <MetricCard
                title="Avg Turns"
                value={metrics.avgTurns}
                change={3}
                icon={<MessageCircle className="w-5 h-5" />}
                color="bg-green-500/20 text-green-400"
              />
              <MetricCard
                title="Resolution Rate"
                value={`${metrics.resolutionRate}%`}
                change={8}
                icon={<CheckCircle className="w-5 h-5" />}
                color="bg-emerald-500/20 text-emerald-400"
              />
              <MetricCard
                title="Escalation Rate"
                value={`${metrics.escalationRate}%`}
                change={-15}
                icon={<ArrowUp className="w-5 h-5" />}
                color="bg-orange-500/20 text-orange-400"
              />
              <MetricCard
                title="Positive Sentiment"
                value={`${metrics.sentimentBreakdown.positive}%`}
                change={10}
                icon={<Smile className="w-5 h-5" />}
                color="bg-green-500/20 text-green-400"
              />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Sentiment Distribution */}
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Sentiment Distribution</h3>
                <div className="flex items-center justify-center gap-8">
                  <div className="relative w-40 h-40">
                    {/* Placeholder for pie chart */}
                    <div className="absolute inset-0 rounded-full border-8 border-green-500/50" style={{ clipPath: `polygon(50% 50%, 50% 0%, 100% 0%, 100% 100%, 50% 100%)` }} />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-white">{metrics.sentimentBreakdown.positive}%</p>
                        <p className="text-xs text-gray-400">Positive</p>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-3">
                    {sentimentData.map((item) => (
                      <div key={item.name} className="flex items-center gap-3">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: item.color }}
                        />
                        <span className="text-gray-400 text-sm w-20">{item.name}</span>
                        <span className="text-white font-medium">{item.value}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Top Intents */}
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Top Intents</h3>
                <div className="space-y-3">
                  {metrics.topIntents.map((intent, i) => (
                    <div key={intent.name} className="flex items-center gap-3">
                      <span className="text-gray-500 text-sm w-6">{i + 1}</span>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-white text-sm">{intent.name}</span>
                          <span className="text-gray-400 text-sm">{intent.percentage}%</span>
                        </div>
                        <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                            style={{ width: `${intent.percentage}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Topics & Hourly Distribution */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Top Topics */}
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Trending Topics</h3>
                <div className="space-y-3">
                  {metrics.topTopics.map((topic) => (
                    <div key={topic.name} className="flex items-center justify-between p-3 bg-black/20 rounded-lg">
                      <div className="flex items-center gap-3">
                        <Tag className="w-4 h-4 text-purple-400" />
                        <span className="text-white">{topic.name}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-gray-400">{formatNumber(topic.count)}</span>
                        <span className={`flex items-center gap-1 text-sm ${topic.trend >= 0 ? "text-green-400" : "text-red-400"}`}>
                          {topic.trend >= 0 ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
                          {Math.abs(topic.trend)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Hourly Distribution */}
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Hourly Distribution</h3>
                <div className="flex items-end justify-between h-40 gap-1">
                  {metrics.hourlyDistribution.map((item) => {
                    const maxCount = Math.max(...metrics.hourlyDistribution.map((h) => h.count));
                    const height = (item.count / maxCount) * 100;
                    return (
                      <div
                        key={item.hour}
                        className="flex-1 bg-gradient-to-t from-purple-500/50 to-purple-500/20 rounded-t hover:from-purple-500/70 hover:to-purple-500/40 transition-all cursor-pointer group relative"
                        style={{ height: `${height}%` }}
                        title={`${item.hour}:00 - ${item.count} conversations`}
                      >
                        <div className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 bg-black/90 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                          {item.hour}:00 - {item.count}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between mt-2 text-xs text-gray-500">
                  <span>12am</span>
                  <span>6am</span>
                  <span>12pm</span>
                  <span>6pm</span>
                  <span>12am</span>
                </div>
              </div>
            </div>

            {/* Recent Conversations */}
            <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Recent Conversations</h3>
                <button
                  onClick={() => setActiveView("conversations")}
                  className="text-sm text-purple-400 hover:text-purple-300 flex items-center gap-1"
                >
                  View All
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {conversations.slice(0, 4).map((conv) => (
                  <ConversationCard
                    key={conv.id}
                    conversation={conv}
                    onClick={() => {
                      setSelectedConversation(conv);
                      setShowDetailDialog(true);
                    }}
                  />
                ))}
              </div>
            </div>
          </>
        )}

        {activeView === "conversations" && (
          <div className="space-y-4">
            {/* Search & Filters */}
            <div className="flex flex-col md:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search conversations..."
                  className="w-full pl-10 pr-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div className="flex gap-2">
                <select className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500">
                  <option value="all">All Sentiments</option>
                  <option value="positive">Positive</option>
                  <option value="neutral">Neutral</option>
                  <option value="negative">Negative</option>
                </select>
                <select className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500">
                  <option value="all">All Intents</option>
                  <option value="inquiry">Inquiry</option>
                  <option value="complaint">Complaint</option>
                  <option value="purchase">Purchase</option>
                  <option value="support">Support</option>
                </select>
              </div>
            </div>

            {/* Conversations List */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {conversations.map((conv) => (
                <ConversationCard
                  key={conv.id}
                  conversation={conv}
                  onClick={() => {
                    setSelectedConversation(conv);
                    setShowDetailDialog(true);
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {activeView === "insights" && (
          <div className="space-y-6">
            {/* Insights Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4 text-center">
                <p className="text-3xl font-bold text-white">{insights.length}</p>
                <p className="text-sm text-gray-400">Total Insights</p>
              </div>
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4 text-center">
                <p className="text-3xl font-bold text-red-400">{insights.filter((i) => i.priority === "high").length}</p>
                <p className="text-sm text-gray-400">High Priority</p>
              </div>
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4 text-center">
                <p className="text-3xl font-bold text-green-400">{insights.filter((i) => i.type === "opportunity").length}</p>
                <p className="text-sm text-gray-400">Opportunities</p>
              </div>
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4 text-center">
                <p className="text-3xl font-bold text-orange-400">{insights.filter((i) => i.type === "anomaly").length}</p>
                <p className="text-sm text-gray-400">Anomalies</p>
              </div>
            </div>

            {/* Insights List */}
            <div className="space-y-4">
              {insights.map((insight) => (
                <InsightCard key={insight.id} insight={insight} />
              ))}
            </div>
          </div>
        )}

        {/* Conversation Detail Dialog */}
        <ConversationDetailDialog
          isOpen={showDetailDialog}
          onClose={() => {
            setShowDetailDialog(false);
            setSelectedConversation(null);
          }}
          conversation={selectedConversation}
        />
      </div>
    </DashboardLayout>
  );
}
