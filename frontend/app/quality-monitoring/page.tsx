"use client";

import React, { useState, useMemo } from "react";
import DashboardLayout from "../components/DashboardLayout";
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Award,
  BarChart3,
  Bot,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  Clock,
  Download,
  Eye,
  FileText,
  Filter,
  Flag,
  Gauge,
  Headphones,
  Heart,
  Info,
  Lightbulb,
  MessageSquare,
  Mic,
  MoreVertical,
  Phone,
  Play,
  RefreshCw,
  Search,
  Settings,
  Sparkles,
  Star,
  Target,
  ThumbsDown,
  ThumbsUp,
  TrendingDown,
  TrendingUp,
  User,
  Users,
  Volume2,
  X,
  XCircle,
  Zap,
} from "lucide-react";

// Types
type ScoreRating = "excellent" | "good" | "fair" | "poor" | "critical";
type ReviewStatus = "pending" | "in_review" | "reviewed" | "flagged" | "escalated";
type QualityCategory = "accuracy" | "tone" | "compliance" | "resolution" | "efficiency" | "empathy";

interface QualityScore {
  overall: number;
  categories: {
    name: QualityCategory;
    score: number;
    trend: number;
    weight: number;
  }[];
}

interface CallReview {
  id: string;
  callId: string;
  agentId: string;
  agentName: string;
  agentType: "ai" | "human";
  customerId: string;
  customerName: string;
  timestamp: Date;
  duration: number;
  status: ReviewStatus;
  qualityScore: QualityScore;
  sentiment: {
    customer: number;
    agent: number;
    overall: number;
  };
  compliance: {
    passed: boolean;
    violations: string[];
  };
  highlights: {
    type: "positive" | "negative" | "neutral";
    timestamp: number;
    text: string;
    category: QualityCategory;
  }[];
  feedback?: {
    reviewer: string;
    rating: number;
    comments: string;
    actionItems: string[];
    reviewedAt: Date;
  };
  tags: string[];
  flagReason?: string;
}

interface QualityMetrics {
  totalReviews: number;
  avgScore: number;
  scoreTrend: number;
  complianceRate: number;
  flaggedCalls: number;
  reviewedToday: number;
  pendingReviews: number;
  topPerformers: { agentId: string; agentName: string; score: number }[];
  needsImprovement: { agentId: string; agentName: string; score: number }[];
  categoryScores: { name: QualityCategory; score: number; trend: number }[];
  scoreDistribution: { range: string; count: number; percentage: number }[];
}

// Mock data
const mockMetrics: QualityMetrics = {
  totalReviews: 2847,
  avgScore: 87.5,
  scoreTrend: 3.2,
  complianceRate: 96.8,
  flaggedCalls: 23,
  reviewedToday: 142,
  pendingReviews: 89,
  topPerformers: [
    { agentId: "agent-1", agentName: "Sales AI", score: 94.2 },
    { agentId: "agent-2", agentName: "Support AI", score: 92.8 },
    { agentId: "agent-3", agentName: "Onboarding AI", score: 91.5 },
  ],
  needsImprovement: [
    { agentId: "agent-4", agentName: "Collections AI", score: 72.3 },
    { agentId: "agent-5", agentName: "Outreach AI", score: 75.1 },
  ],
  categoryScores: [
    { name: "accuracy", score: 92.1, trend: 2.5 },
    { name: "tone", score: 88.4, trend: 1.8 },
    { name: "compliance", score: 96.8, trend: 0.5 },
    { name: "resolution", score: 84.2, trend: 4.1 },
    { name: "efficiency", score: 86.5, trend: -1.2 },
    { name: "empathy", score: 82.3, trend: 3.4 },
  ],
  scoreDistribution: [
    { range: "90-100", count: 1245, percentage: 43.7 },
    { range: "80-89", count: 892, percentage: 31.3 },
    { range: "70-79", count: 456, percentage: 16.0 },
    { range: "60-69", count: 178, percentage: 6.3 },
    { range: "0-59", count: 76, percentage: 2.7 },
  ],
};

const mockReviews: CallReview[] = [
  {
    id: "review-1",
    callId: "call-12345",
    agentId: "agent-1",
    agentName: "Sales AI",
    agentType: "ai",
    customerId: "cust-1",
    customerName: "Sarah Johnson",
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
    duration: 8.5,
    status: "reviewed",
    qualityScore: {
      overall: 94,
      categories: [
        { name: "accuracy", score: 96, trend: 2, weight: 25 },
        { name: "tone", score: 92, trend: 1, weight: 20 },
        { name: "compliance", score: 100, trend: 0, weight: 20 },
        { name: "resolution", score: 90, trend: 5, weight: 15 },
        { name: "efficiency", score: 92, trend: -1, weight: 10 },
        { name: "empathy", score: 88, trend: 3, weight: 10 },
      ],
    },
    sentiment: { customer: 0.85, agent: 0.90, overall: 0.87 },
    compliance: { passed: true, violations: [] },
    highlights: [
      { type: "positive", timestamp: 45, text: "Excellent product explanation with relevant examples", category: "accuracy" },
      { type: "positive", timestamp: 180, text: "Proactive upsell opportunity identified", category: "efficiency" },
      { type: "neutral", timestamp: 320, text: "Customer asked for pricing clarification", category: "resolution" },
    ],
    feedback: {
      reviewer: "Quality Team",
      rating: 5,
      comments: "Excellent call handling with strong product knowledge",
      actionItems: [],
      reviewedAt: new Date(Date.now() - 1 * 60 * 60 * 1000),
    },
    tags: ["excellent", "sales", "upsell"],
  },
  {
    id: "review-2",
    callId: "call-12346",
    agentId: "agent-2",
    agentName: "Support AI",
    agentType: "ai",
    customerId: "cust-2",
    customerName: "Michael Chen",
    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
    duration: 12.3,
    status: "flagged",
    qualityScore: {
      overall: 68,
      categories: [
        { name: "accuracy", score: 72, trend: -5, weight: 25 },
        { name: "tone", score: 78, trend: -2, weight: 20 },
        { name: "compliance", score: 85, trend: -10, weight: 20 },
        { name: "resolution", score: 55, trend: -8, weight: 15 },
        { name: "efficiency", score: 62, trend: -5, weight: 10 },
        { name: "empathy", score: 70, trend: 0, weight: 10 },
      ],
    },
    sentiment: { customer: 0.35, agent: 0.72, overall: 0.48 },
    compliance: { passed: false, violations: ["Missing disclosure statement", "Incorrect pricing quoted"] },
    highlights: [
      { type: "negative", timestamp: 120, text: "Provided incorrect pricing information", category: "accuracy" },
      { type: "negative", timestamp: 280, text: "Failed to acknowledge customer frustration", category: "empathy" },
      { type: "negative", timestamp: 450, text: "Issue not resolved - customer still dissatisfied", category: "resolution" },
    ],
    flagReason: "Multiple compliance violations and unresolved customer issue",
    tags: ["flagged", "compliance", "needs-review"],
  },
  {
    id: "review-3",
    callId: "call-12347",
    agentId: "agent-3",
    agentName: "Onboarding AI",
    agentType: "ai",
    customerId: "cust-3",
    customerName: "Emily Davis",
    timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000),
    duration: 15.7,
    status: "pending",
    qualityScore: {
      overall: 86,
      categories: [
        { name: "accuracy", score: 90, trend: 3, weight: 25 },
        { name: "tone", score: 88, trend: 2, weight: 20 },
        { name: "compliance", score: 100, trend: 0, weight: 20 },
        { name: "resolution", score: 82, trend: 1, weight: 15 },
        { name: "efficiency", score: 78, trend: -3, weight: 10 },
        { name: "empathy", score: 85, trend: 4, weight: 10 },
      ],
    },
    sentiment: { customer: 0.75, agent: 0.82, overall: 0.78 },
    compliance: { passed: true, violations: [] },
    highlights: [
      { type: "positive", timestamp: 90, text: "Clear step-by-step guidance provided", category: "accuracy" },
      { type: "neutral", timestamp: 420, text: "Customer needed additional clarification on features", category: "resolution" },
      { type: "positive", timestamp: 780, text: "Successfully completed onboarding flow", category: "resolution" },
    ],
    tags: ["onboarding", "new-customer"],
  },
  {
    id: "review-4",
    callId: "call-12348",
    agentId: "agent-1",
    agentName: "Sales AI",
    agentType: "ai",
    customerId: "cust-4",
    customerName: "James Wilson",
    timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000),
    duration: 6.2,
    status: "in_review",
    qualityScore: {
      overall: 91,
      categories: [
        { name: "accuracy", score: 94, trend: 1, weight: 25 },
        { name: "tone", score: 90, trend: 2, weight: 20 },
        { name: "compliance", score: 100, trend: 0, weight: 20 },
        { name: "resolution", score: 88, trend: 3, weight: 15 },
        { name: "efficiency", score: 95, trend: 5, weight: 10 },
        { name: "empathy", score: 82, trend: 1, weight: 10 },
      ],
    },
    sentiment: { customer: 0.82, agent: 0.88, overall: 0.85 },
    compliance: { passed: true, violations: [] },
    highlights: [
      { type: "positive", timestamp: 30, text: "Quick identification of customer needs", category: "efficiency" },
      { type: "positive", timestamp: 180, text: "Effective objection handling", category: "tone" },
    ],
    tags: ["sales", "renewal"],
  },
];

// Helper functions
const getScoreColor = (score: number): string => {
  if (score >= 90) return "text-green-400";
  if (score >= 80) return "text-blue-400";
  if (score >= 70) return "text-yellow-400";
  if (score >= 60) return "text-orange-400";
  return "text-red-400";
};

const getScoreBgColor = (score: number): string => {
  if (score >= 90) return "bg-green-500/20";
  if (score >= 80) return "bg-blue-500/20";
  if (score >= 70) return "bg-yellow-500/20";
  if (score >= 60) return "bg-orange-500/20";
  return "bg-red-500/20";
};

const getScoreRating = (score: number): ScoreRating => {
  if (score >= 90) return "excellent";
  if (score >= 80) return "good";
  if (score >= 70) return "fair";
  if (score >= 60) return "poor";
  return "critical";
};

const getStatusColor = (status: ReviewStatus) => {
  switch (status) {
    case "pending":
      return "text-gray-400 bg-gray-500/20";
    case "in_review":
      return "text-blue-400 bg-blue-500/20";
    case "reviewed":
      return "text-green-400 bg-green-500/20";
    case "flagged":
      return "text-red-400 bg-red-500/20";
    case "escalated":
      return "text-orange-400 bg-orange-500/20";
  }
};

const getCategoryIcon = (category: QualityCategory) => {
  switch (category) {
    case "accuracy":
      return <Target className="w-4 h-4" />;
    case "tone":
      return <MessageSquare className="w-4 h-4" />;
    case "compliance":
      return <CheckCircle className="w-4 h-4" />;
    case "resolution":
      return <Zap className="w-4 h-4" />;
    case "efficiency":
      return <Clock className="w-4 h-4" />;
    case "empathy":
      return <Heart className="w-4 h-4" />;
  }
};

const formatDuration = (minutes: number) => {
  return `${Math.floor(minutes)}:${String(Math.round((minutes % 1) * 60)).padStart(2, "0")}`;
};

const formatRelativeTime = (date: Date) => {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const hours = Math.floor(diff / (1000 * 60 * 60));
  if (hours < 1) return "Just now";
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
};

// Score Gauge Component
const ScoreGauge: React.FC<{ score: number; size?: "sm" | "md" | "lg" }> = ({ score, size = "md" }) => {
  const sizeClasses = {
    sm: "w-16 h-16 text-lg",
    md: "w-24 h-24 text-2xl",
    lg: "w-32 h-32 text-3xl",
  };

  const strokeWidth = size === "sm" ? 4 : size === "md" ? 6 : 8;
  const radius = size === "sm" ? 28 : size === "md" ? 42 : 56;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className={`relative ${sizeClasses[size]}`}>
      <svg className="w-full h-full -rotate-90">
        <circle
          cx="50%"
          cy="50%"
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={strokeWidth}
        />
        <circle
          cx="50%"
          cy="50%"
          r={radius}
          fill="none"
          stroke={score >= 90 ? "#22c55e" : score >= 80 ? "#3b82f6" : score >= 70 ? "#eab308" : score >= 60 ? "#f97316" : "#ef4444"}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-1000"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className={`font-bold ${getScoreColor(score)}`}>{score}</span>
      </div>
    </div>
  );
};

// Review Card Component
const ReviewCard: React.FC<{
  review: CallReview;
  onClick: () => void;
  onAction: (action: string) => void;
}> = ({ review, onClick, onAction }) => {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <div
      onClick={onClick}
      className={`bg-[#1a1a2e]/80 backdrop-blur-xl border rounded-xl p-4 hover:border-purple-500/30 transition-all duration-300 cursor-pointer ${
        review.status === "flagged" ? "border-red-500/30" : "border-white/10"
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-4">
          <ScoreGauge score={review.qualityScore.overall} size="sm" />

          <div>
            <div className="flex items-center gap-2">
              <h4 className="font-semibold text-white">{review.customerName}</h4>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(review.status)}`}>
                {review.status.replace("_", " ")}
              </span>
            </div>

            <div className="flex items-center gap-3 mt-1 text-sm text-gray-400">
              <span className="flex items-center gap-1">
                {review.agentType === "ai" ? <Bot className="w-3.5 h-3.5" /> : <User className="w-3.5 h-3.5" />}
                {review.agentName}
              </span>
              <span className="flex items-center gap-1">
                <Clock className="w-3.5 h-3.5" />
                {formatDuration(review.duration)}
              </span>
              <span>{formatRelativeTime(review.timestamp)}</span>
            </div>

            {!review.compliance.passed && (
              <div className="mt-2 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400" />
                <span className="text-sm text-red-400">
                  {review.compliance.violations.length} compliance violation(s)
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="relative">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowMenu(!showMenu);
            }}
            className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
          >
            <MoreVertical className="w-4 h-4" />
          </button>

          {showMenu && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setShowMenu(false)} />
              <div className="absolute right-0 top-full mt-1 w-44 bg-[#252540] border border-white/10 rounded-lg shadow-xl z-20 py-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onAction("review");
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                >
                  <Eye className="w-4 h-4" />
                  View Details
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onAction("play");
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Play Recording
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onAction("flag");
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                >
                  <Flag className="w-4 h-4" />
                  Flag for Review
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Category Scores */}
      <div className="mt-4 grid grid-cols-6 gap-2">
        {review.qualityScore.categories.map((cat) => (
          <div key={cat.name} className="text-center">
            <div className={`mx-auto w-8 h-8 rounded-lg flex items-center justify-center ${getScoreBgColor(cat.score)} ${getScoreColor(cat.score)}`}>
              {getCategoryIcon(cat.name)}
            </div>
            <p className="text-xs text-gray-500 mt-1 capitalize">{cat.name.slice(0, 3)}</p>
            <p className={`text-xs font-medium ${getScoreColor(cat.score)}`}>{cat.score}</p>
          </div>
        ))}
      </div>

      {/* Highlights Preview */}
      {review.highlights.length > 0 && (
        <div className="mt-4 pt-4 border-t border-white/5">
          <div className="flex items-center gap-2">
            {review.highlights.slice(0, 3).map((highlight, i) => (
              <span
                key={i}
                className={`px-2 py-1 rounded text-xs ${
                  highlight.type === "positive"
                    ? "bg-green-500/20 text-green-400"
                    : highlight.type === "negative"
                    ? "bg-red-500/20 text-red-400"
                    : "bg-gray-500/20 text-gray-400"
                }`}
              >
                {highlight.type === "positive" ? <ThumbsUp className="w-3 h-3 inline mr-1" /> :
                 highlight.type === "negative" ? <ThumbsDown className="w-3 h-3 inline mr-1" /> : null}
                {highlight.category}
              </span>
            ))}
            {review.highlights.length > 3 && (
              <span className="text-xs text-gray-500">+{review.highlights.length - 3} more</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Main Component
export default function QualityMonitoringPage() {
  const [metrics] = useState<QualityMetrics>(mockMetrics);
  const [reviews] = useState<CallReview[]>(mockReviews);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<ReviewStatus | "all">("all");
  const [scoreFilter, setScoreFilter] = useState<string>("all");
  const [activeView, setActiveView] = useState<"overview" | "reviews" | "agents">("overview");

  const filteredReviews = useMemo(() => {
    return reviews.filter((review) => {
      const matchesSearch =
        review.customerName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        review.agentName.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesStatus = statusFilter === "all" || review.status === statusFilter;

      const matchesScore =
        scoreFilter === "all" ||
        (scoreFilter === "high" && review.qualityScore.overall >= 90) ||
        (scoreFilter === "medium" && review.qualityScore.overall >= 70 && review.qualityScore.overall < 90) ||
        (scoreFilter === "low" && review.qualityScore.overall < 70);

      return matchesSearch && matchesStatus && matchesScore;
    });
  }, [reviews, searchQuery, statusFilter, scoreFilter]);

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Quality Monitoring</h1>
            <p className="text-gray-400 mt-1">Monitor and improve AI agent call quality</p>
          </div>

          <div className="flex items-center gap-3">
            <button className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors">
              <Download className="w-4 h-4" />
              Export Report
            </button>
            <button className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white transition-colors">
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* View Tabs */}
        <div className="flex gap-2 p-1 bg-[#1a1a2e]/80 rounded-lg w-fit">
          {(["overview", "reviews", "agents"] as const).map((view) => (
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
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <Gauge className="w-5 h-5 text-purple-400" />
                  <span className={`flex items-center gap-1 text-sm ${metrics.scoreTrend >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {metrics.scoreTrend >= 0 ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
                    {Math.abs(metrics.scoreTrend)}%
                  </span>
                </div>
                <p className={`text-2xl font-bold mt-2 ${getScoreColor(metrics.avgScore)}`}>{metrics.avgScore}</p>
                <p className="text-sm text-gray-400">Avg Score</p>
              </div>

              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <Activity className="w-5 h-5 text-blue-400" />
                <p className="text-2xl font-bold text-white mt-2">{metrics.totalReviews.toLocaleString()}</p>
                <p className="text-sm text-gray-400">Total Reviews</p>
              </div>

              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <p className="text-2xl font-bold text-green-400 mt-2">{metrics.complianceRate}%</p>
                <p className="text-sm text-gray-400">Compliance</p>
              </div>

              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <Flag className="w-5 h-5 text-red-400" />
                <p className="text-2xl font-bold text-red-400 mt-2">{metrics.flaggedCalls}</p>
                <p className="text-sm text-gray-400">Flagged</p>
              </div>

              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <Eye className="w-5 h-5 text-yellow-400" />
                <p className="text-2xl font-bold text-white mt-2">{metrics.reviewedToday}</p>
                <p className="text-sm text-gray-400">Today</p>
              </div>

              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <Clock className="w-5 h-5 text-orange-400" />
                <p className="text-2xl font-bold text-orange-400 mt-2">{metrics.pendingReviews}</p>
                <p className="text-sm text-gray-400">Pending</p>
              </div>

              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <Award className="w-5 h-5 text-yellow-400" />
                <p className="text-2xl font-bold text-yellow-400 mt-2">{metrics.topPerformers[0]?.score || 0}</p>
                <p className="text-sm text-gray-400">Top Score</p>
              </div>
            </div>

            {/* Category Scores & Distribution */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Category Scores */}
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Category Performance</h3>
                <div className="space-y-4">
                  {metrics.categoryScores.map((cat) => (
                    <div key={cat.name} className="flex items-center gap-4">
                      <div className={`p-2 rounded-lg ${getScoreBgColor(cat.score)} ${getScoreColor(cat.score)}`}>
                        {getCategoryIcon(cat.name)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-white capitalize">{cat.name}</span>
                          <div className="flex items-center gap-2">
                            <span className={`font-medium ${getScoreColor(cat.score)}`}>{cat.score}%</span>
                            <span className={`text-xs flex items-center gap-1 ${cat.trend >= 0 ? "text-green-400" : "text-red-400"}`}>
                              {cat.trend >= 0 ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
                              {Math.abs(cat.trend)}%
                            </span>
                          </div>
                        </div>
                        <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              cat.score >= 90 ? "bg-green-500" :
                              cat.score >= 80 ? "bg-blue-500" :
                              cat.score >= 70 ? "bg-yellow-500" :
                              cat.score >= 60 ? "bg-orange-500" : "bg-red-500"
                            }`}
                            style={{ width: `${cat.score}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Score Distribution */}
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Score Distribution</h3>
                <div className="space-y-3">
                  {metrics.scoreDistribution.map((bucket) => (
                    <div key={bucket.range} className="flex items-center gap-3">
                      <span className="text-sm text-gray-400 w-16">{bucket.range}</span>
                      <div className="flex-1 h-8 bg-white/5 rounded-lg overflow-hidden">
                        <div
                          className={`h-full rounded-lg transition-all duration-500 flex items-center px-2 ${
                            bucket.range.startsWith("90") ? "bg-green-500/50" :
                            bucket.range.startsWith("80") ? "bg-blue-500/50" :
                            bucket.range.startsWith("70") ? "bg-yellow-500/50" :
                            bucket.range.startsWith("60") ? "bg-orange-500/50" : "bg-red-500/50"
                          }`}
                          style={{ width: `${bucket.percentage}%` }}
                        >
                          <span className="text-xs text-white font-medium">{bucket.count}</span>
                        </div>
                      </div>
                      <span className="text-sm text-gray-400 w-12 text-right">{bucket.percentage}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Top & Bottom Performers */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <Award className="w-5 h-5 text-yellow-400" />
                  Top Performers
                </h3>
                <div className="space-y-3">
                  {metrics.topPerformers.map((agent, i) => (
                    <div key={agent.agentId} className="flex items-center justify-between p-3 bg-green-500/10 rounded-lg">
                      <div className="flex items-center gap-3">
                        <span className="w-6 h-6 rounded-full bg-yellow-500/20 text-yellow-400 flex items-center justify-center text-sm font-bold">
                          {i + 1}
                        </span>
                        <span className="text-white">{agent.agentName}</span>
                      </div>
                      <span className="text-green-400 font-bold">{agent.score}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-orange-400" />
                  Needs Improvement
                </h3>
                <div className="space-y-3">
                  {metrics.needsImprovement.map((agent) => (
                    <div key={agent.agentId} className="flex items-center justify-between p-3 bg-orange-500/10 rounded-lg">
                      <span className="text-white">{agent.agentName}</span>
                      <span className="text-orange-400 font-bold">{agent.score}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        {activeView === "reviews" && (
          <>
            {/* Filters */}
            <div className="flex flex-col md:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search by customer or agent..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div className="flex gap-2">
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value as ReviewStatus | "all")}
                  className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Statuses</option>
                  <option value="pending">Pending</option>
                  <option value="in_review">In Review</option>
                  <option value="reviewed">Reviewed</option>
                  <option value="flagged">Flagged</option>
                </select>
                <select
                  value={scoreFilter}
                  onChange={(e) => setScoreFilter(e.target.value)}
                  className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Scores</option>
                  <option value="high">High (90+)</option>
                  <option value="medium">Medium (70-89)</option>
                  <option value="low">Low (&lt;70)</option>
                </select>
              </div>
            </div>

            {/* Reviews List */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {filteredReviews.map((review) => (
                <ReviewCard
                  key={review.id}
                  review={review}
                  onClick={() => console.log("View review", review.id)}
                  onAction={(action) => console.log(action, review.id)}
                />
              ))}
            </div>
          </>
        )}

        {activeView === "agents" && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[...metrics.topPerformers, ...metrics.needsImprovement].map((agent) => (
              <div
                key={agent.agentId}
                className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6 hover:border-purple-500/30 transition-all"
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-3 rounded-xl bg-purple-500/20">
                      <Bot className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white">{agent.agentName}</h3>
                      <p className="text-sm text-gray-400">AI Agent</p>
                    </div>
                  </div>
                  <ScoreGauge score={agent.score} size="sm" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Rating</span>
                    <span className={`capitalize ${getScoreColor(agent.score)}`}>
                      {getScoreRating(agent.score)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Reviews</span>
                    <span className="text-white">{Math.floor(Math.random() * 500) + 100}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
