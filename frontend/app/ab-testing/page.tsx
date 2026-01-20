"use client";

import React, { useState, useEffect, useMemo } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import {
  Beaker,
  FlaskConical,
  Split,
  GitBranch,
  Play,
  Pause,
  StopCircle,
  Plus,
  Search,
  Filter,
  MoreVertical,
  Edit,
  Trash2,
  Copy,
  Eye,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Activity,
  TrendingUp,
  TrendingDown,
  Target,
  Users,
  BarChart3,
  PieChart,
  ArrowUp,
  ArrowDown,
  ArrowRight,
  Percent,
  Award,
  Zap,
  Settings,
  RefreshCw,
  Download,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Info,
  X,
  Check,
  Sparkles,
  Crown,
  Medal,
  Trophy,
  Gauge,
  Hash,
  Calendar,
  Globe,
  Layers,
  FileText,
  MessageSquare,
  Phone,
  Bot,
  Mic,
  Volume2,
  Sliders,
  ToggleLeft,
  ToggleRight,
  Scale,
  Lightbulb,
  CircleDot,
  GitCompare,
} from "lucide-react";

// Types
type ExperimentStatus = "draft" | "running" | "paused" | "completed" | "stopped";
type ExperimentType = "agent" | "prompt" | "voice" | "routing" | "timing" | "script";
type MetricType = "conversion" | "satisfaction" | "duration" | "resolution" | "engagement" | "cost";

interface Variant {
  id: string;
  name: string;
  description: string;
  trafficPercentage: number;
  isControl: boolean;
  config: Record<string, any>;
  metrics: {
    participants: number;
    conversions: number;
    conversionRate: number;
    avgSatisfaction: number;
    avgDuration: number;
    cost: number;
  };
}

interface Experiment {
  id: string;
  name: string;
  description: string;
  type: ExperimentType;
  status: ExperimentStatus;
  hypothesis: string;
  primaryMetric: MetricType;
  secondaryMetrics: MetricType[];
  variants: Variant[];
  targetAudience: {
    type: "all" | "percentage" | "segment";
    percentage?: number;
    segmentId?: string;
  };
  schedule: {
    startDate: Date;
    endDate?: Date;
    autoStop: boolean;
    sampleSize?: number;
    statisticalSignificance: number;
  };
  results?: {
    winner?: string;
    confidence: number;
    uplift: number;
    isSignificant: boolean;
  };
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  tags: string[];
}

// Mock data
const mockExperiments: Experiment[] = [
  {
    id: "exp-1",
    name: "Greeting Script Optimization",
    description: "Testing different greeting approaches to improve customer engagement",
    type: "script",
    status: "running",
    hypothesis: "A more personalized greeting will increase customer satisfaction by 15%",
    primaryMetric: "satisfaction",
    secondaryMetrics: ["conversion", "duration"],
    variants: [
      {
        id: "var-1a",
        name: "Control - Standard Greeting",
        description: "Current standard greeting script",
        trafficPercentage: 50,
        isControl: true,
        config: { greeting: "Hello, thank you for calling. How can I assist you today?" },
        metrics: {
          participants: 2450,
          conversions: 1225,
          conversionRate: 50.0,
          avgSatisfaction: 4.2,
          avgDuration: 5.8,
          cost: 1225.00,
        },
      },
      {
        id: "var-1b",
        name: "Personalized Greeting",
        description: "Personalized greeting using customer name",
        trafficPercentage: 50,
        isControl: false,
        config: { greeting: "Hi {{customer_name}}! Great to hear from you. What can I help with today?" },
        metrics: {
          participants: 2480,
          conversions: 1364,
          conversionRate: 55.0,
          avgSatisfaction: 4.5,
          avgDuration: 5.4,
          cost: 1240.00,
        },
      },
    ],
    targetAudience: { type: "all" },
    schedule: {
      startDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      autoStop: true,
      sampleSize: 10000,
      statisticalSignificance: 95,
    },
    results: {
      winner: "var-1b",
      confidence: 94.2,
      uplift: 10.0,
      isSignificant: false,
    },
    createdAt: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
    createdBy: "John Smith",
    tags: ["greeting", "customer-experience", "priority"],
  },
  {
    id: "exp-2",
    name: "Voice Model Comparison",
    description: "Testing different AI voice models for customer preference",
    type: "voice",
    status: "running",
    hypothesis: "Natural-sounding voice will improve customer trust and call duration",
    primaryMetric: "satisfaction",
    secondaryMetrics: ["duration", "engagement"],
    variants: [
      {
        id: "var-2a",
        name: "Control - Standard Voice",
        description: "Current ElevenLabs voice model",
        trafficPercentage: 33,
        isControl: true,
        config: { voiceModel: "eleven-v1", voiceId: "default" },
        metrics: {
          participants: 1820,
          conversions: 1001,
          conversionRate: 55.0,
          avgSatisfaction: 4.0,
          avgDuration: 6.2,
          cost: 910.00,
        },
      },
      {
        id: "var-2b",
        name: "Natural Voice A",
        description: "New natural voice model - Female",
        trafficPercentage: 33,
        isControl: false,
        config: { voiceModel: "eleven-v2", voiceId: "natural-female" },
        metrics: {
          participants: 1845,
          conversions: 1125,
          conversionRate: 61.0,
          avgSatisfaction: 4.3,
          avgDuration: 6.8,
          cost: 922.50,
        },
      },
      {
        id: "var-2c",
        name: "Natural Voice B",
        description: "New natural voice model - Male",
        trafficPercentage: 34,
        isControl: false,
        config: { voiceModel: "eleven-v2", voiceId: "natural-male" },
        metrics: {
          participants: 1890,
          conversions: 1115,
          conversionRate: 59.0,
          avgSatisfaction: 4.2,
          avgDuration: 6.5,
          cost: 945.00,
        },
      },
    ],
    targetAudience: { type: "all" },
    schedule: {
      startDate: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
      autoStop: true,
      sampleSize: 15000,
      statisticalSignificance: 95,
    },
    results: {
      winner: "var-2b",
      confidence: 87.5,
      uplift: 10.9,
      isSignificant: false,
    },
    createdAt: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
    createdBy: "Sarah Johnson",
    tags: ["voice", "ai-model", "customer-experience"],
  },
  {
    id: "exp-3",
    name: "Call Routing Strategy",
    description: "Testing AI-based vs rule-based call routing",
    type: "routing",
    status: "completed",
    hypothesis: "AI-based routing will reduce call transfer rate by 20%",
    primaryMetric: "resolution",
    secondaryMetrics: ["satisfaction", "cost"],
    variants: [
      {
        id: "var-3a",
        name: "Control - Rule-based",
        description: "Current rule-based routing system",
        trafficPercentage: 50,
        isControl: true,
        config: { routingType: "rule-based", rules: ["department", "priority"] },
        metrics: {
          participants: 8420,
          conversions: 6736,
          conversionRate: 80.0,
          avgSatisfaction: 3.9,
          avgDuration: 7.2,
          cost: 4210.00,
        },
      },
      {
        id: "var-3b",
        name: "AI-based Routing",
        description: "ML-powered intelligent routing",
        trafficPercentage: 50,
        isControl: false,
        config: { routingType: "ai", model: "routing-ml-v2" },
        metrics: {
          participants: 8380,
          conversions: 7542,
          conversionRate: 90.0,
          avgSatisfaction: 4.4,
          avgDuration: 5.8,
          cost: 4190.00,
        },
      },
    ],
    targetAudience: { type: "all" },
    schedule: {
      startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      endDate: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
      autoStop: true,
      sampleSize: 15000,
      statisticalSignificance: 95,
    },
    results: {
      winner: "var-3b",
      confidence: 99.2,
      uplift: 12.5,
      isSignificant: true,
    },
    createdAt: new Date(Date.now() - 35 * 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
    createdBy: "Michael Chen",
    tags: ["routing", "ai", "automation"],
  },
  {
    id: "exp-4",
    name: "Agent Persona Test",
    description: "Testing different AI agent personalities",
    type: "agent",
    status: "paused",
    hypothesis: "A friendly, casual persona will increase engagement for younger demographics",
    primaryMetric: "engagement",
    secondaryMetrics: ["satisfaction", "conversion"],
    variants: [
      {
        id: "var-4a",
        name: "Control - Professional",
        description: "Professional and formal agent persona",
        trafficPercentage: 50,
        isControl: true,
        config: { persona: "professional", tone: "formal" },
        metrics: {
          participants: 1250,
          conversions: 750,
          conversionRate: 60.0,
          avgSatisfaction: 4.1,
          avgDuration: 5.5,
          cost: 625.00,
        },
      },
      {
        id: "var-4b",
        name: "Friendly Casual",
        description: "Friendly and casual agent persona",
        trafficPercentage: 50,
        isControl: false,
        config: { persona: "friendly", tone: "casual" },
        metrics: {
          participants: 1280,
          conversions: 832,
          conversionRate: 65.0,
          avgSatisfaction: 4.3,
          avgDuration: 6.1,
          cost: 640.00,
        },
      },
    ],
    targetAudience: { type: "segment", segmentId: "age-18-35" },
    schedule: {
      startDate: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
      autoStop: true,
      sampleSize: 5000,
      statisticalSignificance: 90,
    },
    createdAt: new Date(Date.now() - 20 * 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
    createdBy: "Emily Davis",
    tags: ["persona", "demographics", "engagement"],
  },
  {
    id: "exp-5",
    name: "Prompt Engineering Test",
    description: "Testing different system prompts for better responses",
    type: "prompt",
    status: "draft",
    hypothesis: "Detailed prompts with examples will reduce hallucinations by 30%",
    primaryMetric: "resolution",
    secondaryMetrics: ["satisfaction"],
    variants: [
      {
        id: "var-5a",
        name: "Control - Basic Prompt",
        description: "Current basic system prompt",
        trafficPercentage: 50,
        isControl: true,
        config: { promptTemplate: "basic-v1" },
        metrics: {
          participants: 0,
          conversions: 0,
          conversionRate: 0,
          avgSatisfaction: 0,
          avgDuration: 0,
          cost: 0,
        },
      },
      {
        id: "var-5b",
        name: "Enhanced Prompt",
        description: "Detailed prompt with examples and guardrails",
        trafficPercentage: 50,
        isControl: false,
        config: { promptTemplate: "enhanced-v1", includeExamples: true },
        metrics: {
          participants: 0,
          conversions: 0,
          conversionRate: 0,
          avgSatisfaction: 0,
          avgDuration: 0,
          cost: 0,
        },
      },
    ],
    targetAudience: { type: "percentage", percentage: 20 },
    schedule: {
      startDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
      autoStop: true,
      sampleSize: 5000,
      statisticalSignificance: 95,
    },
    createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
    createdBy: "David Wilson",
    tags: ["prompt", "ai", "quality"],
  },
];

// Helper functions
const getStatusColor = (status: ExperimentStatus) => {
  switch (status) {
    case "draft":
      return "text-gray-400 bg-gray-500/20";
    case "running":
      return "text-green-400 bg-green-500/20";
    case "paused":
      return "text-yellow-400 bg-yellow-500/20";
    case "completed":
      return "text-blue-400 bg-blue-500/20";
    case "stopped":
      return "text-red-400 bg-red-500/20";
  }
};

const getStatusIcon = (status: ExperimentStatus) => {
  switch (status) {
    case "draft":
      return <FileText className="w-4 h-4" />;
    case "running":
      return <Play className="w-4 h-4" />;
    case "paused":
      return <Pause className="w-4 h-4" />;
    case "completed":
      return <CheckCircle className="w-4 h-4" />;
    case "stopped":
      return <StopCircle className="w-4 h-4" />;
  }
};

const getTypeIcon = (type: ExperimentType) => {
  switch (type) {
    case "agent":
      return <Bot className="w-4 h-4" />;
    case "prompt":
      return <MessageSquare className="w-4 h-4" />;
    case "voice":
      return <Mic className="w-4 h-4" />;
    case "routing":
      return <GitBranch className="w-4 h-4" />;
    case "timing":
      return <Clock className="w-4 h-4" />;
    case "script":
      return <FileText className="w-4 h-4" />;
  }
};

const getTypeColor = (type: ExperimentType) => {
  switch (type) {
    case "agent":
      return "text-purple-400 bg-purple-500/20";
    case "prompt":
      return "text-blue-400 bg-blue-500/20";
    case "voice":
      return "text-pink-400 bg-pink-500/20";
    case "routing":
      return "text-green-400 bg-green-500/20";
    case "timing":
      return "text-yellow-400 bg-yellow-500/20";
    case "script":
      return "text-orange-400 bg-orange-500/20";
  }
};

const formatNumber = (num: number) => {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
};

const formatRelativeTime = (date: Date) => {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
  return date.toLocaleDateString();
};

// Experiment Card Component
const ExperimentCard: React.FC<{
  experiment: Experiment;
  onView: () => void;
  onEdit: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onStatusChange: (status: ExperimentStatus) => void;
}> = ({ experiment, onView, onEdit, onDuplicate, onDelete, onStatusChange }) => {
  const [showMenu, setShowMenu] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const totalParticipants = experiment.variants.reduce((sum, v) => sum + v.metrics.participants, 0);
  const controlVariant = experiment.variants.find((v) => v.isControl);
  const testVariants = experiment.variants.filter((v) => !v.isControl);

  return (
    <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden hover:border-purple-500/30 transition-all duration-300">
      {/* Header */}
      <div className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3 flex-1 min-w-0">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1 rounded hover:bg-white/10 transition-colors mt-0.5"
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4 text-gray-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-gray-400" />
              )}
            </button>

            <div className={`p-2 rounded-lg ${getTypeColor(experiment.type)}`}>
              {getTypeIcon(experiment.type)}
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-white truncate">{experiment.name}</h3>
                {experiment.results?.isSignificant && (
                  <span title="Statistically Significant">
                    <Trophy className="w-4 h-4 text-yellow-400" />
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-400 line-clamp-1 mt-0.5">{experiment.description}</p>

              <div className="flex items-center gap-2 mt-2 flex-wrap">
                <span className={`px-2 py-0.5 rounded-full text-xs font-medium flex items-center gap-1 ${getStatusColor(experiment.status)}`}>
                  {getStatusIcon(experiment.status)}
                  {experiment.status.charAt(0).toUpperCase() + experiment.status.slice(1)}
                </span>
                <span className="px-2 py-0.5 rounded text-xs bg-white/5 text-gray-400">
                  {experiment.variants.length} variants
                </span>
                <span className="px-2 py-0.5 rounded text-xs bg-white/5 text-gray-400">
                  {formatNumber(totalParticipants)} participants
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {experiment.status === "draft" && (
              <button
                onClick={() => onStatusChange("running")}
                className="p-2 rounded-lg bg-green-500/20 hover:bg-green-500/30 text-green-400 transition-colors"
                title="Start Experiment"
              >
                <Play className="w-4 h-4" />
              </button>
            )}
            {experiment.status === "running" && (
              <button
                onClick={() => onStatusChange("paused")}
                className="p-2 rounded-lg bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400 transition-colors"
                title="Pause Experiment"
              >
                <Pause className="w-4 h-4" />
              </button>
            )}
            {experiment.status === "paused" && (
              <button
                onClick={() => onStatusChange("running")}
                className="p-2 rounded-lg bg-green-500/20 hover:bg-green-500/30 text-green-400 transition-colors"
                title="Resume Experiment"
              >
                <Play className="w-4 h-4" />
              </button>
            )}

            <div className="relative">
              <button
                onClick={() => setShowMenu(!showMenu)}
                className="p-2 rounded-lg hover:bg-white/10 text-gray-400 transition-colors"
              >
                <MoreVertical className="w-4 h-4" />
              </button>

              {showMenu && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => setShowMenu(false)} />
                  <div className="absolute right-0 top-full mt-1 w-48 bg-[#252540] border border-white/10 rounded-lg shadow-xl z-20 py-1">
                    <button
                      onClick={() => {
                        onView();
                        setShowMenu(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                    >
                      <Eye className="w-4 h-4" />
                      View Details
                    </button>
                    <button
                      onClick={() => {
                        onEdit();
                        setShowMenu(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                    >
                      <Edit className="w-4 h-4" />
                      Edit
                    </button>
                    <button
                      onClick={() => {
                        onDuplicate();
                        setShowMenu(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                    >
                      <Copy className="w-4 h-4" />
                      Duplicate
                    </button>
                    {experiment.status === "running" && (
                      <button
                        onClick={() => {
                          onStatusChange("stopped");
                          setShowMenu(false);
                        }}
                        className="w-full px-4 py-2 text-left text-sm text-red-400 hover:bg-red-500/10 flex items-center gap-2"
                      >
                        <StopCircle className="w-4 h-4" />
                        Stop Experiment
                      </button>
                    )}
                    <hr className="my-1 border-white/10" />
                    <button
                      onClick={() => {
                        onDelete();
                        setShowMenu(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-red-400 hover:bg-red-500/10 flex items-center gap-2"
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

        {/* Progress & Results */}
        {experiment.status !== "draft" && (
          <div className="mt-4 pt-4 border-t border-white/10">
            {/* Progress Bar */}
            {experiment.schedule.sampleSize && (
              <div className="mb-4">
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-gray-400">Progress</span>
                  <span className="text-white">
                    {formatNumber(totalParticipants)} / {formatNumber(experiment.schedule.sampleSize)}
                  </span>
                </div>
                <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-500"
                    style={{ width: `${Math.min((totalParticipants / experiment.schedule.sampleSize) * 100, 100)}%` }}
                  />
                </div>
              </div>
            )}

            {/* Variant Comparison */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {experiment.variants.map((variant) => (
                <div
                  key={variant.id}
                  className={`p-3 rounded-lg ${
                    experiment.results?.winner === variant.id
                      ? "bg-green-500/10 border border-green-500/30"
                      : "bg-black/20"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400 truncate">{variant.name}</span>
                    {variant.isControl && (
                      <span className="px-1.5 py-0.5 text-xs bg-blue-500/20 text-blue-400 rounded">Control</span>
                    )}
                    {experiment.results?.winner === variant.id && (
                      <Crown className="w-4 h-4 text-yellow-400" />
                    )}
                  </div>
                  <p className="text-xl font-bold text-white">{variant.metrics.conversionRate.toFixed(1)}%</p>
                  <p className="text-xs text-gray-500">{formatNumber(variant.metrics.participants)} users</p>
                </div>
              ))}
            </div>

            {/* Results Summary */}
            {experiment.results && (
              <div className="mt-3 flex items-center gap-4">
                {experiment.results.winner && (
                  <div className="flex items-center gap-2">
                    <span className={`flex items-center gap-1 ${experiment.results.uplift >= 0 ? "text-green-400" : "text-red-400"}`}>
                      {experiment.results.uplift >= 0 ? (
                        <ArrowUp className="w-4 h-4" />
                      ) : (
                        <ArrowDown className="w-4 h-4" />
                      )}
                      {Math.abs(experiment.results.uplift).toFixed(1)}% uplift
                    </span>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <span className={`text-sm ${experiment.results.isSignificant ? "text-green-400" : "text-yellow-400"}`}>
                    {experiment.results.confidence.toFixed(1)}% confidence
                  </span>
                  {experiment.results.isSignificant ? (
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="border-t border-white/10 p-4 bg-black/20">
          {/* Hypothesis */}
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-gray-300 mb-2 flex items-center gap-2">
              <Lightbulb className="w-4 h-4 text-yellow-400" />
              Hypothesis
            </h4>
            <p className="text-sm text-gray-400">{experiment.hypothesis}</p>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <h4 className="text-sm font-semibold text-gray-300 mb-2">Primary Metric</h4>
              <span className="px-3 py-1.5 rounded-lg bg-purple-500/20 text-purple-400 text-sm capitalize">
                {experiment.primaryMetric}
              </span>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-300 mb-2">Secondary Metrics</h4>
              <div className="flex flex-wrap gap-2">
                {experiment.secondaryMetrics.map((metric) => (
                  <span
                    key={metric}
                    className="px-2 py-1 rounded bg-white/5 text-gray-400 text-xs capitalize"
                  >
                    {metric}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Variant Details */}
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-gray-300 mb-3">Variants</h4>
            <div className="space-y-3">
              {experiment.variants.map((variant) => (
                <div
                  key={variant.id}
                  className="p-3 bg-black/30 rounded-lg border border-white/5"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-white">{variant.name}</span>
                      {variant.isControl && (
                        <span className="px-1.5 py-0.5 text-xs bg-blue-500/20 text-blue-400 rounded">Control</span>
                      )}
                    </div>
                    <span className="text-sm text-purple-400">{variant.trafficPercentage}% traffic</span>
                  </div>
                  <p className="text-sm text-gray-400 mb-3">{variant.description}</p>
                  <div className="grid grid-cols-4 gap-2 text-center">
                    <div>
                      <p className="text-lg font-bold text-white">{variant.metrics.conversionRate.toFixed(1)}%</p>
                      <p className="text-xs text-gray-500">Conversion</p>
                    </div>
                    <div>
                      <p className="text-lg font-bold text-white">{variant.metrics.avgSatisfaction.toFixed(1)}</p>
                      <p className="text-xs text-gray-500">Satisfaction</p>
                    </div>
                    <div>
                      <p className="text-lg font-bold text-white">{variant.metrics.avgDuration.toFixed(1)}m</p>
                      <p className="text-xs text-gray-500">Avg Duration</p>
                    </div>
                    <div>
                      <p className="text-lg font-bold text-white">${variant.metrics.cost.toFixed(0)}</p>
                      <p className="text-xs text-gray-500">Cost</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Schedule & Metadata */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Started:</span>
              <span className="text-white ml-2">{formatRelativeTime(experiment.schedule.startDate)}</span>
            </div>
            {experiment.schedule.endDate && (
              <div>
                <span className="text-gray-500">Ended:</span>
                <span className="text-white ml-2">{formatRelativeTime(experiment.schedule.endDate)}</span>
              </div>
            )}
            <div>
              <span className="text-gray-500">Confidence Level:</span>
              <span className="text-white ml-2">{experiment.schedule.statisticalSignificance}%</span>
            </div>
            <div>
              <span className="text-gray-500">Created by:</span>
              <span className="text-white ml-2">{experiment.createdBy}</span>
            </div>
          </div>

          {/* Tags */}
          {experiment.tags.length > 0 && (
            <div className="mt-4 pt-4 border-t border-white/5 flex items-center gap-2 flex-wrap">
              {experiment.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-0.5 rounded-full text-xs bg-white/5 text-gray-400"
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

// Create Experiment Dialog
const CreateExperimentDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onSave: (experiment: Partial<Experiment>) => void;
}> = ({ isOpen, onClose, onSave }) => {
  const [step, setStep] = useState(1);
  const [experimentData, setExperimentData] = useState<Partial<Experiment>>({
    name: "",
    description: "",
    type: "script",
    hypothesis: "",
    primaryMetric: "conversion",
    secondaryMetrics: [],
    variants: [
      {
        id: "control",
        name: "Control",
        description: "Current implementation",
        trafficPercentage: 50,
        isControl: true,
        config: {},
        metrics: { participants: 0, conversions: 0, conversionRate: 0, avgSatisfaction: 0, avgDuration: 0, cost: 0 },
      },
      {
        id: "variant-a",
        name: "Variant A",
        description: "",
        trafficPercentage: 50,
        isControl: false,
        config: {},
        metrics: { participants: 0, conversions: 0, conversionRate: 0, avgSatisfaction: 0, avgDuration: 0, cost: 0 },
      },
    ],
    targetAudience: { type: "all" },
    schedule: {
      startDate: new Date(),
      autoStop: true,
      sampleSize: 5000,
      statisticalSignificance: 95,
    },
    tags: [],
  });

  if (!isOpen) return null;

  const experimentTypes: { value: ExperimentType; label: string; description: string }[] = [
    { value: "script", label: "Script", description: "Test different conversation scripts" },
    { value: "voice", label: "Voice", description: "Compare AI voice models" },
    { value: "agent", label: "Agent", description: "Test agent personalities" },
    { value: "prompt", label: "Prompt", description: "Optimize system prompts" },
    { value: "routing", label: "Routing", description: "Test call routing strategies" },
    { value: "timing", label: "Timing", description: "Optimize call timing" },
  ];

  const metrics: MetricType[] = ["conversion", "satisfaction", "duration", "resolution", "engagement", "cost"];

  const addVariant = () => {
    const variants = experimentData.variants || [];
    const newVariant: Variant = {
      id: `variant-${Date.now()}`,
      name: `Variant ${String.fromCharCode(65 + variants.length - 1)}`,
      description: "",
      trafficPercentage: 0,
      isControl: false,
      config: {},
      metrics: { participants: 0, conversions: 0, conversionRate: 0, avgSatisfaction: 0, avgDuration: 0, cost: 0 },
    };
    setExperimentData({ ...experimentData, variants: [...variants, newVariant] });
  };

  const updateVariantTraffic = () => {
    const variants = experimentData.variants || [];
    const equalShare = Math.floor(100 / variants.length);
    const remainder = 100 - equalShare * variants.length;

    const updated = variants.map((v, i) => ({
      ...v,
      trafficPercentage: equalShare + (i === 0 ? remainder : 0),
    }));
    setExperimentData({ ...experimentData, variants: updated });
  };

  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Experiment Name</label>
              <input
                type="text"
                value={experimentData.name}
                onChange={(e) => setExperimentData({ ...experimentData, name: e.target.value })}
                className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                placeholder="e.g., Greeting Script A/B Test"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Description</label>
              <textarea
                value={experimentData.description}
                onChange={(e) => setExperimentData({ ...experimentData, description: e.target.value })}
                className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                rows={3}
                placeholder="What are you testing?"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-3">Experiment Type</label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {experimentTypes.map((type) => (
                  <button
                    key={type.value}
                    onClick={() => setExperimentData({ ...experimentData, type: type.value })}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      experimentData.type === type.value
                        ? "border-purple-500 bg-purple-500/20"
                        : "border-white/10 hover:border-white/20"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`p-1.5 rounded-lg ${getTypeColor(type.value)}`}>
                        {getTypeIcon(type.value)}
                      </span>
                      <span className="text-white font-medium">{type.label}</span>
                    </div>
                    <p className="text-xs text-gray-500">{type.description}</p>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Hypothesis</label>
              <textarea
                value={experimentData.hypothesis}
                onChange={(e) => setExperimentData({ ...experimentData, hypothesis: e.target.value })}
                className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                rows={2}
                placeholder="If we change X, then Y will increase by Z%"
              />
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium text-gray-300">Variants</label>
                <button
                  onClick={addVariant}
                  className="flex items-center gap-1 text-sm text-purple-400 hover:text-purple-300"
                >
                  <Plus className="w-4 h-4" />
                  Add Variant
                </button>
              </div>

              <div className="space-y-3">
                {experimentData.variants?.map((variant, index) => (
                  <div
                    key={variant.id}
                    className="p-4 bg-black/30 rounded-lg border border-white/10"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={variant.name}
                          onChange={(e) => {
                            const variants = [...(experimentData.variants || [])];
                            variants[index] = { ...variants[index], name: e.target.value };
                            setExperimentData({ ...experimentData, variants });
                          }}
                          className="px-3 py-1.5 bg-black/30 border border-white/10 rounded text-white text-sm focus:outline-none focus:border-purple-500"
                        />
                        {variant.isControl && (
                          <span className="px-2 py-0.5 text-xs bg-blue-500/20 text-blue-400 rounded">Control</span>
                        )}
                      </div>
                      {!variant.isControl && experimentData.variants!.length > 2 && (
                        <button
                          onClick={() => {
                            const variants = experimentData.variants?.filter((_, i) => i !== index);
                            setExperimentData({ ...experimentData, variants });
                          }}
                          className="p-1 rounded hover:bg-red-500/20 text-red-400"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      )}
                    </div>

                    <input
                      type="text"
                      value={variant.description}
                      onChange={(e) => {
                        const variants = [...(experimentData.variants || [])];
                        variants[index] = { ...variants[index], description: e.target.value };
                        setExperimentData({ ...experimentData, variants });
                      }}
                      className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded text-white text-sm placeholder-gray-500 focus:outline-none focus:border-purple-500"
                      placeholder="Variant description"
                    />

                    <div className="flex items-center gap-3 mt-3">
                      <span className="text-sm text-gray-400">Traffic:</span>
                      <input
                        type="number"
                        value={variant.trafficPercentage}
                        onChange={(e) => {
                          const variants = [...(experimentData.variants || [])];
                          variants[index] = { ...variants[index], trafficPercentage: parseInt(e.target.value) || 0 };
                          setExperimentData({ ...experimentData, variants });
                        }}
                        className="w-20 px-3 py-1.5 bg-black/30 border border-white/10 rounded text-white text-sm focus:outline-none focus:border-purple-500"
                        min={0}
                        max={100}
                      />
                      <span className="text-sm text-gray-400">%</span>
                    </div>
                  </div>
                ))}
              </div>

              <button
                onClick={updateVariantTraffic}
                className="mt-3 text-sm text-purple-400 hover:text-purple-300"
              >
                Split traffic equally
              </button>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-3">Primary Metric</label>
              <div className="grid grid-cols-3 gap-2">
                {metrics.map((metric) => (
                  <button
                    key={metric}
                    onClick={() => setExperimentData({ ...experimentData, primaryMetric: metric })}
                    className={`py-2 px-3 rounded-lg border transition-all text-sm capitalize ${
                      experimentData.primaryMetric === metric
                        ? "border-purple-500 bg-purple-500/20 text-white"
                        : "border-white/10 text-gray-400 hover:border-white/20"
                    }`}
                  >
                    {metric}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-3">Secondary Metrics</label>
              <div className="grid grid-cols-3 gap-2">
                {metrics.filter((m) => m !== experimentData.primaryMetric).map((metric) => (
                  <button
                    key={metric}
                    onClick={() => {
                      const secondaryMetrics = experimentData.secondaryMetrics || [];
                      const updated = secondaryMetrics.includes(metric)
                        ? secondaryMetrics.filter((m) => m !== metric)
                        : [...secondaryMetrics, metric];
                      setExperimentData({ ...experimentData, secondaryMetrics: updated });
                    }}
                    className={`py-2 px-3 rounded-lg border transition-all text-sm capitalize ${
                      experimentData.secondaryMetrics?.includes(metric)
                        ? "border-purple-500 bg-purple-500/20 text-white"
                        : "border-white/10 text-gray-400 hover:border-white/20"
                    }`}
                  >
                    {metric}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Sample Size</label>
                <input
                  type="number"
                  value={experimentData.schedule?.sampleSize || 5000}
                  onChange={(e) => setExperimentData({
                    ...experimentData,
                    schedule: { ...experimentData.schedule!, sampleSize: parseInt(e.target.value) || 5000 },
                  })}
                  className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                  min={100}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Statistical Significance</label>
                <select
                  value={experimentData.schedule?.statisticalSignificance || 95}
                  onChange={(e) => setExperimentData({
                    ...experimentData,
                    schedule: { ...experimentData.schedule!, statisticalSignificance: parseInt(e.target.value) },
                  })}
                  className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value={90}>90%</option>
                  <option value={95}>95%</option>
                  <option value={99}>99%</option>
                </select>
              </div>
            </div>

            <div>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={experimentData.schedule?.autoStop || false}
                  onChange={(e) => setExperimentData({
                    ...experimentData,
                    schedule: { ...experimentData.schedule!, autoStop: e.target.checked },
                  })}
                  className="w-4 h-4 rounded border-white/20 bg-black/30 text-purple-500 focus:ring-purple-500"
                />
                <div>
                  <p className="text-sm text-white">Auto-stop when significant</p>
                  <p className="text-xs text-gray-400">Automatically stop when statistical significance is reached</p>
                </div>
              </label>
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
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">Create A/B Test</h2>
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10 text-gray-400">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="flex items-center gap-2 mt-4">
            {[1, 2, 3].map((s) => (
              <React.Fragment key={s}>
                <button
                  onClick={() => setStep(s)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
                    step === s ? "bg-purple-500/20 text-purple-400" : step > s ? "text-green-400" : "text-gray-500"
                  }`}
                >
                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${
                    step === s ? "bg-purple-500 text-white" : step > s ? "bg-green-500 text-white" : "bg-white/10 text-gray-500"
                  }`}>
                    {step > s ? <Check className="w-4 h-4" /> : s}
                  </span>
                  <span className="text-sm hidden sm:inline">
                    {s === 1 ? "Setup" : s === 2 ? "Variants" : "Metrics"}
                  </span>
                </button>
                {s < 3 && <div className={`flex-1 h-0.5 ${step > s ? "bg-green-500" : "bg-white/10"}`} />}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {renderStep()}
        </div>

        <div className="p-6 border-t border-white/10 flex justify-between">
          <button
            onClick={() => step > 1 && setStep(step - 1)}
            className={`px-4 py-2 rounded-lg text-gray-400 hover:bg-white/10 transition-colors ${step === 1 ? "invisible" : ""}`}
          >
            Back
          </button>
          <div className="flex gap-3">
            <button onClick={onClose} className="px-4 py-2 rounded-lg text-gray-400 hover:bg-white/10 transition-colors">
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
                onClick={() => onSave(experimentData)}
                className="px-6 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium hover:opacity-90 transition-opacity"
              >
                Create Experiment
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Component
export default function ABTestingPage() {
  const [experiments, setExperiments] = useState<Experiment[]>(mockExperiments);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<ExperimentStatus | "all">("all");
  const [typeFilter, setTypeFilter] = useState<ExperimentType | "all">("all");
  const [showCreateDialog, setShowCreateDialog] = useState(false);

  const filteredExperiments = useMemo(() => {
    return experiments.filter((exp) => {
      const matchesSearch =
        exp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        exp.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        exp.tags.some((t) => t.toLowerCase().includes(searchQuery.toLowerCase()));

      const matchesStatus = statusFilter === "all" || exp.status === statusFilter;
      const matchesType = typeFilter === "all" || exp.type === typeFilter;

      return matchesSearch && matchesStatus && matchesType;
    });
  }, [experiments, searchQuery, statusFilter, typeFilter]);

  const stats = useMemo(() => {
    const total = experiments.length;
    const running = experiments.filter((e) => e.status === "running").length;
    const completed = experiments.filter((e) => e.status === "completed").length;
    const significant = experiments.filter((e) => e.results?.isSignificant).length;

    return { total, running, completed, significant };
  }, [experiments]);

  const handleStatusChange = (experimentId: string, newStatus: ExperimentStatus) => {
    setExperiments((prev) =>
      prev.map((exp) =>
        exp.id === experimentId ? { ...exp, status: newStatus, updatedAt: new Date() } : exp
      )
    );
  };

  const handleDelete = (experimentId: string) => {
    if (confirm("Are you sure you want to delete this experiment?")) {
      setExperiments((prev) => prev.filter((exp) => exp.id !== experimentId));
    }
  };

  const handleSave = (data: Partial<Experiment>) => {
    const newExperiment: Experiment = {
      ...data,
      id: `exp-${Date.now()}`,
      status: "draft",
      createdAt: new Date(),
      updatedAt: new Date(),
      createdBy: "Current User",
    } as Experiment;
    setExperiments((prev) => [newExperiment, ...prev]);
    setShowCreateDialog(false);
  };

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">A/B Testing</h1>
            <p className="text-gray-400 mt-1">Run experiments to optimize your AI voice agents</p>
          </div>

          <button
            onClick={() => setShowCreateDialog(true)}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90 transition-opacity"
          >
            <Plus className="w-4 h-4" />
            New Experiment
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400">
                <FlaskConical className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.total}</p>
                <p className="text-sm text-gray-400">Total Experiments</p>
              </div>
            </div>
          </div>

          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-500/20 text-green-400">
                <Play className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.running}</p>
                <p className="text-sm text-gray-400">Running</p>
              </div>
            </div>
          </div>

          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
                <CheckCircle className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.completed}</p>
                <p className="text-sm text-gray-400">Completed</p>
              </div>
            </div>
          </div>

          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-yellow-500/20 text-yellow-400">
                <Trophy className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.significant}</p>
                <p className="text-sm text-gray-400">Significant Results</p>
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
              placeholder="Search experiments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
            />
          </div>

          <div className="flex gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as ExperimentStatus | "all")}
              className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Statuses</option>
              <option value="draft">Draft</option>
              <option value="running">Running</option>
              <option value="paused">Paused</option>
              <option value="completed">Completed</option>
              <option value="stopped">Stopped</option>
            </select>

            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value as ExperimentType | "all")}
              className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Types</option>
              <option value="script">Script</option>
              <option value="voice">Voice</option>
              <option value="agent">Agent</option>
              <option value="prompt">Prompt</option>
              <option value="routing">Routing</option>
              <option value="timing">Timing</option>
            </select>
          </div>
        </div>

        {/* Experiments List */}
        <div className="space-y-4">
          {filteredExperiments.length > 0 ? (
            filteredExperiments.map((experiment) => (
              <ExperimentCard
                key={experiment.id}
                experiment={experiment}
                onView={() => console.log("View", experiment.id)}
                onEdit={() => console.log("Edit", experiment.id)}
                onDuplicate={() => console.log("Duplicate", experiment.id)}
                onDelete={() => handleDelete(experiment.id)}
                onStatusChange={(status) => handleStatusChange(experiment.id, status)}
              />
            ))
          ) : (
            <div className="text-center py-16 bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl">
              <FlaskConical className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">No experiments found</h3>
              <p className="text-gray-400 mb-6">
                {searchQuery || statusFilter !== "all" || typeFilter !== "all"
                  ? "No experiments match your current filters"
                  : "Create your first A/B test to start optimizing"}
              </p>
              <button
                onClick={() => setShowCreateDialog(true)}
                className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors"
              >
                <Plus className="w-4 h-4" />
                New Experiment
              </button>
            </div>
          )}
        </div>

        {/* Create Dialog */}
        <CreateExperimentDialog
          isOpen={showCreateDialog}
          onClose={() => setShowCreateDialog(false)}
          onSave={handleSave}
        />
      </div>
    </DashboardLayout>
  );
}
