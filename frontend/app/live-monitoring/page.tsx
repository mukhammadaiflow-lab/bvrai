"use client";

import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import {
  Phone,
  PhoneCall,
  PhoneOff,
  PhoneIncoming,
  PhoneOutgoing,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Headphones,
  Radio,
  Signal,
  SignalHigh,
  SignalLow,
  SignalMedium,
  Clock,
  Timer,
  User,
  Users,
  Bot,
  MessageSquare,
  AlertTriangle,
  AlertCircle,
  CheckCircle,
  XCircle,
  Info,
  Play,
  Pause,
  Square,
  SkipForward,
  RotateCcw,
  RefreshCw,
  Settings,
  Filter,
  Search,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  MoreVertical,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  X,
  Plus,
  Minus,
  Copy,
  Download,
  Share2,
  ExternalLink,
  Zap,
  Activity,
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart,
  Sparkles,
  Heart,
  ThumbsUp,
  ThumbsDown,
  Frown,
  Meh,
  Smile,
  SmilePlus,
  Target,
  Flag,
  Tag,
  Hash,
  Globe,
  MapPin,
  Building2,
  Calendar,
  ArrowUpRight,
  ArrowDownRight,
  Waves,
  AudioLines,
  CircleDot,
  Circle,
  Loader2,
  BellRing,
  Send,
  MessageCircle,
} from "lucide-react";

// Types
type CallStatus = "ringing" | "connected" | "on_hold" | "transferring" | "ending";
type SentimentScore = "very_negative" | "negative" | "neutral" | "positive" | "very_positive";
type CallDirection = "inbound" | "outbound";

interface TranscriptEntry {
  id: string;
  speaker: "agent" | "customer";
  text: string;
  timestamp: string;
  sentiment?: SentimentScore;
  confidence?: number;
}

interface LiveCall {
  id: string;
  agentId: string;
  agentName: string;
  agentAvatar?: string;
  customerName: string;
  customerPhone: string;
  customerLocation?: string;
  direction: CallDirection;
  status: CallStatus;
  startTime: string;
  duration: number;
  holdTime: number;
  sentiment: SentimentScore;
  sentimentTrend: "improving" | "declining" | "stable";
  confidenceScore: number;
  transcript: TranscriptEntry[];
  tags: string[];
  priority: "low" | "normal" | "high" | "urgent";
  whisperEnabled: boolean;
  bargeEnabled: boolean;
  isListening: boolean;
  audioLevel: number;
  qualityScore: number;
  intent?: string;
  notes?: string;
}

interface QueueStats {
  totalInQueue: number;
  avgWaitTime: number;
  longestWait: number;
  abandonRate: number;
}

interface LiveStats {
  activeCalls: number;
  totalAgents: number;
  availableAgents: number;
  avgHandleTime: number;
  avgSentiment: number;
  callsPerHour: number;
  successRate: number;
  queueStats: QueueStats;
}

// Sample Data
const generateLiveCalls = (): LiveCall[] => [
  {
    id: "call_001",
    agentId: "agent_1",
    agentName: "Sales Assistant AI",
    customerName: "John Smith",
    customerPhone: "+1 (555) 123-4567",
    customerLocation: "New York, NY",
    direction: "inbound",
    status: "connected",
    startTime: new Date(Date.now() - 245000).toISOString(),
    duration: 245,
    holdTime: 0,
    sentiment: "positive",
    sentimentTrend: "improving",
    confidenceScore: 92,
    transcript: [
      { id: "t1", speaker: "agent", text: "Hello! Thank you for calling. How can I help you today?", timestamp: new Date(Date.now() - 245000).toISOString(), sentiment: "positive", confidence: 95 },
      { id: "t2", speaker: "customer", text: "Hi, I'm interested in learning more about your enterprise plan.", timestamp: new Date(Date.now() - 230000).toISOString(), sentiment: "positive", confidence: 88 },
      { id: "t3", speaker: "agent", text: "Absolutely! Our enterprise plan includes unlimited agents, priority support, and custom integrations. Would you like me to go over the pricing details?", timestamp: new Date(Date.now() - 210000).toISOString(), sentiment: "positive", confidence: 94 },
      { id: "t4", speaker: "customer", text: "Yes, please. Also, do you offer any discounts for annual billing?", timestamp: new Date(Date.now() - 180000).toISOString(), sentiment: "positive", confidence: 91 },
      { id: "t5", speaker: "agent", text: "We do! Annual billing comes with a 20% discount. Our enterprise plan starts at $499 per month, which would be $399 per month with annual billing.", timestamp: new Date(Date.now() - 150000).toISOString(), sentiment: "positive", confidence: 96 },
      { id: "t6", speaker: "customer", text: "That sounds reasonable. Can I get a demo before committing?", timestamp: new Date(Date.now() - 120000).toISOString(), sentiment: "positive", confidence: 93 },
    ],
    tags: ["enterprise", "sales", "high-value"],
    priority: "high",
    whisperEnabled: false,
    bargeEnabled: false,
    isListening: false,
    audioLevel: 75,
    qualityScore: 98,
    intent: "Purchase inquiry",
  },
  {
    id: "call_002",
    agentId: "agent_2",
    agentName: "Support Bot",
    customerName: "Sarah Johnson",
    customerPhone: "+1 (555) 234-5678",
    customerLocation: "Los Angeles, CA",
    direction: "inbound",
    status: "connected",
    startTime: new Date(Date.now() - 180000).toISOString(),
    duration: 180,
    holdTime: 15,
    sentiment: "neutral",
    sentimentTrend: "stable",
    confidenceScore: 85,
    transcript: [
      { id: "t1", speaker: "agent", text: "Thank you for calling support. What issue can I help you with?", timestamp: new Date(Date.now() - 180000).toISOString(), sentiment: "neutral", confidence: 92 },
      { id: "t2", speaker: "customer", text: "I'm having trouble connecting my CRM integration.", timestamp: new Date(Date.now() - 165000).toISOString(), sentiment: "negative", confidence: 85 },
      { id: "t3", speaker: "agent", text: "I understand. Let me look into that for you. Which CRM are you trying to connect?", timestamp: new Date(Date.now() - 150000).toISOString(), sentiment: "neutral", confidence: 90 },
      { id: "t4", speaker: "customer", text: "Salesforce. It was working fine until yesterday.", timestamp: new Date(Date.now() - 135000).toISOString(), sentiment: "negative", confidence: 88 },
      { id: "t5", speaker: "agent", text: "I see there was a temporary issue with our Salesforce connector yesterday. It should be resolved now. Can you try reconnecting?", timestamp: new Date(Date.now() - 110000).toISOString(), sentiment: "neutral", confidence: 94 },
    ],
    tags: ["support", "integration", "salesforce"],
    priority: "normal",
    whisperEnabled: false,
    bargeEnabled: false,
    isListening: false,
    audioLevel: 65,
    qualityScore: 95,
    intent: "Technical support",
  },
  {
    id: "call_003",
    agentId: "agent_3",
    agentName: "Collections Agent",
    customerName: "Michael Chen",
    customerPhone: "+1 (555) 345-6789",
    customerLocation: "Chicago, IL",
    direction: "outbound",
    status: "connected",
    startTime: new Date(Date.now() - 120000).toISOString(),
    duration: 120,
    holdTime: 0,
    sentiment: "negative",
    sentimentTrend: "declining",
    confidenceScore: 78,
    transcript: [
      { id: "t1", speaker: "agent", text: "Hello, this is a courtesy call regarding your account. Is this Michael Chen?", timestamp: new Date(Date.now() - 120000).toISOString(), sentiment: "neutral", confidence: 90 },
      { id: "t2", speaker: "customer", text: "Yes, speaking. What is this about?", timestamp: new Date(Date.now() - 110000).toISOString(), sentiment: "neutral", confidence: 85 },
      { id: "t3", speaker: "agent", text: "I'm calling about your outstanding balance of $150. We noticed the payment is 15 days overdue.", timestamp: new Date(Date.now() - 95000).toISOString(), sentiment: "neutral", confidence: 92 },
      { id: "t4", speaker: "customer", text: "I've been meaning to pay that. Things have been tight lately.", timestamp: new Date(Date.now() - 80000).toISOString(), sentiment: "negative", confidence: 80 },
      { id: "t5", speaker: "agent", text: "I understand. Would you like to set up a payment plan? We can split it into three monthly payments.", timestamp: new Date(Date.now() - 65000).toISOString(), sentiment: "positive", confidence: 88 },
    ],
    tags: ["collections", "payment-plan"],
    priority: "normal",
    whisperEnabled: false,
    bargeEnabled: false,
    isListening: false,
    audioLevel: 55,
    qualityScore: 92,
    intent: "Payment collection",
  },
  {
    id: "call_004",
    agentId: "agent_1",
    agentName: "Sales Assistant AI",
    customerName: "Emily Davis",
    customerPhone: "+1 (555) 456-7890",
    customerLocation: "Austin, TX",
    direction: "inbound",
    status: "ringing",
    startTime: new Date(Date.now() - 15000).toISOString(),
    duration: 15,
    holdTime: 0,
    sentiment: "neutral",
    sentimentTrend: "stable",
    confidenceScore: 0,
    transcript: [],
    tags: ["new-lead"],
    priority: "normal",
    whisperEnabled: false,
    bargeEnabled: false,
    isListening: false,
    audioLevel: 0,
    qualityScore: 100,
  },
  {
    id: "call_005",
    agentId: "agent_4",
    agentName: "Appointment Scheduler",
    customerName: "Robert Wilson",
    customerPhone: "+1 (555) 567-8901",
    customerLocation: "Seattle, WA",
    direction: "outbound",
    status: "connected",
    startTime: new Date(Date.now() - 90000).toISOString(),
    duration: 90,
    holdTime: 0,
    sentiment: "very_positive",
    sentimentTrend: "improving",
    confidenceScore: 95,
    transcript: [
      { id: "t1", speaker: "agent", text: "Hi! I'm calling to confirm your appointment for tomorrow at 2 PM.", timestamp: new Date(Date.now() - 90000).toISOString(), sentiment: "positive", confidence: 94 },
      { id: "t2", speaker: "customer", text: "Oh yes, I remember. 2 PM works perfectly!", timestamp: new Date(Date.now() - 80000).toISOString(), sentiment: "very_positive", confidence: 92 },
      { id: "t3", speaker: "agent", text: "Great! You'll be meeting with Dr. Johnson. Is there anything specific you'd like to discuss?", timestamp: new Date(Date.now() - 65000).toISOString(), sentiment: "positive", confidence: 90 },
      { id: "t4", speaker: "customer", text: "Just my annual checkup. I'm looking forward to it!", timestamp: new Date(Date.now() - 50000).toISOString(), sentiment: "very_positive", confidence: 95 },
    ],
    tags: ["appointment", "confirmation", "healthcare"],
    priority: "low",
    whisperEnabled: false,
    bargeEnabled: false,
    isListening: false,
    audioLevel: 70,
    qualityScore: 99,
    intent: "Appointment confirmation",
  },
  {
    id: "call_006",
    agentId: "agent_2",
    agentName: "Support Bot",
    customerName: "Lisa Anderson",
    customerPhone: "+1 (555) 678-9012",
    customerLocation: "Miami, FL",
    direction: "inbound",
    status: "on_hold",
    startTime: new Date(Date.now() - 300000).toISOString(),
    duration: 300,
    holdTime: 45,
    sentiment: "negative",
    sentimentTrend: "declining",
    confidenceScore: 72,
    transcript: [
      { id: "t1", speaker: "agent", text: "Thank you for your patience. How can I assist you?", timestamp: new Date(Date.now() - 300000).toISOString(), sentiment: "neutral", confidence: 88 },
      { id: "t2", speaker: "customer", text: "I've been trying to cancel my subscription for weeks and no one is helping me!", timestamp: new Date(Date.now() - 285000).toISOString(), sentiment: "very_negative", confidence: 95 },
      { id: "t3", speaker: "agent", text: "I'm so sorry to hear about your experience. Let me help you with that right away.", timestamp: new Date(Date.now() - 270000).toISOString(), sentiment: "positive", confidence: 90 },
      { id: "t4", speaker: "customer", text: "I just want this resolved. It's been frustrating.", timestamp: new Date(Date.now() - 255000).toISOString(), sentiment: "negative", confidence: 92 },
      { id: "t5", speaker: "agent", text: "I completely understand. I'm going to process your cancellation now and ensure you receive a confirmation email. Please hold for just a moment.", timestamp: new Date(Date.now() - 240000).toISOString(), sentiment: "neutral", confidence: 88 },
    ],
    tags: ["cancellation", "escalation", "at-risk"],
    priority: "urgent",
    whisperEnabled: true,
    bargeEnabled: false,
    isListening: false,
    audioLevel: 0,
    qualityScore: 85,
    intent: "Subscription cancellation",
    notes: "Customer frustrated - previous tickets unresolved",
  },
];

const sampleLiveStats: LiveStats = {
  activeCalls: 6,
  totalAgents: 12,
  availableAgents: 6,
  avgHandleTime: 185,
  avgSentiment: 72,
  callsPerHour: 45,
  successRate: 87,
  queueStats: {
    totalInQueue: 3,
    avgWaitTime: 28,
    longestWait: 65,
    abandonRate: 4.2,
  },
};

// Utility functions
const getSentimentIcon = (sentiment: SentimentScore) => {
  switch (sentiment) {
    case "very_positive":
      return <SmilePlus className="h-4 w-4 text-green-400" />;
    case "positive":
      return <Smile className="h-4 w-4 text-green-400" />;
    case "neutral":
      return <Meh className="h-4 w-4 text-gray-400" />;
    case "negative":
      return <Frown className="h-4 w-4 text-orange-400" />;
    case "very_negative":
      return <Frown className="h-4 w-4 text-red-400" />;
    default:
      return <Meh className="h-4 w-4 text-gray-400" />;
  }
};

const getSentimentColor = (sentiment: SentimentScore) => {
  switch (sentiment) {
    case "very_positive":
      return "text-green-400 bg-green-500/20";
    case "positive":
      return "text-green-400 bg-green-500/20";
    case "neutral":
      return "text-gray-400 bg-gray-500/20";
    case "negative":
      return "text-orange-400 bg-orange-500/20";
    case "very_negative":
      return "text-red-400 bg-red-500/20";
    default:
      return "text-gray-400 bg-gray-500/20";
  }
};

const getStatusColor = (status: CallStatus) => {
  switch (status) {
    case "ringing":
      return "text-yellow-400 bg-yellow-500/20 animate-pulse";
    case "connected":
      return "text-green-400 bg-green-500/20";
    case "on_hold":
      return "text-orange-400 bg-orange-500/20";
    case "transferring":
      return "text-blue-400 bg-blue-500/20";
    case "ending":
      return "text-red-400 bg-red-500/20";
    default:
      return "text-gray-400 bg-gray-500/20";
  }
};

const getPriorityColor = (priority: string) => {
  switch (priority) {
    case "urgent":
      return "text-red-400 bg-red-500/20 border-red-500/30";
    case "high":
      return "text-orange-400 bg-orange-500/20 border-orange-500/30";
    case "normal":
      return "text-blue-400 bg-blue-500/20 border-blue-500/30";
    case "low":
      return "text-gray-400 bg-gray-500/20 border-gray-500/30";
    default:
      return "text-gray-400 bg-gray-500/20 border-gray-500/30";
  }
};

const formatDuration = (seconds: number) => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
};

const formatTime = (dateString: string) => {
  return new Date(dateString).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
};

// Components
function StatsCard({
  title,
  value,
  change,
  changeType,
  icon: Icon,
  color,
  subtitle,
  pulse,
}: {
  title: string;
  value: string | number;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: any;
  color: string;
  subtitle?: string;
  pulse?: boolean;
}) {
  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4 hover:border-white/10 transition-all">
      <div className="flex items-center justify-between">
        <div
          className={`p-2 rounded-lg ${pulse ? "animate-pulse" : ""}`}
          style={{ backgroundColor: `${color}20` }}
        >
          <Icon className={`h-4 w-4`} style={{ color }} />
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
        {subtitle && <div className="text-xs text-gray-500 mt-0.5">{subtitle}</div>}
      </div>
    </div>
  );
}

function AudioVisualizer({ level, isActive }: { level: number; isActive: boolean }) {
  const bars = 5;
  return (
    <div className="flex items-end gap-0.5 h-4">
      {Array.from({ length: bars }).map((_, i) => {
        const barHeight = isActive ? Math.random() * 100 : 0;
        const active = isActive && barHeight > (i + 1) * (100 / bars) - 20;
        return (
          <div
            key={i}
            className={`w-1 rounded-full transition-all duration-75 ${
              active ? "bg-green-400" : "bg-gray-600"
            }`}
            style={{ height: `${Math.max(20, (i + 1) * 20)}%` }}
          />
        );
      })}
    </div>
  );
}

function LiveCallCard({
  call,
  isExpanded,
  onToggleExpand,
  onAction,
}: {
  call: LiveCall;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onAction: (action: string, call: LiveCall) => void;
}) {
  const [localDuration, setLocalDuration] = useState(call.duration);
  const transcriptRef = useRef<HTMLDivElement>(null);

  // Update duration every second
  useEffect(() => {
    if (call.status === "connected" || call.status === "on_hold") {
      const interval = setInterval(() => {
        setLocalDuration((d) => d + 1);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [call.status]);

  // Auto-scroll transcript
  useEffect(() => {
    if (transcriptRef.current && isExpanded) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [call.transcript, isExpanded]);

  return (
    <div
      className={`bg-[#1a1a2e]/80 rounded-xl border transition-all ${
        call.priority === "urgent"
          ? "border-red-500/30 shadow-lg shadow-red-500/10"
          : call.priority === "high"
          ? "border-orange-500/20"
          : "border-white/5 hover:border-white/10"
      }`}
    >
      {/* Header */}
      <div
        className="p-4 cursor-pointer"
        onClick={onToggleExpand}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Status Indicator */}
            <div className="relative">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center ${getStatusColor(
                  call.status
                )}`}
              >
                {call.status === "ringing" ? (
                  <Phone className="h-5 w-5 animate-bounce" />
                ) : call.status === "connected" ? (
                  <PhoneCall className="h-5 w-5" />
                ) : call.status === "on_hold" ? (
                  <Pause className="h-5 w-5" />
                ) : (
                  <Phone className="h-5 w-5" />
                )}
              </div>
              {call.status === "connected" && (
                <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-green-500 rounded-full border-2 border-[#1a1a2e]" />
              )}
            </div>

            {/* Call Info */}
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white">{call.customerName}</span>
                <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${getPriorityColor(call.priority)}`}>
                  {call.priority}
                </span>
                {call.direction === "inbound" ? (
                  <PhoneIncoming className="h-3.5 w-3.5 text-green-400" />
                ) : (
                  <PhoneOutgoing className="h-3.5 w-3.5 text-blue-400" />
                )}
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <span>{call.customerPhone}</span>
                {call.customerLocation && (
                  <>
                    <span>•</span>
                    <span className="flex items-center gap-1">
                      <MapPin className="h-3 w-3" />
                      {call.customerLocation}
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Right Side */}
          <div className="flex items-center gap-4">
            {/* Sentiment */}
            <div className="flex items-center gap-2">
              {getSentimentIcon(call.sentiment)}
              <span className="text-sm font-medium text-gray-300">
                {call.confidenceScore}%
              </span>
              {call.sentimentTrend === "improving" ? (
                <TrendingUp className="h-3.5 w-3.5 text-green-400" />
              ) : call.sentimentTrend === "declining" ? (
                <TrendingDown className="h-3.5 w-3.5 text-red-400" />
              ) : null}
            </div>

            {/* Duration */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-lg">
              <Clock className="h-4 w-4 text-gray-400" />
              <span className="font-mono text-white">{formatDuration(localDuration)}</span>
            </div>

            {/* Audio Level */}
            {call.status === "connected" && (
              <AudioVisualizer level={call.audioLevel} isActive={true} />
            )}

            {/* Expand Toggle */}
            <button className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400">
              {isExpanded ? (
                <ChevronUp className="h-5 w-5" />
              ) : (
                <ChevronDown className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>

        {/* Agent & Tags */}
        <div className="flex items-center justify-between mt-3">
          <div className="flex items-center gap-2 text-sm">
            <Bot className="h-4 w-4 text-purple-400" />
            <span className="text-gray-300">{call.agentName}</span>
            {call.intent && (
              <>
                <span className="text-gray-600">•</span>
                <span className="text-gray-400">{call.intent}</span>
              </>
            )}
          </div>
          <div className="flex items-center gap-1.5">
            {call.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs"
              >
                {tag}
              </span>
            ))}
            {call.tags.length > 3 && (
              <span className="text-xs text-gray-500">+{call.tags.length - 3}</span>
            )}
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t border-white/5">
          {/* Action Bar */}
          <div className="px-4 py-3 bg-white/[0.02] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button
                onClick={() => onAction("listen", call)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors ${
                  call.isListening
                    ? "bg-green-500/20 text-green-400"
                    : "bg-white/5 text-gray-300 hover:bg-white/10"
                }`}
              >
                <Headphones className="h-4 w-4" />
                {call.isListening ? "Listening" : "Listen"}
              </button>
              <button
                onClick={() => onAction("whisper", call)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors ${
                  call.whisperEnabled
                    ? "bg-yellow-500/20 text-yellow-400"
                    : "bg-white/5 text-gray-300 hover:bg-white/10"
                }`}
              >
                <MessageCircle className="h-4 w-4" />
                Whisper
              </button>
              <button
                onClick={() => onAction("barge", call)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors ${
                  call.bargeEnabled
                    ? "bg-red-500/20 text-red-400"
                    : "bg-white/5 text-gray-300 hover:bg-white/10"
                }`}
              >
                <Mic className="h-4 w-4" />
                Barge In
              </button>
            </div>
            <div className="flex items-center gap-2">
              {call.status === "connected" && (
                <>
                  <button
                    onClick={() => onAction("hold", call)}
                    className="px-3 py-1.5 bg-orange-500/20 text-orange-400 rounded-lg text-sm font-medium hover:bg-orange-500/30 transition-colors flex items-center gap-2"
                  >
                    <Pause className="h-4 w-4" />
                    Hold
                  </button>
                  <button
                    onClick={() => onAction("transfer", call)}
                    className="px-3 py-1.5 bg-blue-500/20 text-blue-400 rounded-lg text-sm font-medium hover:bg-blue-500/30 transition-colors flex items-center gap-2"
                  >
                    <Share2 className="h-4 w-4" />
                    Transfer
                  </button>
                </>
              )}
              <button
                onClick={() => onAction("end", call)}
                className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded-lg text-sm font-medium hover:bg-red-500/30 transition-colors flex items-center gap-2"
              >
                <PhoneOff className="h-4 w-4" />
                End Call
              </button>
            </div>
          </div>

          {/* Transcript */}
          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium text-gray-300">Live Transcript</h4>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 text-xs text-gray-500">
                  <CircleDot className="h-3 w-3 text-red-500 animate-pulse" />
                  Live
                </div>
              </div>
            </div>
            <div
              ref={transcriptRef}
              className="bg-[#0d0d1a] rounded-xl p-4 max-h-64 overflow-y-auto space-y-3"
            >
              {call.transcript.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Waves className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>Waiting for conversation to begin...</p>
                </div>
              ) : (
                call.transcript.map((entry) => (
                  <div
                    key={entry.id}
                    className={`flex gap-3 ${
                      entry.speaker === "agent" ? "justify-start" : "justify-end"
                    }`}
                  >
                    <div
                      className={`max-w-[80%] p-3 rounded-xl ${
                        entry.speaker === "agent"
                          ? "bg-purple-500/10 border border-purple-500/20"
                          : "bg-white/5 border border-white/10"
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        {entry.speaker === "agent" ? (
                          <Bot className="h-3.5 w-3.5 text-purple-400" />
                        ) : (
                          <User className="h-3.5 w-3.5 text-gray-400" />
                        )}
                        <span className="text-xs text-gray-500">
                          {formatTime(entry.timestamp)}
                        </span>
                        {entry.sentiment && getSentimentIcon(entry.sentiment)}
                      </div>
                      <p className="text-sm text-gray-300">{entry.text}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Notes */}
          {call.notes && (
            <div className="px-4 pb-4">
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                <div className="flex items-center gap-2 text-yellow-400 text-sm">
                  <AlertTriangle className="h-4 w-4" />
                  <span className="font-medium">Note:</span>
                  <span className="text-yellow-300">{call.notes}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function QueuePanel({ stats }: { stats: QueueStats }) {
  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <Users className="h-4 w-4 text-purple-400" />
          Queue Status
        </h3>
        {stats.totalInQueue > 0 && (
          <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded-full text-xs font-medium animate-pulse">
            {stats.totalInQueue} waiting
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-lg font-bold text-white">{stats.totalInQueue}</div>
          <div className="text-xs text-gray-400">In Queue</div>
        </div>
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-lg font-bold text-white">{stats.avgWaitTime}s</div>
          <div className="text-xs text-gray-400">Avg Wait</div>
        </div>
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-lg font-bold text-white">{stats.longestWait}s</div>
          <div className="text-xs text-gray-400">Longest Wait</div>
        </div>
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-lg font-bold text-red-400">{stats.abandonRate}%</div>
          <div className="text-xs text-gray-400">Abandon Rate</div>
        </div>
      </div>
    </div>
  );
}

function AgentStatusPanel({ totalAgents, availableAgents }: { totalAgents: number; availableAgents: number }) {
  const busyAgents = totalAgents - availableAgents;
  const utilizationRate = Math.round((busyAgents / totalAgents) * 100);

  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <Bot className="h-4 w-4 text-purple-400" />
          Agent Status
        </h3>
        <span className="text-sm text-gray-400">{utilizationRate}% utilized</span>
      </div>
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">Available</span>
          <span className="text-sm font-medium text-green-400">{availableAgents}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">On Call</span>
          <span className="text-sm font-medium text-yellow-400">{busyAgents}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">Total</span>
          <span className="text-sm font-medium text-white">{totalAgents}</span>
        </div>
        <div className="mt-2">
          <div className="h-2 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-green-500 to-yellow-500 rounded-full transition-all"
              style={{ width: `${utilizationRate}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function SentimentOverview({ calls }: { calls: LiveCall[] }) {
  const sentimentCounts = useMemo(() => {
    const counts = {
      very_positive: 0,
      positive: 0,
      neutral: 0,
      negative: 0,
      very_negative: 0,
    };
    calls.forEach((call) => {
      if (call.status !== "ringing") {
        counts[call.sentiment]++;
      }
    });
    return counts;
  }, [calls]);

  const total = Object.values(sentimentCounts).reduce((a, b) => a + b, 0);

  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4">
      <h3 className="font-semibold text-white flex items-center gap-2 mb-4">
        <Heart className="h-4 w-4 text-purple-400" />
        Live Sentiment
      </h3>
      <div className="space-y-2">
        {[
          { key: "very_positive", label: "Very Positive", color: "bg-green-500" },
          { key: "positive", label: "Positive", color: "bg-green-400" },
          { key: "neutral", label: "Neutral", color: "bg-gray-400" },
          { key: "negative", label: "Negative", color: "bg-orange-400" },
          { key: "very_negative", label: "Very Negative", color: "bg-red-500" },
        ].map((item) => {
          const count = sentimentCounts[item.key as SentimentScore];
          const percent = total > 0 ? Math.round((count / total) * 100) : 0;
          return (
            <div key={item.key} className="flex items-center gap-2">
              <div className="w-24 text-xs text-gray-400">{item.label}</div>
              <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                <div
                  className={`h-full ${item.color} rounded-full transition-all`}
                  style={{ width: `${percent}%` }}
                />
              </div>
              <div className="w-8 text-xs text-gray-400 text-right">{count}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Main Page Component
export default function LiveMonitoringPage() {
  const [liveCalls, setLiveCalls] = useState<LiveCall[]>(generateLiveCalls());
  const [liveStats, setLiveStats] = useState<LiveStats>(sampleLiveStats);
  const [expandedCallId, setExpandedCallId] = useState<string | null>("call_001");
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<CallStatus | "all">("all");
  const [sentimentFilter, setSentimentFilter] = useState<SentimentScore | "all">("all");
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [viewMode, setViewMode] = useState<"grid" | "list">("list");

  // Simulate real-time updates
  useEffect(() => {
    if (!isAutoRefresh) return;

    const interval = setInterval(() => {
      setLiveCalls((prev) =>
        prev.map((call) => {
          if (call.status === "connected" || call.status === "on_hold") {
            // Randomly add new transcript entries
            if (Math.random() > 0.7 && call.transcript.length < 20) {
              const newEntry: TranscriptEntry = {
                id: `t${call.transcript.length + 1}`,
                speaker: Math.random() > 0.5 ? "agent" : "customer",
                text: "Sample transcript text...",
                timestamp: new Date().toISOString(),
                sentiment: ["positive", "neutral", "negative"][Math.floor(Math.random() * 3)] as SentimentScore,
                confidence: 80 + Math.floor(Math.random() * 20),
              };
              return {
                ...call,
                duration: call.duration + 1,
                audioLevel: 40 + Math.floor(Math.random() * 60),
                transcript: [...call.transcript, newEntry],
              };
            }
            return {
              ...call,
              duration: call.duration + 1,
              audioLevel: 40 + Math.floor(Math.random() * 60),
            };
          }
          return call;
        })
      );
    }, 1000);

    return () => clearInterval(interval);
  }, [isAutoRefresh]);

  // Filter calls
  const filteredCalls = useMemo(() => {
    return liveCalls.filter((call) => {
      const matchesSearch =
        call.customerName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        call.customerPhone.includes(searchQuery) ||
        call.agentName.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = statusFilter === "all" || call.status === statusFilter;
      const matchesSentiment = sentimentFilter === "all" || call.sentiment === sentimentFilter;
      return matchesSearch && matchesStatus && matchesSentiment;
    });
  }, [liveCalls, searchQuery, statusFilter, sentimentFilter]);

  // Handle call actions
  const handleCallAction = useCallback((action: string, call: LiveCall) => {
    switch (action) {
      case "listen":
        setLiveCalls((prev) =>
          prev.map((c) =>
            c.id === call.id ? { ...c, isListening: !c.isListening } : c
          )
        );
        break;
      case "whisper":
        setLiveCalls((prev) =>
          prev.map((c) =>
            c.id === call.id ? { ...c, whisperEnabled: !c.whisperEnabled } : c
          )
        );
        break;
      case "barge":
        setLiveCalls((prev) =>
          prev.map((c) =>
            c.id === call.id ? { ...c, bargeEnabled: !c.bargeEnabled } : c
          )
        );
        break;
      case "hold":
        setLiveCalls((prev) =>
          prev.map((c) =>
            c.id === call.id
              ? { ...c, status: c.status === "on_hold" ? "connected" : "on_hold" }
              : c
          )
        );
        break;
      case "end":
        setLiveCalls((prev) => prev.filter((c) => c.id !== call.id));
        break;
      default:
        break;
    }
  }, []);

  return (
    <DashboardLayout>
      <div className="p-6 lg:p-8 max-w-[1800px] mx-auto">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
          <div>
            <h1 className="text-2xl lg:text-3xl font-bold text-white flex items-center gap-3">
              <div className="relative">
                <Radio className="h-7 w-7 text-purple-400" />
                <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse" />
              </div>
              Live Monitoring
            </h1>
            <p className="text-gray-400 mt-1">Real-time call monitoring and supervision</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsAutoRefresh(!isAutoRefresh)}
              className={`px-4 py-2.5 rounded-xl text-sm font-medium flex items-center gap-2 transition-colors ${
                isAutoRefresh
                  ? "bg-green-500/20 text-green-400"
                  : "bg-white/5 text-gray-400"
              }`}
            >
              {isAutoRefresh ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              {isAutoRefresh ? "Live" : "Paused"}
            </button>
            <button className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white font-medium hover:bg-white/10 transition-colors flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Settings
            </button>
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 lg:grid-cols-8 gap-4 mb-6">
          <StatsCard
            title="Active Calls"
            value={liveStats.activeCalls}
            icon={PhoneCall}
            color="#10B981"
            pulse
          />
          <StatsCard
            title="In Queue"
            value={liveStats.queueStats.totalInQueue}
            icon={Users}
            color="#F59E0B"
          />
          <StatsCard
            title="Available Agents"
            value={`${liveStats.availableAgents}/${liveStats.totalAgents}`}
            icon={Bot}
            color="#8B5CF6"
          />
          <StatsCard
            title="Avg Handle Time"
            value={formatDuration(liveStats.avgHandleTime)}
            icon={Timer}
            color="#3B82F6"
          />
          <StatsCard
            title="Avg Sentiment"
            value={`${liveStats.avgSentiment}%`}
            icon={Heart}
            color="#EC4899"
            change="+2.5%"
            changeType="positive"
          />
          <StatsCard
            title="Calls/Hour"
            value={liveStats.callsPerHour}
            icon={Activity}
            color="#06B6D4"
          />
          <StatsCard
            title="Success Rate"
            value={`${liveStats.successRate}%`}
            icon={Target}
            color="#10B981"
          />
          <StatsCard
            title="Abandon Rate"
            value={`${liveStats.queueStats.abandonRate}%`}
            icon={AlertTriangle}
            color="#EF4444"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Main Call List */}
          <div className="lg:col-span-3 space-y-4">
            {/* Filters */}
            <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4">
              <div className="flex flex-col lg:flex-row gap-4">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search calls..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                  />
                </div>
                <div className="flex items-center gap-3">
                  <select
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value as CallStatus | "all")}
                    className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  >
                    <option value="all">All Status</option>
                    <option value="ringing">Ringing</option>
                    <option value="connected">Connected</option>
                    <option value="on_hold">On Hold</option>
                    <option value="transferring">Transferring</option>
                  </select>
                  <select
                    value={sentimentFilter}
                    onChange={(e) => setSentimentFilter(e.target.value as SentimentScore | "all")}
                    className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  >
                    <option value="all">All Sentiment</option>
                    <option value="very_positive">Very Positive</option>
                    <option value="positive">Positive</option>
                    <option value="neutral">Neutral</option>
                    <option value="negative">Negative</option>
                    <option value="very_negative">Very Negative</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Call Cards */}
            <div className="space-y-4">
              {filteredCalls.length === 0 ? (
                <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-12 text-center">
                  <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mx-auto mb-4">
                    <PhoneCall className="h-8 w-8 text-purple-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">No active calls</h3>
                  <p className="text-gray-400">
                    {searchQuery || statusFilter !== "all" || sentimentFilter !== "all"
                      ? "No calls match your current filters."
                      : "There are currently no calls in progress."}
                  </p>
                </div>
              ) : (
                filteredCalls.map((call) => (
                  <LiveCallCard
                    key={call.id}
                    call={call}
                    isExpanded={expandedCallId === call.id}
                    onToggleExpand={() =>
                      setExpandedCallId(expandedCallId === call.id ? null : call.id)
                    }
                    onAction={handleCallAction}
                  />
                ))
              )}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            <QueuePanel stats={liveStats.queueStats} />
            <AgentStatusPanel
              totalAgents={liveStats.totalAgents}
              availableAgents={liveStats.availableAgents}
            />
            <SentimentOverview calls={liveCalls} />

            {/* Alerts Panel */}
            <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4">
              <h3 className="font-semibold text-white flex items-center gap-2 mb-4">
                <BellRing className="h-4 w-4 text-purple-400" />
                Live Alerts
              </h3>
              <div className="space-y-2">
                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <div className="flex items-center gap-2 text-red-400 text-sm">
                    <AlertCircle className="h-4 w-4" />
                    <span className="font-medium">High frustration detected</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">Call #006 - Lisa Anderson</p>
                </div>
                <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                  <div className="flex items-center gap-2 text-yellow-400 text-sm">
                    <Clock className="h-4 w-4" />
                    <span className="font-medium">Long hold time</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">Call #006 - 45s on hold</p>
                </div>
                <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                  <div className="flex items-center gap-2 text-green-400 text-sm">
                    <Target className="h-4 w-4" />
                    <span className="font-medium">High-value lead detected</span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">Call #001 - Enterprise inquiry</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
