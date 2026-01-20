"use client";

import React, { useState, useMemo } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import {
  MessageSquare,
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart,
  LineChart,
  Activity,
  Zap,
  Target,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Calendar,
  Filter,
  Search,
  Download,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Eye,
  Play,
  Pause,
  Volume2,
  Lightbulb,
  Brain,
  Sparkles,
  ThumbsUp,
  ThumbsDown,
  Meh,
  Smile,
  Frown,
  Heart,
  Star,
  Tag,
  Hash,
  Users,
  Bot,
  Phone,
  Globe,
  MapPin,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  MoreVertical,
  Info,
  HelpCircle,
  ExternalLink,
  Maximize2,
  X
} from 'lucide-react';

// Types
interface SentimentData {
  positive: number;
  neutral: number;
  negative: number;
  trend: number;
}

interface TopicData {
  id: string;
  name: string;
  count: number;
  percentage: number;
  sentiment: number;
  trend: 'up' | 'down' | 'stable';
  examples: string[];
}

interface IntentData {
  id: string;
  name: string;
  count: number;
  successRate: number;
  avgHandleTime: number;
  trend: number;
}

interface KeywordData {
  word: string;
  count: number;
  sentiment: 'positive' | 'neutral' | 'negative';
  category: string;
}

interface ConversationPattern {
  id: string;
  pattern: string;
  frequency: number;
  outcome: 'success' | 'escalation' | 'dropout';
  avgDuration: number;
  recommendation: string;
}

interface InsightCard {
  id: string;
  type: 'opportunity' | 'risk' | 'trend' | 'achievement';
  title: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  metric?: string;
  metricChange?: number;
  actionable: boolean;
  actions?: string[];
}

interface ConversationSample {
  id: string;
  date: string;
  agent: string;
  duration: number;
  sentiment: number;
  topics: string[];
  intents: string[];
  outcome: 'resolved' | 'escalated' | 'dropped';
  satisfaction: number;
  transcript: TranscriptMessage[];
}

interface TranscriptMessage {
  speaker: 'agent' | 'customer';
  text: string;
  timestamp: number;
  sentiment: number;
}

// Mock Data
const sentimentData: SentimentData = {
  positive: 68,
  neutral: 22,
  negative: 10,
  trend: 5.2
};

const topicsData: TopicData[] = [
  { id: 't1', name: 'Billing Inquiries', count: 3240, percentage: 28, sentiment: 0.65, trend: 'up', examples: ['payment status', 'invoice question', 'refund request'] },
  { id: 't2', name: 'Technical Support', count: 2890, percentage: 25, sentiment: 0.72, trend: 'stable', examples: ['connection issue', 'feature not working', 'error message'] },
  { id: 't3', name: 'Product Information', count: 2100, percentage: 18, sentiment: 0.85, trend: 'up', examples: ['pricing details', 'feature comparison', 'new features'] },
  { id: 't4', name: 'Account Management', count: 1560, percentage: 13, sentiment: 0.78, trend: 'down', examples: ['password reset', 'update profile', 'cancel subscription'] },
  { id: 't5', name: 'Complaints', count: 980, percentage: 8, sentiment: 0.35, trend: 'down', examples: ['service quality', 'wait time', 'agent behavior'] },
  { id: 't6', name: 'Sales Inquiries', count: 920, percentage: 8, sentiment: 0.82, trend: 'up', examples: ['upgrade options', 'discounts', 'enterprise plan'] },
];

const intentsData: IntentData[] = [
  { id: 'i1', name: 'Get Information', count: 4520, successRate: 94, avgHandleTime: 180, trend: 3.2 },
  { id: 'i2', name: 'Report Issue', count: 2890, successRate: 88, avgHandleTime: 420, trend: -1.5 },
  { id: 'i3', name: 'Make Purchase', count: 1560, successRate: 76, avgHandleTime: 540, trend: 8.4 },
  { id: 'i4', name: 'Cancel Service', count: 890, successRate: 92, avgHandleTime: 320, trend: -4.2 },
  { id: 'i5', name: 'Update Account', count: 780, successRate: 96, avgHandleTime: 240, trend: 1.8 },
  { id: 'i6', name: 'Request Refund', count: 650, successRate: 82, avgHandleTime: 380, trend: 2.1 },
];

const keywordsData: KeywordData[] = [
  { word: 'thank you', count: 4520, sentiment: 'positive', category: 'gratitude' },
  { word: 'helpful', count: 3890, sentiment: 'positive', category: 'satisfaction' },
  { word: 'waiting', count: 2340, sentiment: 'negative', category: 'frustration' },
  { word: 'great', count: 2120, sentiment: 'positive', category: 'satisfaction' },
  { word: 'problem', count: 1980, sentiment: 'negative', category: 'issue' },
  { word: 'excellent', count: 1560, sentiment: 'positive', category: 'satisfaction' },
  { word: 'frustrated', count: 1240, sentiment: 'negative', category: 'frustration' },
  { word: 'resolved', count: 1180, sentiment: 'positive', category: 'resolution' },
  { word: 'confused', count: 980, sentiment: 'negative', category: 'confusion' },
  { word: 'appreciate', count: 920, sentiment: 'positive', category: 'gratitude' },
  { word: 'issue', count: 890, sentiment: 'neutral', category: 'issue' },
  { word: 'quickly', count: 780, sentiment: 'positive', category: 'efficiency' },
];

const patternsData: ConversationPattern[] = [
  {
    id: 'p1',
    pattern: 'Greeting → Issue Description → Solution → Confirmation → Closing',
    frequency: 45,
    outcome: 'success',
    avgDuration: 240,
    recommendation: 'This is your most effective pattern. Consider training more agents on this flow.'
  },
  {
    id: 'p2',
    pattern: 'Greeting → Multiple Clarifications → Transfer → Resolution',
    frequency: 18,
    outcome: 'escalation',
    avgDuration: 480,
    recommendation: 'Improve initial intent detection to reduce clarification loops.'
  },
  {
    id: 'p3',
    pattern: 'Greeting → Long Hold → Customer Hangup',
    frequency: 8,
    outcome: 'dropout',
    avgDuration: 320,
    recommendation: 'Implement callback option when wait time exceeds 2 minutes.'
  },
  {
    id: 'p4',
    pattern: 'Greeting → FAQ Response → Quick Resolution',
    frequency: 22,
    outcome: 'success',
    avgDuration: 120,
    recommendation: 'Expand FAQ coverage to handle more queries automatically.'
  },
];

const insightsData: InsightCard[] = [
  {
    id: 'ins1',
    type: 'opportunity',
    title: 'Increase Self-Service Resolution',
    description: '34% of billing inquiries could be resolved through self-service. Implementing a billing FAQ bot could save 450+ agent hours monthly.',
    impact: 'high',
    metric: '34%',
    metricChange: 8,
    actionable: true,
    actions: ['Create billing FAQ', 'Add payment status checker', 'Implement invoice download']
  },
  {
    id: 'ins2',
    type: 'risk',
    title: 'Rising Wait Time Complaints',
    description: 'Mentions of "waiting" and "hold time" increased 23% this week. Consider adding more agents during peak hours (2-4 PM EST).',
    impact: 'high',
    metric: '+23%',
    metricChange: -23,
    actionable: true,
    actions: ['Add peak hour staffing', 'Implement callback system', 'Review queue routing']
  },
  {
    id: 'ins3',
    type: 'trend',
    title: 'Product Feature Questions Trending',
    description: 'Questions about the new dashboard feature increased 156% since launch. Consider creating a tutorial or walkthrough.',
    impact: 'medium',
    metric: '+156%',
    metricChange: 156,
    actionable: true,
    actions: ['Create video tutorial', 'Add tooltips', 'Update help docs']
  },
  {
    id: 'ins4',
    type: 'achievement',
    title: 'First Contact Resolution Improved',
    description: 'FCR rate reached 78%, up from 72% last month. Technical support team showed the biggest improvement.',
    impact: 'medium',
    metric: '78%',
    metricChange: 6,
    actionable: false
  },
];

const sampleConversations: ConversationSample[] = [
  {
    id: 'conv1',
    date: '2024-01-20T14:30:00Z',
    agent: 'AI Agent - Sarah',
    duration: 245,
    sentiment: 0.85,
    topics: ['Billing Inquiries', 'Account Management'],
    intents: ['Get Information', 'Update Account'],
    outcome: 'resolved',
    satisfaction: 5,
    transcript: [
      { speaker: 'agent', text: "Hello! Thank you for calling. How can I help you today?", timestamp: 0, sentiment: 0.9 },
      { speaker: 'customer', text: "Hi, I have a question about my recent invoice. There's a charge I don't recognize.", timestamp: 5, sentiment: 0.4 },
      { speaker: 'agent', text: "I'd be happy to help you with that. Let me pull up your account. Can you tell me the charge amount you're referring to?", timestamp: 12, sentiment: 0.85 },
      { speaker: 'customer', text: "It's $29.99 from last Tuesday.", timestamp: 20, sentiment: 0.5 },
      { speaker: 'agent', text: "I found it. That charge is for the premium add-on feature that was activated on your account. Would you like me to explain what that includes?", timestamp: 28, sentiment: 0.8 },
      { speaker: 'customer', text: "Oh, I see! Yes, I did sign up for that. I forgot. Thank you for clarifying!", timestamp: 38, sentiment: 0.95 },
      { speaker: 'agent', text: "You're welcome! Is there anything else I can help you with today?", timestamp: 45, sentiment: 0.9 },
    ]
  },
  {
    id: 'conv2',
    date: '2024-01-20T11:15:00Z',
    agent: 'AI Agent - Michael',
    duration: 480,
    sentiment: 0.45,
    topics: ['Technical Support', 'Complaints'],
    intents: ['Report Issue'],
    outcome: 'escalated',
    satisfaction: 2,
    transcript: [
      { speaker: 'agent', text: "Hello! Thank you for calling technical support. How can I assist you?", timestamp: 0, sentiment: 0.9 },
      { speaker: 'customer', text: "This is so frustrating! Your app hasn't been working for two days now!", timestamp: 5, sentiment: 0.1 },
      { speaker: 'agent', text: "I'm sorry to hear you're experiencing issues. Let me help you troubleshoot this. What error message are you seeing?", timestamp: 15, sentiment: 0.75 },
      { speaker: 'customer', text: "It just says 'connection failed' every time I try to log in.", timestamp: 25, sentiment: 0.2 },
      { speaker: 'agent', text: "I understand how frustrating that must be. Let's try a few things. Have you tried clearing your cache and cookies?", timestamp: 35, sentiment: 0.8 },
    ]
  }
];

const timeRanges = ['Today', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'Custom'];

export default function ConversationInsightsPage() {
  const [selectedTimeRange, setSelectedTimeRange] = useState('Last 7 Days');
  const [activeTab, setActiveTab] = useState<'overview' | 'topics' | 'intents' | 'keywords' | 'patterns' | 'samples'>('overview');
  const [selectedTopic, setSelectedTopic] = useState<TopicData | null>(null);
  const [selectedConversation, setSelectedConversation] = useState<ConversationSample | null>(null);
  const [showConversationDialog, setShowConversationDialog] = useState(false);
  const [sentimentFilter, setSentimentFilter] = useState<'all' | 'positive' | 'neutral' | 'negative'>('all');

  const getSentimentColor = (sentiment: number) => {
    if (sentiment >= 0.7) return 'text-green-400';
    if (sentiment >= 0.4) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getSentimentBg = (sentiment: number) => {
    if (sentiment >= 0.7) return 'bg-green-500/20';
    if (sentiment >= 0.4) return 'bg-yellow-500/20';
    return 'bg-red-500/20';
  };

  const getSentimentIcon = (sentiment: number) => {
    if (sentiment >= 0.7) return Smile;
    if (sentiment >= 0.4) return Meh;
    return Frown;
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Sentiment Donut Chart
  const SentimentDonut = () => {
    const total = sentimentData.positive + sentimentData.neutral + sentimentData.negative;
    const positiveAngle = (sentimentData.positive / total) * 360;
    const neutralAngle = (sentimentData.neutral / total) * 360;

    return (
      <div className="relative w-48 h-48 mx-auto">
        <svg viewBox="0 0 100 100" className="transform -rotate-90">
          {/* Negative (background) */}
          <circle
            cx="50" cy="50" r="40"
            fill="transparent"
            stroke="#EF4444"
            strokeWidth="12"
            strokeDasharray="251.2"
            strokeDashoffset="0"
            className="opacity-30"
          />
          {/* Neutral */}
          <circle
            cx="50" cy="50" r="40"
            fill="transparent"
            stroke="#EAB308"
            strokeWidth="12"
            strokeDasharray={`${(sentimentData.neutral / total) * 251.2} 251.2`}
            strokeDashoffset={`-${(sentimentData.positive / total) * 251.2}`}
            className="opacity-60"
          />
          {/* Positive */}
          <circle
            cx="50" cy="50" r="40"
            fill="transparent"
            stroke="#22C55E"
            strokeWidth="12"
            strokeDasharray={`${(sentimentData.positive / total) * 251.2} 251.2`}
            strokeDashoffset="0"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <Smile className="w-8 h-8 text-green-400 mb-1" />
          <span className="text-2xl font-bold text-white">{sentimentData.positive}%</span>
          <span className="text-xs text-gray-500">Positive</span>
        </div>
      </div>
    );
  };

  // Topic Bar Component
  const TopicBar = ({ topic }: { topic: TopicData }) => {
    const SentimentIcon = getSentimentIcon(topic.sentiment);
    return (
      <div
        className="p-4 bg-gray-800/50 rounded-xl border border-gray-700 hover:border-purple-500/50 transition-all cursor-pointer"
        onClick={() => setSelectedTopic(topic)}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg ${getSentimentBg(topic.sentiment)} flex items-center justify-center`}>
              <SentimentIcon className={`w-5 h-5 ${getSentimentColor(topic.sentiment)}`} />
            </div>
            <div>
              <h4 className="font-medium text-white">{topic.name}</h4>
              <p className="text-sm text-gray-400">{topic.count.toLocaleString()} conversations</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold text-white">{topic.percentage}%</span>
            {topic.trend === 'up' && <TrendingUp className="w-4 h-4 text-green-400" />}
            {topic.trend === 'down' && <TrendingDown className="w-4 h-4 text-red-400" />}
            {topic.trend === 'stable' && <Minus className="w-4 h-4 text-gray-400" />}
          </div>
        </div>
        <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
            style={{ width: `${topic.percentage}%` }}
          />
        </div>
        <div className="flex flex-wrap gap-2 mt-3">
          {topic.examples.map(ex => (
            <span key={ex} className="px-2 py-0.5 bg-gray-700/50 rounded text-xs text-gray-400">
              {ex}
            </span>
          ))}
        </div>
      </div>
    );
  };

  // Intent Card Component
  const IntentCard = ({ intent }: { intent: IntentData }) => (
    <div className="p-4 bg-gray-800/50 rounded-xl border border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-medium text-white">{intent.name}</h4>
        <div className={`flex items-center gap-1 text-sm ${intent.trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {intent.trend >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          {Math.abs(intent.trend)}%
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-lg font-semibold text-white">{intent.count.toLocaleString()}</p>
          <p className="text-xs text-gray-500">Total</p>
        </div>
        <div>
          <p className="text-lg font-semibold text-green-400">{intent.successRate}%</p>
          <p className="text-xs text-gray-500">Success</p>
        </div>
        <div>
          <p className="text-lg font-semibold text-white">{formatDuration(intent.avgHandleTime)}</p>
          <p className="text-xs text-gray-500">Avg Time</p>
        </div>
      </div>
    </div>
  );

  // Keyword Cloud Component
  const KeywordCloud = () => {
    const maxCount = Math.max(...keywordsData.map(k => k.count));

    return (
      <div className="flex flex-wrap gap-3 justify-center p-6">
        {keywordsData.map(keyword => {
          const size = 0.8 + (keyword.count / maxCount) * 0.8;
          const colorClass = keyword.sentiment === 'positive' ? 'text-green-400' :
                            keyword.sentiment === 'negative' ? 'text-red-400' : 'text-gray-400';
          return (
            <span
              key={keyword.word}
              className={`${colorClass} hover:opacity-80 cursor-pointer transition-opacity`}
              style={{ fontSize: `${size}rem` }}
              title={`${keyword.count} occurrences`}
            >
              {keyword.word}
            </span>
          );
        })}
      </div>
    );
  };

  // Insight Card Component
  const InsightCardComponent = ({ insight }: { insight: InsightCard }) => {
    const getTypeIcon = () => {
      switch (insight.type) {
        case 'opportunity': return Lightbulb;
        case 'risk': return AlertTriangle;
        case 'trend': return TrendingUp;
        case 'achievement': return Star;
        default: return Info;
      }
    };

    const getTypeColor = () => {
      switch (insight.type) {
        case 'opportunity': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
        case 'risk': return 'bg-red-500/20 text-red-400 border-red-500/30';
        case 'trend': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
        case 'achievement': return 'bg-green-500/20 text-green-400 border-green-500/30';
        default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
      }
    };

    const Icon = getTypeIcon();

    return (
      <div className={`p-4 rounded-xl border ${getTypeColor()}`}>
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getTypeColor().split(' ')[0]}`}>
              <Icon className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-medium text-white">{insight.title}</h4>
              <span className={`text-xs px-2 py-0.5 rounded capitalize ${getTypeColor()}`}>
                {insight.impact} impact
              </span>
            </div>
          </div>
          {insight.metric && (
            <div className="text-right">
              <span className="text-xl font-bold text-white">{insight.metric}</span>
              {insight.metricChange !== undefined && (
                <div className={`flex items-center justify-end gap-1 text-sm ${
                  insight.metricChange >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {insight.metricChange >= 0 ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                  {Math.abs(insight.metricChange)}%
                </div>
              )}
            </div>
          )}
        </div>
        <p className="text-sm text-gray-400 mb-3">{insight.description}</p>
        {insight.actionable && insight.actions && (
          <div className="space-y-2">
            <p className="text-xs text-gray-500 uppercase">Suggested Actions</p>
            <div className="flex flex-wrap gap-2">
              {insight.actions.map(action => (
                <button
                  key={action}
                  className="px-3 py-1.5 bg-gray-700/50 hover:bg-gray-700 rounded-lg text-xs text-gray-300 transition-colors"
                >
                  {action}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Pattern Card Component
  const PatternCard = ({ pattern }: { pattern: ConversationPattern }) => {
    const outcomeConfig = {
      success: { color: 'bg-green-500/20 text-green-400 border-green-500/30', icon: CheckCircle },
      escalation: { color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30', icon: AlertTriangle },
      dropout: { color: 'bg-red-500/20 text-red-400 border-red-500/30', icon: XCircle }
    };

    const config = outcomeConfig[pattern.outcome];
    const Icon = config.icon;

    return (
      <div className="p-4 bg-gray-800/50 rounded-xl border border-gray-700">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${config.color.split(' ')[0]}`}>
              <Icon className={`w-5 h-5 ${config.color.split(' ')[1]}`} />
            </div>
            <div>
              <span className={`text-xs px-2 py-0.5 rounded capitalize ${config.color}`}>
                {pattern.outcome}
              </span>
              <p className="text-sm text-gray-400 mt-1">{pattern.frequency}% of conversations</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-lg font-semibold text-white">{formatDuration(pattern.avgDuration)}</p>
            <p className="text-xs text-gray-500">Avg duration</p>
          </div>
        </div>
        <div className="mb-3 p-3 bg-gray-900/50 rounded-lg">
          <p className="text-sm text-white font-mono">{pattern.pattern}</p>
        </div>
        <div className="flex items-start gap-2 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <Lightbulb className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-blue-400">{pattern.recommendation}</p>
        </div>
      </div>
    );
  };

  // Conversation Sample Card
  const ConversationSampleCard = ({ conversation }: { conversation: ConversationSample }) => {
    const SentimentIcon = getSentimentIcon(conversation.sentiment);

    return (
      <div
        className="p-4 bg-gray-800/50 rounded-xl border border-gray-700 hover:border-purple-500/50 transition-all cursor-pointer"
        onClick={() => {
          setSelectedConversation(conversation);
          setShowConversationDialog(true);
        }}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg ${getSentimentBg(conversation.sentiment)} flex items-center justify-center`}>
              <SentimentIcon className={`w-5 h-5 ${getSentimentColor(conversation.sentiment)}`} />
            </div>
            <div>
              <p className="font-medium text-white">{conversation.agent}</p>
              <p className="text-sm text-gray-400">{formatDate(conversation.date)}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={`px-2 py-1 rounded text-xs capitalize ${
              conversation.outcome === 'resolved' ? 'bg-green-500/20 text-green-400' :
              conversation.outcome === 'escalated' ? 'bg-yellow-500/20 text-yellow-400' :
              'bg-red-500/20 text-red-400'
            }`}>
              {conversation.outcome}
            </span>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4 mb-3">
          <div>
            <p className="text-lg font-semibold text-white">{formatDuration(conversation.duration)}</p>
            <p className="text-xs text-gray-500">Duration</p>
          </div>
          <div>
            <p className="text-lg font-semibold text-white">{Math.round(conversation.sentiment * 100)}%</p>
            <p className="text-xs text-gray-500">Sentiment</p>
          </div>
          <div>
            <div className="flex items-center gap-1">
              {Array.from({ length: 5 }).map((_, i) => (
                <Star
                  key={i}
                  className={`w-4 h-4 ${i < conversation.satisfaction ? 'text-yellow-400 fill-yellow-400' : 'text-gray-600'}`}
                />
              ))}
            </div>
            <p className="text-xs text-gray-500">CSAT</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {conversation.topics.map(topic => (
            <span key={topic} className="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs">
              {topic}
            </span>
          ))}
          {conversation.intents.map(intent => (
            <span key={intent} className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded text-xs">
              {intent}
            </span>
          ))}
        </div>
      </div>
    );
  };

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">Conversation Insights</h1>
            <p className="text-gray-400">AI-powered analysis of customer conversations</p>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={selectedTimeRange}
              onChange={(e) => setSelectedTimeRange(e.target.value)}
              className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white"
            >
              {timeRanges.map(range => (
                <option key={range} value={range}>{range}</option>
              ))}
            </select>
            <button className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-2">
              <Download className="w-5 h-5" />
              Export
            </button>
            <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90 transition-opacity flex items-center gap-2">
              <RefreshCw className="w-5 h-5" />
              Refresh
            </button>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Total Conversations</span>
              <MessageSquare className="w-5 h-5 text-purple-400" />
            </div>
            <p className="text-2xl font-bold text-white">11,690</p>
            <div className="flex items-center gap-1 text-sm text-green-400">
              <TrendingUp className="w-4 h-4" />
              <span>+12.3%</span>
            </div>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Avg Sentiment</span>
              <Smile className="w-5 h-5 text-green-400" />
            </div>
            <p className="text-2xl font-bold text-white">{sentimentData.positive}%</p>
            <div className="flex items-center gap-1 text-sm text-green-400">
              <TrendingUp className="w-4 h-4" />
              <span>+{sentimentData.trend}%</span>
            </div>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Resolution Rate</span>
              <CheckCircle className="w-5 h-5 text-green-400" />
            </div>
            <p className="text-2xl font-bold text-white">78%</p>
            <div className="flex items-center gap-1 text-sm text-green-400">
              <TrendingUp className="w-4 h-4" />
              <span>+6%</span>
            </div>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Avg Handle Time</span>
              <Clock className="w-5 h-5 text-blue-400" />
            </div>
            <p className="text-2xl font-bold text-white">4:32</p>
            <div className="flex items-center gap-1 text-sm text-green-400">
              <TrendingDown className="w-4 h-4" />
              <span>-18s</span>
            </div>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">CSAT Score</span>
              <Star className="w-5 h-5 text-yellow-400" />
            </div>
            <p className="text-2xl font-bold text-white">4.6</p>
            <div className="flex items-center gap-1 text-sm text-green-400">
              <TrendingUp className="w-4 h-4" />
              <span>+0.3</span>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-1 p-1 bg-gray-800/50 rounded-lg w-fit">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'topics', label: 'Topics', icon: Tag },
            { id: 'intents', label: 'Intents', icon: Target },
            { id: 'keywords', label: 'Keywords', icon: Hash },
            { id: 'patterns', label: 'Patterns', icon: Activity },
            { id: 'samples', label: 'Samples', icon: MessageSquare }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                activeTab === tab.id
                  ? 'bg-purple-500 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Sentiment Distribution */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6">
              <h3 className="text-lg font-medium text-white mb-6">Sentiment Distribution</h3>
              <SentimentDonut />
              <div className="grid grid-cols-3 gap-4 mt-6">
                <div className="text-center">
                  <div className="flex items-center justify-center gap-2 mb-1">
                    <div className="w-3 h-3 rounded-full bg-green-500" />
                    <span className="text-sm text-gray-400">Positive</span>
                  </div>
                  <p className="text-lg font-semibold text-white">{sentimentData.positive}%</p>
                </div>
                <div className="text-center">
                  <div className="flex items-center justify-center gap-2 mb-1">
                    <div className="w-3 h-3 rounded-full bg-yellow-500" />
                    <span className="text-sm text-gray-400">Neutral</span>
                  </div>
                  <p className="text-lg font-semibold text-white">{sentimentData.neutral}%</p>
                </div>
                <div className="text-center">
                  <div className="flex items-center justify-center gap-2 mb-1">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <span className="text-sm text-gray-400">Negative</span>
                  </div>
                  <p className="text-lg font-semibold text-white">{sentimentData.negative}%</p>
                </div>
              </div>
            </div>

            {/* Top Topics */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6">
              <h3 className="text-lg font-medium text-white mb-4">Top Topics</h3>
              <div className="space-y-3">
                {topicsData.slice(0, 5).map((topic, index) => (
                  <div key={topic.id} className="flex items-center gap-3">
                    <span className="w-6 h-6 rounded-full bg-purple-500/20 text-purple-400 text-xs flex items-center justify-center">
                      {index + 1}
                    </span>
                    <div className="flex-grow">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-white">{topic.name}</span>
                        <span className="text-sm text-gray-400">{topic.percentage}%</span>
                      </div>
                      <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                          style={{ width: `${topic.percentage}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* AI Insights */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-5 h-5 text-purple-400" />
                <h3 className="text-lg font-medium text-white">AI Insights</h3>
                <Sparkles className="w-4 h-4 text-yellow-400" />
              </div>
              <div className="space-y-3">
                {insightsData.slice(0, 3).map(insight => (
                  <InsightCardComponent key={insight.id} insight={insight} />
                ))}
              </div>
            </div>

            {/* Keyword Cloud */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6 lg:col-span-2">
              <h3 className="text-lg font-medium text-white mb-4">Keyword Cloud</h3>
              <KeywordCloud />
              <div className="flex items-center justify-center gap-6 mt-4 pt-4 border-t border-gray-700">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                  <span className="text-sm text-gray-400">Positive</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-gray-500" />
                  <span className="text-sm text-gray-400">Neutral</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500" />
                  <span className="text-sm text-gray-400">Negative</span>
                </div>
              </div>
            </div>

            {/* Top Intents */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6">
              <h3 className="text-lg font-medium text-white mb-4">Top Intents</h3>
              <div className="space-y-3">
                {intentsData.slice(0, 4).map(intent => (
                  <div key={intent.id} className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                    <div>
                      <p className="text-sm text-white">{intent.name}</p>
                      <p className="text-xs text-gray-500">{intent.count.toLocaleString()} calls</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-semibold text-green-400">{intent.successRate}%</p>
                      <p className="text-xs text-gray-500">success</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Topics Tab */}
        {activeTab === 'topics' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {topicsData.map(topic => (
              <TopicBar key={topic.id} topic={topic} />
            ))}
          </div>
        )}

        {/* Intents Tab */}
        {activeTab === 'intents' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {intentsData.map(intent => (
              <IntentCard key={intent.id} intent={intent} />
            ))}
          </div>
        )}

        {/* Keywords Tab */}
        {activeTab === 'keywords' && (
          <div className="space-y-6">
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6">
              <h3 className="text-lg font-medium text-white mb-4">Keyword Analysis</h3>
              <KeywordCloud />
            </div>
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left p-4 text-gray-400 font-medium">Keyword</th>
                    <th className="text-left p-4 text-gray-400 font-medium">Count</th>
                    <th className="text-left p-4 text-gray-400 font-medium">Sentiment</th>
                    <th className="text-left p-4 text-gray-400 font-medium">Category</th>
                  </tr>
                </thead>
                <tbody>
                  {keywordsData.map(keyword => (
                    <tr key={keyword.word} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                      <td className="p-4 text-white font-medium">{keyword.word}</td>
                      <td className="p-4 text-gray-400">{keyword.count.toLocaleString()}</td>
                      <td className="p-4">
                        <span className={`px-2 py-1 rounded text-xs capitalize ${
                          keyword.sentiment === 'positive' ? 'bg-green-500/20 text-green-400' :
                          keyword.sentiment === 'negative' ? 'bg-red-500/20 text-red-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {keyword.sentiment}
                        </span>
                      </td>
                      <td className="p-4 text-gray-400 capitalize">{keyword.category}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Patterns Tab */}
        {activeTab === 'patterns' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {patternsData.map(pattern => (
              <PatternCard key={pattern.id} pattern={pattern} />
            ))}
          </div>
        )}

        {/* Samples Tab */}
        {activeTab === 'samples' && (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="relative flex-grow max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search conversations..."
                  className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div className="flex items-center gap-2">
                {['all', 'positive', 'neutral', 'negative'].map(filter => (
                  <button
                    key={filter}
                    onClick={() => setSentimentFilter(filter as any)}
                    className={`px-3 py-1.5 rounded-lg text-sm capitalize transition-all ${
                      sentimentFilter === filter
                        ? 'bg-purple-500 text-white'
                        : 'bg-gray-800/50 text-gray-400 hover:text-white border border-gray-700'
                    }`}
                  >
                    {filter}
                  </button>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {sampleConversations.map(conversation => (
                <ConversationSampleCard key={conversation.id} conversation={conversation} />
              ))}
            </div>
          </div>
        )}

        {/* Conversation Detail Dialog */}
        {showConversationDialog && selectedConversation && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`w-12 h-12 rounded-xl ${getSentimentBg(selectedConversation.sentiment)} flex items-center justify-center`}>
                      {React.createElement(getSentimentIcon(selectedConversation.sentiment), {
                        className: `w-6 h-6 ${getSentimentColor(selectedConversation.sentiment)}`
                      })}
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white">{selectedConversation.agent}</h2>
                      <p className="text-gray-400">{formatDate(selectedConversation.date)}</p>
                    </div>
                  </div>
                  <button onClick={() => setShowConversationDialog(false)} className="p-2 hover:bg-gray-700 rounded-lg">
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              <div className="p-6 overflow-y-auto max-h-[60vh]">
                {/* Metrics */}
                <div className="grid grid-cols-4 gap-4 mb-6">
                  <div className="bg-gray-700/30 rounded-xl p-3 text-center">
                    <p className="text-lg font-semibold text-white">{formatDuration(selectedConversation.duration)}</p>
                    <p className="text-xs text-gray-500">Duration</p>
                  </div>
                  <div className="bg-gray-700/30 rounded-xl p-3 text-center">
                    <p className={`text-lg font-semibold ${getSentimentColor(selectedConversation.sentiment)}`}>
                      {Math.round(selectedConversation.sentiment * 100)}%
                    </p>
                    <p className="text-xs text-gray-500">Sentiment</p>
                  </div>
                  <div className="bg-gray-700/30 rounded-xl p-3 text-center">
                    <span className={`px-2 py-1 rounded text-sm capitalize ${
                      selectedConversation.outcome === 'resolved' ? 'bg-green-500/20 text-green-400' :
                      selectedConversation.outcome === 'escalated' ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {selectedConversation.outcome}
                    </span>
                    <p className="text-xs text-gray-500 mt-1">Outcome</p>
                  </div>
                  <div className="bg-gray-700/30 rounded-xl p-3 text-center">
                    <div className="flex items-center justify-center gap-0.5">
                      {Array.from({ length: 5 }).map((_, i) => (
                        <Star
                          key={i}
                          className={`w-4 h-4 ${i < selectedConversation.satisfaction ? 'text-yellow-400 fill-yellow-400' : 'text-gray-600'}`}
                        />
                      ))}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">CSAT</p>
                  </div>
                </div>

                {/* Tags */}
                <div className="flex flex-wrap gap-2 mb-6">
                  {selectedConversation.topics.map(topic => (
                    <span key={topic} className="px-3 py-1 bg-purple-500/20 text-purple-400 rounded-lg text-sm">
                      {topic}
                    </span>
                  ))}
                  {selectedConversation.intents.map(intent => (
                    <span key={intent} className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-lg text-sm">
                      {intent}
                    </span>
                  ))}
                </div>

                {/* Transcript */}
                <h3 className="text-lg font-medium text-white mb-4">Transcript</h3>
                <div className="space-y-4">
                  {selectedConversation.transcript.map((message, index) => (
                    <div
                      key={index}
                      className={`flex gap-3 ${message.speaker === 'agent' ? '' : 'flex-row-reverse'}`}
                    >
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                        message.speaker === 'agent' ? 'bg-purple-500/20' : 'bg-blue-500/20'
                      }`}>
                        {message.speaker === 'agent' ? (
                          <Bot className="w-4 h-4 text-purple-400" />
                        ) : (
                          <Users className="w-4 h-4 text-blue-400" />
                        )}
                      </div>
                      <div className={`max-w-[70%] ${message.speaker === 'agent' ? '' : 'text-right'}`}>
                        <div className={`p-3 rounded-xl ${
                          message.speaker === 'agent' ? 'bg-gray-700/50' : 'bg-blue-500/20'
                        }`}>
                          <p className="text-white text-sm">{message.text}</p>
                        </div>
                        <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
                          <span>{formatDuration(message.timestamp)}</span>
                          <span>•</span>
                          <span className={getSentimentColor(message.sentiment)}>
                            {Math.round(message.sentiment * 100)}% sentiment
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-6 border-t border-gray-700 flex justify-between">
                <button className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-2">
                  <Play className="w-4 h-4" />
                  Play Recording
                </button>
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2">
                  <ExternalLink className="w-4 h-4" />
                  View Full Details
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
