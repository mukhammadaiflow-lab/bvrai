"use client";

import React, { useState, useMemo } from "react";
import DashboardLayout from "@/components/DashboardLayout";

// Types
type FeedbackType = "rating" | "survey" | "review" | "suggestion" | "complaint" | "praise";
type FeedbackStatus = "new" | "in_review" | "resolved" | "archived" | "escalated";
type FeedbackPriority = "low" | "medium" | "high" | "critical";
type SentimentType = "positive" | "neutral" | "negative" | "mixed";
type SurveyQuestionType = "rating" | "nps" | "text" | "multiple_choice" | "checkbox" | "scale";

interface FeedbackResponse {
  id: string;
  questionId: string;
  answer: string | number | string[];
}

interface FeedbackItem {
  id: string;
  type: FeedbackType;
  status: FeedbackStatus;
  priority: FeedbackPriority;
  customerId: string;
  customerName: string;
  customerEmail: string;
  customerPhone?: string;
  customerAvatar?: string;
  agentId: string;
  agentName: string;
  callId?: string;
  rating?: number;
  npsScore?: number;
  sentiment: SentimentType;
  title: string;
  content: string;
  responses?: FeedbackResponse[];
  tags: string[];
  createdAt: string;
  updatedAt: string;
  resolvedAt?: string;
  resolvedBy?: string;
  internalNotes?: string[];
  followUpRequired: boolean;
  followUpDate?: string;
  source: "call" | "email" | "web" | "sms" | "chat";
}

interface SurveyQuestion {
  id: string;
  type: SurveyQuestionType;
  question: string;
  required: boolean;
  options?: string[];
  minValue?: number;
  maxValue?: number;
  minLabel?: string;
  maxLabel?: string;
}

interface Survey {
  id: string;
  title: string;
  description: string;
  questions: SurveyQuestion[];
  isActive: boolean;
  responseCount: number;
  averageCompletion: number;
  createdAt: string;
  triggerCondition: "after_call" | "after_resolution" | "scheduled" | "manual";
}

interface FeedbackMetrics {
  totalFeedback: number;
  newFeedback: number;
  avgRating: number;
  npsScore: number;
  responseRate: number;
  avgResolutionTime: number;
  positiveSentiment: number;
  negativeSentiment: number;
  pendingFollowUps: number;
  escalatedCount: number;
  skillsAcquired: number;
}

interface TrendDataPoint {
  date: string;
  rating: number;
  nps: number;
  volume: number;
  sentiment: number;
}

// Mock Data
const mockFeedback: FeedbackItem[] = [
  {
    id: "fb-1",
    type: "rating",
    status: "new",
    priority: "high",
    customerId: "cust-1",
    customerName: "John Anderson",
    customerEmail: "john.anderson@email.com",
    customerPhone: "+1 (555) 123-4567",
    agentId: "agent-1",
    agentName: "Sales Bot Pro",
    callId: "call-12345",
    rating: 5,
    npsScore: 10,
    sentiment: "positive",
    title: "Excellent service experience",
    content: "The AI agent was incredibly helpful and resolved my issue quickly. Very impressed with how natural the conversation felt. Would definitely recommend to others!",
    tags: ["excellent-service", "quick-resolution", "natural-conversation"],
    createdAt: "2024-01-20T10:30:00Z",
    updatedAt: "2024-01-20T10:30:00Z",
    followUpRequired: false,
    source: "call",
  },
  {
    id: "fb-2",
    type: "complaint",
    status: "escalated",
    priority: "critical",
    customerId: "cust-2",
    customerName: "Sarah Miller",
    customerEmail: "sarah.miller@company.com",
    customerPhone: "+1 (555) 234-5678",
    agentId: "agent-2",
    agentName: "Support Master",
    callId: "call-12346",
    rating: 2,
    npsScore: 2,
    sentiment: "negative",
    title: "Long wait time and unresolved issue",
    content: "I had to wait for over 10 minutes before the agent could help me, and even then my issue wasn't fully resolved. I was promised a callback that never happened. Very disappointed with the service.",
    tags: ["long-wait", "unresolved", "missed-callback"],
    createdAt: "2024-01-20T09:15:00Z",
    updatedAt: "2024-01-20T14:30:00Z",
    followUpRequired: true,
    followUpDate: "2024-01-21T10:00:00Z",
    source: "call",
    internalNotes: ["Escalated to supervisor", "Customer requested manager callback"],
  },
  {
    id: "fb-3",
    type: "suggestion",
    status: "in_review",
    priority: "medium",
    customerId: "cust-3",
    customerName: "Michael Chen",
    customerEmail: "m.chen@techcorp.io",
    agentId: "agent-1",
    agentName: "Sales Bot Pro",
    sentiment: "neutral",
    title: "Feature suggestion for appointment booking",
    content: "It would be great if the agent could offer multiple time slots at once instead of suggesting them one by one. This would save time during the scheduling process.",
    tags: ["feature-request", "appointment-booking", "efficiency"],
    createdAt: "2024-01-19T16:45:00Z",
    updatedAt: "2024-01-20T08:00:00Z",
    followUpRequired: false,
    source: "email",
  },
  {
    id: "fb-4",
    type: "review",
    status: "resolved",
    priority: "low",
    customerId: "cust-4",
    customerName: "Emily Watson",
    customerEmail: "emily.w@gmail.com",
    customerPhone: "+1 (555) 345-6789",
    agentId: "agent-3",
    agentName: "Customer Care AI",
    callId: "call-12347",
    rating: 4,
    npsScore: 8,
    sentiment: "positive",
    title: "Good service, minor improvements needed",
    content: "Overall a good experience. The agent understood my request well but took a bit longer than expected to process my refund. Still, would use the service again.",
    tags: ["good-service", "processing-time", "refund"],
    createdAt: "2024-01-19T11:20:00Z",
    updatedAt: "2024-01-19T15:00:00Z",
    resolvedAt: "2024-01-19T15:00:00Z",
    resolvedBy: "admin-1",
    followUpRequired: false,
    source: "web",
  },
  {
    id: "fb-5",
    type: "praise",
    status: "archived",
    priority: "low",
    customerId: "cust-5",
    customerName: "David Thompson",
    customerEmail: "d.thompson@business.net",
    agentId: "agent-1",
    agentName: "Sales Bot Pro",
    callId: "call-12348",
    rating: 5,
    npsScore: 10,
    sentiment: "positive",
    title: "Best AI agent experience ever!",
    content: "I've interacted with many AI assistants but this one is by far the best. It understood exactly what I needed, anticipated my follow-up questions, and completed everything efficiently. Kudos to the team!",
    tags: ["best-experience", "efficient", "anticipatory"],
    createdAt: "2024-01-18T14:30:00Z",
    updatedAt: "2024-01-18T16:00:00Z",
    resolvedAt: "2024-01-18T16:00:00Z",
    followUpRequired: false,
    source: "call",
  },
  {
    id: "fb-6",
    type: "survey",
    status: "new",
    priority: "medium",
    customerId: "cust-6",
    customerName: "Lisa Brown",
    customerEmail: "lisa.brown@email.com",
    agentId: "agent-2",
    agentName: "Support Master",
    callId: "call-12349",
    rating: 3,
    npsScore: 6,
    sentiment: "mixed",
    title: "Survey Response",
    content: "The agent was polite but couldn't help with my specific technical issue. Had to be transferred twice before getting the right assistance.",
    responses: [
      { id: "r1", questionId: "q1", answer: 3 },
      { id: "r2", questionId: "q2", answer: 6 },
      { id: "r3", questionId: "q3", answer: ["transferred-multiple-times", "technical-limitation"] },
      { id: "r4", questionId: "q4", answer: "Better technical training for agents" },
    ],
    tags: ["transfer-issue", "technical-limitation"],
    createdAt: "2024-01-20T08:45:00Z",
    updatedAt: "2024-01-20T08:45:00Z",
    followUpRequired: true,
    followUpDate: "2024-01-22T09:00:00Z",
    source: "sms",
  },
  {
    id: "fb-7",
    type: "complaint",
    status: "in_review",
    priority: "high",
    customerId: "cust-7",
    customerName: "Robert Garcia",
    customerEmail: "r.garcia@company.com",
    customerPhone: "+1 (555) 456-7890",
    agentId: "agent-4",
    agentName: "Tech Helper",
    callId: "call-12350",
    rating: 1,
    npsScore: 0,
    sentiment: "negative",
    title: "Completely frustrated with the service",
    content: "The agent kept repeating the same scripted responses and couldn't understand my problem at all. I had to explain the issue 5 times before any progress was made. This is unacceptable.",
    tags: ["scripted-responses", "understanding-issue", "frustration"],
    createdAt: "2024-01-20T07:00:00Z",
    updatedAt: "2024-01-20T11:30:00Z",
    followUpRequired: true,
    followUpDate: "2024-01-20T16:00:00Z",
    source: "call",
    internalNotes: ["Agent needs additional training", "Issue escalated to QA team"],
  },
  {
    id: "fb-8",
    type: "rating",
    status: "new",
    priority: "low",
    customerId: "cust-8",
    customerName: "Jennifer Lee",
    customerEmail: "j.lee@personal.com",
    agentId: "agent-3",
    agentName: "Customer Care AI",
    callId: "call-12351",
    rating: 4,
    npsScore: 7,
    sentiment: "positive",
    title: "Quick response to my inquiry",
    content: "Got my question answered quickly. The agent was professional and courteous.",
    tags: ["quick-response", "professional"],
    createdAt: "2024-01-20T06:30:00Z",
    updatedAt: "2024-01-20T06:30:00Z",
    followUpRequired: false,
    source: "chat",
  },
];

const mockSurveys: Survey[] = [
  {
    id: "survey-1",
    title: "Post-Call Satisfaction Survey",
    description: "Quick survey to measure customer satisfaction after each call",
    questions: [
      { id: "q1", type: "rating", question: "How would you rate your overall experience?", required: true, minValue: 1, maxValue: 5 },
      { id: "q2", type: "nps", question: "How likely are you to recommend our service to others?", required: true, minValue: 0, maxValue: 10 },
      { id: "q3", type: "checkbox", question: "What aspects of the service could be improved?", required: false, options: ["Wait time", "Agent knowledge", "Communication clarity", "Problem resolution", "Follow-up"] },
      { id: "q4", type: "text", question: "Any additional comments or suggestions?", required: false },
    ],
    isActive: true,
    responseCount: 1247,
    averageCompletion: 78,
    createdAt: "2024-01-01T00:00:00Z",
    triggerCondition: "after_call",
  },
  {
    id: "survey-2",
    title: "Resolution Quality Survey",
    description: "Survey sent after issue resolution to measure quality",
    questions: [
      { id: "q1", type: "rating", question: "How satisfied are you with the resolution?", required: true, minValue: 1, maxValue: 5 },
      { id: "q2", type: "multiple_choice", question: "Was your issue fully resolved?", required: true, options: ["Yes, completely", "Partially resolved", "Not resolved", "Need more time to verify"] },
      { id: "q3", type: "scale", question: "How much effort did you have to put in to resolve your issue?", required: true, minValue: 1, maxValue: 7, minLabel: "Very low effort", maxLabel: "Very high effort" },
      { id: "q4", type: "text", question: "What could we have done better?", required: false },
    ],
    isActive: true,
    responseCount: 892,
    averageCompletion: 65,
    createdAt: "2024-01-05T00:00:00Z",
    triggerCondition: "after_resolution",
  },
  {
    id: "survey-3",
    title: "Monthly Service Review",
    description: "Comprehensive monthly survey for premium customers",
    questions: [
      { id: "q1", type: "nps", question: "Based on your interactions this month, how likely are you to recommend us?", required: true, minValue: 0, maxValue: 10 },
      { id: "q2", type: "rating", question: "Rate the overall quality of our AI agents", required: true, minValue: 1, maxValue: 5 },
      { id: "q3", type: "rating", question: "Rate the speed of service", required: true, minValue: 1, maxValue: 5 },
      { id: "q4", type: "rating", question: "Rate the accuracy of information provided", required: true, minValue: 1, maxValue: 5 },
      { id: "q5", type: "checkbox", question: "Which features do you use most?", required: false, options: ["Phone support", "Chat support", "Appointment booking", "Order tracking", "Technical support", "Billing inquiries"] },
      { id: "q6", type: "text", question: "Share your detailed feedback", required: false },
    ],
    isActive: false,
    responseCount: 234,
    averageCompletion: 52,
    createdAt: "2024-01-10T00:00:00Z",
    triggerCondition: "scheduled",
  },
];

const mockMetrics: FeedbackMetrics = {
  totalFeedback: 1892,
  newFeedback: 47,
  avgRating: 4.2,
  npsScore: 42,
  responseRate: 68,
  avgResolutionTime: 4.5,
  positiveSentiment: 72,
  negativeSentiment: 15,
  pendingFollowUps: 12,
  escalatedCount: 3,
  skillsAcquired: 156,
};

const mockTrendData: TrendDataPoint[] = [
  { date: "Jan 14", rating: 4.1, nps: 38, volume: 145, sentiment: 68 },
  { date: "Jan 15", rating: 4.2, nps: 41, volume: 162, sentiment: 71 },
  { date: "Jan 16", rating: 4.0, nps: 39, volume: 138, sentiment: 65 },
  { date: "Jan 17", rating: 4.3, nps: 44, volume: 175, sentiment: 74 },
  { date: "Jan 18", rating: 4.2, nps: 42, volume: 168, sentiment: 72 },
  { date: "Jan 19", rating: 4.4, nps: 45, volume: 182, sentiment: 76 },
  { date: "Jan 20", rating: 4.2, nps: 42, volume: 156, sentiment: 72 },
];

const feedbackCategories = [
  { type: "rating", label: "Ratings", icon: "⭐", color: "yellow" },
  { type: "survey", label: "Surveys", icon: "📋", color: "blue" },
  { type: "review", label: "Reviews", icon: "📝", color: "purple" },
  { type: "suggestion", label: "Suggestions", icon: "💡", color: "cyan" },
  { type: "complaint", label: "Complaints", icon: "⚠️", color: "red" },
  { type: "praise", label: "Praise", icon: "🌟", color: "green" },
];

// Components
const MetricCard: React.FC<{
  title: string;
  value: string | number;
  icon: string;
  trend?: number;
  subtitle?: string;
  color?: string;
}> = ({ title, value, icon, trend, subtitle, color = "purple" }) => {
  const colorClasses: Record<string, string> = {
    purple: "from-purple-500/20 to-pink-500/20",
    green: "from-green-500/20 to-emerald-500/20",
    blue: "from-blue-500/20 to-cyan-500/20",
    yellow: "from-yellow-500/20 to-orange-500/20",
    red: "from-red-500/20 to-rose-500/20",
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-gray-400 text-sm">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {subtitle && <p className="text-gray-500 text-xs mt-1">{subtitle}</p>}
          {trend !== undefined && (
            <div className={`flex items-center gap-1 mt-2 text-sm ${trend >= 0 ? "text-green-400" : "text-red-400"}`}>
              <span>{trend >= 0 ? "↑" : "↓"}</span>
              <span>{Math.abs(trend)}% vs last week</span>
            </div>
          )}
        </div>
        <div className={`w-12 h-12 bg-gradient-to-br ${colorClasses[color]} rounded-xl flex items-center justify-center text-2xl`}>
          {icon}
        </div>
      </div>
    </div>
  );
};

const RatingStars: React.FC<{ rating: number; size?: "sm" | "md" | "lg" }> = ({ rating, size = "md" }) => {
  const sizeClasses = { sm: "text-sm", md: "text-lg", lg: "text-xl" };
  return (
    <div className={`flex items-center gap-0.5 ${sizeClasses[size]}`}>
      {[1, 2, 3, 4, 5].map((star) => (
        <span key={star} className={star <= rating ? "text-yellow-400" : "text-gray-600"}>
          ★
        </span>
      ))}
    </div>
  );
};

const NPSIndicator: React.FC<{ score: number }> = ({ score }) => {
  const getColor = () => {
    if (score >= 9) return "text-green-400";
    if (score >= 7) return "text-yellow-400";
    return "text-red-400";
  };

  const getLabel = () => {
    if (score >= 9) return "Promoter";
    if (score >= 7) return "Passive";
    return "Detractor";
  };

  return (
    <div className="flex items-center gap-2">
      <div className={`text-lg font-bold ${getColor()}`}>{score}</div>
      <span className={`text-xs px-2 py-0.5 rounded-full ${getColor()} bg-current/10`}>{getLabel()}</span>
    </div>
  );
};

const SentimentBadge: React.FC<{ sentiment: SentimentType }> = ({ sentiment }) => {
  const config: Record<SentimentType, { color: string; icon: string }> = {
    positive: { color: "bg-green-500/20 text-green-400 border-green-500/30", icon: "😊" },
    neutral: { color: "bg-gray-500/20 text-gray-400 border-gray-500/30", icon: "😐" },
    negative: { color: "bg-red-500/20 text-red-400 border-red-500/30", icon: "😞" },
    mixed: { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", icon: "🤔" },
  };

  const { color, icon } = config[sentiment];

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${color} capitalize`}>
      <span>{icon}</span>
      <span>{sentiment}</span>
    </span>
  );
};

const PriorityBadge: React.FC<{ priority: FeedbackPriority }> = ({ priority }) => {
  const colors: Record<FeedbackPriority, string> = {
    low: "bg-gray-500/20 text-gray-400 border-gray-500/30",
    medium: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    high: "bg-orange-500/20 text-orange-400 border-orange-500/30",
    critical: "bg-red-500/20 text-red-400 border-red-500/30",
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${colors[priority]} capitalize`}>
      {priority}
    </span>
  );
};

const StatusBadge: React.FC<{ status: FeedbackStatus }> = ({ status }) => {
  const config: Record<FeedbackStatus, { color: string; label: string }> = {
    new: { color: "bg-blue-500/20 text-blue-400 border-blue-500/30", label: "New" },
    in_review: { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", label: "In Review" },
    resolved: { color: "bg-green-500/20 text-green-400 border-green-500/30", label: "Resolved" },
    archived: { color: "bg-gray-500/20 text-gray-400 border-gray-500/30", label: "Archived" },
    escalated: { color: "bg-red-500/20 text-red-400 border-red-500/30", label: "Escalated" },
  };

  const { color, label } = config[status];

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${color}`}>{label}</span>
  );
};

const TypeBadge: React.FC<{ type: FeedbackType }> = ({ type }) => {
  const config: Record<FeedbackType, { color: string; icon: string }> = {
    rating: { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", icon: "⭐" },
    survey: { color: "bg-blue-500/20 text-blue-400 border-blue-500/30", icon: "📋" },
    review: { color: "bg-purple-500/20 text-purple-400 border-purple-500/30", icon: "📝" },
    suggestion: { color: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30", icon: "💡" },
    complaint: { color: "bg-red-500/20 text-red-400 border-red-500/30", icon: "⚠️" },
    praise: { color: "bg-green-500/20 text-green-400 border-green-500/30", icon: "🌟" },
  };

  const { color, icon } = config[type];

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${color} capitalize`}>
      <span>{icon}</span>
      <span>{type}</span>
    </span>
  );
};

const SourceIcon: React.FC<{ source: FeedbackItem["source"] }> = ({ source }) => {
  const icons: Record<FeedbackItem["source"], string> = {
    call: "📞",
    email: "✉️",
    web: "🌐",
    sms: "💬",
    chat: "💭",
  };
  return <span title={source}>{icons[source]}</span>;
};

const TrendChart: React.FC<{ data: TrendDataPoint[]; metric: "rating" | "nps" | "volume" | "sentiment" }> = ({ data, metric }) => {
  const maxValue = Math.max(...data.map((d) => d[metric]));
  const minValue = Math.min(...data.map((d) => d[metric]));
  const range = maxValue - minValue || 1;

  return (
    <div className="h-24 flex items-end gap-1">
      {data.map((point, index) => {
        const height = ((point[metric] - minValue) / range) * 100;
        const normalizedHeight = Math.max(10, height);

        return (
          <div key={index} className="flex-1 flex flex-col items-center gap-1">
            <div
              className="w-full bg-gradient-to-t from-purple-500 to-pink-500 rounded-t-sm transition-all duration-300 hover:opacity-80"
              style={{ height: `${normalizedHeight}%` }}
              title={`${point.date}: ${point[metric]}`}
            />
            <span className="text-[10px] text-gray-500 truncate max-w-full">{point.date.split(" ")[1]}</span>
          </div>
        );
      })}
    </div>
  );
};

const FeedbackCard: React.FC<{
  feedback: FeedbackItem;
  onSelect: (feedback: FeedbackItem) => void;
}> = ({ feedback, onSelect }) => {
  return (
    <div
      className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5 hover:border-purple-500/50 transition-all duration-300 cursor-pointer"
      onClick={() => onSelect(feedback)}
    >
      <div className="flex items-start gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center text-xl flex-shrink-0">
          {feedback.customerAvatar || feedback.customerName.charAt(0)}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-2">
            <TypeBadge type={feedback.type} />
            <StatusBadge status={feedback.status} />
            <PriorityBadge priority={feedback.priority} />
            <SentimentBadge sentiment={feedback.sentiment} />
          </div>
          <h3 className="font-semibold text-white truncate">{feedback.title}</h3>
          <p className="text-gray-400 text-sm mt-1 line-clamp-2">{feedback.content}</p>

          <div className="flex items-center gap-4 mt-3 text-sm text-gray-500">
            <span className="flex items-center gap-1">
              <span>👤</span>
              <span>{feedback.customerName}</span>
            </span>
            <span className="flex items-center gap-1">
              <SourceIcon source={feedback.source} />
              <span className="capitalize">{feedback.source}</span>
            </span>
            <span>{new Date(feedback.createdAt).toLocaleDateString()}</span>
          </div>

          {feedback.rating !== undefined && (
            <div className="flex items-center gap-4 mt-3">
              <RatingStars rating={feedback.rating} size="sm" />
              {feedback.npsScore !== undefined && <NPSIndicator score={feedback.npsScore} />}
            </div>
          )}

          {feedback.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-3">
              {feedback.tags.slice(0, 3).map((tag) => (
                <span key={tag} className="px-2 py-0.5 bg-gray-700/50 text-gray-400 text-xs rounded-full">
                  #{tag}
                </span>
              ))}
              {feedback.tags.length > 3 && (
                <span className="px-2 py-0.5 text-gray-500 text-xs">+{feedback.tags.length - 3} more</span>
              )}
            </div>
          )}
        </div>

        <div className="flex flex-col items-end gap-2">
          {feedback.followUpRequired && (
            <span className="px-2 py-1 bg-orange-500/20 text-orange-400 text-xs rounded-lg flex items-center gap-1">
              <span>⏰</span>
              <span>Follow-up</span>
            </span>
          )}
          <span className="text-gray-400">→</span>
        </div>
      </div>
    </div>
  );
};

const SurveyCard: React.FC<{ survey: Survey; onSelect: (survey: Survey) => void }> = ({ survey, onSelect }) => (
  <div
    className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5 hover:border-purple-500/50 transition-all duration-300 cursor-pointer"
    onClick={() => onSelect(survey)}
  >
    <div className="flex items-start justify-between mb-4">
      <div>
        <div className="flex items-center gap-2 mb-2">
          <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${
            survey.isActive
              ? "bg-green-500/20 text-green-400 border-green-500/30"
              : "bg-gray-500/20 text-gray-400 border-gray-500/30"
          }`}>
            {survey.isActive ? "Active" : "Inactive"}
          </span>
          <span className="px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded-full border border-purple-500/30 capitalize">
            {survey.triggerCondition.replace("_", " ")}
          </span>
        </div>
        <h3 className="font-semibold text-white">{survey.title}</h3>
        <p className="text-gray-400 text-sm mt-1">{survey.description}</p>
      </div>
      <span className="text-2xl">📋</span>
    </div>

    <div className="grid grid-cols-3 gap-4 mt-4">
      <div className="text-center p-3 bg-gray-700/30 rounded-lg">
        <p className="text-xl font-bold text-white">{survey.questions.length}</p>
        <p className="text-xs text-gray-400">Questions</p>
      </div>
      <div className="text-center p-3 bg-gray-700/30 rounded-lg">
        <p className="text-xl font-bold text-white">{survey.responseCount.toLocaleString()}</p>
        <p className="text-xs text-gray-400">Responses</p>
      </div>
      <div className="text-center p-3 bg-gray-700/30 rounded-lg">
        <p className="text-xl font-bold text-white">{survey.averageCompletion}%</p>
        <p className="text-xs text-gray-400">Completion</p>
      </div>
    </div>
  </div>
);

const FeedbackDetailDialog: React.FC<{
  feedback: FeedbackItem | null;
  onClose: () => void;
  onUpdateStatus: (id: string, status: FeedbackStatus) => void;
}> = ({ feedback, onClose, onUpdateStatus }) => {
  const [internalNote, setInternalNote] = useState("");

  if (!feedback) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-3xl max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-4">
              <div className="w-14 h-14 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center text-2xl">
                {feedback.customerAvatar || feedback.customerName.charAt(0)}
              </div>
              <div>
                <div className="flex items-center gap-2 flex-wrap mb-2">
                  <TypeBadge type={feedback.type} />
                  <StatusBadge status={feedback.status} />
                  <PriorityBadge priority={feedback.priority} />
                  <SentimentBadge sentiment={feedback.sentiment} />
                </div>
                <h2 className="text-xl font-bold text-white">{feedback.title}</h2>
                <p className="text-gray-400 text-sm mt-1">
                  From {feedback.customerName} • {new Date(feedback.createdAt).toLocaleString()}
                </p>
              </div>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-280px)]">
          {/* Customer Info */}
          <div className="bg-gray-800/50 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Customer Information</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-gray-500">Name</p>
                <p className="text-white">{feedback.customerName}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Email</p>
                <p className="text-white">{feedback.customerEmail}</p>
              </div>
              {feedback.customerPhone && (
                <div>
                  <p className="text-xs text-gray-500">Phone</p>
                  <p className="text-white">{feedback.customerPhone}</p>
                </div>
              )}
              <div>
                <p className="text-xs text-gray-500">Source</p>
                <p className="text-white capitalize flex items-center gap-2">
                  <SourceIcon source={feedback.source} />
                  {feedback.source}
                </p>
              </div>
            </div>
          </div>

          {/* Rating & NPS */}
          {(feedback.rating !== undefined || feedback.npsScore !== undefined) && (
            <div className="bg-gray-800/50 rounded-lg p-4 mb-6">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Scores</h3>
              <div className="flex items-center gap-8">
                {feedback.rating !== undefined && (
                  <div>
                    <p className="text-xs text-gray-500 mb-1">Rating</p>
                    <div className="flex items-center gap-2">
                      <RatingStars rating={feedback.rating} />
                      <span className="text-white font-bold">{feedback.rating}/5</span>
                    </div>
                  </div>
                )}
                {feedback.npsScore !== undefined && (
                  <div>
                    <p className="text-xs text-gray-500 mb-1">NPS Score</p>
                    <NPSIndicator score={feedback.npsScore} />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Feedback Content */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Feedback Content</h3>
            <p className="text-white leading-relaxed">{feedback.content}</p>
          </div>

          {/* Tags */}
          {feedback.tags.length > 0 && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {feedback.tags.map((tag) => (
                  <span key={tag} className="px-3 py-1 bg-purple-500/20 text-purple-400 text-sm rounded-full border border-purple-500/30">
                    #{tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Agent Info */}
          <div className="bg-gray-800/50 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Related Information</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-gray-500">Agent</p>
                <p className="text-white">{feedback.agentName}</p>
              </div>
              {feedback.callId && (
                <div>
                  <p className="text-xs text-gray-500">Call ID</p>
                  <p className="text-purple-400">{feedback.callId}</p>
                </div>
              )}
            </div>
          </div>

          {/* Follow-up */}
          {feedback.followUpRequired && feedback.followUpDate && (
            <div className="bg-orange-500/10 border border-orange-500/20 rounded-lg p-4 mb-6">
              <h3 className="text-sm font-medium text-orange-400 mb-2 flex items-center gap-2">
                <span>⏰</span>
                <span>Follow-up Required</span>
              </h3>
              <p className="text-white">
                Scheduled for {new Date(feedback.followUpDate).toLocaleString()}
              </p>
            </div>
          )}

          {/* Internal Notes */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Internal Notes</h3>
            {feedback.internalNotes && feedback.internalNotes.length > 0 && (
              <div className="space-y-2 mb-4">
                {feedback.internalNotes.map((note, index) => (
                  <div key={index} className="bg-gray-800/50 rounded-lg p-3 text-sm text-gray-300">
                    {note}
                  </div>
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <input
                type="text"
                value={internalNote}
                onChange={(e) => setInternalNote(e.target.value)}
                placeholder="Add internal note..."
                className="flex-1 px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
              <button className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors">
                Add Note
              </button>
            </div>
          </div>
        </div>

        <div className="p-6 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">Update Status:</span>
              <select
                value={feedback.status}
                onChange={(e) => onUpdateStatus(feedback.id, e.target.value as FeedbackStatus)}
                className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-purple-500"
              >
                <option value="new">New</option>
                <option value="in_review">In Review</option>
                <option value="resolved">Resolved</option>
                <option value="escalated">Escalated</option>
                <option value="archived">Archived</option>
              </select>
            </div>
            <div className="flex items-center gap-3">
              <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
                View Call Recording
              </button>
              <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
                Contact Customer
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const CreateSurveyDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
}> = ({ isOpen, onClose }) => {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [triggerCondition, setTriggerCondition] = useState<Survey["triggerCondition"]>("after_call");
  const [questions, setQuestions] = useState<Partial<SurveyQuestion>[]>([
    { type: "rating", question: "", required: true, minValue: 1, maxValue: 5 },
  ]);

  if (!isOpen) return null;

  const addQuestion = () => {
    setQuestions([...questions, { type: "rating", question: "", required: false, minValue: 1, maxValue: 5 }]);
  };

  const updateQuestion = (index: number, updates: Partial<SurveyQuestion>) => {
    const newQuestions = [...questions];
    newQuestions[index] = { ...newQuestions[index], ...updates };
    setQuestions(newQuestions);
  };

  const removeQuestion = (index: number) => {
    setQuestions(questions.filter((_, i) => i !== index));
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white">Create Survey</h2>
              <p className="text-gray-400 text-sm mt-1">Design a new customer feedback survey</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          <div className="space-y-6">
            {/* Survey Details */}
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Survey Title</label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g., Post-Call Satisfaction Survey"
                className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of the survey purpose"
                rows={2}
                className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Trigger Condition</label>
              <select
                value={triggerCondition}
                onChange={(e) => setTriggerCondition(e.target.value as Survey["triggerCondition"])}
                className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
              >
                <option value="after_call">After Call</option>
                <option value="after_resolution">After Resolution</option>
                <option value="scheduled">Scheduled</option>
                <option value="manual">Manual</option>
              </select>
            </div>

            {/* Questions */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <label className="text-sm font-medium text-gray-400">Questions</label>
                <button
                  onClick={addQuestion}
                  className="text-sm text-purple-400 hover:text-purple-300 flex items-center gap-1"
                >
                  <span>+</span>
                  <span>Add Question</span>
                </button>
              </div>

              <div className="space-y-4">
                {questions.map((question, index) => (
                  <div key={index} className="bg-gray-800/50 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-3">
                      <span className="text-sm text-gray-400">Question {index + 1}</span>
                      {questions.length > 1 && (
                        <button
                          onClick={() => removeQuestion(index)}
                          className="text-red-400 hover:text-red-300 text-sm"
                        >
                          Remove
                        </button>
                      )}
                    </div>

                    <div className="space-y-3">
                      <select
                        value={question.type}
                        onChange={(e) => updateQuestion(index, { type: e.target.value as SurveyQuestionType })}
                        className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:border-purple-500"
                      >
                        <option value="rating">Star Rating (1-5)</option>
                        <option value="nps">NPS Score (0-10)</option>
                        <option value="scale">Custom Scale</option>
                        <option value="multiple_choice">Multiple Choice</option>
                        <option value="checkbox">Checkbox</option>
                        <option value="text">Text Response</option>
                      </select>

                      <input
                        type="text"
                        value={question.question || ""}
                        onChange={(e) => updateQuestion(index, { question: e.target.value })}
                        placeholder="Enter question text"
                        className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-purple-500"
                      />

                      <label className="flex items-center gap-2 text-sm text-gray-400">
                        <input
                          type="checkbox"
                          checked={question.required}
                          onChange={(e) => updateQuestion(index, { required: e.target.checked })}
                          className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-purple-500 focus:ring-purple-500"
                        />
                        Required
                      </label>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 border-t border-gray-700">
          <div className="flex items-center justify-end gap-3">
            <button onClick={onClose} className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
              Cancel
            </button>
            <button className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
              Create Survey
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Component
export default function FeedbackPortalPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "feedback" | "surveys" | "analytics">("overview");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState<FeedbackType | "all">("all");
  const [selectedStatus, setSelectedStatus] = useState<FeedbackStatus | "all">("all");
  const [selectedPriority, setSelectedPriority] = useState<FeedbackPriority | "all">("all");
  const [selectedSentiment, setSelectedSentiment] = useState<SentimentType | "all">("all");
  const [selectedFeedback, setSelectedFeedback] = useState<FeedbackItem | null>(null);
  const [selectedSurvey, setSelectedSurvey] = useState<Survey | null>(null);
  const [showCreateSurvey, setShowCreateSurvey] = useState(false);

  const filteredFeedback = useMemo(() => {
    return mockFeedback.filter((item) => {
      const matchesSearch =
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.customerName.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesType = selectedType === "all" || item.type === selectedType;
      const matchesStatus = selectedStatus === "all" || item.status === selectedStatus;
      const matchesPriority = selectedPriority === "all" || item.priority === selectedPriority;
      const matchesSentiment = selectedSentiment === "all" || item.sentiment === selectedSentiment;
      return matchesSearch && matchesType && matchesStatus && matchesPriority && matchesSentiment;
    });
  }, [searchQuery, selectedType, selectedStatus, selectedPriority, selectedSentiment]);

  const handleUpdateStatus = (id: string, status: FeedbackStatus) => {
    console.log("Update status:", id, status);
    // In real app, update the feedback status
  };

  const tabs = [
    { id: "overview", label: "Overview", icon: "📊" },
    { id: "feedback", label: "All Feedback", icon: "💬" },
    { id: "surveys", label: "Surveys", icon: "📋" },
    { id: "analytics", label: "Analytics", icon: "📈" },
  ];

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-900">
        {/* Header */}
        <div className="border-b border-gray-800 bg-gray-900/95 backdrop-blur-sm sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-2xl font-bold text-white">Customer Feedback Portal</h1>
                <p className="text-gray-400 mt-1">Monitor and manage customer feedback across all channels</p>
              </div>
              <div className="flex items-center gap-3">
                <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
                  Export Report
                </button>
                <button
                  onClick={() => setShowCreateSurvey(true)}
                  className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
                >
                  <span>+</span>
                  <span>Create Survey</span>
                </button>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex items-center gap-2">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg whitespace-nowrap transition-colors ${
                    activeTab === tab.id
                      ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                      : "text-gray-400 hover:text-white hover:bg-gray-800"
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Overview Tab */}
          {activeTab === "overview" && (
            <div className="space-y-8">
              {/* Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard title="Total Feedback" value={mockMetrics.totalFeedback.toLocaleString()} icon="💬" trend={8} color="purple" />
                <MetricCard title="Average Rating" value={`${mockMetrics.avgRating} / 5`} icon="⭐" trend={3} color="yellow" />
                <MetricCard title="NPS Score" value={mockMetrics.npsScore} icon="📊" trend={5} color="blue" />
                <MetricCard title="Response Rate" value={`${mockMetrics.responseRate}%`} icon="📈" trend={-2} color="green" />
              </div>

              {/* Two Column Layout */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Sentiment Distribution */}
                <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Sentiment Distribution</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400 flex items-center gap-2">
                          <span>😊</span>
                          <span>Positive</span>
                        </span>
                        <span className="text-sm font-medium text-green-400">{mockMetrics.positiveSentiment}%</span>
                      </div>
                      <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                        <div className="h-full bg-green-500 rounded-full" style={{ width: `${mockMetrics.positiveSentiment}%` }} />
                      </div>
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400 flex items-center gap-2">
                          <span>😐</span>
                          <span>Neutral</span>
                        </span>
                        <span className="text-sm font-medium text-gray-400">{100 - mockMetrics.positiveSentiment - mockMetrics.negativeSentiment}%</span>
                      </div>
                      <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                        <div className="h-full bg-gray-500 rounded-full" style={{ width: `${100 - mockMetrics.positiveSentiment - mockMetrics.negativeSentiment}%` }} />
                      </div>
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400 flex items-center gap-2">
                          <span>😞</span>
                          <span>Negative</span>
                        </span>
                        <span className="text-sm font-medium text-red-400">{mockMetrics.negativeSentiment}%</span>
                      </div>
                      <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                        <div className="h-full bg-red-500 rounded-full" style={{ width: `${mockMetrics.negativeSentiment}%` }} />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Feedback by Type */}
                <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Feedback by Type</h3>
                  <div className="grid grid-cols-2 gap-3">
                    {feedbackCategories.map((cat) => {
                      const count = mockFeedback.filter((f) => f.type === cat.type).length;
                      return (
                        <div key={cat.type} className="bg-gray-700/30 rounded-lg p-4 flex items-center gap-3">
                          <span className="text-2xl">{cat.icon}</span>
                          <div>
                            <p className="text-xl font-bold text-white">{count}</p>
                            <p className="text-xs text-gray-400">{cat.label}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Rating Trend */}
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Rating Trend (Last 7 Days)</h3>
                <TrendChart data={mockTrendData} metric="rating" />
              </div>

              {/* Recent Feedback */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Recent Feedback</h3>
                  <button
                    onClick={() => setActiveTab("feedback")}
                    className="text-sm text-purple-400 hover:text-purple-300"
                  >
                    View All →
                  </button>
                </div>
                <div className="space-y-4">
                  {mockFeedback.slice(0, 4).map((feedback) => (
                    <FeedbackCard key={feedback.id} feedback={feedback} onSelect={setSelectedFeedback} />
                  ))}
                </div>
              </div>

              {/* Action Items */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-orange-500/10 border border-orange-500/20 rounded-xl p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <span className="text-2xl">⏰</span>
                    <div>
                      <p className="text-2xl font-bold text-orange-400">{mockMetrics.pendingFollowUps}</p>
                      <p className="text-sm text-gray-400">Pending Follow-ups</p>
                    </div>
                  </div>
                  <button className="w-full px-4 py-2 bg-orange-500/20 text-orange-400 rounded-lg hover:bg-orange-500/30 transition-colors text-sm font-medium">
                    View Follow-ups
                  </button>
                </div>
                <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <span className="text-2xl">🚨</span>
                    <div>
                      <p className="text-2xl font-bold text-red-400">{mockMetrics.escalatedCount}</p>
                      <p className="text-sm text-gray-400">Escalated Issues</p>
                    </div>
                  </div>
                  <button className="w-full px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors text-sm font-medium">
                    Handle Escalations
                  </button>
                </div>
                <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <span className="text-2xl">📥</span>
                    <div>
                      <p className="text-2xl font-bold text-blue-400">{mockMetrics.newFeedback}</p>
                      <p className="text-sm text-gray-400">New Today</p>
                    </div>
                  </div>
                  <button className="w-full px-4 py-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors text-sm font-medium">
                    Review New Feedback
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Feedback Tab */}
          {activeTab === "feedback" && (
            <div className="space-y-6">
              {/* Filters */}
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex-1 min-w-[200px]">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search feedback..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full px-4 py-2 pl-10 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                    />
                    <svg className="w-5 h-5 text-gray-500 absolute left-3 top-1/2 -translate-y-1/2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                </div>

                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value as FeedbackType | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Types</option>
                  <option value="rating">Ratings</option>
                  <option value="survey">Surveys</option>
                  <option value="review">Reviews</option>
                  <option value="suggestion">Suggestions</option>
                  <option value="complaint">Complaints</option>
                  <option value="praise">Praise</option>
                </select>

                <select
                  value={selectedStatus}
                  onChange={(e) => setSelectedStatus(e.target.value as FeedbackStatus | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Status</option>
                  <option value="new">New</option>
                  <option value="in_review">In Review</option>
                  <option value="resolved">Resolved</option>
                  <option value="escalated">Escalated</option>
                  <option value="archived">Archived</option>
                </select>

                <select
                  value={selectedPriority}
                  onChange={(e) => setSelectedPriority(e.target.value as FeedbackPriority | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Priority</option>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>

                <select
                  value={selectedSentiment}
                  onChange={(e) => setSelectedSentiment(e.target.value as SentimentType | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Sentiment</option>
                  <option value="positive">Positive</option>
                  <option value="neutral">Neutral</option>
                  <option value="negative">Negative</option>
                  <option value="mixed">Mixed</option>
                </select>
              </div>

              {/* Results count */}
              <p className="text-sm text-gray-400">
                Showing {filteredFeedback.length} of {mockFeedback.length} feedback items
              </p>

              {/* Feedback List */}
              <div className="space-y-4">
                {filteredFeedback.map((feedback) => (
                  <FeedbackCard key={feedback.id} feedback={feedback} onSelect={setSelectedFeedback} />
                ))}
              </div>

              {filteredFeedback.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-gray-400">No feedback found matching your criteria</p>
                </div>
              )}
            </div>
          )}

          {/* Surveys Tab */}
          {activeTab === "surveys" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-400">{mockSurveys.length} surveys</p>
                <button
                  onClick={() => setShowCreateSurvey(true)}
                  className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
                >
                  <span>+</span>
                  <span>Create Survey</span>
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {mockSurveys.map((survey) => (
                  <SurveyCard key={survey.id} survey={survey} onSelect={setSelectedSurvey} />
                ))}
              </div>
            </div>
          )}

          {/* Analytics Tab */}
          {activeTab === "analytics" && (
            <div className="space-y-8">
              {/* Metric Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard title="Avg Resolution Time" value={`${mockMetrics.avgResolutionTime}h`} icon="⏱️" trend={-12} color="green" subtitle="Hours to resolve" />
                <MetricCard title="Skills Feedback" value={mockMetrics.skillsAcquired} icon="🎯" trend={15} color="purple" />
                <MetricCard title="Response Rate" value={`${mockMetrics.responseRate}%`} icon="📬" trend={3} color="blue" />
                <MetricCard title="Customer Effort" value="2.4" icon="💪" trend={-8} color="yellow" subtitle="Avg effort score" />
              </div>

              {/* Charts Row */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">NPS Trend</h3>
                  <TrendChart data={mockTrendData} metric="nps" />
                </div>
                <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Feedback Volume</h3>
                  <TrendChart data={mockTrendData} metric="volume" />
                </div>
              </div>

              {/* Sentiment Trend */}
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Sentiment Trend</h3>
                <TrendChart data={mockTrendData} metric="sentiment" />
              </div>

              {/* Top Issues */}
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Top Reported Issues</h3>
                <div className="space-y-3">
                  {[
                    { issue: "Long wait times", count: 45, percentage: 28 },
                    { issue: "Transfer issues", count: 32, percentage: 20 },
                    { issue: "Resolution quality", count: 28, percentage: 17 },
                    { issue: "Agent knowledge", count: 24, percentage: 15 },
                    { issue: "Follow-up delays", count: 18, percentage: 11 },
                  ].map((item) => (
                    <div key={item.issue} className="flex items-center gap-4">
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm text-white">{item.issue}</span>
                          <span className="text-sm text-gray-400">{item.count} reports</span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-red-500 to-orange-500 rounded-full"
                            style={{ width: `${item.percentage}%` }}
                          />
                        </div>
                      </div>
                      <span className="text-sm text-gray-500 w-12 text-right">{item.percentage}%</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Agent Performance */}
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Agent Feedback Scores</h3>
                <div className="space-y-4">
                  {[
                    { name: "Sales Bot Pro", rating: 4.8, nps: 72, feedback: 245 },
                    { name: "Customer Care AI", rating: 4.5, nps: 58, feedback: 189 },
                    { name: "Support Master", rating: 4.2, nps: 45, feedback: 156 },
                    { name: "Tech Helper", rating: 3.9, nps: 32, feedback: 98 },
                  ].map((agent) => (
                    <div key={agent.name} className="flex items-center gap-4 p-4 bg-gray-700/30 rounded-lg">
                      <div className="w-10 h-10 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center">
                        🤖
                      </div>
                      <div className="flex-1">
                        <p className="font-medium text-white">{agent.name}</p>
                        <p className="text-sm text-gray-400">{agent.feedback} feedback received</p>
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="text-center">
                          <RatingStars rating={Math.round(agent.rating)} size="sm" />
                          <p className="text-xs text-gray-500 mt-1">{agent.rating} avg</p>
                        </div>
                        <div className="text-center">
                          <p className={`text-lg font-bold ${agent.nps >= 50 ? "text-green-400" : agent.nps >= 30 ? "text-yellow-400" : "text-red-400"}`}>
                            {agent.nps}
                          </p>
                          <p className="text-xs text-gray-500">NPS</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Dialogs */}
        <FeedbackDetailDialog
          feedback={selectedFeedback}
          onClose={() => setSelectedFeedback(null)}
          onUpdateStatus={handleUpdateStatus}
        />
        <CreateSurveyDialog isOpen={showCreateSurvey} onClose={() => setShowCreateSurvey(false)} />
      </div>
    </DashboardLayout>
  );
}
