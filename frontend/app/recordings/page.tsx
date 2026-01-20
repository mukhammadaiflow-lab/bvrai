"use client";

import React, { useState, useMemo, useRef } from "react";
import DashboardLayout from "../components/DashboardLayout";

// Types
type RecordingStatus = "completed" | "processing" | "failed" | "archived";
type TranscriptionStatus = "pending" | "processing" | "completed" | "failed";
type CallOutcome = "successful" | "voicemail" | "no_answer" | "busy" | "failed" | "transferred" | "callback_scheduled";
type SentimentType = "positive" | "neutral" | "negative" | "mixed";

interface TranscriptSegment {
  id: string;
  speaker: "agent" | "customer";
  speakerName: string;
  text: string;
  timestamp: number;
  duration: number;
  sentiment?: SentimentType;
  keywords?: string[];
}

interface Recording {
  id: string;
  callId: string;
  agentId: string;
  agentName: string;
  customerId: string;
  customerName: string;
  customerPhone: string;
  direction: "inbound" | "outbound";
  status: RecordingStatus;
  outcome: CallOutcome;
  duration: number;
  fileSize: number;
  fileUrl: string;
  waveformData?: number[];
  transcriptionStatus: TranscriptionStatus;
  transcript?: TranscriptSegment[];
  sentiment: SentimentType;
  qualityScore?: number;
  tags: string[];
  notes?: string;
  createdAt: string;
  campaign?: string;
  topics?: string[];
  highlights?: { timestamp: number; text: string; type: string }[];
}

interface RecordingMetrics {
  totalRecordings: number;
  totalDuration: number;
  avgDuration: number;
  transcribedCount: number;
  avgQualityScore: number;
  storageUsed: number;
  recordingsThisWeek: number;
  positiveSentiment: number;
}

interface PlaylistItem {
  id: string;
  recordingId: string;
  addedAt: string;
}

interface Playlist {
  id: string;
  name: string;
  description: string;
  items: PlaylistItem[];
  createdAt: string;
  isPublic: boolean;
}

// Mock Data
const generateWaveform = (): number[] => {
  return Array.from({ length: 100 }, () => Math.random() * 0.8 + 0.1);
};

const mockTranscript: TranscriptSegment[] = [
  { id: "t1", speaker: "agent", speakerName: "Sales Bot Pro", text: "Good morning! Thank you for calling BVRAI. My name is Alex, and I'm here to help you today. How can I assist you?", timestamp: 0, duration: 6, sentiment: "positive" },
  { id: "t2", speaker: "customer", speakerName: "John Anderson", text: "Hi Alex, I'm calling about upgrading my current plan. I've been with you guys for about a year now.", timestamp: 6, duration: 5, sentiment: "neutral" },
  { id: "t3", speaker: "agent", speakerName: "Sales Bot Pro", text: "That's wonderful to hear, John! I can see you've been a valued customer. Let me pull up your account details. I see you're currently on our Standard plan. Are you looking at our Premium or Enterprise options?", timestamp: 11, duration: 9, sentiment: "positive", keywords: ["upgrade", "Premium", "Enterprise"] },
  { id: "t4", speaker: "customer", speakerName: "John Anderson", text: "I'm thinking Premium. What additional features would I get?", timestamp: 20, duration: 4, sentiment: "positive", keywords: ["Premium", "features"] },
  { id: "t5", speaker: "agent", speakerName: "Sales Bot Pro", text: "Great choice! Premium includes unlimited calls, priority support, advanced analytics, and custom integrations. It's our most popular upgrade. Based on your usage, you'd also save about 20% compared to your current per-minute charges.", timestamp: 24, duration: 12, sentiment: "positive", keywords: ["unlimited calls", "priority support", "analytics", "integrations", "20% savings"] },
  { id: "t6", speaker: "customer", speakerName: "John Anderson", text: "That sounds perfect. And what's the pricing difference?", timestamp: 36, duration: 3, sentiment: "neutral" },
  { id: "t7", speaker: "agent", speakerName: "Sales Bot Pro", text: "Premium is $199 per month, which is $50 more than your current plan. However, with your usage patterns, you'd actually save around $75 per month overall. Would you like me to process the upgrade today?", timestamp: 39, duration: 10, sentiment: "positive", keywords: ["$199", "savings", "upgrade"] },
  { id: "t8", speaker: "customer", speakerName: "John Anderson", text: "Yes, let's do it! Can you also add the API access feature?", timestamp: 49, duration: 4, sentiment: "positive", keywords: ["API access"] },
  { id: "t9", speaker: "agent", speakerName: "Sales Bot Pro", text: "Absolutely! I've upgraded your account to Premium and added API access. You'll receive a confirmation email shortly with all the details. Is there anything else I can help you with today?", timestamp: 53, duration: 8, sentiment: "positive" },
  { id: "t10", speaker: "customer", speakerName: "John Anderson", text: "No, that's all. Thank you so much for your help, Alex!", timestamp: 61, duration: 3, sentiment: "positive" },
  { id: "t11", speaker: "agent", speakerName: "Sales Bot Pro", text: "You're very welcome, John! Thank you for choosing BVRAI. Have a great day!", timestamp: 64, duration: 4, sentiment: "positive" },
];

const mockRecordings: Recording[] = [
  {
    id: "rec-1",
    callId: "call-12345",
    agentId: "agent-1",
    agentName: "Sales Bot Pro",
    customerId: "cust-1",
    customerName: "John Anderson",
    customerPhone: "+1 (555) 123-4567",
    direction: "inbound",
    status: "completed",
    outcome: "successful",
    duration: 68,
    fileSize: 1240000,
    fileUrl: "/recordings/rec-1.mp3",
    waveformData: generateWaveform(),
    transcriptionStatus: "completed",
    transcript: mockTranscript,
    sentiment: "positive",
    qualityScore: 95,
    tags: ["upgrade", "premium", "satisfied-customer"],
    notes: "Customer upgraded to Premium plan. Very positive interaction.",
    createdAt: "2024-01-20T10:30:00Z",
    campaign: "Q1 Retention",
    topics: ["Plan Upgrade", "Pricing", "API Access"],
    highlights: [
      { timestamp: 24, text: "Premium features explanation", type: "key_moment" },
      { timestamp: 49, text: "Successful close - upgrade confirmed", type: "conversion" },
    ],
  },
  {
    id: "rec-2",
    callId: "call-12346",
    agentId: "agent-2",
    agentName: "Support Master",
    customerId: "cust-2",
    customerName: "Sarah Miller",
    customerPhone: "+1 (555) 234-5678",
    direction: "inbound",
    status: "completed",
    outcome: "transferred",
    duration: 245,
    fileSize: 4520000,
    fileUrl: "/recordings/rec-2.mp3",
    waveformData: generateWaveform(),
    transcriptionStatus: "completed",
    sentiment: "negative",
    qualityScore: 72,
    tags: ["escalation", "complaint", "billing-issue"],
    notes: "Customer had billing dispute. Transferred to supervisor.",
    createdAt: "2024-01-20T09:15:00Z",
    topics: ["Billing", "Dispute", "Escalation"],
  },
  {
    id: "rec-3",
    callId: "call-12347",
    agentId: "agent-1",
    agentName: "Sales Bot Pro",
    customerId: "cust-3",
    customerName: "Michael Chen",
    customerPhone: "+1 (555) 345-6789",
    direction: "outbound",
    status: "completed",
    outcome: "callback_scheduled",
    duration: 156,
    fileSize: 2890000,
    fileUrl: "/recordings/rec-3.mp3",
    waveformData: generateWaveform(),
    transcriptionStatus: "completed",
    sentiment: "neutral",
    qualityScore: 88,
    tags: ["follow-up", "demo-request", "enterprise"],
    createdAt: "2024-01-20T08:45:00Z",
    campaign: "Enterprise Outreach",
    topics: ["Demo Request", "Enterprise Features"],
  },
  {
    id: "rec-4",
    callId: "call-12348",
    agentId: "agent-3",
    agentName: "Customer Care AI",
    customerId: "cust-4",
    customerName: "Emily Watson",
    customerPhone: "+1 (555) 456-7890",
    direction: "inbound",
    status: "completed",
    outcome: "successful",
    duration: 89,
    fileSize: 1650000,
    fileUrl: "/recordings/rec-4.mp3",
    waveformData: generateWaveform(),
    transcriptionStatus: "completed",
    sentiment: "positive",
    qualityScore: 91,
    tags: ["quick-resolution", "account-inquiry"],
    createdAt: "2024-01-20T07:30:00Z",
    topics: ["Account Information", "Password Reset"],
  },
  {
    id: "rec-5",
    callId: "call-12349",
    agentId: "agent-1",
    agentName: "Sales Bot Pro",
    customerId: "cust-5",
    customerName: "David Thompson",
    customerPhone: "+1 (555) 567-8901",
    direction: "outbound",
    status: "completed",
    outcome: "voicemail",
    duration: 32,
    fileSize: 590000,
    fileUrl: "/recordings/rec-5.mp3",
    waveformData: generateWaveform(),
    transcriptionStatus: "completed",
    sentiment: "neutral",
    tags: ["voicemail", "follow-up-needed"],
    createdAt: "2024-01-19T16:45:00Z",
    campaign: "Q1 Retention",
  },
  {
    id: "rec-6",
    callId: "call-12350",
    agentId: "agent-4",
    agentName: "Tech Helper",
    customerId: "cust-6",
    customerName: "Lisa Brown",
    customerPhone: "+1 (555) 678-9012",
    direction: "inbound",
    status: "processing",
    outcome: "successful",
    duration: 312,
    fileSize: 5780000,
    fileUrl: "/recordings/rec-6.mp3",
    transcriptionStatus: "processing",
    sentiment: "mixed",
    tags: ["technical-support", "troubleshooting"],
    createdAt: "2024-01-19T15:20:00Z",
    topics: ["Technical Issue", "Integration Setup"],
  },
  {
    id: "rec-7",
    callId: "call-12351",
    agentId: "agent-2",
    agentName: "Support Master",
    customerId: "cust-7",
    customerName: "Robert Garcia",
    customerPhone: "+1 (555) 789-0123",
    direction: "inbound",
    status: "completed",
    outcome: "failed",
    duration: 45,
    fileSize: 830000,
    fileUrl: "/recordings/rec-7.mp3",
    waveformData: generateWaveform(),
    transcriptionStatus: "completed",
    sentiment: "negative",
    qualityScore: 45,
    tags: ["dropped-call", "technical-issue"],
    createdAt: "2024-01-19T14:10:00Z",
  },
  {
    id: "rec-8",
    callId: "call-12352",
    agentId: "agent-1",
    agentName: "Sales Bot Pro",
    customerId: "cust-8",
    customerName: "Jennifer Lee",
    customerPhone: "+1 (555) 890-1234",
    direction: "outbound",
    status: "archived",
    outcome: "successful",
    duration: 198,
    fileSize: 3670000,
    fileUrl: "/recordings/rec-8.mp3",
    waveformData: generateWaveform(),
    transcriptionStatus: "completed",
    sentiment: "positive",
    qualityScore: 94,
    tags: ["new-customer", "onboarding", "successful-sale"],
    createdAt: "2024-01-18T11:30:00Z",
    campaign: "New Customer Acquisition",
    topics: ["Product Demo", "Pricing", "Onboarding"],
  },
];

const mockMetrics: RecordingMetrics = {
  totalRecordings: 2847,
  totalDuration: 142350, // seconds
  avgDuration: 145,
  transcribedCount: 2756,
  avgQualityScore: 82,
  storageUsed: 45.8, // GB
  recordingsThisWeek: 234,
  positiveSentiment: 68,
};

const mockPlaylists: Playlist[] = [
  {
    id: "pl-1",
    name: "Best Sales Calls",
    description: "Top performing sales calls for training",
    items: [
      { id: "pli-1", recordingId: "rec-1", addedAt: "2024-01-20T10:00:00Z" },
      { id: "pli-2", recordingId: "rec-8", addedAt: "2024-01-19T14:00:00Z" },
    ],
    createdAt: "2024-01-15T00:00:00Z",
    isPublic: true,
  },
  {
    id: "pl-2",
    name: "Training Examples - Escalations",
    description: "Examples of proper escalation handling",
    items: [
      { id: "pli-3", recordingId: "rec-2", addedAt: "2024-01-20T09:30:00Z" },
    ],
    createdAt: "2024-01-10T00:00:00Z",
    isPublic: false,
  },
];

// Utility Functions
const formatDuration = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
};

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

// Components
const MetricCard: React.FC<{
  title: string;
  value: string | number;
  icon: string;
  trend?: number;
  subtitle?: string;
}> = ({ title, value, icon, trend, subtitle }) => (
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
      <div className="w-12 h-12 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center text-2xl">
        {icon}
      </div>
    </div>
  </div>
);

const StatusBadge: React.FC<{ status: RecordingStatus }> = ({ status }) => {
  const config: Record<RecordingStatus, { color: string; label: string }> = {
    completed: { color: "bg-green-500/20 text-green-400 border-green-500/30", label: "Completed" },
    processing: { color: "bg-blue-500/20 text-blue-400 border-blue-500/30", label: "Processing" },
    failed: { color: "bg-red-500/20 text-red-400 border-red-500/30", label: "Failed" },
    archived: { color: "bg-gray-500/20 text-gray-400 border-gray-500/30", label: "Archived" },
  };

  const { color, label } = config[status];

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${color}`}>{label}</span>
  );
};

const OutcomeBadge: React.FC<{ outcome: CallOutcome }> = ({ outcome }) => {
  const config: Record<CallOutcome, { color: string; label: string; icon: string }> = {
    successful: { color: "bg-green-500/20 text-green-400", label: "Successful", icon: "✓" },
    voicemail: { color: "bg-yellow-500/20 text-yellow-400", label: "Voicemail", icon: "📫" },
    no_answer: { color: "bg-gray-500/20 text-gray-400", label: "No Answer", icon: "📵" },
    busy: { color: "bg-orange-500/20 text-orange-400", label: "Busy", icon: "⏳" },
    failed: { color: "bg-red-500/20 text-red-400", label: "Failed", icon: "✗" },
    transferred: { color: "bg-purple-500/20 text-purple-400", label: "Transferred", icon: "↗️" },
    callback_scheduled: { color: "bg-blue-500/20 text-blue-400", label: "Callback", icon: "📅" },
  };

  const { color, label, icon } = config[outcome];

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full ${color}`}>
      <span>{icon}</span>
      <span>{label}</span>
    </span>
  );
};

const SentimentBadge: React.FC<{ sentiment: SentimentType }> = ({ sentiment }) => {
  const config: Record<SentimentType, { color: string; icon: string }> = {
    positive: { color: "text-green-400", icon: "😊" },
    neutral: { color: "text-gray-400", icon: "😐" },
    negative: { color: "text-red-400", icon: "😞" },
    mixed: { color: "text-yellow-400", icon: "🤔" },
  };

  const { color, icon } = config[sentiment];

  return <span className={color} title={sentiment}>{icon}</span>;
};

const QualityScoreBadge: React.FC<{ score: number }> = ({ score }) => {
  const getColor = () => {
    if (score >= 90) return "text-green-400 bg-green-500/20";
    if (score >= 70) return "text-yellow-400 bg-yellow-500/20";
    return "text-red-400 bg-red-500/20";
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-bold rounded-full ${getColor()}`}>
      {score}
    </span>
  );
};

const WaveformDisplay: React.FC<{
  data: number[];
  currentTime?: number;
  duration: number;
  isPlaying?: boolean;
  onSeek?: (time: number) => void;
}> = ({ data, currentTime = 0, duration, isPlaying, onSeek }) => {
  const progress = (currentTime / duration) * 100;

  return (
    <div
      className="h-12 flex items-center gap-0.5 cursor-pointer group"
      onClick={(e) => {
        if (onSeek) {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const percentage = x / rect.width;
          onSeek(percentage * duration);
        }
      }}
    >
      {data.map((value, index) => {
        const isPlayed = (index / data.length) * 100 <= progress;
        return (
          <div
            key={index}
            className={`flex-1 rounded-full transition-all ${
              isPlayed ? "bg-purple-500" : "bg-gray-600 group-hover:bg-gray-500"
            }`}
            style={{ height: `${value * 100}%` }}
          />
        );
      })}
    </div>
  );
};

const RecordingCard: React.FC<{
  recording: Recording;
  onSelect: (recording: Recording) => void;
  onPlay: (recording: Recording) => void;
  isPlaying: boolean;
}> = ({ recording, onSelect, onPlay, isPlaying }) => {
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5 hover:border-purple-500/50 transition-all duration-300">
      <div className="flex items-start gap-4">
        <button
          onClick={() => onPlay(recording)}
          className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 transition-all ${
            isPlaying
              ? "bg-purple-500 text-white"
              : "bg-gray-700/50 text-gray-400 hover:bg-purple-500/20 hover:text-purple-400"
          }`}
        >
          {isPlaying ? (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            <svg className="w-5 h-5 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
              <polygon points="5,3 19,12 5,21" />
            </svg>
          )}
        </button>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-2">
            <StatusBadge status={recording.status} />
            <OutcomeBadge outcome={recording.outcome} />
            <SentimentBadge sentiment={recording.sentiment} />
            {recording.qualityScore && <QualityScoreBadge score={recording.qualityScore} />}
          </div>

          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold text-white truncate">{recording.customerName}</h3>
            <span className="text-gray-500 text-sm">•</span>
            <span className="text-gray-400 text-sm">{recording.customerPhone}</span>
          </div>

          <div className="flex items-center gap-4 text-sm text-gray-500">
            <span className="flex items-center gap-1">
              {recording.direction === "inbound" ? "📥" : "📤"}
              <span className="capitalize">{recording.direction}</span>
            </span>
            <span className="flex items-center gap-1">
              <span>🤖</span>
              <span>{recording.agentName}</span>
            </span>
            <span className="flex items-center gap-1">
              <span>⏱️</span>
              <span>{formatDuration(recording.duration)}</span>
            </span>
            <span>{new Date(recording.createdAt).toLocaleDateString()}</span>
          </div>

          {recording.waveformData && (
            <div className="mt-3">
              <WaveformDisplay data={recording.waveformData} duration={recording.duration} />
            </div>
          )}

          {recording.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-3">
              {recording.tags.slice(0, 4).map((tag) => (
                <span key={tag} className="px-2 py-0.5 bg-gray-700/50 text-gray-400 text-xs rounded-full">
                  #{tag}
                </span>
              ))}
              {recording.tags.length > 4 && (
                <span className="text-gray-500 text-xs">+{recording.tags.length - 4}</span>
              )}
            </div>
          )}
        </div>

        <div className="flex flex-col items-end gap-2">
          <span className="text-xs text-gray-500">{formatFileSize(recording.fileSize)}</span>
          <button
            onClick={() => onSelect(recording)}
            className="p-2 text-gray-400 hover:text-purple-400 hover:bg-purple-500/10 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

const TranscriptView: React.FC<{
  transcript: TranscriptSegment[];
  currentTime?: number;
  onSeek?: (time: number) => void;
}> = ({ transcript, currentTime = 0, onSeek }) => {
  return (
    <div className="space-y-4">
      {transcript.map((segment) => {
        const isActive = currentTime >= segment.timestamp && currentTime < segment.timestamp + segment.duration;

        return (
          <div
            key={segment.id}
            className={`flex gap-4 p-3 rounded-lg cursor-pointer transition-all ${
              isActive ? "bg-purple-500/10 border border-purple-500/30" : "hover:bg-gray-800/50"
            }`}
            onClick={() => onSeek?.(segment.timestamp)}
          >
            <div className="flex-shrink-0">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                segment.speaker === "agent" ? "bg-purple-500/20 text-purple-400" : "bg-blue-500/20 text-blue-400"
              }`}>
                {segment.speaker === "agent" ? "🤖" : "👤"}
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className={`font-medium ${segment.speaker === "agent" ? "text-purple-400" : "text-blue-400"}`}>
                  {segment.speakerName}
                </span>
                <span className="text-xs text-gray-500">{formatDuration(segment.timestamp)}</span>
                {segment.sentiment && <SentimentBadge sentiment={segment.sentiment} />}
              </div>
              <p className="text-gray-300 text-sm">{segment.text}</p>
              {segment.keywords && segment.keywords.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {segment.keywords.map((keyword) => (
                    <span key={keyword} className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded-full">
                      {keyword}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

const AudioPlayer: React.FC<{
  recording: Recording;
  isPlaying: boolean;
  onTogglePlay: () => void;
  currentTime: number;
  onSeek: (time: number) => void;
}> = ({ recording, isPlaying, onTogglePlay, currentTime, onSeek }) => {
  return (
    <div className="bg-gray-800/80 backdrop-blur-md border border-gray-700 rounded-xl p-4">
      <div className="flex items-center gap-4">
        <button
          onClick={onTogglePlay}
          className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white hover:opacity-90 transition-opacity"
        >
          {isPlaying ? (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            <svg className="w-5 h-5 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
              <polygon points="5,3 19,12 5,21" />
            </svg>
          )}
        </button>

        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-white font-medium">{recording.customerName}</span>
              <span className="text-gray-500">•</span>
              <span className="text-gray-400 text-sm">{recording.agentName}</span>
            </div>
            <span className="text-sm text-gray-400">
              {formatDuration(Math.floor(currentTime))} / {formatDuration(recording.duration)}
            </span>
          </div>

          {recording.waveformData && (
            <WaveformDisplay
              data={recording.waveformData}
              currentTime={currentTime}
              duration={recording.duration}
              isPlaying={isPlaying}
              onSeek={onSeek}
            />
          )}
        </div>

        <div className="flex items-center gap-2">
          <button className="p-2 text-gray-400 hover:text-white transition-colors" title="Download">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
          </button>
          <button className="p-2 text-gray-400 hover:text-white transition-colors" title="Share">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

const RecordingDetailDialog: React.FC<{
  recording: Recording | null;
  onClose: () => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  currentTime: number;
  onSeek: (time: number) => void;
}> = ({ recording, onClose, isPlaying, onTogglePlay, currentTime, onSeek }) => {
  const [activeTab, setActiveTab] = useState<"transcript" | "details" | "analytics">("transcript");

  if (!recording) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <StatusBadge status={recording.status} />
                <OutcomeBadge outcome={recording.outcome} />
                <SentimentBadge sentiment={recording.sentiment} />
                {recording.qualityScore && <QualityScoreBadge score={recording.qualityScore} />}
              </div>
              <h2 className="text-xl font-bold text-white">
                {recording.customerName} - {formatDuration(recording.duration)}
              </h2>
              <p className="text-gray-400 text-sm mt-1">
                {recording.direction === "inbound" ? "Inbound" : "Outbound"} call with {recording.agentName}
              </p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Audio Player */}
          <div className="mt-4">
            <AudioPlayer
              recording={recording}
              isPlaying={isPlaying}
              onTogglePlay={onTogglePlay}
              currentTime={currentTime}
              onSeek={onSeek}
            />
          </div>

          {/* Tabs */}
          <div className="flex items-center gap-2 mt-4">
            {(["transcript", "details", "analytics"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors capitalize ${
                  activeTab === tab
                    ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                    : "text-gray-400 hover:text-white hover:bg-gray-800"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-350px)]">
          {activeTab === "transcript" && recording.transcript && (
            <TranscriptView transcript={recording.transcript} currentTime={currentTime} onSeek={onSeek} />
          )}

          {activeTab === "transcript" && !recording.transcript && (
            <div className="text-center py-12">
              {recording.transcriptionStatus === "processing" ? (
                <div>
                  <div className="w-12 h-12 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin mx-auto mb-4" />
                  <p className="text-gray-400">Transcription in progress...</p>
                </div>
              ) : (
                <p className="text-gray-400">Transcript not available</p>
              )}
            </div>
          )}

          {activeTab === "details" && (
            <div className="space-y-6">
              {/* Call Details */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 mb-1">Customer</p>
                  <p className="text-white font-medium">{recording.customerName}</p>
                  <p className="text-gray-400 text-sm">{recording.customerPhone}</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 mb-1">Agent</p>
                  <p className="text-white font-medium">{recording.agentName}</p>
                  <p className="text-gray-400 text-sm">ID: {recording.agentId}</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 mb-1">Call ID</p>
                  <p className="text-white font-medium font-mono">{recording.callId}</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 mb-1">Date & Time</p>
                  <p className="text-white font-medium">{new Date(recording.createdAt).toLocaleString()}</p>
                </div>
              </div>

              {/* Campaign */}
              {recording.campaign && (
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <p className="text-xs text-gray-500 mb-1">Campaign</p>
                  <p className="text-white font-medium">{recording.campaign}</p>
                </div>
              )}

              {/* Topics */}
              {recording.topics && recording.topics.length > 0 && (
                <div>
                  <p className="text-sm text-gray-400 mb-2">Topics Discussed</p>
                  <div className="flex flex-wrap gap-2">
                    {recording.topics.map((topic) => (
                      <span key={topic} className="px-3 py-1 bg-purple-500/20 text-purple-400 text-sm rounded-full">
                        {topic}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Tags */}
              {recording.tags.length > 0 && (
                <div>
                  <p className="text-sm text-gray-400 mb-2">Tags</p>
                  <div className="flex flex-wrap gap-2">
                    {recording.tags.map((tag) => (
                      <span key={tag} className="px-3 py-1 bg-gray-700/50 text-gray-300 text-sm rounded-full">
                        #{tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Highlights */}
              {recording.highlights && recording.highlights.length > 0 && (
                <div>
                  <p className="text-sm text-gray-400 mb-2">Key Moments</p>
                  <div className="space-y-2">
                    {recording.highlights.map((highlight, index) => (
                      <div
                        key={index}
                        className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg cursor-pointer hover:bg-gray-700/50"
                        onClick={() => onSeek(highlight.timestamp)}
                      >
                        <span className="text-purple-400 font-mono text-sm">{formatDuration(highlight.timestamp)}</span>
                        <span className="text-white">{highlight.text}</span>
                        <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded-full capitalize">
                          {highlight.type.replace("_", " ")}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Notes */}
              {recording.notes && (
                <div>
                  <p className="text-sm text-gray-400 mb-2">Notes</p>
                  <p className="text-gray-300 bg-gray-800/50 rounded-lg p-4">{recording.notes}</p>
                </div>
              )}

              {/* File Info */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-gray-800/50 rounded-lg p-4 text-center">
                  <p className="text-2xl font-bold text-white">{formatDuration(recording.duration)}</p>
                  <p className="text-xs text-gray-500">Duration</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-4 text-center">
                  <p className="text-2xl font-bold text-white">{formatFileSize(recording.fileSize)}</p>
                  <p className="text-xs text-gray-500">File Size</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-4 text-center">
                  <p className="text-2xl font-bold text-white">{recording.qualityScore || "-"}</p>
                  <p className="text-xs text-gray-500">Quality Score</p>
                </div>
              </div>
            </div>
          )}

          {activeTab === "analytics" && (
            <div className="space-y-6">
              {/* Sentiment Analysis */}
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-400 mb-4">Sentiment Analysis</h4>
                <div className="flex items-center gap-4">
                  <SentimentBadge sentiment={recording.sentiment} />
                  <span className="text-white capitalize">{recording.sentiment} Overall</span>
                </div>
              </div>

              {/* Speaker Stats */}
              {recording.transcript && (
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-400 mb-4">Speaking Time</h4>
                  <div className="space-y-3">
                    {["agent", "customer"].map((speaker) => {
                      const segments = recording.transcript!.filter((s) => s.speaker === speaker);
                      const totalTime = segments.reduce((sum, s) => sum + s.duration, 0);
                      const percentage = (totalTime / recording.duration) * 100;

                      return (
                        <div key={speaker}>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm text-gray-300 capitalize">{speaker}</span>
                            <span className="text-sm text-gray-400">{Math.round(percentage)}%</span>
                          </div>
                          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${speaker === "agent" ? "bg-purple-500" : "bg-blue-500"}`}
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Quality Breakdown */}
              {recording.qualityScore && (
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-400 mb-4">Quality Breakdown</h4>
                  <div className="space-y-3">
                    {[
                      { label: "Communication", score: 92 },
                      { label: "Professionalism", score: 95 },
                      { label: "Problem Resolution", score: 88 },
                      { label: "Compliance", score: 100 },
                    ].map((item) => (
                      <div key={item.label}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm text-gray-300">{item.label}</span>
                          <span className="text-sm text-gray-400">{item.score}%</span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              item.score >= 90 ? "bg-green-500" : item.score >= 70 ? "bg-yellow-500" : "bg-red-500"
                            }`}
                            style={{ width: `${item.score}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="p-6 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors flex items-center gap-2">
                <span>📥</span>
                <span>Download</span>
              </button>
              <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors flex items-center gap-2">
                <span>📋</span>
                <span>Copy Transcript</span>
              </button>
              <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors flex items-center gap-2">
                <span>➕</span>
                <span>Add to Playlist</span>
              </button>
            </div>
            <button
              onClick={onClose}
              className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

const PlaylistCard: React.FC<{ playlist: Playlist }> = ({ playlist }) => (
  <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5 hover:border-purple-500/50 transition-all duration-300 cursor-pointer">
    <div className="flex items-start justify-between mb-4">
      <div>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-2xl">📁</span>
          <span className={`px-2 py-0.5 text-xs rounded-full ${
            playlist.isPublic
              ? "bg-green-500/20 text-green-400 border border-green-500/30"
              : "bg-gray-500/20 text-gray-400 border border-gray-500/30"
          }`}>
            {playlist.isPublic ? "Public" : "Private"}
          </span>
        </div>
        <h3 className="font-semibold text-white">{playlist.name}</h3>
        <p className="text-gray-400 text-sm mt-1">{playlist.description}</p>
      </div>
    </div>
    <div className="flex items-center gap-4 text-sm text-gray-500">
      <span>{playlist.items.length} recordings</span>
      <span>•</span>
      <span>Created {new Date(playlist.createdAt).toLocaleDateString()}</span>
    </div>
  </div>
);

// Main Component
export default function RecordingsPage() {
  const [activeTab, setActiveTab] = useState<"library" | "playlists" | "analytics">("library");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedStatus, setSelectedStatus] = useState<RecordingStatus | "all">("all");
  const [selectedOutcome, setSelectedOutcome] = useState<CallOutcome | "all">("all");
  const [selectedSentiment, setSelectedSentiment] = useState<SentimentType | "all">("all");
  const [selectedDirection, setSelectedDirection] = useState<"all" | "inbound" | "outbound">("all");
  const [selectedRecording, setSelectedRecording] = useState<Recording | null>(null);
  const [playingRecordingId, setPlayingRecordingId] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);

  const filteredRecordings = useMemo(() => {
    return mockRecordings.filter((rec) => {
      const matchesSearch =
        rec.customerName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        rec.agentName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        rec.customerPhone.includes(searchQuery) ||
        rec.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()));
      const matchesStatus = selectedStatus === "all" || rec.status === selectedStatus;
      const matchesOutcome = selectedOutcome === "all" || rec.outcome === selectedOutcome;
      const matchesSentiment = selectedSentiment === "all" || rec.sentiment === selectedSentiment;
      const matchesDirection = selectedDirection === "all" || rec.direction === selectedDirection;
      return matchesSearch && matchesStatus && matchesOutcome && matchesSentiment && matchesDirection;
    });
  }, [searchQuery, selectedStatus, selectedOutcome, selectedSentiment, selectedDirection]);

  const handlePlay = (recording: Recording) => {
    if (playingRecordingId === recording.id) {
      setPlayingRecordingId(null);
    } else {
      setPlayingRecordingId(recording.id);
      setCurrentTime(0);
    }
  };

  const handleSeek = (time: number) => {
    setCurrentTime(time);
  };

  const tabs = [
    { id: "library", label: "Recording Library", icon: "🎵" },
    { id: "playlists", label: "Playlists", icon: "📁" },
    { id: "analytics", label: "Analytics", icon: "📊" },
  ];

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-900">
        {/* Header */}
        <div className="border-b border-gray-800 bg-gray-900/95 backdrop-blur-sm sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-2xl font-bold text-white">Call Recordings</h1>
                <p className="text-gray-400 mt-1">Listen, transcribe, and analyze call recordings</p>
              </div>
              <div className="flex items-center gap-3">
                <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
                  Bulk Download
                </button>
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2">
                  <span>+</span>
                  <span>Create Playlist</span>
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
          {/* Library Tab */}
          {activeTab === "library" && (
            <div className="space-y-6">
              {/* Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard title="Total Recordings" value={mockMetrics.totalRecordings.toLocaleString()} icon="🎵" trend={12} />
                <MetricCard title="This Week" value={mockMetrics.recordingsThisWeek} icon="📅" trend={8} />
                <MetricCard title="Avg Duration" value={formatDuration(mockMetrics.avgDuration)} icon="⏱️" />
                <MetricCard title="Storage Used" value={`${mockMetrics.storageUsed} GB`} icon="💾" subtitle="of 100 GB" />
              </div>

              {/* Filters */}
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex-1 min-w-[200px]">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search recordings..."
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
                  value={selectedDirection}
                  onChange={(e) => setSelectedDirection(e.target.value as any)}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Directions</option>
                  <option value="inbound">Inbound</option>
                  <option value="outbound">Outbound</option>
                </select>

                <select
                  value={selectedOutcome}
                  onChange={(e) => setSelectedOutcome(e.target.value as CallOutcome | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Outcomes</option>
                  <option value="successful">Successful</option>
                  <option value="voicemail">Voicemail</option>
                  <option value="no_answer">No Answer</option>
                  <option value="busy">Busy</option>
                  <option value="failed">Failed</option>
                  <option value="transferred">Transferred</option>
                  <option value="callback_scheduled">Callback</option>
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

                <select
                  value={selectedStatus}
                  onChange={(e) => setSelectedStatus(e.target.value as RecordingStatus | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Status</option>
                  <option value="completed">Completed</option>
                  <option value="processing">Processing</option>
                  <option value="failed">Failed</option>
                  <option value="archived">Archived</option>
                </select>
              </div>

              {/* Results Count */}
              <p className="text-sm text-gray-400">
                Showing {filteredRecordings.length} of {mockRecordings.length} recordings
              </p>

              {/* Recordings List */}
              <div className="space-y-4">
                {filteredRecordings.map((recording) => (
                  <RecordingCard
                    key={recording.id}
                    recording={recording}
                    onSelect={setSelectedRecording}
                    onPlay={handlePlay}
                    isPlaying={playingRecordingId === recording.id}
                  />
                ))}
              </div>

              {filteredRecordings.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-gray-400">No recordings found matching your criteria</p>
                </div>
              )}
            </div>
          )}

          {/* Playlists Tab */}
          {activeTab === "playlists" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {mockPlaylists.map((playlist) => (
                  <PlaylistCard key={playlist.id} playlist={playlist} />
                ))}

                {/* Create New Playlist Card */}
                <div className="bg-gray-800/30 border-2 border-dashed border-gray-700 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer hover:border-purple-500/50 transition-colors">
                  <div className="w-12 h-12 bg-gray-700/50 rounded-full flex items-center justify-center text-2xl mb-3">
                    ➕
                  </div>
                  <p className="text-white font-medium">Create New Playlist</p>
                  <p className="text-gray-500 text-sm mt-1">Organize recordings for training</p>
                </div>
              </div>
            </div>
          )}

          {/* Analytics Tab */}
          {activeTab === "analytics" && (
            <div className="space-y-8">
              {/* Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard title="Transcribed" value={mockMetrics.transcribedCount.toLocaleString()} icon="📝" subtitle={`${Math.round((mockMetrics.transcribedCount / mockMetrics.totalRecordings) * 100)}% of total`} />
                <MetricCard title="Avg Quality Score" value={mockMetrics.avgQualityScore} icon="⭐" trend={3} />
                <MetricCard title="Total Duration" value={`${Math.floor(mockMetrics.totalDuration / 3600)}h`} icon="⏱️" />
                <MetricCard title="Positive Sentiment" value={`${mockMetrics.positiveSentiment}%`} icon="😊" trend={5} />
              </div>

              {/* Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Outcome Distribution */}
                <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Call Outcomes</h3>
                  <div className="space-y-3">
                    {[
                      { outcome: "Successful", count: 1456, percentage: 51 },
                      { outcome: "Voicemail", count: 568, percentage: 20 },
                      { outcome: "No Answer", count: 398, percentage: 14 },
                      { outcome: "Transferred", count: 227, percentage: 8 },
                      { outcome: "Failed", count: 142, percentage: 5 },
                      { outcome: "Callback", count: 56, percentage: 2 },
                    ].map((item) => (
                      <div key={item.outcome}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm text-gray-300">{item.outcome}</span>
                          <span className="text-sm text-gray-400">{item.count} ({item.percentage}%)</span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                            style={{ width: `${item.percentage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Duration Distribution */}
                <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Duration Distribution</h3>
                  <div className="space-y-3">
                    {[
                      { range: "0-1 min", count: 234, percentage: 8 },
                      { range: "1-3 min", count: 712, percentage: 25 },
                      { range: "3-5 min", count: 1124, percentage: 39 },
                      { range: "5-10 min", count: 598, percentage: 21 },
                      { range: "10+ min", count: 179, percentage: 7 },
                    ].map((item) => (
                      <div key={item.range}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm text-gray-300">{item.range}</span>
                          <span className="text-sm text-gray-400">{item.count} calls</span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full"
                            style={{ width: `${item.percentage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Agent Performance */}
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Agent Recording Stats</h3>
                <div className="space-y-4">
                  {[
                    { name: "Sales Bot Pro", recordings: 892, avgDuration: "3:24", quality: 94 },
                    { name: "Support Master", recordings: 756, avgDuration: "4:12", quality: 87 },
                    { name: "Customer Care AI", recordings: 623, avgDuration: "2:45", quality: 91 },
                    { name: "Tech Helper", recordings: 412, avgDuration: "5:38", quality: 82 },
                  ].map((agent) => (
                    <div key={agent.name} className="flex items-center gap-4 p-4 bg-gray-700/30 rounded-lg">
                      <div className="w-10 h-10 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center">
                        🤖
                      </div>
                      <div className="flex-1">
                        <p className="font-medium text-white">{agent.name}</p>
                        <p className="text-sm text-gray-400">{agent.recordings} recordings</p>
                      </div>
                      <div className="text-center px-4">
                        <p className="text-white font-medium">{agent.avgDuration}</p>
                        <p className="text-xs text-gray-500">Avg Duration</p>
                      </div>
                      <div className="text-center px-4">
                        <QualityScoreBadge score={agent.quality} />
                        <p className="text-xs text-gray-500 mt-1">Quality</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Storage Usage */}
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Storage Usage</h3>
                <div className="flex items-center gap-8">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-400">Used</span>
                      <span className="text-white font-medium">{mockMetrics.storageUsed} GB / 100 GB</span>
                    </div>
                    <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                        style={{ width: `${(mockMetrics.storageUsed / 100) * 100}%` }}
                      />
                    </div>
                  </div>
                  <button className="px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors">
                    Manage Storage
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Recording Detail Dialog */}
        <RecordingDetailDialog
          recording={selectedRecording}
          onClose={() => setSelectedRecording(null)}
          isPlaying={playingRecordingId === selectedRecording?.id}
          onTogglePlay={() => selectedRecording && handlePlay(selectedRecording)}
          currentTime={currentTime}
          onSeek={handleSeek}
        />
      </div>
    </DashboardLayout>
  );
}
