"use client";

import { useState, useMemo, useCallback } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import {
  Phone,
  Plus,
  Search,
  Filter,
  MoreVertical,
  Play,
  Pause,
  Square,
  Clock,
  Users,
  CheckCircle,
  XCircle,
  PhoneCall,
  PhoneOff,
  PhoneMissed,
  Calendar,
  Target,
  TrendingUp,
  BarChart3,
  Settings,
  Copy,
  Trash2,
  Edit,
  Eye,
  Download,
  Upload,
  FileText,
  UserPlus,
  RefreshCw,
  ChevronRight,
  ChevronDown,
  ArrowUpRight,
  ArrowDownRight,
  Zap,
  Globe,
  Bot,
  MessageSquare,
  Mail,
  Volume2,
  VolumeX,
  AlertTriangle,
  Info,
  Sparkles,
  Layers,
  Timer,
  Activity,
  PieChart,
  LineChart,
  CheckCheck,
  X,
  AlertCircle,
  List,
  Grid3X3,
  SlidersHorizontal,
  Tag,
  Mic,
  Building2,
  Star,
  Archive,
  Send,
  History,
  ClipboardList,
  Workflow,
  Voicemail,
  PhoneIncoming,
  PhoneOutgoing,
} from "lucide-react";

// Types
type CampaignStatus = "draft" | "scheduled" | "running" | "paused" | "completed" | "failed";
type CampaignType = "outbound" | "inbound" | "blended" | "survey" | "notification" | "appointment";
type ContactStatus = "pending" | "calling" | "completed" | "failed" | "dnc" | "voicemail" | "callback";
type DialingMode = "predictive" | "progressive" | "power" | "preview";

interface CampaignStats {
  totalContacts: number;
  contacted: number;
  answered: number;
  voicemail: number;
  noAnswer: number;
  busy: number;
  failed: number;
  dnc: number;
  remaining: number;
  avgCallDuration: number;
  successRate: number;
  conversionRate: number;
  costPerCall: number;
  totalCost: number;
}

interface CampaignSchedule {
  timezone: string;
  startDate: string;
  endDate: string;
  startTime: string;
  endTime: string;
  daysOfWeek: number[];
  maxConcurrentCalls: number;
  callsPerHour: number;
  retryAttempts: number;
  retryInterval: number;
}

interface Campaign {
  id: string;
  name: string;
  description: string;
  type: CampaignType;
  status: CampaignStatus;
  agentId: string;
  agentName: string;
  phoneNumberId: string;
  phoneNumber: string;
  contactListId: string;
  contactListName: string;
  dialingMode: DialingMode;
  stats: CampaignStats;
  schedule: CampaignSchedule;
  script: string;
  tags: string[];
  priority: number;
  createdAt: string;
  updatedAt: string;
  startedAt?: string;
  completedAt?: string;
  createdBy: string;
}

interface ContactList {
  id: string;
  name: string;
  description: string;
  totalContacts: number;
  validContacts: number;
  dncCount: number;
  duplicates: number;
  source: "csv" | "crm" | "api" | "manual";
  lastUpdated: string;
  status: "ready" | "processing" | "error";
}

interface Contact {
  id: string;
  firstName: string;
  lastName: string;
  phone: string;
  email?: string;
  company?: string;
  status: ContactStatus;
  attempts: number;
  lastAttempt?: string;
  callDuration?: number;
  outcome?: string;
  notes?: string;
  customFields: Record<string, string>;
}

// Sample Data
const sampleCampaigns: Campaign[] = [
  {
    id: "camp_1",
    name: "Q1 Sales Outreach",
    description: "Quarterly sales outreach campaign targeting enterprise prospects",
    type: "outbound",
    status: "running",
    agentId: "agent_1",
    agentName: "Sales Assistant AI",
    phoneNumberId: "phone_1",
    phoneNumber: "+1 (555) 123-4567",
    contactListId: "list_1",
    contactListName: "Enterprise Leads Q1",
    dialingMode: "predictive",
    stats: {
      totalContacts: 5000,
      contacted: 3250,
      answered: 2100,
      voicemail: 650,
      noAnswer: 350,
      busy: 100,
      failed: 50,
      dnc: 120,
      remaining: 1750,
      avgCallDuration: 185,
      successRate: 64.6,
      conversionRate: 18.5,
      costPerCall: 0.12,
      totalCost: 390.0,
    },
    schedule: {
      timezone: "America/New_York",
      startDate: "2024-01-15",
      endDate: "2024-03-31",
      startTime: "09:00",
      endTime: "18:00",
      daysOfWeek: [1, 2, 3, 4, 5],
      maxConcurrentCalls: 25,
      callsPerHour: 150,
      retryAttempts: 3,
      retryInterval: 60,
    },
    script: "sales_pitch_v2",
    tags: ["enterprise", "q1", "sales"],
    priority: 1,
    createdAt: "2024-01-10T10:00:00Z",
    updatedAt: "2024-01-20T15:30:00Z",
    startedAt: "2024-01-15T09:00:00Z",
    createdBy: "John Smith",
  },
  {
    id: "camp_2",
    name: "Appointment Reminders",
    description: "Automated appointment reminder calls for healthcare clinic",
    type: "appointment",
    status: "scheduled",
    agentId: "agent_2",
    agentName: "Healthcare Scheduler",
    phoneNumberId: "phone_2",
    phoneNumber: "+1 (555) 234-5678",
    contactListId: "list_2",
    contactListName: "Upcoming Appointments",
    dialingMode: "progressive",
    stats: {
      totalContacts: 450,
      contacted: 0,
      answered: 0,
      voicemail: 0,
      noAnswer: 0,
      busy: 0,
      failed: 0,
      dnc: 0,
      remaining: 450,
      avgCallDuration: 0,
      successRate: 0,
      conversionRate: 0,
      costPerCall: 0.08,
      totalCost: 0,
    },
    schedule: {
      timezone: "America/Los_Angeles",
      startDate: "2024-01-22",
      endDate: "2024-01-22",
      startTime: "08:00",
      endTime: "20:00",
      daysOfWeek: [1, 2, 3, 4, 5, 6],
      maxConcurrentCalls: 10,
      callsPerHour: 60,
      retryAttempts: 2,
      retryInterval: 30,
    },
    script: "appointment_reminder_v1",
    tags: ["healthcare", "reminder", "appointment"],
    priority: 2,
    createdAt: "2024-01-18T14:00:00Z",
    updatedAt: "2024-01-18T14:00:00Z",
    createdBy: "Dr. Sarah Johnson",
  },
  {
    id: "camp_3",
    name: "Customer Satisfaction Survey",
    description: "Post-purchase customer satisfaction survey",
    type: "survey",
    status: "paused",
    agentId: "agent_3",
    agentName: "Survey Bot",
    phoneNumberId: "phone_1",
    phoneNumber: "+1 (555) 123-4567",
    contactListId: "list_3",
    contactListName: "Recent Customers",
    dialingMode: "power",
    stats: {
      totalContacts: 2000,
      contacted: 850,
      answered: 520,
      voicemail: 180,
      noAnswer: 120,
      busy: 25,
      failed: 5,
      dnc: 35,
      remaining: 1150,
      avgCallDuration: 240,
      successRate: 61.2,
      conversionRate: 72.3,
      costPerCall: 0.15,
      totalCost: 127.5,
    },
    schedule: {
      timezone: "America/Chicago",
      startDate: "2024-01-08",
      endDate: "2024-02-15",
      startTime: "10:00",
      endTime: "19:00",
      daysOfWeek: [1, 2, 3, 4, 5],
      maxConcurrentCalls: 15,
      callsPerHour: 80,
      retryAttempts: 2,
      retryInterval: 120,
    },
    script: "csat_survey_v3",
    tags: ["survey", "csat", "customer-feedback"],
    priority: 3,
    createdAt: "2024-01-05T09:00:00Z",
    updatedAt: "2024-01-19T11:20:00Z",
    startedAt: "2024-01-08T10:00:00Z",
    createdBy: "Emily Davis",
  },
  {
    id: "camp_4",
    name: "Lead Qualification",
    description: "Inbound lead qualification and scheduling",
    type: "inbound",
    status: "running",
    agentId: "agent_4",
    agentName: "Lead Qualifier AI",
    phoneNumberId: "phone_3",
    phoneNumber: "+1 (800) 555-1234",
    contactListId: "list_4",
    contactListName: "Inbound Queue",
    dialingMode: "preview",
    stats: {
      totalContacts: 1250,
      contacted: 980,
      answered: 890,
      voicemail: 45,
      noAnswer: 30,
      busy: 10,
      failed: 5,
      dnc: 15,
      remaining: 270,
      avgCallDuration: 320,
      successRate: 90.8,
      conversionRate: 42.5,
      costPerCall: 0.10,
      totalCost: 98.0,
    },
    schedule: {
      timezone: "UTC",
      startDate: "2024-01-01",
      endDate: "2024-12-31",
      startTime: "00:00",
      endTime: "23:59",
      daysOfWeek: [0, 1, 2, 3, 4, 5, 6],
      maxConcurrentCalls: 50,
      callsPerHour: 200,
      retryAttempts: 1,
      retryInterval: 15,
    },
    script: "lead_qualification_v2",
    tags: ["inbound", "leads", "qualification"],
    priority: 1,
    createdAt: "2024-01-01T00:00:00Z",
    updatedAt: "2024-01-20T16:45:00Z",
    startedAt: "2024-01-01T00:00:00Z",
    createdBy: "Mike Wilson",
  },
  {
    id: "camp_5",
    name: "Product Launch Notification",
    description: "VIP customer notification about new product launch",
    type: "notification",
    status: "completed",
    agentId: "agent_5",
    agentName: "Notification Agent",
    phoneNumberId: "phone_2",
    phoneNumber: "+1 (555) 234-5678",
    contactListId: "list_5",
    contactListName: "VIP Customers",
    dialingMode: "power",
    stats: {
      totalContacts: 500,
      contacted: 500,
      answered: 385,
      voicemail: 80,
      noAnswer: 25,
      busy: 8,
      failed: 2,
      dnc: 10,
      remaining: 0,
      avgCallDuration: 95,
      successRate: 77.0,
      conversionRate: 35.8,
      costPerCall: 0.09,
      totalCost: 45.0,
    },
    schedule: {
      timezone: "America/New_York",
      startDate: "2024-01-10",
      endDate: "2024-01-12",
      startTime: "10:00",
      endTime: "18:00",
      daysOfWeek: [1, 2, 3, 4, 5],
      maxConcurrentCalls: 20,
      callsPerHour: 100,
      retryAttempts: 2,
      retryInterval: 60,
    },
    script: "product_launch_notification",
    tags: ["vip", "product-launch", "notification"],
    priority: 1,
    createdAt: "2024-01-08T15:00:00Z",
    updatedAt: "2024-01-12T18:30:00Z",
    startedAt: "2024-01-10T10:00:00Z",
    completedAt: "2024-01-12T16:45:00Z",
    createdBy: "Lisa Chen",
  },
  {
    id: "camp_6",
    name: "Debt Collection",
    description: "Friendly payment reminder campaign",
    type: "outbound",
    status: "draft",
    agentId: "agent_6",
    agentName: "Collections Agent",
    phoneNumberId: "phone_4",
    phoneNumber: "+1 (555) 345-6789",
    contactListId: "list_6",
    contactListName: "Past Due Accounts",
    dialingMode: "progressive",
    stats: {
      totalContacts: 3500,
      contacted: 0,
      answered: 0,
      voicemail: 0,
      noAnswer: 0,
      busy: 0,
      failed: 0,
      dnc: 0,
      remaining: 3500,
      avgCallDuration: 0,
      successRate: 0,
      conversionRate: 0,
      costPerCall: 0.12,
      totalCost: 0,
    },
    schedule: {
      timezone: "America/Denver",
      startDate: "2024-02-01",
      endDate: "2024-02-28",
      startTime: "09:00",
      endTime: "20:00",
      daysOfWeek: [1, 2, 3, 4, 5, 6],
      maxConcurrentCalls: 30,
      callsPerHour: 120,
      retryAttempts: 5,
      retryInterval: 1440,
    },
    script: "payment_reminder_v1",
    tags: ["collections", "payment", "reminder"],
    priority: 2,
    createdAt: "2024-01-19T10:00:00Z",
    updatedAt: "2024-01-19T10:00:00Z",
    createdBy: "Robert Taylor",
  },
];

const sampleContactLists: ContactList[] = [
  {
    id: "list_1",
    name: "Enterprise Leads Q1",
    description: "Enterprise prospects for Q1 sales campaign",
    totalContacts: 5000,
    validContacts: 4850,
    dncCount: 120,
    duplicates: 30,
    source: "crm",
    lastUpdated: "2024-01-14T12:00:00Z",
    status: "ready",
  },
  {
    id: "list_2",
    name: "Upcoming Appointments",
    description: "Patients with appointments in the next 48 hours",
    totalContacts: 450,
    validContacts: 445,
    dncCount: 3,
    duplicates: 2,
    source: "api",
    lastUpdated: "2024-01-21T06:00:00Z",
    status: "ready",
  },
  {
    id: "list_3",
    name: "Recent Customers",
    description: "Customers who made purchases in the last 30 days",
    totalContacts: 2000,
    validContacts: 1950,
    dncCount: 35,
    duplicates: 15,
    source: "crm",
    lastUpdated: "2024-01-20T00:00:00Z",
    status: "ready",
  },
];

// Utility functions
const getStatusColor = (status: CampaignStatus) => {
  switch (status) {
    case "draft":
      return "bg-gray-500/20 text-gray-400";
    case "scheduled":
      return "bg-blue-500/20 text-blue-400";
    case "running":
      return "bg-green-500/20 text-green-400";
    case "paused":
      return "bg-yellow-500/20 text-yellow-400";
    case "completed":
      return "bg-purple-500/20 text-purple-400";
    case "failed":
      return "bg-red-500/20 text-red-400";
    default:
      return "bg-gray-500/20 text-gray-400";
  }
};

const getStatusIcon = (status: CampaignStatus) => {
  switch (status) {
    case "draft":
      return <Edit className="h-3.5 w-3.5" />;
    case "scheduled":
      return <Clock className="h-3.5 w-3.5" />;
    case "running":
      return <Play className="h-3.5 w-3.5" />;
    case "paused":
      return <Pause className="h-3.5 w-3.5" />;
    case "completed":
      return <CheckCircle className="h-3.5 w-3.5" />;
    case "failed":
      return <XCircle className="h-3.5 w-3.5" />;
    default:
      return <Info className="h-3.5 w-3.5" />;
  }
};

const getCampaignTypeIcon = (type: CampaignType) => {
  switch (type) {
    case "outbound":
      return <PhoneOutgoing className="h-4 w-4" />;
    case "inbound":
      return <PhoneIncoming className="h-4 w-4" />;
    case "blended":
      return <Phone className="h-4 w-4" />;
    case "survey":
      return <ClipboardList className="h-4 w-4" />;
    case "notification":
      return <MessageSquare className="h-4 w-4" />;
    case "appointment":
      return <Calendar className="h-4 w-4" />;
    default:
      return <Phone className="h-4 w-4" />;
  }
};

const formatDuration = (seconds: number) => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
};

const formatNumber = (num: number) => {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
};

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(amount);
};

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
};

const formatDateTime = (dateString: string) => {
  return new Date(dateString).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
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
}: {
  title: string;
  value: string | number;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: any;
  color: string;
  subtitle?: string;
}) {
  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-5 hover:border-white/10 transition-all">
      <div className="flex items-center justify-between">
        <div
          className={`p-2.5 rounded-lg ${color} bg-opacity-20`}
          style={{ backgroundColor: `${color}20` }}
        >
          <Icon className={`h-5 w-5`} style={{ color }} />
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
      <div className="mt-3">
        <div className="text-2xl font-bold text-white">{value}</div>
        <div className="text-sm text-gray-400 mt-0.5">{title}</div>
        {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
      </div>
    </div>
  );
}

function CampaignCard({ campaign, onAction }: { campaign: Campaign; onAction: (action: string, campaign: Campaign) => void }) {
  const [showMenu, setShowMenu] = useState(false);
  const progressPercent = campaign.stats.totalContacts > 0
    ? Math.round((campaign.stats.contacted / campaign.stats.totalContacts) * 100)
    : 0;

  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 hover:border-white/10 transition-all group">
      <div className="p-5">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div
              className={`p-2.5 rounded-lg ${
                campaign.status === "running"
                  ? "bg-green-500/20"
                  : campaign.status === "paused"
                  ? "bg-yellow-500/20"
                  : campaign.status === "completed"
                  ? "bg-purple-500/20"
                  : "bg-gray-500/20"
              }`}
            >
              {getCampaignTypeIcon(campaign.type)}
            </div>
            <div>
              <h3 className="font-semibold text-white group-hover:text-purple-400 transition-colors">
                {campaign.name}
              </h3>
              <p className="text-sm text-gray-400 line-clamp-1">{campaign.description}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={`px-2.5 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 ${getStatusColor(campaign.status)}`}>
              {getStatusIcon(campaign.status)}
              {campaign.status.charAt(0).toUpperCase() + campaign.status.slice(1)}
            </span>
            <div className="relative">
              <button
                onClick={() => setShowMenu(!showMenu)}
                className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
              >
                <MoreVertical className="h-4 w-4" />
              </button>
              {showMenu && (
                <div className="absolute right-0 top-full mt-1 w-48 bg-[#252542] rounded-lg border border-white/10 shadow-xl z-10 py-1">
                  <button
                    onClick={() => {
                      onAction("view", campaign);
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                  >
                    <Eye className="h-4 w-4" />
                    View Details
                  </button>
                  <button
                    onClick={() => {
                      onAction("edit", campaign);
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                  >
                    <Edit className="h-4 w-4" />
                    Edit Campaign
                  </button>
                  <button
                    onClick={() => {
                      onAction("duplicate", campaign);
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                  >
                    <Copy className="h-4 w-4" />
                    Duplicate
                  </button>
                  <button
                    onClick={() => {
                      onAction("export", campaign);
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                  >
                    <Download className="h-4 w-4" />
                    Export Results
                  </button>
                  <div className="border-t border-white/5 my-1" />
                  <button
                    onClick={() => {
                      onAction("delete", campaign);
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-2 text-left text-sm text-red-400 hover:bg-red-500/10 flex items-center gap-2"
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Meta Info */}
        <div className="flex flex-wrap gap-4 text-sm text-gray-400 mb-4">
          <div className="flex items-center gap-1.5">
            <Bot className="h-4 w-4" />
            {campaign.agentName}
          </div>
          <div className="flex items-center gap-1.5">
            <Phone className="h-4 w-4" />
            {campaign.phoneNumber}
          </div>
          <div className="flex items-center gap-1.5">
            <Users className="h-4 w-4" />
            {formatNumber(campaign.stats.totalContacts)} contacts
          </div>
        </div>

        {/* Progress */}
        <div className="mb-4">
          <div className="flex items-center justify-between text-sm mb-1.5">
            <span className="text-gray-400">Progress</span>
            <span className="text-white font-medium">{progressPercent}%</span>
          </div>
          <div className="h-2 bg-white/5 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                campaign.status === "running"
                  ? "bg-gradient-to-r from-green-500 to-emerald-500"
                  : campaign.status === "paused"
                  ? "bg-gradient-to-r from-yellow-500 to-orange-500"
                  : campaign.status === "completed"
                  ? "bg-gradient-to-r from-purple-500 to-pink-500"
                  : "bg-gradient-to-r from-gray-500 to-gray-600"
              }`}
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-4 gap-3 mb-4">
          <div className="text-center p-2 bg-white/5 rounded-lg">
            <div className="text-sm font-semibold text-green-400">
              {formatNumber(campaign.stats.answered)}
            </div>
            <div className="text-xs text-gray-500">Answered</div>
          </div>
          <div className="text-center p-2 bg-white/5 rounded-lg">
            <div className="text-sm font-semibold text-yellow-400">
              {formatNumber(campaign.stats.voicemail)}
            </div>
            <div className="text-xs text-gray-500">Voicemail</div>
          </div>
          <div className="text-center p-2 bg-white/5 rounded-lg">
            <div className="text-sm font-semibold text-gray-400">
              {formatNumber(campaign.stats.noAnswer)}
            </div>
            <div className="text-xs text-gray-500">No Answer</div>
          </div>
          <div className="text-center p-2 bg-white/5 rounded-lg">
            <div className="text-sm font-semibold text-blue-400">
              {formatNumber(campaign.stats.remaining)}
            </div>
            <div className="text-xs text-gray-500">Remaining</div>
          </div>
        </div>

        {/* Footer Stats */}
        <div className="flex items-center justify-between pt-4 border-t border-white/5">
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-1.5">
              <TrendingUp className="h-4 w-4 text-green-400" />
              <span className="text-white">{campaign.stats.successRate}%</span>
              <span className="text-gray-500">success</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Target className="h-4 w-4 text-purple-400" />
              <span className="text-white">{campaign.stats.conversionRate}%</span>
              <span className="text-gray-500">conversion</span>
            </div>
          </div>
          <div className="text-sm text-gray-400">
            {formatCurrency(campaign.stats.totalCost)} spent
          </div>
        </div>
      </div>

      {/* Action Bar */}
      <div className="px-5 py-3 border-t border-white/5 bg-white/[0.02] flex items-center justify-between">
        <div className="text-xs text-gray-500">
          {campaign.status === "running" || campaign.status === "paused" ? (
            <>Started {formatDateTime(campaign.startedAt || campaign.createdAt)}</>
          ) : campaign.status === "completed" ? (
            <>Completed {formatDateTime(campaign.completedAt || campaign.updatedAt)}</>
          ) : campaign.status === "scheduled" ? (
            <>Scheduled for {formatDateTime(campaign.schedule.startDate)}</>
          ) : (
            <>Created {formatDateTime(campaign.createdAt)}</>
          )}
        </div>
        <div className="flex items-center gap-2">
          {campaign.status === "draft" && (
            <button
              onClick={() => onAction("start", campaign)}
              className="px-3 py-1.5 bg-green-500/20 text-green-400 rounded-lg text-sm font-medium hover:bg-green-500/30 transition-colors flex items-center gap-1.5"
            >
              <Play className="h-3.5 w-3.5" />
              Start
            </button>
          )}
          {campaign.status === "running" && (
            <>
              <button
                onClick={() => onAction("pause", campaign)}
                className="px-3 py-1.5 bg-yellow-500/20 text-yellow-400 rounded-lg text-sm font-medium hover:bg-yellow-500/30 transition-colors flex items-center gap-1.5"
              >
                <Pause className="h-3.5 w-3.5" />
                Pause
              </button>
              <button
                onClick={() => onAction("stop", campaign)}
                className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded-lg text-sm font-medium hover:bg-red-500/30 transition-colors flex items-center gap-1.5"
              >
                <Square className="h-3.5 w-3.5" />
                Stop
              </button>
            </>
          )}
          {campaign.status === "paused" && (
            <button
              onClick={() => onAction("resume", campaign)}
              className="px-3 py-1.5 bg-green-500/20 text-green-400 rounded-lg text-sm font-medium hover:bg-green-500/30 transition-colors flex items-center gap-1.5"
            >
              <Play className="h-3.5 w-3.5" />
              Resume
            </button>
          )}
          {campaign.status === "scheduled" && (
            <button
              onClick={() => onAction("cancel", campaign)}
              className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded-lg text-sm font-medium hover:bg-red-500/30 transition-colors flex items-center gap-1.5"
            >
              <X className="h-3.5 w-3.5" />
              Cancel
            </button>
          )}
          <button
            onClick={() => onAction("view", campaign)}
            className="px-3 py-1.5 bg-white/5 text-gray-300 rounded-lg text-sm font-medium hover:bg-white/10 transition-colors flex items-center gap-1.5"
          >
            <Eye className="h-3.5 w-3.5" />
            Details
          </button>
        </div>
      </div>
    </div>
  );
}

function CreateCampaignDialog({
  isOpen,
  onClose,
  contactLists,
}: {
  isOpen: boolean;
  onClose: () => void;
  contactLists: ContactList[];
}) {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    type: "outbound" as CampaignType,
    agentId: "",
    phoneNumberId: "",
    contactListId: "",
    dialingMode: "predictive" as DialingMode,
    timezone: "America/New_York",
    startDate: "",
    endDate: "",
    startTime: "09:00",
    endTime: "18:00",
    daysOfWeek: [1, 2, 3, 4, 5],
    maxConcurrentCalls: 25,
    callsPerHour: 100,
    retryAttempts: 3,
    retryInterval: 60,
    tags: [] as string[],
    priority: 2,
  });

  if (!isOpen) return null;

  const campaignTypes: { value: CampaignType; label: string; description: string; icon: any }[] = [
    { value: "outbound", label: "Outbound", description: "Make calls to contact list", icon: PhoneOutgoing },
    { value: "inbound", label: "Inbound", description: "Handle incoming calls", icon: PhoneIncoming },
    { value: "blended", label: "Blended", description: "Both inbound and outbound", icon: Phone },
    { value: "survey", label: "Survey", description: "Conduct surveys via phone", icon: ClipboardList },
    { value: "notification", label: "Notification", description: "Send automated notifications", icon: MessageSquare },
    { value: "appointment", label: "Appointment", description: "Reminder and scheduling", icon: Calendar },
  ];

  const dialingModes: { value: DialingMode; label: string; description: string }[] = [
    { value: "predictive", label: "Predictive", description: "AI predicts agent availability for maximum efficiency" },
    { value: "progressive", label: "Progressive", description: "Dials next number when agent becomes available" },
    { value: "power", label: "Power", description: "Multiple lines dialed per agent" },
    { value: "preview", label: "Preview", description: "Agent reviews contact info before each call" },
  ];

  const timezones = [
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Phoenix",
    "America/Anchorage",
    "Pacific/Honolulu",
    "UTC",
  ];

  const daysOfWeek = [
    { value: 0, label: "Sun" },
    { value: 1, label: "Mon" },
    { value: 2, label: "Tue" },
    { value: 3, label: "Wed" },
    { value: 4, label: "Thu" },
    { value: 5, label: "Fri" },
    { value: 6, label: "Sat" },
  ];

  const toggleDay = (day: number) => {
    if (formData.daysOfWeek.includes(day)) {
      setFormData({ ...formData, daysOfWeek: formData.daysOfWeek.filter((d) => d !== day) });
    } else {
      setFormData({ ...formData, daysOfWeek: [...formData.daysOfWeek, day].sort() });
    }
  };

  const sampleAgents = [
    { id: "agent_1", name: "Sales Assistant AI" },
    { id: "agent_2", name: "Healthcare Scheduler" },
    { id: "agent_3", name: "Survey Bot" },
    { id: "agent_4", name: "Lead Qualifier AI" },
  ];

  const samplePhoneNumbers = [
    { id: "phone_1", number: "+1 (555) 123-4567", label: "Main Line" },
    { id: "phone_2", number: "+1 (555) 234-5678", label: "Support Line" },
    { id: "phone_3", number: "+1 (800) 555-1234", label: "Toll Free" },
  ];

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a2e] rounded-2xl border border-white/10 w-full max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">Create Campaign</h2>
            <p className="text-sm text-gray-400 mt-0.5">Step {step} of 4</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Progress */}
        <div className="px-6 py-3 border-b border-white/5">
          <div className="flex items-center gap-2">
            {[1, 2, 3, 4].map((s) => (
              <div key={s} className="flex items-center flex-1">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    s < step
                      ? "bg-purple-500 text-white"
                      : s === step
                      ? "bg-purple-500/20 text-purple-400 border border-purple-500"
                      : "bg-white/5 text-gray-500"
                  }`}
                >
                  {s < step ? <CheckCircle className="h-4 w-4" /> : s}
                </div>
                {s < 4 && (
                  <div
                    className={`flex-1 h-0.5 mx-2 ${
                      s < step ? "bg-purple-500" : "bg-white/10"
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-between mt-2 text-xs text-gray-500">
            <span>Basic Info</span>
            <span>Configuration</span>
            <span>Schedule</span>
            <span>Review</span>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Step 1: Basic Info */}
          {step === 1 && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Campaign Name
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="e.g., Q1 Sales Outreach"
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Description
                </label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="Describe the purpose and goals of this campaign..."
                  rows={3}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Campaign Type
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {campaignTypes.map((type) => (
                    <button
                      key={type.value}
                      onClick={() => setFormData({ ...formData, type: type.value })}
                      className={`p-4 rounded-xl border text-left transition-all ${
                        formData.type === type.value
                          ? "bg-purple-500/20 border-purple-500"
                          : "bg-white/5 border-white/10 hover:border-white/20"
                      }`}
                    >
                      <type.icon
                        className={`h-5 w-5 mb-2 ${
                          formData.type === type.value ? "text-purple-400" : "text-gray-400"
                        }`}
                      />
                      <div className="font-medium text-white text-sm">{type.label}</div>
                      <div className="text-xs text-gray-500 mt-0.5">{type.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Configuration */}
          {step === 2 && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    AI Agent
                  </label>
                  <select
                    value={formData.agentId}
                    onChange={(e) => setFormData({ ...formData, agentId: e.target.value })}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  >
                    <option value="">Select an agent</option>
                    {sampleAgents.map((agent) => (
                      <option key={agent.id} value={agent.id}>
                        {agent.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Phone Number
                  </label>
                  <select
                    value={formData.phoneNumberId}
                    onChange={(e) => setFormData({ ...formData, phoneNumberId: e.target.value })}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  >
                    <option value="">Select a phone number</option>
                    {samplePhoneNumbers.map((phone) => (
                      <option key={phone.id} value={phone.id}>
                        {phone.number} - {phone.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Contact List
                </label>
                <select
                  value={formData.contactListId}
                  onChange={(e) => setFormData({ ...formData, contactListId: e.target.value })}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="">Select a contact list</option>
                  {contactLists.map((list) => (
                    <option key={list.id} value={list.id}>
                      {list.name} ({formatNumber(list.validContacts)} contacts)
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Dialing Mode
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {dialingModes.map((mode) => (
                    <button
                      key={mode.value}
                      onClick={() => setFormData({ ...formData, dialingMode: mode.value })}
                      className={`p-4 rounded-xl border text-left transition-all ${
                        formData.dialingMode === mode.value
                          ? "bg-purple-500/20 border-purple-500"
                          : "bg-white/5 border-white/10 hover:border-white/20"
                      }`}
                    >
                      <div className="font-medium text-white text-sm">{mode.label}</div>
                      <div className="text-xs text-gray-500 mt-1">{mode.description}</div>
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Max Concurrent Calls
                  </label>
                  <input
                    type="number"
                    value={formData.maxConcurrentCalls}
                    onChange={(e) =>
                      setFormData({ ...formData, maxConcurrentCalls: parseInt(e.target.value) || 1 })
                    }
                    min={1}
                    max={100}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Calls Per Hour
                  </label>
                  <input
                    type="number"
                    value={formData.callsPerHour}
                    onChange={(e) =>
                      setFormData({ ...formData, callsPerHour: parseInt(e.target.value) || 1 })
                    }
                    min={1}
                    max={500}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Schedule */}
          {step === 3 && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Timezone
                </label>
                <select
                  value={formData.timezone}
                  onChange={(e) => setFormData({ ...formData, timezone: e.target.value })}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                >
                  {timezones.map((tz) => (
                    <option key={tz} value={tz}>
                      {tz}
                    </option>
                  ))}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Start Date
                  </label>
                  <input
                    type="date"
                    value={formData.startDate}
                    onChange={(e) => setFormData({ ...formData, startDate: e.target.value })}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    End Date
                  </label>
                  <input
                    type="date"
                    value={formData.endDate}
                    onChange={(e) => setFormData({ ...formData, endDate: e.target.value })}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Daily Start Time
                  </label>
                  <input
                    type="time"
                    value={formData.startTime}
                    onChange={(e) => setFormData({ ...formData, startTime: e.target.value })}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Daily End Time
                  </label>
                  <input
                    type="time"
                    value={formData.endTime}
                    onChange={(e) => setFormData({ ...formData, endTime: e.target.value })}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Days of Week
                </label>
                <div className="flex gap-2">
                  {daysOfWeek.map((day) => (
                    <button
                      key={day.value}
                      onClick={() => toggleDay(day.value)}
                      className={`w-12 h-10 rounded-lg text-sm font-medium transition-all ${
                        formData.daysOfWeek.includes(day.value)
                          ? "bg-purple-500 text-white"
                          : "bg-white/5 text-gray-400 hover:bg-white/10"
                      }`}
                    >
                      {day.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Retry Attempts
                  </label>
                  <input
                    type="number"
                    value={formData.retryAttempts}
                    onChange={(e) =>
                      setFormData({ ...formData, retryAttempts: parseInt(e.target.value) || 0 })
                    }
                    min={0}
                    max={10}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">Number of retry attempts for failed calls</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Retry Interval (minutes)
                  </label>
                  <input
                    type="number"
                    value={formData.retryInterval}
                    onChange={(e) =>
                      setFormData({ ...formData, retryInterval: parseInt(e.target.value) || 1 })
                    }
                    min={1}
                    max={1440}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">Time between retry attempts</p>
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Review */}
          {step === 4 && (
            <div className="space-y-6">
              <div className="bg-white/5 rounded-xl p-5">
                <h3 className="font-semibold text-white mb-4">Campaign Summary</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Name</div>
                    <div className="text-white mt-1">{formData.name || "-"}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Type</div>
                    <div className="text-white mt-1 capitalize">{formData.type}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Dialing Mode</div>
                    <div className="text-white mt-1 capitalize">{formData.dialingMode}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Max Concurrent</div>
                    <div className="text-white mt-1">{formData.maxConcurrentCalls} calls</div>
                  </div>
                </div>
              </div>
              <div className="bg-white/5 rounded-xl p-5">
                <h3 className="font-semibold text-white mb-4">Schedule</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Date Range</div>
                    <div className="text-white mt-1">
                      {formData.startDate || "-"} to {formData.endDate || "-"}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Time Window</div>
                    <div className="text-white mt-1">
                      {formData.startTime} - {formData.endTime}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Timezone</div>
                    <div className="text-white mt-1">{formData.timezone}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Active Days</div>
                    <div className="text-white mt-1">
                      {formData.daysOfWeek.map((d) => daysOfWeek[d].label).join(", ")}
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-purple-500/10 rounded-xl p-5 border border-purple-500/20">
                <div className="flex items-start gap-3">
                  <Info className="h-5 w-5 text-purple-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="font-medium text-purple-400">Ready to Launch</div>
                    <div className="text-sm text-gray-400 mt-1">
                      Your campaign will be created as a draft. You can review and start it from the
                      campaigns dashboard.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-white/5 flex items-center justify-between">
          <button
            onClick={() => (step > 1 ? setStep(step - 1) : onClose())}
            className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            {step === 1 ? "Cancel" : "Back"}
          </button>
          <button
            onClick={() => (step < 4 ? setStep(step + 1) : onClose())}
            className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
          >
            {step === 4 ? (
              <>
                <Sparkles className="h-4 w-4" />
                Create Campaign
              </>
            ) : (
              <>
                Next
                <ChevronRight className="h-4 w-4" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

function ContactListDialog({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const [step, setStep] = useState(1);
  const [uploadMethod, setUploadMethod] = useState<"csv" | "crm" | "manual">("csv");
  const [listName, setListName] = useState("");
  const [description, setDescription] = useState("");

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a2e] rounded-2xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">Create Contact List</h2>
            <p className="text-sm text-gray-400 mt-0.5">Step {step} of 3</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {step === 1 && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  List Name
                </label>
                <input
                  type="text"
                  value={listName}
                  onChange={(e) => setListName(e.target.value)}
                  placeholder="e.g., Q1 Sales Leads"
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe the contacts in this list..."
                  rows={3}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Import Method
                </label>
                <div className="grid grid-cols-3 gap-3">
                  <button
                    onClick={() => setUploadMethod("csv")}
                    className={`p-4 rounded-xl border text-center transition-all ${
                      uploadMethod === "csv"
                        ? "bg-purple-500/20 border-purple-500"
                        : "bg-white/5 border-white/10 hover:border-white/20"
                    }`}
                  >
                    <Upload className={`h-6 w-6 mx-auto mb-2 ${uploadMethod === "csv" ? "text-purple-400" : "text-gray-400"}`} />
                    <div className="font-medium text-white text-sm">CSV Upload</div>
                    <div className="text-xs text-gray-500 mt-1">Upload a file</div>
                  </button>
                  <button
                    onClick={() => setUploadMethod("crm")}
                    className={`p-4 rounded-xl border text-center transition-all ${
                      uploadMethod === "crm"
                        ? "bg-purple-500/20 border-purple-500"
                        : "bg-white/5 border-white/10 hover:border-white/20"
                    }`}
                  >
                    <Building2 className={`h-6 w-6 mx-auto mb-2 ${uploadMethod === "crm" ? "text-purple-400" : "text-gray-400"}`} />
                    <div className="font-medium text-white text-sm">CRM Sync</div>
                    <div className="text-xs text-gray-500 mt-1">Import from CRM</div>
                  </button>
                  <button
                    onClick={() => setUploadMethod("manual")}
                    className={`p-4 rounded-xl border text-center transition-all ${
                      uploadMethod === "manual"
                        ? "bg-purple-500/20 border-purple-500"
                        : "bg-white/5 border-white/10 hover:border-white/20"
                    }`}
                  >
                    <UserPlus className={`h-6 w-6 mx-auto mb-2 ${uploadMethod === "manual" ? "text-purple-400" : "text-gray-400"}`} />
                    <div className="font-medium text-white text-sm">Manual Entry</div>
                    <div className="text-xs text-gray-500 mt-1">Add one by one</div>
                  </button>
                </div>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-6">
              {uploadMethod === "csv" && (
                <div className="border-2 border-dashed border-white/10 rounded-xl p-12 text-center hover:border-purple-500/50 transition-colors cursor-pointer">
                  <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <div className="text-lg font-medium text-white mb-2">
                    Drop your CSV file here
                  </div>
                  <p className="text-sm text-gray-400 mb-4">
                    or click to browse from your computer
                  </p>
                  <button className="px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg text-sm font-medium hover:bg-purple-500/30 transition-colors">
                    Select File
                  </button>
                  <div className="mt-6 text-xs text-gray-500">
                    Supported: CSV, XLSX, XLS (max 100MB)
                  </div>
                </div>
              )}
              {uploadMethod === "crm" && (
                <div className="space-y-4">
                  <p className="text-sm text-gray-400">Select your CRM to import contacts:</p>
                  <div className="grid grid-cols-2 gap-3">
                    {["Salesforce", "HubSpot", "Pipedrive", "Zoho CRM"].map((crm) => (
                      <button
                        key={crm}
                        className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-purple-500/50 transition-all text-left"
                      >
                        <div className="font-medium text-white">{crm}</div>
                        <div className="text-xs text-gray-500 mt-1">Connect and import</div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {uploadMethod === "manual" && (
                <div className="space-y-4">
                  <p className="text-sm text-gray-400">Add contacts manually:</p>
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                      <input
                        type="text"
                        placeholder="First Name"
                        className="px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                      />
                      <input
                        type="text"
                        placeholder="Last Name"
                        className="px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                      />
                    </div>
                    <input
                      type="tel"
                      placeholder="Phone Number"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                    />
                    <input
                      type="email"
                      placeholder="Email (optional)"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                    />
                    <button className="w-full px-4 py-3 bg-purple-500/20 text-purple-400 rounded-xl font-medium hover:bg-purple-500/30 transition-colors flex items-center justify-center gap-2">
                      <Plus className="h-4 w-4" />
                      Add Contact
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {step === 3 && (
            <div className="space-y-6">
              <div className="bg-green-500/10 rounded-xl p-5 border border-green-500/20">
                <div className="flex items-start gap-3">
                  <CheckCircle className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="font-medium text-green-400">List Ready</div>
                    <div className="text-sm text-gray-400 mt-1">
                      Your contact list has been processed and is ready for campaigns.
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-white/5 rounded-xl p-5">
                <h3 className="font-semibold text-white mb-4">Summary</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">List Name</div>
                    <div className="text-white mt-1">{listName || "Untitled List"}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Import Method</div>
                    <div className="text-white mt-1 capitalize">{uploadMethod}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Total Contacts</div>
                    <div className="text-white mt-1">0</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Valid Contacts</div>
                    <div className="text-white mt-1">0</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-white/5 flex items-center justify-between">
          <button
            onClick={() => (step > 1 ? setStep(step - 1) : onClose())}
            className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            {step === 1 ? "Cancel" : "Back"}
          </button>
          <button
            onClick={() => (step < 3 ? setStep(step + 1) : onClose())}
            className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
          >
            {step === 3 ? (
              <>
                <CheckCircle className="h-4 w-4" />
                Create List
              </>
            ) : (
              <>
                Next
                <ChevronRight className="h-4 w-4" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

function CampaignDetailsDialog({
  campaign,
  isOpen,
  onClose,
}: {
  campaign: Campaign | null;
  isOpen: boolean;
  onClose: () => void;
}) {
  const [activeTab, setActiveTab] = useState<"overview" | "contacts" | "analytics" | "settings">("overview");

  if (!isOpen || !campaign) return null;

  const tabs = [
    { id: "overview", label: "Overview", icon: Eye },
    { id: "contacts", label: "Contacts", icon: Users },
    { id: "analytics", label: "Analytics", icon: BarChart3 },
    { id: "settings", label: "Settings", icon: Settings },
  ];

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a2e] rounded-2xl border border-white/10 w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div
              className={`p-3 rounded-xl ${
                campaign.status === "running"
                  ? "bg-green-500/20"
                  : campaign.status === "paused"
                  ? "bg-yellow-500/20"
                  : "bg-gray-500/20"
              }`}
            >
              {getCampaignTypeIcon(campaign.type)}
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">{campaign.name}</h2>
              <p className="text-sm text-gray-400 mt-0.5">{campaign.description}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span
              className={`px-3 py-1.5 rounded-full text-sm font-medium flex items-center gap-2 ${getStatusColor(
                campaign.status
              )}`}
            >
              {getStatusIcon(campaign.status)}
              {campaign.status.charAt(0).toUpperCase() + campaign.status.slice(1)}
            </span>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="px-6 border-b border-white/5">
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
                  activeTab === tab.id
                    ? "text-purple-400 border-purple-500"
                    : "text-gray-400 border-transparent hover:text-white"
                }`}
              >
                <tab.icon className="h-4 w-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === "overview" && (
            <div className="space-y-6">
              {/* Stats Grid */}
              <div className="grid grid-cols-4 gap-4">
                <StatsCard
                  title="Total Contacts"
                  value={formatNumber(campaign.stats.totalContacts)}
                  icon={Users}
                  color="#8B5CF6"
                />
                <StatsCard
                  title="Answered"
                  value={formatNumber(campaign.stats.answered)}
                  subtitle={`${campaign.stats.successRate}% success rate`}
                  icon={PhoneCall}
                  color="#10B981"
                />
                <StatsCard
                  title="Conversion Rate"
                  value={`${campaign.stats.conversionRate}%`}
                  icon={Target}
                  color="#F59E0B"
                />
                <StatsCard
                  title="Total Cost"
                  value={formatCurrency(campaign.stats.totalCost)}
                  subtitle={`${formatCurrency(campaign.stats.costPerCall)}/call`}
                  icon={TrendingUp}
                  color="#EC4899"
                />
              </div>

              {/* Progress */}
              <div className="bg-white/5 rounded-xl p-5">
                <h3 className="font-semibold text-white mb-4">Campaign Progress</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-400">Overall Progress</span>
                      <span className="text-white">
                        {formatNumber(campaign.stats.contacted)} / {formatNumber(campaign.stats.totalContacts)}
                      </span>
                    </div>
                    <div className="h-3 bg-white/5 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                        style={{
                          width: `${(campaign.stats.contacted / campaign.stats.totalContacts) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-6 gap-3">
                    <div className="text-center p-3 bg-green-500/10 rounded-lg">
                      <div className="text-lg font-semibold text-green-400">
                        {formatNumber(campaign.stats.answered)}
                      </div>
                      <div className="text-xs text-gray-500">Answered</div>
                    </div>
                    <div className="text-center p-3 bg-yellow-500/10 rounded-lg">
                      <div className="text-lg font-semibold text-yellow-400">
                        {formatNumber(campaign.stats.voicemail)}
                      </div>
                      <div className="text-xs text-gray-500">Voicemail</div>
                    </div>
                    <div className="text-center p-3 bg-gray-500/10 rounded-lg">
                      <div className="text-lg font-semibold text-gray-400">
                        {formatNumber(campaign.stats.noAnswer)}
                      </div>
                      <div className="text-xs text-gray-500">No Answer</div>
                    </div>
                    <div className="text-center p-3 bg-orange-500/10 rounded-lg">
                      <div className="text-lg font-semibold text-orange-400">
                        {formatNumber(campaign.stats.busy)}
                      </div>
                      <div className="text-xs text-gray-500">Busy</div>
                    </div>
                    <div className="text-center p-3 bg-red-500/10 rounded-lg">
                      <div className="text-lg font-semibold text-red-400">
                        {formatNumber(campaign.stats.failed)}
                      </div>
                      <div className="text-xs text-gray-500">Failed</div>
                    </div>
                    <div className="text-center p-3 bg-purple-500/10 rounded-lg">
                      <div className="text-lg font-semibold text-purple-400">
                        {formatNumber(campaign.stats.dnc)}
                      </div>
                      <div className="text-xs text-gray-500">DNC</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Campaign Details */}
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-white/5 rounded-xl p-5">
                  <h3 className="font-semibold text-white mb-4">Configuration</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Agent</span>
                      <span className="text-white">{campaign.agentName}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Phone Number</span>
                      <span className="text-white">{campaign.phoneNumber}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Contact List</span>
                      <span className="text-white">{campaign.contactListName}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Dialing Mode</span>
                      <span className="text-white capitalize">{campaign.dialingMode}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Avg Call Duration</span>
                      <span className="text-white">{formatDuration(campaign.stats.avgCallDuration)}</span>
                    </div>
                  </div>
                </div>
                <div className="bg-white/5 rounded-xl p-5">
                  <h3 className="font-semibold text-white mb-4">Schedule</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Timezone</span>
                      <span className="text-white">{campaign.schedule.timezone}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Date Range</span>
                      <span className="text-white">
                        {formatDate(campaign.schedule.startDate)} - {formatDate(campaign.schedule.endDate)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Time Window</span>
                      <span className="text-white">
                        {campaign.schedule.startTime} - {campaign.schedule.endTime}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Max Concurrent</span>
                      <span className="text-white">{campaign.schedule.maxConcurrentCalls} calls</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Retry Attempts</span>
                      <span className="text-white">{campaign.schedule.retryAttempts}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "contacts" && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search contacts..."
                    className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                  />
                </div>
                <select className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500">
                  <option value="">All Statuses</option>
                  <option value="pending">Pending</option>
                  <option value="completed">Completed</option>
                  <option value="failed">Failed</option>
                </select>
              </div>
              <div className="bg-white/5 rounded-xl overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/5">
                      <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                        Contact
                      </th>
                      <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                        Phone
                      </th>
                      <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                        Attempts
                      </th>
                      <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                        Duration
                      </th>
                      <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                        Last Attempt
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {[1, 2, 3, 4, 5].map((i) => (
                      <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                        <td className="px-4 py-3">
                          <div className="text-white font-medium">John Smith {i}</div>
                          <div className="text-xs text-gray-500">Acme Corp</div>
                        </td>
                        <td className="px-4 py-3 text-gray-300">+1 (555) 123-456{i}</td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-1 rounded-full text-xs font-medium bg-green-500/20 text-green-400">
                            Completed
                          </span>
                        </td>
                        <td className="px-4 py-3 text-gray-300">{i}</td>
                        <td className="px-4 py-3 text-gray-300">{formatDuration(120 + i * 30)}</td>
                        <td className="px-4 py-3 text-gray-400 text-sm">Jan 20, 2024</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === "analytics" && (
            <div className="space-y-6">
              <div className="bg-white/5 rounded-xl p-5">
                <h3 className="font-semibold text-white mb-4">Call Volume Over Time</h3>
                <div className="h-64 flex items-center justify-center text-gray-500">
                  <LineChart className="h-12 w-12 mr-3" />
                  Chart visualization would go here
                </div>
              </div>
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-white/5 rounded-xl p-5">
                  <h3 className="font-semibold text-white mb-4">Outcome Distribution</h3>
                  <div className="h-48 flex items-center justify-center text-gray-500">
                    <PieChart className="h-12 w-12 mr-3" />
                    Pie chart would go here
                  </div>
                </div>
                <div className="bg-white/5 rounded-xl p-5">
                  <h3 className="font-semibold text-white mb-4">Hourly Performance</h3>
                  <div className="h-48 flex items-center justify-center text-gray-500">
                    <BarChart3 className="h-12 w-12 mr-3" />
                    Bar chart would go here
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "settings" && (
            <div className="space-y-6">
              <div className="bg-white/5 rounded-xl p-5">
                <h3 className="font-semibold text-white mb-4">Campaign Settings</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Campaign Name
                    </label>
                    <input
                      type="text"
                      value={campaign.name}
                      readOnly
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Description
                    </label>
                    <textarea
                      value={campaign.description}
                      readOnly
                      rows={3}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white resize-none"
                    />
                  </div>
                </div>
              </div>
              <div className="bg-red-500/10 rounded-xl p-5 border border-red-500/20">
                <h3 className="font-semibold text-red-400 mb-2">Danger Zone</h3>
                <p className="text-sm text-gray-400 mb-4">
                  These actions are irreversible. Please be certain.
                </p>
                <div className="flex gap-3">
                  <button className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg text-sm font-medium hover:bg-red-500/30 transition-colors">
                    Delete Campaign
                  </button>
                  <button className="px-4 py-2 bg-white/5 text-gray-300 rounded-lg text-sm font-medium hover:bg-white/10 transition-colors">
                    Archive Campaign
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Main Page Component
export default function CampaignsPage() {
  const [campaigns, setCampaigns] = useState<Campaign[]>(sampleCampaigns);
  const [contactLists] = useState<ContactList[]>(sampleContactLists);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<CampaignStatus | "all">("all");
  const [typeFilter, setTypeFilter] = useState<CampaignType | "all">("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showContactListDialog, setShowContactListDialog] = useState(false);
  const [selectedCampaign, setSelectedCampaign] = useState<Campaign | null>(null);
  const [showDetailsDialog, setShowDetailsDialog] = useState(false);

  // Calculate overall stats
  const overallStats = useMemo(() => {
    const activeCampaigns = campaigns.filter((c) => c.status === "running");
    const totalContacts = campaigns.reduce((sum, c) => sum + c.stats.totalContacts, 0);
    const totalAnswered = campaigns.reduce((sum, c) => sum + c.stats.answered, 0);
    const totalCost = campaigns.reduce((sum, c) => sum + c.stats.totalCost, 0);
    const avgSuccessRate =
      campaigns.length > 0
        ? campaigns.reduce((sum, c) => sum + c.stats.successRate, 0) / campaigns.length
        : 0;

    return {
      totalCampaigns: campaigns.length,
      activeCampaigns: activeCampaigns.length,
      totalContacts,
      totalAnswered,
      totalCost,
      avgSuccessRate,
    };
  }, [campaigns]);

  // Filter campaigns
  const filteredCampaigns = useMemo(() => {
    return campaigns.filter((campaign) => {
      const matchesSearch =
        campaign.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        campaign.description.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = statusFilter === "all" || campaign.status === statusFilter;
      const matchesType = typeFilter === "all" || campaign.type === typeFilter;
      return matchesSearch && matchesStatus && matchesType;
    });
  }, [campaigns, searchQuery, statusFilter, typeFilter]);

  // Handle campaign actions
  const handleCampaignAction = useCallback((action: string, campaign: Campaign) => {
    switch (action) {
      case "view":
        setSelectedCampaign(campaign);
        setShowDetailsDialog(true);
        break;
      case "start":
      case "resume":
        setCampaigns((prev) =>
          prev.map((c) =>
            c.id === campaign.id
              ? { ...c, status: "running" as CampaignStatus, startedAt: new Date().toISOString() }
              : c
          )
        );
        break;
      case "pause":
        setCampaigns((prev) =>
          prev.map((c) =>
            c.id === campaign.id ? { ...c, status: "paused" as CampaignStatus } : c
          )
        );
        break;
      case "stop":
      case "cancel":
        setCampaigns((prev) =>
          prev.map((c) =>
            c.id === campaign.id
              ? { ...c, status: "completed" as CampaignStatus, completedAt: new Date().toISOString() }
              : c
          )
        );
        break;
      case "delete":
        setCampaigns((prev) => prev.filter((c) => c.id !== campaign.id));
        break;
      default:
        break;
    }
  }, []);

  return (
    <DashboardLayout>
      <div className="p-6 lg:p-8 max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="text-2xl lg:text-3xl font-bold text-white">Campaigns</h1>
            <p className="text-gray-400 mt-1">Manage your outbound calling campaigns</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowContactListDialog(true)}
              className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white font-medium hover:bg-white/10 transition-colors flex items-center gap-2"
            >
              <Upload className="h-4 w-4" />
              Import Contacts
            </button>
            <button
              onClick={() => setShowCreateDialog(true)}
              className="px-4 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
            >
              <Plus className="h-4 w-4" />
              Create Campaign
            </button>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-2 lg:grid-cols-6 gap-4 mb-8">
          <StatsCard
            title="Total Campaigns"
            value={overallStats.totalCampaigns}
            icon={Layers}
            color="#8B5CF6"
          />
          <StatsCard
            title="Active Campaigns"
            value={overallStats.activeCampaigns}
            icon={Play}
            color="#10B981"
          />
          <StatsCard
            title="Total Contacts"
            value={formatNumber(overallStats.totalContacts)}
            icon={Users}
            color="#3B82F6"
          />
          <StatsCard
            title="Calls Answered"
            value={formatNumber(overallStats.totalAnswered)}
            icon={PhoneCall}
            color="#F59E0B"
          />
          <StatsCard
            title="Avg Success Rate"
            value={`${overallStats.avgSuccessRate.toFixed(1)}%`}
            icon={Target}
            color="#EC4899"
          />
          <StatsCard
            title="Total Spent"
            value={formatCurrency(overallStats.totalCost)}
            icon={TrendingUp}
            color="#06B6D4"
          />
        </div>

        {/* Filters */}
        <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4 mb-6">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
              <input
                type="text"
                placeholder="Search campaigns..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>
            <div className="flex items-center gap-3">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value as CampaignStatus | "all")}
                className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
              >
                <option value="all">All Status</option>
                <option value="draft">Draft</option>
                <option value="scheduled">Scheduled</option>
                <option value="running">Running</option>
                <option value="paused">Paused</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value as CampaignType | "all")}
                className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
              >
                <option value="all">All Types</option>
                <option value="outbound">Outbound</option>
                <option value="inbound">Inbound</option>
                <option value="blended">Blended</option>
                <option value="survey">Survey</option>
                <option value="notification">Notification</option>
                <option value="appointment">Appointment</option>
              </select>
              <div className="flex items-center bg-white/5 rounded-xl p-1">
                <button
                  onClick={() => setViewMode("grid")}
                  className={`p-2 rounded-lg transition-colors ${
                    viewMode === "grid" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:text-white"
                  }`}
                >
                  <Grid3X3 className="h-4 w-4" />
                </button>
                <button
                  onClick={() => setViewMode("list")}
                  className={`p-2 rounded-lg transition-colors ${
                    viewMode === "list" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:text-white"
                  }`}
                >
                  <List className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Campaigns Grid/List */}
        {filteredCampaigns.length === 0 ? (
          <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-12 text-center">
            <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mx-auto mb-4">
              <Phone className="h-8 w-8 text-purple-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">No campaigns found</h3>
            <p className="text-gray-400 mb-6 max-w-md mx-auto">
              {searchQuery || statusFilter !== "all" || typeFilter !== "all"
                ? "Try adjusting your filters to find what you're looking for."
                : "Create your first outbound campaign to start reaching your contacts."}
            </p>
            <button
              onClick={() => setShowCreateDialog(true)}
              className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity inline-flex items-center gap-2"
            >
              <Plus className="h-4 w-4" />
              Create Campaign
            </button>
          </div>
        ) : viewMode === "grid" ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredCampaigns.map((campaign) => (
              <CampaignCard
                key={campaign.id}
                campaign={campaign}
                onAction={handleCampaignAction}
              />
            ))}
          </div>
        ) : (
          <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Campaign
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Progress
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Success Rate
                  </th>
                  <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Cost
                  </th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredCampaigns.map((campaign) => (
                  <tr
                    key={campaign.id}
                    className="border-b border-white/5 hover:bg-white/[0.02] cursor-pointer"
                    onClick={() => handleCampaignAction("view", campaign)}
                  >
                    <td className="px-4 py-4">
                      <div className="flex items-center gap-3">
                        <div
                          className={`p-2 rounded-lg ${
                            campaign.status === "running"
                              ? "bg-green-500/20"
                              : campaign.status === "paused"
                              ? "bg-yellow-500/20"
                              : "bg-gray-500/20"
                          }`}
                        >
                          {getCampaignTypeIcon(campaign.type)}
                        </div>
                        <div>
                          <div className="font-medium text-white">{campaign.name}</div>
                          <div className="text-xs text-gray-500">{campaign.agentName}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <span
                        className={`px-2.5 py-1 rounded-full text-xs font-medium inline-flex items-center gap-1.5 ${getStatusColor(
                          campaign.status
                        )}`}
                      >
                        {getStatusIcon(campaign.status)}
                        {campaign.status.charAt(0).toUpperCase() + campaign.status.slice(1)}
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-24 h-2 bg-white/5 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-purple-500 rounded-full"
                            style={{
                              width: `${
                                (campaign.stats.contacted / campaign.stats.totalContacts) * 100
                              }%`,
                            }}
                          />
                        </div>
                        <span className="text-sm text-gray-400">
                          {Math.round(
                            (campaign.stats.contacted / campaign.stats.totalContacts) * 100
                          )}
                          %
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <span className="text-white">{campaign.stats.successRate}%</span>
                    </td>
                    <td className="px-4 py-4">
                      <span className="text-white">{formatCurrency(campaign.stats.totalCost)}</span>
                    </td>
                    <td className="px-4 py-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCampaignAction("view", campaign);
                        }}
                        className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
                      >
                        <MoreVertical className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Dialogs */}
        <CreateCampaignDialog
          isOpen={showCreateDialog}
          onClose={() => setShowCreateDialog(false)}
          contactLists={contactLists}
        />
        <ContactListDialog
          isOpen={showContactListDialog}
          onClose={() => setShowContactListDialog(false)}
        />
        <CampaignDetailsDialog
          campaign={selectedCampaign}
          isOpen={showDetailsDialog}
          onClose={() => {
            setShowDetailsDialog(false);
            setSelectedCampaign(null);
          }}
        />
      </div>
    </DashboardLayout>
  );
}
