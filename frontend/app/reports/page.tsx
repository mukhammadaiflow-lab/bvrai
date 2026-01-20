"use client";

import { useState, useMemo } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import {
  FileText,
  Plus,
  Search,
  Filter,
  MoreVertical,
  Download,
  Eye,
  Trash2,
  Copy,
  Calendar,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Info,
  Star,
  RefreshCw,
  Settings,
  Sparkles,
  Zap,
  BarChart3,
  LineChart,
  PieChart,
  TrendingUp,
  TrendingDown,
  Activity,
  Users,
  Phone,
  PhoneCall,
  Bot,
  DollarSign,
  Target,
  Timer,
  ChevronRight,
  ChevronDown,
  X,
  Save,
  Send,
  Mail,
  Globe,
  Table,
  FileSpreadsheet,
  FileJson,
  File,
  FolderOpen,
  LayoutGrid,
  List,
  Play,
  Pause,
  ArrowUpRight,
  ArrowDownRight,
  Layers,
  Share2,
  ExternalLink,
  Loader2,
  CalendarDays,
  CalendarRange,
  History,
  Building2,
  Tag,
} from "lucide-react";

// Types
type ReportType = "calls" | "agents" | "campaigns" | "billing" | "performance" | "custom";
type ReportStatus = "ready" | "generating" | "scheduled" | "failed";
type ReportFormat = "pdf" | "csv" | "xlsx" | "json";
type ScheduleFrequency = "daily" | "weekly" | "monthly" | "quarterly";

interface ReportFilter {
  field: string;
  operator: "equals" | "contains" | "greater_than" | "less_than" | "between";
  value: any;
}

interface ReportColumn {
  id: string;
  name: string;
  enabled: boolean;
}

interface Report {
  id: string;
  name: string;
  description: string;
  type: ReportType;
  status: ReportStatus;
  format: ReportFormat;
  dateRange: {
    start: string;
    end: string;
  };
  filters: ReportFilter[];
  columns: string[];
  schedule?: {
    frequency: ScheduleFrequency;
    time: string;
    recipients: string[];
    nextRun: string;
  };
  stats: {
    rows: number;
    fileSize: string;
  };
  createdAt: string;
  generatedAt?: string;
  downloadUrl?: string;
  createdBy: string;
  favorite: boolean;
  tags: string[];
}

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  type: ReportType;
  icon: any;
  columns: ReportColumn[];
  defaultFilters: ReportFilter[];
}

// Sample Data
const reportTemplates: ReportTemplate[] = [
  {
    id: "tmpl_calls",
    name: "Call Analytics Report",
    description: "Comprehensive call data with duration, outcomes, and sentiment",
    type: "calls",
    icon: PhoneCall,
    columns: [
      { id: "call_id", name: "Call ID", enabled: true },
      { id: "agent_name", name: "Agent", enabled: true },
      { id: "customer_phone", name: "Phone Number", enabled: true },
      { id: "direction", name: "Direction", enabled: true },
      { id: "duration", name: "Duration", enabled: true },
      { id: "outcome", name: "Outcome", enabled: true },
      { id: "sentiment", name: "Sentiment", enabled: true },
      { id: "cost", name: "Cost", enabled: false },
      { id: "timestamp", name: "Date/Time", enabled: true },
    ],
    defaultFilters: [],
  },
  {
    id: "tmpl_agents",
    name: "Agent Performance Report",
    description: "Agent metrics including calls handled, success rates, and ratings",
    type: "agents",
    icon: Bot,
    columns: [
      { id: "agent_id", name: "Agent ID", enabled: true },
      { id: "agent_name", name: "Agent Name", enabled: true },
      { id: "total_calls", name: "Total Calls", enabled: true },
      { id: "avg_duration", name: "Avg Duration", enabled: true },
      { id: "success_rate", name: "Success Rate", enabled: true },
      { id: "avg_sentiment", name: "Avg Sentiment", enabled: true },
      { id: "total_cost", name: "Total Cost", enabled: false },
    ],
    defaultFilters: [],
  },
  {
    id: "tmpl_campaigns",
    name: "Campaign Performance Report",
    description: "Campaign results with contact rates, conversions, and ROI",
    type: "campaigns",
    icon: Target,
    columns: [
      { id: "campaign_id", name: "Campaign ID", enabled: true },
      { id: "campaign_name", name: "Campaign Name", enabled: true },
      { id: "status", name: "Status", enabled: true },
      { id: "total_contacts", name: "Total Contacts", enabled: true },
      { id: "contacted", name: "Contacted", enabled: true },
      { id: "answered", name: "Answered", enabled: true },
      { id: "conversion_rate", name: "Conversion Rate", enabled: true },
      { id: "cost", name: "Cost", enabled: true },
    ],
    defaultFilters: [],
  },
  {
    id: "tmpl_billing",
    name: "Billing & Usage Report",
    description: "Detailed billing breakdown by service and time period",
    type: "billing",
    icon: DollarSign,
    columns: [
      { id: "date", name: "Date", enabled: true },
      { id: "service", name: "Service", enabled: true },
      { id: "usage", name: "Usage", enabled: true },
      { id: "unit_cost", name: "Unit Cost", enabled: true },
      { id: "total_cost", name: "Total Cost", enabled: true },
    ],
    defaultFilters: [],
  },
  {
    id: "tmpl_performance",
    name: "System Performance Report",
    description: "System health, API usage, and latency metrics",
    type: "performance",
    icon: Activity,
    columns: [
      { id: "timestamp", name: "Timestamp", enabled: true },
      { id: "metric", name: "Metric", enabled: true },
      { id: "value", name: "Value", enabled: true },
      { id: "status", name: "Status", enabled: true },
    ],
    defaultFilters: [],
  },
];

const sampleReports: Report[] = [
  {
    id: "rpt_1",
    name: "January Call Analytics",
    description: "All calls from January 2024",
    type: "calls",
    status: "ready",
    format: "xlsx",
    dateRange: { start: "2024-01-01", end: "2024-01-31" },
    filters: [],
    columns: ["call_id", "agent_name", "customer_phone", "duration", "outcome", "sentiment", "timestamp"],
    stats: { rows: 15420, fileSize: "2.4 MB" },
    createdAt: "2024-02-01T10:00:00Z",
    generatedAt: "2024-02-01T10:05:00Z",
    downloadUrl: "/reports/january-calls.xlsx",
    createdBy: "Admin",
    favorite: true,
    tags: ["monthly", "calls"],
  },
  {
    id: "rpt_2",
    name: "Q4 Agent Performance",
    description: "Agent performance metrics for Q4 2023",
    type: "agents",
    status: "ready",
    format: "pdf",
    dateRange: { start: "2023-10-01", end: "2023-12-31" },
    filters: [],
    columns: ["agent_name", "total_calls", "avg_duration", "success_rate", "avg_sentiment"],
    stats: { rows: 12, fileSize: "1.8 MB" },
    createdAt: "2024-01-05T09:00:00Z",
    generatedAt: "2024-01-05T09:12:00Z",
    downloadUrl: "/reports/q4-agents.pdf",
    createdBy: "Admin",
    favorite: true,
    tags: ["quarterly", "agents"],
  },
  {
    id: "rpt_3",
    name: "Weekly Sales Campaign",
    description: "Weekly report for sales outreach campaign",
    type: "campaigns",
    status: "scheduled",
    format: "csv",
    dateRange: { start: "2024-01-15", end: "2024-01-21" },
    filters: [{ field: "campaign_type", operator: "equals", value: "sales" }],
    columns: ["campaign_name", "total_contacts", "contacted", "answered", "conversion_rate"],
    schedule: {
      frequency: "weekly",
      time: "09:00",
      recipients: ["team@company.com"],
      nextRun: "2024-01-22T09:00:00Z",
    },
    stats: { rows: 0, fileSize: "-" },
    createdAt: "2024-01-15T14:00:00Z",
    createdBy: "Sales Manager",
    favorite: false,
    tags: ["weekly", "campaigns", "automated"],
  },
  {
    id: "rpt_4",
    name: "Monthly Billing Summary",
    description: "Billing breakdown for December",
    type: "billing",
    status: "ready",
    format: "pdf",
    dateRange: { start: "2023-12-01", end: "2023-12-31" },
    filters: [],
    columns: ["date", "service", "usage", "unit_cost", "total_cost"],
    stats: { rows: 45, fileSize: "890 KB" },
    createdAt: "2024-01-02T08:00:00Z",
    generatedAt: "2024-01-02T08:03:00Z",
    downloadUrl: "/reports/dec-billing.pdf",
    createdBy: "Finance",
    favorite: false,
    tags: ["billing", "monthly"],
  },
  {
    id: "rpt_5",
    name: "Real-time Performance",
    description: "Live system performance metrics",
    type: "performance",
    status: "generating",
    format: "json",
    dateRange: { start: "2024-01-20", end: "2024-01-20" },
    filters: [],
    columns: ["timestamp", "metric", "value", "status"],
    stats: { rows: 0, fileSize: "-" },
    createdAt: "2024-01-20T16:30:00Z",
    createdBy: "System",
    favorite: false,
    tags: ["performance", "live"],
  },
];

// Utility functions
const getTypeColor = (type: ReportType) => {
  switch (type) {
    case "calls":
      return "bg-green-500/20 text-green-400";
    case "agents":
      return "bg-purple-500/20 text-purple-400";
    case "campaigns":
      return "bg-blue-500/20 text-blue-400";
    case "billing":
      return "bg-yellow-500/20 text-yellow-400";
    case "performance":
      return "bg-orange-500/20 text-orange-400";
    case "custom":
      return "bg-pink-500/20 text-pink-400";
    default:
      return "bg-gray-500/20 text-gray-400";
  }
};

const getStatusColor = (status: ReportStatus) => {
  switch (status) {
    case "ready":
      return "bg-green-500/20 text-green-400";
    case "generating":
      return "bg-blue-500/20 text-blue-400";
    case "scheduled":
      return "bg-purple-500/20 text-purple-400";
    case "failed":
      return "bg-red-500/20 text-red-400";
    default:
      return "bg-gray-500/20 text-gray-400";
  }
};

const getStatusIcon = (status: ReportStatus) => {
  switch (status) {
    case "ready":
      return <CheckCircle className="h-4 w-4" />;
    case "generating":
      return <Loader2 className="h-4 w-4 animate-spin" />;
    case "scheduled":
      return <Clock className="h-4 w-4" />;
    case "failed":
      return <XCircle className="h-4 w-4" />;
    default:
      return <Info className="h-4 w-4" />;
  }
};

const getFormatIcon = (format: ReportFormat) => {
  switch (format) {
    case "pdf":
      return <FileText className="h-4 w-4 text-red-400" />;
    case "csv":
      return <Table className="h-4 w-4 text-green-400" />;
    case "xlsx":
      return <FileSpreadsheet className="h-4 w-4 text-emerald-400" />;
    case "json":
      return <FileJson className="h-4 w-4 text-yellow-400" />;
    default:
      return <File className="h-4 w-4 text-gray-400" />;
  }
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

const formatNumber = (num: number) => {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
};

// Components
function StatsCard({
  title,
  value,
  change,
  changeType,
  icon: Icon,
  color,
}: {
  title: string;
  value: string | number;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: any;
  color: string;
}) {
  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4 hover:border-white/10 transition-all">
      <div className="flex items-center justify-between">
        <div className="p-2 rounded-lg" style={{ backgroundColor: `${color}20` }}>
          <Icon className="h-4 w-4" style={{ color }} />
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
      </div>
    </div>
  );
}

function ReportCard({
  report,
  onDownload,
  onView,
  onDelete,
  onDuplicate,
  onToggleFavorite,
}: {
  report: Report;
  onDownload: () => void;
  onView: () => void;
  onDelete: () => void;
  onDuplicate: () => void;
  onToggleFavorite: () => void;
}) {
  const [showMenu, setShowMenu] = useState(false);
  const template = reportTemplates.find((t) => t.type === report.type);
  const Icon = template?.icon || FileText;

  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 hover:border-white/10 transition-all group">
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-lg ${getTypeColor(report.type)}`}>
              <Icon className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-white group-hover:text-purple-400 transition-colors">
                  {report.name}
                </h3>
                <button onClick={onToggleFavorite}>
                  {report.favorite ? (
                    <Star className="h-4 w-4 text-yellow-400 fill-yellow-400" />
                  ) : (
                    <Star className="h-4 w-4 text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                  )}
                </button>
              </div>
              <p className="text-sm text-gray-400">{report.description}</p>
            </div>
          </div>
          <div className="relative">
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
            >
              <MoreVertical className="h-4 w-4" />
            </button>
            {showMenu && (
              <div className="absolute right-0 top-full mt-1 w-40 bg-[#252542] rounded-lg border border-white/10 shadow-xl z-10 py-1">
                <button
                  onClick={() => { onView(); setShowMenu(false); }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                >
                  <Eye className="h-4 w-4" /> View
                </button>
                {report.status === "ready" && (
                  <button
                    onClick={() => { onDownload(); setShowMenu(false); }}
                    className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                  >
                    <Download className="h-4 w-4" /> Download
                  </button>
                )}
                <button
                  onClick={() => { onDuplicate(); setShowMenu(false); }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                >
                  <Copy className="h-4 w-4" /> Duplicate
                </button>
                <button
                  onClick={() => { setShowMenu(false); }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                >
                  <Share2 className="h-4 w-4" /> Share
                </button>
                <div className="border-t border-white/5 my-1" />
                <button
                  onClick={() => { onDelete(); setShowMenu(false); }}
                  className="w-full px-4 py-2 text-left text-sm text-red-400 hover:bg-red-500/10 flex items-center gap-2"
                >
                  <Trash2 className="h-4 w-4" /> Delete
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Status & Format */}
        <div className="flex items-center gap-2 mb-3">
          <span className={`px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 ${getStatusColor(report.status)}`}>
            {getStatusIcon(report.status)}
            {report.status.charAt(0).toUpperCase() + report.status.slice(1)}
          </span>
          <span className={`px-2 py-1 rounded text-xs font-medium ${getTypeColor(report.type)}`}>
            {report.type}
          </span>
          <div className="flex items-center gap-1.5 px-2 py-1 bg-white/5 rounded text-xs text-gray-400">
            {getFormatIcon(report.format)}
            {report.format.toUpperCase()}
          </div>
        </div>

        {/* Date Range */}
        <div className="flex items-center gap-2 text-sm text-gray-400 mb-3">
          <CalendarRange className="h-4 w-4" />
          <span>
            {formatDate(report.dateRange.start)} - {formatDate(report.dateRange.end)}
          </span>
        </div>

        {/* Tags */}
        {report.tags.length > 0 && (
          <div className="flex items-center gap-1.5 mb-3">
            {report.tags.slice(0, 3).map((tag) => (
              <span key={tag} className="px-2 py-0.5 bg-white/5 text-gray-400 rounded text-xs">
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Schedule Info */}
        {report.schedule && (
          <div className="p-3 bg-purple-500/10 rounded-lg border border-purple-500/20 mb-3">
            <div className="flex items-center gap-2 text-sm">
              <Clock className="h-4 w-4 text-purple-400" />
              <span className="text-purple-400 font-medium capitalize">{report.schedule.frequency}</span>
              <span className="text-gray-400">at {report.schedule.time}</span>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Next run: {formatDateTime(report.schedule.nextRun)}
            </div>
          </div>
        )}

        {/* Stats & Actions */}
        <div className="flex items-center justify-between pt-3 border-t border-white/5">
          <div className="flex items-center gap-4 text-sm">
            {report.status === "ready" && (
              <>
                <div className="flex items-center gap-1.5">
                  <Layers className="h-3.5 w-3.5 text-gray-500" />
                  <span className="text-gray-400">{formatNumber(report.stats.rows)} rows</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <File className="h-3.5 w-3.5 text-gray-500" />
                  <span className="text-gray-400">{report.stats.fileSize}</span>
                </div>
              </>
            )}
          </div>
          {report.status === "ready" && (
            <button
              onClick={onDownload}
              className="px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg text-sm font-medium hover:bg-purple-500/30 transition-colors flex items-center gap-1.5"
            >
              <Download className="h-4 w-4" />
              Download
            </button>
          )}
          {report.status === "generating" && (
            <div className="flex items-center gap-2 text-sm text-blue-400">
              <Loader2 className="h-4 w-4 animate-spin" />
              Generating...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function CreateReportDialog({
  isOpen,
  onClose,
  onCreate,
}: {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (report: Partial<Report>) => void;
}) {
  const [step, setStep] = useState(1);
  const [selectedTemplate, setSelectedTemplate] = useState<ReportTemplate | null>(null);
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    format: "xlsx" as ReportFormat,
    dateRange: { start: "", end: "" },
    columns: [] as string[],
    schedule: null as null | {
      frequency: ScheduleFrequency;
      time: string;
      recipients: string[];
    },
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a2e] rounded-2xl border border-white/10 w-full max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">Create Report</h2>
            <p className="text-sm text-gray-400 mt-0.5">Step {step} of 3</p>
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
            {[1, 2, 3].map((s) => (
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
                {s < 3 && (
                  <div className={`flex-1 h-0.5 mx-2 ${s < step ? "bg-purple-500" : "bg-white/10"}`} />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-between mt-2 text-xs text-gray-500">
            <span>Select Template</span>
            <span>Configure</span>
            <span>Review</span>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {step === 1 && (
            <div className="space-y-4">
              <p className="text-gray-400 mb-4">Choose a report template to get started</p>
              <div className="grid grid-cols-2 gap-4">
                {reportTemplates.map((template) => (
                  <button
                    key={template.id}
                    onClick={() => {
                      setSelectedTemplate(template);
                      setFormData({
                        ...formData,
                        columns: template.columns.filter((c) => c.enabled).map((c) => c.id),
                      });
                    }}
                    className={`p-4 rounded-xl border text-left transition-all ${
                      selectedTemplate?.id === template.id
                        ? "bg-purple-500/10 border-purple-500"
                        : "bg-white/5 border-white/10 hover:border-white/20"
                    }`}
                  >
                    <div className={`p-2.5 rounded-lg ${getTypeColor(template.type)} w-fit mb-3`}>
                      <template.icon className="h-5 w-5" />
                    </div>
                    <h3 className="font-medium text-white mb-1">{template.name}</h3>
                    <p className="text-sm text-gray-500">{template.description}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {step === 2 && selectedTemplate && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Report Name
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder={`${selectedTemplate.name} - ${new Date().toLocaleDateString()}`}
                    className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Format
                  </label>
                  <select
                    value={formData.format}
                    onChange={(e) => setFormData({ ...formData, format: e.target.value as ReportFormat })}
                    className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  >
                    <option value="xlsx">Excel (.xlsx)</option>
                    <option value="csv">CSV (.csv)</option>
                    <option value="pdf">PDF (.pdf)</option>
                    <option value="json">JSON (.json)</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Date Range
                </label>
                <div className="grid grid-cols-2 gap-4">
                  <input
                    type="date"
                    value={formData.dateRange.start}
                    onChange={(e) =>
                      setFormData({ ...formData, dateRange: { ...formData.dateRange, start: e.target.value } })
                    }
                    className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  />
                  <input
                    type="date"
                    value={formData.dateRange.end}
                    onChange={(e) =>
                      setFormData({ ...formData, dateRange: { ...formData.dateRange, end: e.target.value } })
                    }
                    className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Columns to Include
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {selectedTemplate.columns.map((column) => (
                    <label
                      key={column.id}
                      className="flex items-center gap-2 p-2 bg-white/5 rounded-lg cursor-pointer hover:bg-white/10 transition-colors"
                    >
                      <input
                        type="checkbox"
                        checked={formData.columns.includes(column.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setFormData({ ...formData, columns: [...formData.columns, column.id] });
                          } else {
                            setFormData({
                              ...formData,
                              columns: formData.columns.filter((c) => c !== column.id),
                            });
                          }
                        }}
                        className="w-4 h-4 rounded border-white/20 bg-white/5 text-purple-500 focus:ring-purple-500"
                      />
                      <span className="text-sm text-gray-300">{column.name}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-300">
                    Schedule Report
                  </label>
                  <button
                    onClick={() =>
                      setFormData({
                        ...formData,
                        schedule: formData.schedule
                          ? null
                          : { frequency: "weekly", time: "09:00", recipients: [] },
                      })
                    }
                    className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                      formData.schedule
                        ? "bg-purple-500/20 text-purple-400"
                        : "bg-white/5 text-gray-400"
                    }`}
                  >
                    {formData.schedule ? "Enabled" : "Disabled"}
                  </button>
                </div>
                {formData.schedule && (
                  <div className="grid grid-cols-2 gap-4 p-4 bg-white/5 rounded-xl">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Frequency</label>
                      <select
                        value={formData.schedule.frequency}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            schedule: { ...formData.schedule!, frequency: e.target.value as ScheduleFrequency },
                          })
                        }
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                      >
                        <option value="daily">Daily</option>
                        <option value="weekly">Weekly</option>
                        <option value="monthly">Monthly</option>
                        <option value="quarterly">Quarterly</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Time</label>
                      <input
                        type="time"
                        value={formData.schedule.time}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            schedule: { ...formData.schedule!, time: e.target.value },
                          })
                        }
                        className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {step === 3 && selectedTemplate && (
            <div className="space-y-6">
              <div className="bg-white/5 rounded-xl p-5">
                <h3 className="font-semibold text-white mb-4">Report Summary</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Name</div>
                    <div className="text-white mt-1">{formData.name || "Untitled Report"}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Type</div>
                    <div className="text-white mt-1 capitalize">{selectedTemplate.type}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Format</div>
                    <div className="text-white mt-1 uppercase">{formData.format}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Date Range</div>
                    <div className="text-white mt-1">
                      {formData.dateRange.start && formData.dateRange.end
                        ? `${formatDate(formData.dateRange.start)} - ${formatDate(formData.dateRange.end)}`
                        : "Not set"}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Columns</div>
                    <div className="text-white mt-1">{formData.columns.length} selected</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider">Schedule</div>
                    <div className="text-white mt-1 capitalize">
                      {formData.schedule ? formData.schedule.frequency : "One-time"}
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-green-500/10 rounded-xl p-5 border border-green-500/20">
                <div className="flex items-start gap-3">
                  <CheckCircle className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="font-medium text-green-400">Ready to Generate</div>
                    <div className="text-sm text-gray-400 mt-1">
                      Your report will be generated and available for download shortly.
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
            onClick={() => {
              if (step < 3) {
                setStep(step + 1);
              } else {
                onCreate({
                  name: formData.name || `${selectedTemplate?.name} - ${new Date().toLocaleDateString()}`,
                  type: selectedTemplate?.type,
                  format: formData.format,
                  dateRange: formData.dateRange,
                  columns: formData.columns,
                });
                onClose();
              }
            }}
            disabled={step === 1 && !selectedTemplate}
            className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-medium hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {step === 3 ? (
              <>
                <Sparkles className="h-4 w-4" />
                Generate Report
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

// Main Page Component
export default function ReportsPage() {
  const [reports, setReports] = useState<Report[]>(sampleReports);
  const [searchQuery, setSearchQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState<ReportType | "all">("all");
  const [statusFilter, setStatusFilter] = useState<ReportStatus | "all">("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [showCreateDialog, setShowCreateDialog] = useState(false);

  // Filter reports
  const filteredReports = useMemo(() => {
    let filtered = reports;

    if (typeFilter !== "all") {
      filtered = filtered.filter((r) => r.type === typeFilter);
    }

    if (statusFilter !== "all") {
      filtered = filtered.filter((r) => r.status === statusFilter);
    }

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (r) =>
          r.name.toLowerCase().includes(query) ||
          r.description.toLowerCase().includes(query)
      );
    }

    return filtered;
  }, [reports, typeFilter, statusFilter, searchQuery]);

  // Calculate stats
  const stats = useMemo(() => {
    const ready = reports.filter((r) => r.status === "ready").length;
    const scheduled = reports.filter((r) => r.status === "scheduled").length;
    const totalRows = reports.reduce((sum, r) => sum + r.stats.rows, 0);
    return { total: reports.length, ready, scheduled, totalRows };
  }, [reports]);

  const handleCreateReport = (report: Partial<Report>) => {
    const newReport: Report = {
      id: `rpt_${Date.now()}`,
      name: report.name || "New Report",
      description: "Custom generated report",
      type: report.type || "custom",
      status: "generating",
      format: report.format || "xlsx",
      dateRange: report.dateRange || { start: "", end: "" },
      filters: [],
      columns: report.columns || [],
      stats: { rows: 0, fileSize: "-" },
      createdAt: new Date().toISOString(),
      createdBy: "User",
      favorite: false,
      tags: [],
    };
    setReports((prev) => [newReport, ...prev]);

    // Simulate report generation
    setTimeout(() => {
      setReports((prev) =>
        prev.map((r) =>
          r.id === newReport.id
            ? {
                ...r,
                status: "ready" as ReportStatus,
                generatedAt: new Date().toISOString(),
                downloadUrl: `/reports/${r.id}.${r.format}`,
                stats: { rows: Math.floor(Math.random() * 10000) + 100, fileSize: `${(Math.random() * 5 + 0.5).toFixed(1)} MB` },
              }
            : r
        )
      );
    }, 3000);
  };

  return (
    <DashboardLayout>
      <div className="p-6 lg:p-8 max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
          <div>
            <h1 className="text-2xl lg:text-3xl font-bold text-white">Reports</h1>
            <p className="text-gray-400 mt-1">Generate and download detailed reports</p>
          </div>
          <div className="flex items-center gap-3">
            <button className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white font-medium hover:bg-white/10 transition-colors flex items-center gap-2">
              <History className="h-4 w-4" />
              History
            </button>
            <button
              onClick={() => setShowCreateDialog(true)}
              className="px-4 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
            >
              <Plus className="h-4 w-4" />
              Create Report
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <StatsCard
            title="Total Reports"
            value={stats.total}
            icon={FileText}
            color="#8B5CF6"
          />
          <StatsCard
            title="Ready"
            value={stats.ready}
            icon={CheckCircle}
            color="#10B981"
          />
          <StatsCard
            title="Scheduled"
            value={stats.scheduled}
            icon={Clock}
            color="#3B82F6"
          />
          <StatsCard
            title="Total Rows"
            value={formatNumber(stats.totalRows)}
            icon={Layers}
            color="#EC4899"
          />
        </div>

        {/* Filters */}
        <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4 mb-6">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
              <input
                type="text"
                placeholder="Search reports..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>
            <div className="flex items-center gap-3">
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value as ReportType | "all")}
                className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
              >
                <option value="all">All Types</option>
                <option value="calls">Calls</option>
                <option value="agents">Agents</option>
                <option value="campaigns">Campaigns</option>
                <option value="billing">Billing</option>
                <option value="performance">Performance</option>
              </select>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value as ReportStatus | "all")}
                className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
              >
                <option value="all">All Status</option>
                <option value="ready">Ready</option>
                <option value="generating">Generating</option>
                <option value="scheduled">Scheduled</option>
                <option value="failed">Failed</option>
              </select>
              <div className="flex items-center bg-white/5 rounded-xl p-1">
                <button
                  onClick={() => setViewMode("grid")}
                  className={`p-2 rounded-lg transition-colors ${
                    viewMode === "grid" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:text-white"
                  }`}
                >
                  <LayoutGrid className="h-4 w-4" />
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

        {/* Reports Grid */}
        {filteredReports.length === 0 ? (
          <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-12 text-center">
            <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mx-auto mb-4">
              <FileText className="h-8 w-8 text-purple-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">No reports found</h3>
            <p className="text-gray-400 mb-6">
              {searchQuery || typeFilter !== "all" || statusFilter !== "all"
                ? "Try adjusting your filters."
                : "Generate your first report to get started."}
            </p>
            <button
              onClick={() => setShowCreateDialog(true)}
              className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity inline-flex items-center gap-2"
            >
              <Plus className="h-4 w-4" />
              Create Report
            </button>
          </div>
        ) : (
          <div className={viewMode === "grid" ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" : "space-y-4"}>
            {filteredReports.map((report) => (
              <ReportCard
                key={report.id}
                report={report}
                onDownload={() => {}}
                onView={() => {}}
                onDelete={() => setReports((prev) => prev.filter((r) => r.id !== report.id))}
                onDuplicate={() => {
                  const duplicate = {
                    ...report,
                    id: `rpt_${Date.now()}`,
                    name: `${report.name} (Copy)`,
                    status: "generating" as ReportStatus,
                    createdAt: new Date().toISOString(),
                  };
                  setReports((prev) => [duplicate, ...prev]);
                }}
                onToggleFavorite={() =>
                  setReports((prev) =>
                    prev.map((r) => (r.id === report.id ? { ...r, favorite: !r.favorite } : r))
                  )
                }
              />
            ))}
          </div>
        )}

        {/* Create Dialog */}
        <CreateReportDialog
          isOpen={showCreateDialog}
          onClose={() => setShowCreateDialog(false)}
          onCreate={handleCreateReport}
        />
      </div>
    </DashboardLayout>
  );
}
