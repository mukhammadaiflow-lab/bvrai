"use client";

import { useState, useMemo, useCallback } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import {
  Mail,
  Plus,
  Search,
  Filter,
  MoreVertical,
  Edit,
  Copy,
  Trash2,
  Eye,
  Send,
  Clock,
  Calendar,
  CheckCircle,
  XCircle,
  AlertCircle,
  Info,
  Star,
  StarOff,
  Tag,
  Folder,
  FolderOpen,
  FileText,
  Image,
  Link,
  Bold,
  Italic,
  Underline,
  AlignLeft,
  AlignCenter,
  AlignRight,
  List,
  ListOrdered,
  Code,
  Palette,
  Type,
  AtSign,
  Hash,
  Braces,
  ChevronRight,
  ChevronDown,
  X,
  Maximize2,
  Minimize2,
  Undo,
  Redo,
  Save,
  Download,
  Upload,
  RefreshCw,
  Settings,
  Sparkles,
  Zap,
  Layout,
  LayoutTemplate,
  Layers,
  PanelLeft,
  PanelRight,
  Smartphone,
  Monitor,
  Tablet,
  Globe,
  Users,
  User,
  Building2,
  Phone,
  MessageSquare,
  Heart,
  ThumbsUp,
  Share2,
  ExternalLink,
  ArrowUpRight,
  BarChart3,
  TrendingUp,
  Activity,
  Wand2,
  Heading1,
  Heading2,
  Quote,
  Minus,
  CircleDot,
  Square,
  Circle,
  Triangle,
  Hexagon,
  MousePointer,
  Move,
  GripVertical,
  Columns,
  SplitSquareVertical,
  SplitSquareHorizontal,
  PaintBucket,
  Droplet,
} from "lucide-react";

// Types
type TemplateStatus = "draft" | "active" | "archived";
type TemplateCategory = "transactional" | "marketing" | "notification" | "welcome" | "followup" | "reminder";
type EditorView = "visual" | "html" | "preview";
type PreviewDevice = "desktop" | "tablet" | "mobile";

interface EmailTemplate {
  id: string;
  name: string;
  subject: string;
  description: string;
  category: TemplateCategory;
  status: TemplateStatus;
  content: string;
  htmlContent: string;
  variables: string[];
  stats: {
    sent: number;
    opened: number;
    clicked: number;
    openRate: number;
    clickRate: number;
  };
  thumbnail?: string;
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  favorite: boolean;
  tags: string[];
}

interface TemplateFolder {
  id: string;
  name: string;
  icon: any;
  count: number;
  color: string;
}

// Sample Data
const sampleTemplates: EmailTemplate[] = [
  {
    id: "tmpl_1",
    name: "Welcome Email",
    subject: "Welcome to {{company_name}}! Let's get started",
    description: "Sent to new users after signup",
    category: "welcome",
    status: "active",
    content: `<h1>Welcome aboard, {{first_name}}!</h1>
<p>We're thrilled to have you join {{company_name}}. Here's how to get started:</p>
<ol>
<li>Complete your profile</li>
<li>Create your first AI agent</li>
<li>Make your first call</li>
</ol>
<p>Need help? Our support team is just a click away.</p>
<a href="{{dashboard_url}}">Go to Dashboard</a>`,
    htmlContent: `<!DOCTYPE html>
<html>
<head><title>Welcome</title></head>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
<h1 style="color: #8B5CF6;">Welcome aboard, {{first_name}}!</h1>
<p>We're thrilled to have you join {{company_name}}.</p>
</body>
</html>`,
    variables: ["first_name", "last_name", "company_name", "dashboard_url"],
    stats: { sent: 15420, opened: 11565, clicked: 7710, openRate: 75, clickRate: 50 },
    createdAt: "2024-01-01T10:00:00Z",
    updatedAt: "2024-01-15T14:30:00Z",
    createdBy: "Admin",
    favorite: true,
    tags: ["onboarding", "new-users"],
  },
  {
    id: "tmpl_2",
    name: "Call Summary",
    subject: "Your call summary with {{customer_name}}",
    description: "Sent after each call with transcript and insights",
    category: "transactional",
    status: "active",
    content: `<h2>Call Summary</h2>
<p>Here's a summary of your recent call:</p>
<ul>
<li>Customer: {{customer_name}}</li>
<li>Duration: {{call_duration}}</li>
<li>Sentiment: {{sentiment}}</li>
</ul>
<h3>Key Points</h3>
<p>{{summary}}</p>
<a href="{{transcript_url}}">View Full Transcript</a>`,
    htmlContent: "",
    variables: ["customer_name", "call_duration", "sentiment", "summary", "transcript_url"],
    stats: { sent: 45200, opened: 31640, clicked: 22600, openRate: 70, clickRate: 50 },
    createdAt: "2024-01-05T09:00:00Z",
    updatedAt: "2024-01-18T11:20:00Z",
    createdBy: "Admin",
    favorite: true,
    tags: ["calls", "summary"],
  },
  {
    id: "tmpl_3",
    name: "Monthly Report",
    subject: "Your {{month}} performance report is ready",
    description: "Monthly analytics report for users",
    category: "notification",
    status: "active",
    content: `<h1>{{month}} Performance Report</h1>
<p>Hi {{first_name}},</p>
<p>Your monthly report is ready! Here are the highlights:</p>
<table>
<tr><td>Total Calls</td><td>{{total_calls}}</td></tr>
<tr><td>Avg Duration</td><td>{{avg_duration}}</td></tr>
<tr><td>Success Rate</td><td>{{success_rate}}%</td></tr>
</table>
<a href="{{report_url}}">View Full Report</a>`,
    htmlContent: "",
    variables: ["month", "first_name", "total_calls", "avg_duration", "success_rate", "report_url"],
    stats: { sent: 8500, opened: 5950, clicked: 3400, openRate: 70, clickRate: 40 },
    createdAt: "2024-01-10T08:00:00Z",
    updatedAt: "2024-01-19T16:00:00Z",
    createdBy: "Admin",
    favorite: false,
    tags: ["reports", "analytics"],
  },
  {
    id: "tmpl_4",
    name: "Password Reset",
    subject: "Reset your {{company_name}} password",
    description: "Sent when user requests password reset",
    category: "transactional",
    status: "active",
    content: `<h2>Password Reset Request</h2>
<p>Hi {{first_name}},</p>
<p>We received a request to reset your password. Click the button below to create a new password:</p>
<a href="{{reset_url}}">Reset Password</a>
<p>This link will expire in 24 hours.</p>
<p>If you didn't request this, please ignore this email.</p>`,
    htmlContent: "",
    variables: ["first_name", "reset_url", "company_name"],
    stats: { sent: 2100, opened: 1890, clicked: 1680, openRate: 90, clickRate: 80 },
    createdAt: "2024-01-02T11:00:00Z",
    updatedAt: "2024-01-02T11:00:00Z",
    createdBy: "Admin",
    favorite: false,
    tags: ["security", "auth"],
  },
  {
    id: "tmpl_5",
    name: "Trial Ending Soon",
    subject: "Your free trial ends in {{days_remaining}} days",
    description: "Reminder before trial expiration",
    category: "marketing",
    status: "active",
    content: `<h1>Don't lose access, {{first_name}}!</h1>
<p>Your free trial ends in {{days_remaining}} days.</p>
<p>Upgrade now to keep:</p>
<ul>
<li>All your AI agents</li>
<li>Call history & analytics</li>
<li>Premium features</li>
</ul>
<a href="{{upgrade_url}}">Upgrade Now</a>
<p>Use code <strong>SAVE20</strong> for 20% off your first month!</p>`,
    htmlContent: "",
    variables: ["first_name", "days_remaining", "upgrade_url"],
    stats: { sent: 3200, opened: 2240, clicked: 960, openRate: 70, clickRate: 30 },
    createdAt: "2024-01-08T14:00:00Z",
    updatedAt: "2024-01-17T10:30:00Z",
    createdBy: "Marketing",
    favorite: true,
    tags: ["trial", "conversion"],
  },
  {
    id: "tmpl_6",
    name: "Appointment Reminder",
    subject: "Reminder: Your appointment is tomorrow at {{time}}",
    description: "Sent 24 hours before scheduled appointment",
    category: "reminder",
    status: "active",
    content: `<h2>Appointment Reminder</h2>
<p>Hi {{first_name}},</p>
<p>This is a friendly reminder about your upcoming appointment:</p>
<p><strong>Date:</strong> {{date}}<br>
<strong>Time:</strong> {{time}}<br>
<strong>With:</strong> {{provider_name}}</p>
<a href="{{reschedule_url}}">Reschedule</a>
<a href="{{cancel_url}}">Cancel</a>`,
    htmlContent: "",
    variables: ["first_name", "date", "time", "provider_name", "reschedule_url", "cancel_url"],
    stats: { sent: 12800, opened: 10240, clicked: 5120, openRate: 80, clickRate: 40 },
    createdAt: "2024-01-03T09:30:00Z",
    updatedAt: "2024-01-12T15:45:00Z",
    createdBy: "Admin",
    favorite: false,
    tags: ["appointments", "healthcare"],
  },
  {
    id: "tmpl_7",
    name: "Follow-up After Call",
    subject: "Great chatting with you, {{first_name}}!",
    description: "Sent after a positive sales call",
    category: "followup",
    status: "draft",
    content: `<h1>Thanks for your time!</h1>
<p>Hi {{first_name}},</p>
<p>It was great speaking with you today about {{topic}}.</p>
<p>As discussed, here are the next steps:</p>
<ol>
<li>{{step_1}}</li>
<li>{{step_2}}</li>
<li>{{step_3}}</li>
</ol>
<p>Feel free to reach out if you have any questions!</p>`,
    htmlContent: "",
    variables: ["first_name", "topic", "step_1", "step_2", "step_3"],
    stats: { sent: 0, opened: 0, clicked: 0, openRate: 0, clickRate: 0 },
    createdAt: "2024-01-19T16:00:00Z",
    updatedAt: "2024-01-19T16:00:00Z",
    createdBy: "Sales",
    favorite: false,
    tags: ["sales", "followup"],
  },
];

const templateFolders: TemplateFolder[] = [
  { id: "all", name: "All Templates", icon: Folder, count: 7, color: "#8B5CF6" },
  { id: "favorites", name: "Favorites", icon: Star, count: 3, color: "#F59E0B" },
  { id: "transactional", name: "Transactional", icon: Mail, count: 2, color: "#3B82F6" },
  { id: "marketing", name: "Marketing", icon: Zap, count: 1, color: "#EC4899" },
  { id: "notification", name: "Notifications", icon: MessageSquare, count: 1, color: "#10B981" },
  { id: "welcome", name: "Welcome", icon: Heart, count: 1, color: "#F97316" },
  { id: "followup", name: "Follow-up", icon: RefreshCw, count: 1, color: "#6366F1" },
  { id: "reminder", name: "Reminders", icon: Clock, count: 1, color: "#EF4444" },
];

// Utility functions
const getCategoryColor = (category: TemplateCategory) => {
  switch (category) {
    case "transactional":
      return "bg-blue-500/20 text-blue-400";
    case "marketing":
      return "bg-pink-500/20 text-pink-400";
    case "notification":
      return "bg-green-500/20 text-green-400";
    case "welcome":
      return "bg-orange-500/20 text-orange-400";
    case "followup":
      return "bg-indigo-500/20 text-indigo-400";
    case "reminder":
      return "bg-red-500/20 text-red-400";
    default:
      return "bg-gray-500/20 text-gray-400";
  }
};

const getStatusColor = (status: TemplateStatus) => {
  switch (status) {
    case "active":
      return "bg-green-500/20 text-green-400";
    case "draft":
      return "bg-yellow-500/20 text-yellow-400";
    case "archived":
      return "bg-gray-500/20 text-gray-400";
    default:
      return "bg-gray-500/20 text-gray-400";
  }
};

const formatNumber = (num: number) => {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
};

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
};

// Components
function TemplateCard({
  template,
  onEdit,
  onDuplicate,
  onDelete,
  onToggleFavorite,
}: {
  template: EmailTemplate;
  onEdit: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onToggleFavorite: () => void;
}) {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 hover:border-white/10 transition-all group">
      {/* Preview Thumbnail */}
      <div className="relative h-40 bg-gradient-to-br from-purple-500/10 to-pink-500/10 rounded-t-xl overflow-hidden">
        <div className="absolute inset-0 p-4">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3 text-xs text-gray-300 font-mono overflow-hidden h-full">
            <div className="text-purple-400 mb-1">Subject: {template.subject.substring(0, 40)}...</div>
            <div className="text-gray-500 line-clamp-4" dangerouslySetInnerHTML={{ __html: template.content.replace(/<[^>]*>/g, ' ').substring(0, 150) }} />
          </div>
        </div>
        <div className="absolute top-2 right-2 flex items-center gap-1">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggleFavorite();
            }}
            className="p-1.5 rounded-lg bg-black/30 hover:bg-black/50 transition-colors"
          >
            {template.favorite ? (
              <Star className="h-4 w-4 text-yellow-400 fill-yellow-400" />
            ) : (
              <Star className="h-4 w-4 text-gray-400" />
            )}
          </button>
        </div>
        <div className="absolute bottom-2 left-2">
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(template.status)}`}>
            {template.status}
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div>
            <h3 className="font-semibold text-white group-hover:text-purple-400 transition-colors">
              {template.name}
            </h3>
            <p className="text-sm text-gray-400 line-clamp-1">{template.description}</p>
          </div>
          <div className="relative">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
            >
              <MoreVertical className="h-4 w-4" />
            </button>
            {showMenu && (
              <div className="absolute right-0 top-full mt-1 w-40 bg-[#252542] rounded-lg border border-white/10 shadow-xl z-10 py-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onEdit();
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                >
                  <Edit className="h-4 w-4" />
                  Edit
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDuplicate();
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                >
                  <Copy className="h-4 w-4" />
                  Duplicate
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                >
                  <Eye className="h-4 w-4" />
                  Preview
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                >
                  <Send className="h-4 w-4" />
                  Send Test
                </button>
                <div className="border-t border-white/5 my-1" />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete();
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

        {/* Category & Tags */}
        <div className="flex items-center gap-2 mb-3">
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${getCategoryColor(template.category)}`}>
            {template.category}
          </span>
          {template.tags.slice(0, 2).map((tag) => (
            <span key={tag} className="px-2 py-0.5 bg-white/5 text-gray-400 rounded text-xs">
              {tag}
            </span>
          ))}
        </div>

        {/* Stats */}
        <div className="flex items-center gap-4 pt-3 border-t border-white/5 text-sm">
          <div className="flex items-center gap-1.5">
            <Send className="h-3.5 w-3.5 text-gray-500" />
            <span className="text-gray-400">{formatNumber(template.stats.sent)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Eye className="h-3.5 w-3.5 text-gray-500" />
            <span className="text-gray-400">{template.stats.openRate}%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <MousePointer className="h-3.5 w-3.5 text-gray-500" />
            <span className="text-gray-400">{template.stats.clickRate}%</span>
          </div>
          <div className="ml-auto text-xs text-gray-500">
            Updated {formatDate(template.updatedAt)}
          </div>
        </div>
      </div>
    </div>
  );
}

function TemplateEditorDialog({
  template,
  isOpen,
  onClose,
  onSave,
}: {
  template: EmailTemplate | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: (template: EmailTemplate) => void;
}) {
  const [editorView, setEditorView] = useState<EditorView>("visual");
  const [previewDevice, setPreviewDevice] = useState<PreviewDevice>("desktop");
  const [formData, setFormData] = useState<Partial<EmailTemplate>>(
    template || {
      name: "",
      subject: "",
      description: "",
      category: "transactional",
      content: "",
      variables: [],
      tags: [],
    }
  );
  const [showVariablePanel, setShowVariablePanel] = useState(false);

  if (!isOpen) return null;

  const availableVariables = [
    { name: "first_name", description: "Recipient's first name" },
    { name: "last_name", description: "Recipient's last name" },
    { name: "email", description: "Recipient's email address" },
    { name: "company_name", description: "Company name" },
    { name: "date", description: "Current date" },
    { name: "time", description: "Current time" },
    { name: "dashboard_url", description: "Dashboard URL" },
    { name: "unsubscribe_url", description: "Unsubscribe link" },
  ];

  const insertVariable = (varName: string) => {
    setFormData({
      ...formData,
      content: (formData.content || "") + `{{${varName}}}`,
    });
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-[#1a1a2e] rounded-2xl border border-white/10 w-full max-w-6xl h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-2 rounded-lg bg-purple-500/20">
              <Mail className="h-5 w-5 text-purple-400" />
            </div>
            <div>
              <input
                type="text"
                value={formData.name || ""}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="Template Name"
                className="text-lg font-semibold text-white bg-transparent border-none outline-none placeholder-gray-500"
              />
              <input
                type="text"
                value={formData.description || ""}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="Description"
                className="text-sm text-gray-400 bg-transparent border-none outline-none placeholder-gray-500 w-full"
              />
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="px-4 py-2 bg-white/5 text-gray-300 rounded-lg text-sm font-medium hover:bg-white/10 transition-colors flex items-center gap-2">
              <Send className="h-4 w-4" />
              Send Test
            </button>
            <button
              onClick={() => onSave(formData as EmailTemplate)}
              className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg text-sm font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
            >
              <Save className="h-4 w-4" />
              Save Template
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Toolbar */}
        <div className="px-6 py-3 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* View Toggles */}
            <div className="flex items-center bg-white/5 rounded-lg p-1">
              {(["visual", "html", "preview"] as const).map((view) => (
                <button
                  key={view}
                  onClick={() => setEditorView(view)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                    editorView === view
                      ? "bg-purple-500/20 text-purple-400"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  {view === "visual" ? "Visual" : view === "html" ? "HTML" : "Preview"}
                </button>
              ))}
            </div>

            {/* Formatting Tools */}
            {editorView === "visual" && (
              <div className="flex items-center gap-1 ml-4">
                <button className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <Bold className="h-4 w-4" />
                </button>
                <button className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <Italic className="h-4 w-4" />
                </button>
                <button className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <Underline className="h-4 w-4" />
                </button>
                <div className="w-px h-5 bg-white/10 mx-1" />
                <button className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <AlignLeft className="h-4 w-4" />
                </button>
                <button className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <AlignCenter className="h-4 w-4" />
                </button>
                <button className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <AlignRight className="h-4 w-4" />
                </button>
                <div className="w-px h-5 bg-white/10 mx-1" />
                <button className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <List className="h-4 w-4" />
                </button>
                <button className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <ListOrdered className="h-4 w-4" />
                </button>
                <div className="w-px h-5 bg-white/10 mx-1" />
                <button className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <Link className="h-4 w-4" />
                </button>
                <button className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <Image className="h-4 w-4" />
                </button>
                <div className="w-px h-5 bg-white/10 mx-1" />
                <button
                  onClick={() => setShowVariablePanel(!showVariablePanel)}
                  className={`p-2 rounded-lg transition-colors flex items-center gap-1.5 ${
                    showVariablePanel
                      ? "bg-purple-500/20 text-purple-400"
                      : "hover:bg-white/5 text-gray-400 hover:text-white"
                  }`}
                >
                  <Braces className="h-4 w-4" />
                  <span className="text-sm">Variables</span>
                </button>
              </div>
            )}
          </div>

          {/* Preview Device */}
          {editorView === "preview" && (
            <div className="flex items-center gap-2">
              {(["desktop", "tablet", "mobile"] as const).map((device) => (
                <button
                  key={device}
                  onClick={() => setPreviewDevice(device)}
                  className={`p-2 rounded-lg transition-colors ${
                    previewDevice === device
                      ? "bg-purple-500/20 text-purple-400"
                      : "hover:bg-white/5 text-gray-400 hover:text-white"
                  }`}
                >
                  {device === "desktop" ? (
                    <Monitor className="h-4 w-4" />
                  ) : device === "tablet" ? (
                    <Tablet className="h-4 w-4" />
                  ) : (
                    <Smartphone className="h-4 w-4" />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Editor Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left Panel - Subject & Settings */}
          <div className="w-80 border-r border-white/5 p-4 overflow-y-auto">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Subject Line
                </label>
                <input
                  type="text"
                  value={formData.subject || ""}
                  onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                  placeholder="Enter subject line..."
                  className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Preview Text
                </label>
                <input
                  type="text"
                  placeholder="Text shown in email preview..."
                  className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Category
                </label>
                <select
                  value={formData.category || "transactional"}
                  onChange={(e) => setFormData({ ...formData, category: e.target.value as TemplateCategory })}
                  className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="transactional">Transactional</option>
                  <option value="marketing">Marketing</option>
                  <option value="notification">Notification</option>
                  <option value="welcome">Welcome</option>
                  <option value="followup">Follow-up</option>
                  <option value="reminder">Reminder</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Tags
                </label>
                <input
                  type="text"
                  placeholder="Add tags..."
                  className="w-full px-3 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>

              {/* Variables Panel */}
              {showVariablePanel && (
                <div className="border-t border-white/5 pt-4">
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Available Variables
                  </label>
                  <div className="space-y-1.5">
                    {availableVariables.map((variable) => (
                      <button
                        key={variable.name}
                        onClick={() => insertVariable(variable.name)}
                        className="w-full p-2 text-left bg-white/5 rounded-lg hover:bg-white/10 transition-colors"
                      >
                        <div className="text-sm text-purple-400 font-mono">
                          {`{{${variable.name}}}`}
                        </div>
                        <div className="text-xs text-gray-500">{variable.description}</div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Main Editor */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {editorView === "visual" && (
              <div className="flex-1 p-4 overflow-y-auto">
                <div className="bg-white/5 rounded-xl border border-white/10 min-h-[400px]">
                  <textarea
                    value={formData.content || ""}
                    onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                    placeholder="Start writing your email content..."
                    className="w-full h-full min-h-[400px] p-4 bg-transparent text-white placeholder-gray-500 focus:outline-none resize-none"
                  />
                </div>
              </div>
            )}

            {editorView === "html" && (
              <div className="flex-1 p-4 overflow-y-auto">
                <div className="bg-[#0d0d1a] rounded-xl border border-white/10 h-full">
                  <textarea
                    value={formData.htmlContent || formData.content || ""}
                    onChange={(e) => setFormData({ ...formData, htmlContent: e.target.value })}
                    placeholder="<!DOCTYPE html>..."
                    className="w-full h-full min-h-[400px] p-4 bg-transparent text-green-400 font-mono text-sm placeholder-gray-600 focus:outline-none resize-none"
                  />
                </div>
              </div>
            )}

            {editorView === "preview" && (
              <div className="flex-1 p-4 overflow-y-auto flex justify-center">
                <div
                  className={`bg-white rounded-xl shadow-2xl transition-all ${
                    previewDevice === "desktop"
                      ? "w-full max-w-2xl"
                      : previewDevice === "tablet"
                      ? "w-[768px]"
                      : "w-[375px]"
                  }`}
                >
                  {/* Email Preview */}
                  <div className="p-4 border-b border-gray-200">
                    <div className="text-sm text-gray-500">From: BVRAI &lt;noreply@bvrai.com&gt;</div>
                    <div className="text-sm text-gray-500">To: recipient@example.com</div>
                    <div className="text-lg font-semibold text-gray-900 mt-2">
                      {formData.subject || "No subject"}
                    </div>
                  </div>
                  <div
                    className="p-6 text-gray-800"
                    dangerouslySetInnerHTML={{ __html: formData.content || "<p>Email content...</p>" }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function CreateTemplateDialog({
  isOpen,
  onClose,
  onCreate,
}: {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (template: Partial<EmailTemplate>) => void;
}) {
  const [selectedType, setSelectedType] = useState<"blank" | "template">("blank");
  const [step, setStep] = useState(1);

  if (!isOpen) return null;

  const templateTypes = [
    { id: "blank", name: "Blank Template", description: "Start from scratch", icon: FileText },
    { id: "welcome", name: "Welcome Email", description: "Greet new users", icon: Heart },
    { id: "notification", name: "Notification", description: "Alert or update", icon: MessageSquare },
    { id: "marketing", name: "Marketing", description: "Promotional content", icon: Zap },
    { id: "transactional", name: "Transactional", description: "Order, receipt, etc.", icon: Mail },
    { id: "reminder", name: "Reminder", description: "Appointment or event", icon: Clock },
  ];

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a2e] rounded-2xl border border-white/10 w-full max-w-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">Create Template</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          <p className="text-gray-400 mb-4">Choose a starting point for your new email template</p>
          <div className="grid grid-cols-2 gap-3">
            {templateTypes.map((type) => (
              <button
                key={type.id}
                onClick={() => {
                  if (type.id === "blank") {
                    onCreate({ category: "transactional" });
                  } else {
                    onCreate({ category: type.id as TemplateCategory });
                  }
                }}
                className="p-4 bg-white/5 rounded-xl border border-white/10 hover:border-purple-500/50 transition-all text-left group"
              >
                <type.icon className="h-6 w-6 text-purple-400 mb-2 group-hover:scale-110 transition-transform" />
                <div className="font-medium text-white">{type.name}</div>
                <div className="text-sm text-gray-500">{type.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-white/5 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

// Main Page Component
export default function EmailTemplatesPage() {
  const [templates, setTemplates] = useState<EmailTemplate[]>(sampleTemplates);
  const [selectedFolder, setSelectedFolder] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showEditorDialog, setShowEditorDialog] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<EmailTemplate | null>(null);

  // Filter templates
  const filteredTemplates = useMemo(() => {
    let filtered = templates;

    if (selectedFolder === "favorites") {
      filtered = filtered.filter((t) => t.favorite);
    } else if (selectedFolder !== "all") {
      filtered = filtered.filter((t) => t.category === selectedFolder);
    }

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (t) =>
          t.name.toLowerCase().includes(query) ||
          t.description.toLowerCase().includes(query) ||
          t.subject.toLowerCase().includes(query)
      );
    }

    return filtered;
  }, [templates, selectedFolder, searchQuery]);

  // Calculate stats
  const stats = useMemo(() => {
    const total = templates.length;
    const active = templates.filter((t) => t.status === "active").length;
    const totalSent = templates.reduce((sum, t) => sum + t.stats.sent, 0);
    const avgOpenRate = templates.length > 0
      ? templates.reduce((sum, t) => sum + t.stats.openRate, 0) / templates.length
      : 0;
    return { total, active, totalSent, avgOpenRate };
  }, [templates]);

  const handleEditTemplate = (template: EmailTemplate) => {
    setSelectedTemplate(template);
    setShowEditorDialog(true);
  };

  const handleCreateTemplate = (template: Partial<EmailTemplate>) => {
    setSelectedTemplate(null);
    setShowCreateDialog(false);
    setShowEditorDialog(true);
  };

  const handleSaveTemplate = (template: EmailTemplate) => {
    if (selectedTemplate) {
      setTemplates((prev) =>
        prev.map((t) => (t.id === template.id ? { ...t, ...template, updatedAt: new Date().toISOString() } : t))
      );
    } else {
      setTemplates((prev) => [
        ...prev,
        {
          ...template,
          id: `tmpl_${Date.now()}`,
          status: "draft",
          stats: { sent: 0, opened: 0, clicked: 0, openRate: 0, clickRate: 0 },
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          createdBy: "User",
          favorite: false,
          tags: [],
        },
      ]);
    }
    setShowEditorDialog(false);
    setSelectedTemplate(null);
  };

  const handleToggleFavorite = (templateId: string) => {
    setTemplates((prev) =>
      prev.map((t) => (t.id === templateId ? { ...t, favorite: !t.favorite } : t))
    );
  };

  const handleDeleteTemplate = (templateId: string) => {
    setTemplates((prev) => prev.filter((t) => t.id !== templateId));
  };

  const handleDuplicateTemplate = (template: EmailTemplate) => {
    const duplicate = {
      ...template,
      id: `tmpl_${Date.now()}`,
      name: `${template.name} (Copy)`,
      status: "draft" as TemplateStatus,
      stats: { sent: 0, opened: 0, clicked: 0, openRate: 0, clickRate: 0 },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    setTemplates((prev) => [...prev, duplicate]);
  };

  return (
    <DashboardLayout>
      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar */}
        <div className="w-64 border-r border-white/5 bg-[#0d0d1a]/50 flex flex-col">
          <div className="p-4 border-b border-white/5">
            <button
              onClick={() => setShowCreateDialog(true)}
              className="w-full px-4 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
            >
              <Plus className="h-4 w-4" />
              New Template
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-2">
            {templateFolders.map((folder) => (
              <button
                key={folder.id}
                onClick={() => setSelectedFolder(folder.id)}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-all ${
                  selectedFolder === folder.id
                    ? "bg-purple-500/10 text-purple-400"
                    : "text-gray-400 hover:bg-white/5 hover:text-white"
                }`}
              >
                <div className="flex items-center gap-2">
                  <folder.icon className="h-4 w-4" style={{ color: folder.color }} />
                  <span className="text-sm">{folder.name}</span>
                </div>
                <span className="text-xs text-gray-500">{folder.count}</span>
              </button>
            ))}
          </div>

          {/* Stats */}
          <div className="p-4 border-t border-white/5">
            <div className="grid grid-cols-2 gap-2">
              <div className="p-3 bg-white/5 rounded-lg">
                <div className="text-lg font-bold text-white">{stats.total}</div>
                <div className="text-xs text-gray-500">Templates</div>
              </div>
              <div className="p-3 bg-white/5 rounded-lg">
                <div className="text-lg font-bold text-green-400">{stats.active}</div>
                <div className="text-xs text-gray-500">Active</div>
              </div>
              <div className="p-3 bg-white/5 rounded-lg">
                <div className="text-lg font-bold text-white">{formatNumber(stats.totalSent)}</div>
                <div className="text-xs text-gray-500">Sent</div>
              </div>
              <div className="p-3 bg-white/5 rounded-lg">
                <div className="text-lg font-bold text-purple-400">{stats.avgOpenRate.toFixed(0)}%</div>
                <div className="text-xs text-gray-500">Avg Open</div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <div className="px-6 py-4 border-b border-white/5">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-xl font-semibold text-white">Email Templates</h1>
                <p className="text-sm text-gray-400">
                  {filteredTemplates.length} template{filteredTemplates.length !== 1 ? "s" : ""}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search templates..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 w-64"
                  />
                </div>
                <button className="p-2 rounded-lg bg-white/5 text-gray-400 hover:text-white transition-colors">
                  <Filter className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Template Grid */}
          <div className="flex-1 overflow-y-auto p-6">
            {filteredTemplates.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mb-4">
                  <Mail className="h-8 w-8 text-purple-400" />
                </div>
                <h2 className="text-xl font-semibold text-white mb-2">No templates found</h2>
                <p className="text-gray-400 max-w-md mb-6">
                  {searchQuery
                    ? "Try adjusting your search to find what you're looking for."
                    : "Create your first email template to get started."}
                </p>
                <button
                  onClick={() => setShowCreateDialog(true)}
                  className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity inline-flex items-center gap-2"
                >
                  <Plus className="h-4 w-4" />
                  Create Template
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredTemplates.map((template) => (
                  <TemplateCard
                    key={template.id}
                    template={template}
                    onEdit={() => handleEditTemplate(template)}
                    onDuplicate={() => handleDuplicateTemplate(template)}
                    onDelete={() => handleDeleteTemplate(template.id)}
                    onToggleFavorite={() => handleToggleFavorite(template.id)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Dialogs */}
        <CreateTemplateDialog
          isOpen={showCreateDialog}
          onClose={() => setShowCreateDialog(false)}
          onCreate={handleCreateTemplate}
        />
        <TemplateEditorDialog
          template={selectedTemplate}
          isOpen={showEditorDialog}
          onClose={() => {
            setShowEditorDialog(false);
            setSelectedTemplate(null);
          }}
          onSave={handleSaveTemplate}
        />
      </div>
    </DashboardLayout>
  );
}
