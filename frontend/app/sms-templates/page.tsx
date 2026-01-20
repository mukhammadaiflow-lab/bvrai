"use client";

import { useState, useMemo, useCallback } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import {
  MessageSquare,
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
  AtSign,
  Hash,
  Braces,
  ChevronRight,
  ChevronDown,
  X,
  Save,
  Download,
  Upload,
  RefreshCw,
  Settings,
  Sparkles,
  Zap,
  Smartphone,
  Globe,
  Users,
  User,
  Building2,
  Phone,
  Heart,
  ThumbsUp,
  Share2,
  ExternalLink,
  ArrowUpRight,
  BarChart3,
  TrendingUp,
  Activity,
  Wand2,
  Type,
  MessageCircle,
  Loader2,
  AlertTriangle,
  CheckCheck,
  XOctagon,
  Inbox,
  SendHorizontal,
  Reply,
  Bot,
  Workflow,
  Timer,
  LayoutGrid,
  List,
  ArrowDownRight,
} from "lucide-react";

// Types
type SMSTemplateStatus = "draft" | "active" | "archived";
type SMSTemplateCategory = "marketing" | "transactional" | "notification" | "reminder" | "followup" | "otp";

interface SMSTemplate {
  id: string;
  name: string;
  description: string;
  content: string;
  category: SMSTemplateCategory;
  status: SMSTemplateStatus;
  variables: string[];
  characterCount: number;
  segmentCount: number;
  stats: {
    sent: number;
    delivered: number;
    clicked: number;
    replied: number;
    deliveryRate: number;
    clickRate: number;
    replyRate: number;
  };
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  favorite: boolean;
  tags: string[];
  shortLink?: boolean;
  includeOptOut: boolean;
}

interface SMSConversation {
  id: string;
  phone: string;
  name: string;
  lastMessage: string;
  lastMessageTime: string;
  unread: number;
  status: "active" | "resolved" | "pending";
}

// Sample Data
const sampleSMSTemplates: SMSTemplate[] = [
  {
    id: "sms_1",
    name: "Appointment Reminder",
    description: "Sent 24 hours before appointment",
    content: "Hi {{first_name}}! Reminder: Your appointment is tomorrow at {{time}}. Reply CONFIRM to confirm or RESCHEDULE to change. Questions? Call us at {{phone}}.",
    category: "reminder",
    status: "active",
    variables: ["first_name", "time", "phone"],
    characterCount: 158,
    segmentCount: 1,
    stats: { sent: 25400, delivered: 24638, clicked: 0, replied: 18315, deliveryRate: 97, clickRate: 0, replyRate: 72 },
    createdAt: "2024-01-05T10:00:00Z",
    updatedAt: "2024-01-18T14:30:00Z",
    createdBy: "Admin",
    favorite: true,
    tags: ["appointment", "healthcare"],
    shortLink: false,
    includeOptOut: false,
  },
  {
    id: "sms_2",
    name: "Order Shipped",
    description: "Sent when order is shipped",
    content: "Great news, {{first_name}}! Your order #{{order_number}} has shipped. Track it here: {{tracking_link}}",
    category: "transactional",
    status: "active",
    variables: ["first_name", "order_number", "tracking_link"],
    characterCount: 112,
    segmentCount: 1,
    stats: { sent: 45200, delivered: 44296, clicked: 35436, replied: 1356, deliveryRate: 98, clickRate: 78, replyRate: 3 },
    createdAt: "2024-01-02T09:00:00Z",
    updatedAt: "2024-01-15T11:20:00Z",
    createdBy: "Admin",
    favorite: true,
    tags: ["ecommerce", "shipping"],
    shortLink: true,
    includeOptOut: false,
  },
  {
    id: "sms_3",
    name: "Flash Sale Alert",
    description: "Limited time offer notification",
    content: "{{first_name}}, flash sale! 30% off everything for the next 2 hours only. Shop now: {{link}} Reply STOP to opt out.",
    category: "marketing",
    status: "active",
    variables: ["first_name", "link"],
    characterCount: 128,
    segmentCount: 1,
    stats: { sent: 15000, delivered: 14400, clicked: 4320, replied: 450, deliveryRate: 96, clickRate: 29, replyRate: 3 },
    createdAt: "2024-01-10T15:00:00Z",
    updatedAt: "2024-01-17T09:45:00Z",
    createdBy: "Marketing",
    favorite: false,
    tags: ["sale", "promotion"],
    shortLink: true,
    includeOptOut: true,
  },
  {
    id: "sms_4",
    name: "OTP Verification",
    description: "Two-factor authentication code",
    content: "Your {{company_name}} verification code is: {{code}}. Valid for 10 minutes. Don't share this code with anyone.",
    category: "otp",
    status: "active",
    variables: ["company_name", "code"],
    characterCount: 118,
    segmentCount: 1,
    stats: { sent: 85000, delivered: 84150, clicked: 0, replied: 0, deliveryRate: 99, clickRate: 0, replyRate: 0 },
    createdAt: "2024-01-01T08:00:00Z",
    updatedAt: "2024-01-01T08:00:00Z",
    createdBy: "System",
    favorite: true,
    tags: ["security", "auth"],
    shortLink: false,
    includeOptOut: false,
  },
  {
    id: "sms_5",
    name: "Follow-up After Call",
    description: "Sent after support call",
    content: "Thanks for calling us, {{first_name}}! Here's your support ticket: #{{ticket_id}}. Need more help? Reply to this message.",
    category: "followup",
    status: "active",
    variables: ["first_name", "ticket_id"],
    characterCount: 131,
    segmentCount: 1,
    stats: { sent: 8500, delivered: 8330, clicked: 0, replied: 1666, deliveryRate: 98, clickRate: 0, replyRate: 20 },
    createdAt: "2024-01-08T11:00:00Z",
    updatedAt: "2024-01-16T16:00:00Z",
    createdBy: "Support",
    favorite: false,
    tags: ["support", "followup"],
    shortLink: false,
    includeOptOut: false,
  },
  {
    id: "sms_6",
    name: "Payment Due Reminder",
    description: "Sent 3 days before payment due",
    content: "Hi {{first_name}}, your payment of {{amount}} is due on {{due_date}}. Pay now: {{payment_link}} or call {{phone}} for assistance.",
    category: "notification",
    status: "active",
    variables: ["first_name", "amount", "due_date", "payment_link", "phone"],
    characterCount: 145,
    segmentCount: 1,
    stats: { sent: 12000, delivered: 11640, clicked: 6984, replied: 1164, deliveryRate: 97, clickRate: 58, replyRate: 10 },
    createdAt: "2024-01-06T14:00:00Z",
    updatedAt: "2024-01-19T10:30:00Z",
    createdBy: "Finance",
    favorite: false,
    tags: ["billing", "payment"],
    shortLink: true,
    includeOptOut: false,
  },
  {
    id: "sms_7",
    name: "Welcome Message",
    description: "Sent after sign up",
    content: "Welcome to {{company_name}}, {{first_name}}! 🎉 Get started: {{link}} Need help? Text us anytime!",
    category: "notification",
    status: "draft",
    variables: ["company_name", "first_name", "link"],
    characterCount: 104,
    segmentCount: 1,
    stats: { sent: 0, delivered: 0, clicked: 0, replied: 0, deliveryRate: 0, clickRate: 0, replyRate: 0 },
    createdAt: "2024-01-19T16:00:00Z",
    updatedAt: "2024-01-19T16:00:00Z",
    createdBy: "Admin",
    favorite: false,
    tags: ["onboarding", "welcome"],
    shortLink: true,
    includeOptOut: false,
  },
];

const sampleConversations: SMSConversation[] = [
  { id: "conv_1", phone: "+1 (555) 123-4567", name: "John Smith", lastMessage: "Yes, I'll confirm my appointment", lastMessageTime: "2024-01-20T14:30:00Z", unread: 0, status: "resolved" },
  { id: "conv_2", phone: "+1 (555) 234-5678", name: "Sarah Johnson", lastMessage: "Can I reschedule to next week?", lastMessageTime: "2024-01-20T14:15:00Z", unread: 1, status: "pending" },
  { id: "conv_3", phone: "+1 (555) 345-6789", name: "Mike Davis", lastMessage: "Thanks for the quick response!", lastMessageTime: "2024-01-20T13:45:00Z", unread: 0, status: "resolved" },
];

// Utility functions
const getCategoryColor = (category: SMSTemplateCategory) => {
  switch (category) {
    case "marketing":
      return "bg-pink-500/20 text-pink-400";
    case "transactional":
      return "bg-blue-500/20 text-blue-400";
    case "notification":
      return "bg-green-500/20 text-green-400";
    case "reminder":
      return "bg-orange-500/20 text-orange-400";
    case "followup":
      return "bg-indigo-500/20 text-indigo-400";
    case "otp":
      return "bg-red-500/20 text-red-400";
    default:
      return "bg-gray-500/20 text-gray-400";
  }
};

const getStatusColor = (status: SMSTemplateStatus) => {
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

const formatTime = (dateString: string) => {
  return new Date(dateString).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
};

const calculateSegments = (text: string) => {
  const length = text.length;
  if (length <= 160) return 1;
  return Math.ceil(length / 153);
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

function SMSTemplateCard({
  template,
  onEdit,
  onDuplicate,
  onDelete,
  onToggleFavorite,
  onSendTest,
}: {
  template: SMSTemplate;
  onEdit: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onToggleFavorite: () => void;
  onSendTest: () => void;
}) {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 hover:border-white/10 transition-all group">
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-purple-500/20">
              <MessageSquare className="h-4 w-4 text-purple-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-white group-hover:text-purple-400 transition-colors">
                  {template.name}
                </h3>
                <button onClick={onToggleFavorite}>
                  {template.favorite ? (
                    <Star className="h-4 w-4 text-yellow-400 fill-yellow-400" />
                  ) : (
                    <Star className="h-4 w-4 text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                  )}
                </button>
              </div>
              <p className="text-sm text-gray-400">{template.description}</p>
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
                  onClick={() => { onEdit(); setShowMenu(false); }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                >
                  <Edit className="h-4 w-4" /> Edit
                </button>
                <button
                  onClick={() => { onDuplicate(); setShowMenu(false); }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                >
                  <Copy className="h-4 w-4" /> Duplicate
                </button>
                <button
                  onClick={() => { onSendTest(); setShowMenu(false); }}
                  className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                >
                  <Send className="h-4 w-4" /> Send Test
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

        {/* Message Preview */}
        <div className="bg-[#0d0d1a] rounded-xl p-3 mb-3">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
              BV
            </div>
            <div className="flex-1">
              <div className="bg-purple-500/20 rounded-2xl rounded-tl-none p-3 text-sm text-gray-300 inline-block max-w-full">
                {template.content}
              </div>
              <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                <span>{template.characterCount} characters</span>
                <span>•</span>
                <span>{template.segmentCount} segment{template.segmentCount > 1 ? "s" : ""}</span>
                {template.shortLink && (
                  <>
                    <span>•</span>
                    <span className="text-blue-400">Short link</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Tags & Status */}
        <div className="flex items-center gap-2 mb-3">
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${getStatusColor(template.status)}`}>
            {template.status}
          </span>
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
        <div className="flex items-center justify-between pt-3 border-t border-white/5">
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-1.5">
              <Send className="h-3.5 w-3.5 text-gray-500" />
              <span className="text-gray-400">{formatNumber(template.stats.sent)}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <CheckCheck className="h-3.5 w-3.5 text-gray-500" />
              <span className="text-gray-400">{template.stats.deliveryRate}%</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Reply className="h-3.5 w-3.5 text-gray-500" />
              <span className="text-gray-400">{template.stats.replyRate}%</span>
            </div>
          </div>
          <div className="text-xs text-gray-500">
            {formatDate(template.updatedAt)}
          </div>
        </div>
      </div>
    </div>
  );
}

function SMSEditorDialog({
  template,
  isOpen,
  onClose,
  onSave,
}: {
  template: SMSTemplate | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: (template: SMSTemplate) => void;
}) {
  const [formData, setFormData] = useState<Partial<SMSTemplate>>(
    template || {
      name: "",
      description: "",
      content: "",
      category: "notification",
      variables: [],
      tags: [],
      shortLink: false,
      includeOptOut: false,
    }
  );

  const characterCount = (formData.content || "").length;
  const segmentCount = calculateSegments(formData.content || "");

  const availableVariables = [
    { name: "first_name", description: "Recipient's first name" },
    { name: "last_name", description: "Recipient's last name" },
    { name: "phone", description: "Business phone number" },
    { name: "company_name", description: "Company name" },
    { name: "link", description: "Tracking link" },
    { name: "code", description: "Verification code" },
    { name: "amount", description: "Payment amount" },
    { name: "date", description: "Date" },
    { name: "time", description: "Time" },
  ];

  const insertVariable = (varName: string) => {
    setFormData({
      ...formData,
      content: (formData.content || "") + `{{${varName}}}`,
    });
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a2e] rounded-2xl border border-white/10 w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">
              {template ? "Edit SMS Template" : "Create SMS Template"}
            </h2>
            <p className="text-sm text-gray-400 mt-0.5">
              Compose your SMS message with personalization
            </p>
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
          <div className="grid grid-cols-2 gap-6">
            {/* Left Column - Form */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Template Name
                </label>
                <input
                  type="text"
                  value={formData.name || ""}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="e.g., Appointment Reminder"
                  className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Description
                </label>
                <input
                  type="text"
                  value={formData.description || ""}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="Brief description of when this is used"
                  className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Category
                </label>
                <select
                  value={formData.category || "notification"}
                  onChange={(e) => setFormData({ ...formData, category: e.target.value as SMSTemplateCategory })}
                  className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="notification">Notification</option>
                  <option value="transactional">Transactional</option>
                  <option value="marketing">Marketing</option>
                  <option value="reminder">Reminder</option>
                  <option value="followup">Follow-up</option>
                  <option value="otp">OTP/Verification</option>
                </select>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-300">
                    Message Content
                  </label>
                  <div className="text-xs text-gray-500">
                    <span className={characterCount > 160 ? "text-yellow-400" : "text-gray-400"}>
                      {characterCount}
                    </span>
                    /160 ({segmentCount} segment{segmentCount > 1 ? "s" : ""})
                  </div>
                </div>
                <textarea
                  value={formData.content || ""}
                  onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                  placeholder="Type your SMS message here..."
                  rows={6}
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                />
                {characterCount > 160 && (
                  <p className="text-xs text-yellow-400 mt-1">
                    Messages over 160 characters will be sent as multiple segments
                  </p>
                )}
              </div>

              {/* Variables */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Insert Variable
                </label>
                <div className="flex flex-wrap gap-2">
                  {availableVariables.map((variable) => (
                    <button
                      key={variable.name}
                      onClick={() => insertVariable(variable.name)}
                      className="px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg text-xs font-medium hover:bg-purple-500/30 transition-colors"
                    >
                      {`{{${variable.name}}}`}
                    </button>
                  ))}
                </div>
              </div>

              {/* Options */}
              <div className="flex items-center gap-6">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formData.shortLink}
                    onChange={(e) => setFormData({ ...formData, shortLink: e.target.checked })}
                    className="w-4 h-4 rounded border-white/20 bg-white/5 text-purple-500 focus:ring-purple-500"
                  />
                  <span className="text-sm text-gray-300">Use short links</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formData.includeOptOut}
                    onChange={(e) => setFormData({ ...formData, includeOptOut: e.target.checked })}
                    className="w-4 h-4 rounded border-white/20 bg-white/5 text-purple-500 focus:ring-purple-500"
                  />
                  <span className="text-sm text-gray-300">Include opt-out text</span>
                </label>
              </div>
            </div>

            {/* Right Column - Preview */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Preview
              </label>
              <div className="bg-[#0d0d1a] rounded-2xl p-4 border border-white/10">
                {/* Phone Frame */}
                <div className="bg-gray-900 rounded-3xl p-4 max-w-[280px] mx-auto">
                  {/* Status Bar */}
                  <div className="flex items-center justify-between text-white text-xs mb-3">
                    <span>9:41</span>
                    <div className="flex items-center gap-1">
                      <div className="w-4 h-2 bg-white rounded-sm" />
                    </div>
                  </div>

                  {/* Header */}
                  <div className="text-center mb-4">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 mx-auto mb-2 flex items-center justify-center text-white font-bold">
                      BV
                    </div>
                    <div className="text-white text-sm font-medium">BVRAI</div>
                    <div className="text-gray-500 text-xs">Business SMS</div>
                  </div>

                  {/* Message */}
                  <div className="space-y-2">
                    <div className="bg-gray-800 rounded-2xl rounded-tl-none p-3 text-sm text-gray-200 max-w-[85%]">
                      {(formData.content || "Your message preview will appear here...")
                        .replace(/\{\{(\w+)\}\}/g, "[$1]")}
                    </div>
                    <div className="text-xs text-gray-500 text-center">
                      Just now
                    </div>
                  </div>

                  {/* Input */}
                  <div className="mt-4 bg-gray-800 rounded-full px-4 py-2 flex items-center gap-2">
                    <span className="text-gray-500 text-sm flex-1">iMessage</span>
                    <div className="w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center">
                      <ArrowUpRight className="h-3 w-3 text-white" />
                    </div>
                  </div>
                </div>

                {/* Info */}
                <div className="mt-4 p-3 bg-white/5 rounded-lg">
                  <div className="flex items-center gap-2 text-sm">
                    <Info className="h-4 w-4 text-blue-400" />
                    <span className="text-gray-400">
                      {segmentCount === 1
                        ? "This message will be sent as a single SMS"
                        : `This message will be sent as ${segmentCount} SMS segments`}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-white/5 flex items-center justify-between">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <div className="flex items-center gap-3">
            <button className="px-4 py-2 bg-white/5 text-gray-300 rounded-xl text-sm font-medium hover:bg-white/10 transition-colors flex items-center gap-2">
              <Send className="h-4 w-4" />
              Send Test
            </button>
            <button
              onClick={() => onSave(formData as SMSTemplate)}
              className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
            >
              <Save className="h-4 w-4" />
              Save Template
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ConversationsPanel({ conversations }: { conversations: SMSConversation[] }) {
  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 overflow-hidden">
      <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <Inbox className="h-4 w-4 text-purple-400" />
          Recent Conversations
        </h3>
        <button className="text-sm text-purple-400 hover:text-purple-300 transition-colors">
          View All
        </button>
      </div>
      <div className="divide-y divide-white/5">
        {conversations.map((conv) => (
          <button
            key={conv.id}
            className="w-full px-4 py-3 hover:bg-white/5 transition-colors text-left"
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <span className="font-medium text-white">{conv.name}</span>
                {conv.unread > 0 && (
                  <span className="px-1.5 py-0.5 bg-purple-500 text-white text-xs rounded-full">
                    {conv.unread}
                  </span>
                )}
              </div>
              <span className="text-xs text-gray-500">{formatTime(conv.lastMessageTime)}</span>
            </div>
            <p className="text-sm text-gray-400 truncate">{conv.lastMessage}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

// Main Page Component
export default function SMSTemplatesPage() {
  const [templates, setTemplates] = useState<SMSTemplate[]>(sampleSMSTemplates);
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<SMSTemplateCategory | "all">("all");
  const [statusFilter, setStatusFilter] = useState<SMSTemplateStatus | "all">("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [showEditorDialog, setShowEditorDialog] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<SMSTemplate | null>(null);

  // Filter templates
  const filteredTemplates = useMemo(() => {
    let filtered = templates;

    if (categoryFilter !== "all") {
      filtered = filtered.filter((t) => t.category === categoryFilter);
    }

    if (statusFilter !== "all") {
      filtered = filtered.filter((t) => t.status === statusFilter);
    }

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (t) =>
          t.name.toLowerCase().includes(query) ||
          t.description.toLowerCase().includes(query) ||
          t.content.toLowerCase().includes(query)
      );
    }

    return filtered;
  }, [templates, categoryFilter, statusFilter, searchQuery]);

  // Calculate stats
  const stats = useMemo(() => {
    const totalSent = templates.reduce((sum, t) => sum + t.stats.sent, 0);
    const totalDelivered = templates.reduce((sum, t) => sum + t.stats.delivered, 0);
    const avgDeliveryRate = templates.length > 0
      ? templates.reduce((sum, t) => sum + t.stats.deliveryRate, 0) / templates.length
      : 0;
    const avgReplyRate = templates.length > 0
      ? templates.reduce((sum, t) => sum + t.stats.replyRate, 0) / templates.length
      : 0;
    return { totalSent, totalDelivered, avgDeliveryRate, avgReplyRate };
  }, [templates]);

  const handleSaveTemplate = (template: SMSTemplate) => {
    if (selectedTemplate) {
      setTemplates((prev) =>
        prev.map((t) =>
          t.id === template.id
            ? {
                ...t,
                ...template,
                characterCount: template.content.length,
                segmentCount: calculateSegments(template.content),
                updatedAt: new Date().toISOString(),
              }
            : t
        )
      );
    } else {
      setTemplates((prev) => [
        ...prev,
        {
          ...template,
          id: `sms_${Date.now()}`,
          status: "draft" as SMSTemplateStatus,
          characterCount: template.content?.length || 0,
          segmentCount: calculateSegments(template.content || ""),
          stats: { sent: 0, delivered: 0, clicked: 0, replied: 0, deliveryRate: 0, clickRate: 0, replyRate: 0 },
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          createdBy: "User",
          favorite: false,
        } as SMSTemplate,
      ]);
    }
    setShowEditorDialog(false);
    setSelectedTemplate(null);
  };

  return (
    <DashboardLayout>
      <div className="p-6 lg:p-8 max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
          <div>
            <h1 className="text-2xl lg:text-3xl font-bold text-white">SMS Templates</h1>
            <p className="text-gray-400 mt-1">Create and manage SMS message templates</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                setSelectedTemplate(null);
                setShowEditorDialog(true);
              }}
              className="px-4 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
            >
              <Plus className="h-4 w-4" />
              New Template
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <StatsCard
            title="Total Sent"
            value={formatNumber(stats.totalSent)}
            icon={Send}
            color="#8B5CF6"
            change="+12%"
            changeType="positive"
          />
          <StatsCard
            title="Delivered"
            value={formatNumber(stats.totalDelivered)}
            icon={CheckCheck}
            color="#10B981"
          />
          <StatsCard
            title="Avg Delivery Rate"
            value={`${stats.avgDeliveryRate.toFixed(0)}%`}
            icon={TrendingUp}
            color="#3B82F6"
            change="+2%"
            changeType="positive"
          />
          <StatsCard
            title="Avg Reply Rate"
            value={`${stats.avgReplyRate.toFixed(0)}%`}
            icon={Reply}
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
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>
            <div className="flex items-center gap-3">
              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value as SMSTemplateCategory | "all")}
                className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
              >
                <option value="all">All Categories</option>
                <option value="notification">Notification</option>
                <option value="transactional">Transactional</option>
                <option value="marketing">Marketing</option>
                <option value="reminder">Reminder</option>
                <option value="followup">Follow-up</option>
                <option value="otp">OTP</option>
              </select>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value as SMSTemplateStatus | "all")}
                className="px-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="draft">Draft</option>
                <option value="archived">Archived</option>
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

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Templates Grid */}
          <div className="lg:col-span-3">
            {filteredTemplates.length === 0 ? (
              <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-12 text-center">
                <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mx-auto mb-4">
                  <MessageSquare className="h-8 w-8 text-purple-400" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">No templates found</h3>
                <p className="text-gray-400 mb-6">
                  {searchQuery || categoryFilter !== "all" || statusFilter !== "all"
                    ? "Try adjusting your filters."
                    : "Create your first SMS template to get started."}
                </p>
                <button
                  onClick={() => {
                    setSelectedTemplate(null);
                    setShowEditorDialog(true);
                  }}
                  className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity inline-flex items-center gap-2"
                >
                  <Plus className="h-4 w-4" />
                  Create Template
                </button>
              </div>
            ) : (
              <div className={viewMode === "grid" ? "grid grid-cols-1 md:grid-cols-2 gap-4" : "space-y-4"}>
                {filteredTemplates.map((template) => (
                  <SMSTemplateCard
                    key={template.id}
                    template={template}
                    onEdit={() => {
                      setSelectedTemplate(template);
                      setShowEditorDialog(true);
                    }}
                    onDuplicate={() => {
                      const duplicate = {
                        ...template,
                        id: `sms_${Date.now()}`,
                        name: `${template.name} (Copy)`,
                        status: "draft" as SMSTemplateStatus,
                        stats: { sent: 0, delivered: 0, clicked: 0, replied: 0, deliveryRate: 0, clickRate: 0, replyRate: 0 },
                        createdAt: new Date().toISOString(),
                        updatedAt: new Date().toISOString(),
                      };
                      setTemplates((prev) => [...prev, duplicate]);
                    }}
                    onDelete={() => setTemplates((prev) => prev.filter((t) => t.id !== template.id))}
                    onToggleFavorite={() =>
                      setTemplates((prev) =>
                        prev.map((t) => (t.id === template.id ? { ...t, favorite: !t.favorite } : t))
                      )
                    }
                    onSendTest={() => {}}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            <ConversationsPanel conversations={sampleConversations} />

            {/* Quick Actions */}
            <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-4">
              <h3 className="font-semibold text-white mb-3">Quick Actions</h3>
              <div className="space-y-2">
                <button className="w-full px-4 py-2.5 bg-white/5 rounded-lg text-left hover:bg-white/10 transition-colors flex items-center gap-3">
                  <Bot className="h-4 w-4 text-purple-400" />
                  <span className="text-sm text-gray-300">AI Generate Template</span>
                </button>
                <button className="w-full px-4 py-2.5 bg-white/5 rounded-lg text-left hover:bg-white/10 transition-colors flex items-center gap-3">
                  <Upload className="h-4 w-4 text-blue-400" />
                  <span className="text-sm text-gray-300">Import Templates</span>
                </button>
                <button className="w-full px-4 py-2.5 bg-white/5 rounded-lg text-left hover:bg-white/10 transition-colors flex items-center gap-3">
                  <Workflow className="h-4 w-4 text-green-400" />
                  <span className="text-sm text-gray-300">Automation Rules</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Editor Dialog */}
        <SMSEditorDialog
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
