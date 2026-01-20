"use client";

import React, { useState, useMemo, useCallback } from "react";
import { DashboardLayout } from "@/components/dashboard-layout";
import {
  Search,
  Plus,
  Upload,
  FileText,
  File,
  Folder,
  FolderOpen,
  FolderPlus,
  Trash2,
  Edit,
  Eye,
  Download,
  MoreVertical,
  ChevronRight,
  ChevronDown,
  ExternalLink,
  Link,
  Unlink,
  Bot,
  Settings,
  RefreshCw,
  Copy,
  Check,
  X,
  AlertCircle,
  CheckCircle,
  Clock,
  Loader2,
  Filter,
  Grid,
  List,
  BookOpen,
  FileQuestion,
  Globe,
  Database,
  Sparkles,
  Brain,
  Zap,
  Target,
  Tag,
  Star,
  StarOff,
  Hash,
  FileCode,
  FileJson,
  FileImage,
  FileSpreadsheet,
  FileArchive,
  MessageSquare,
  HelpCircle,
  Info,
  Wand2,
  ArrowRight,
  Play,
  Pause,
  Lightbulb,
} from "lucide-react";

// Types
interface KnowledgeSource {
  id: string;
  name: string;
  type: SourceType;
  status: "active" | "processing" | "error" | "disabled";
  category: string;
  description?: string;
  url?: string;
  filePath?: string;
  fileSize?: number;
  mimeType?: string;
  content?: string;
  metadata: {
    wordCount?: number;
    charCount?: number;
    pageCount?: number;
    lastCrawled?: string;
    crawlFrequency?: string;
    chunkCount?: number;
  };
  agents: string[];
  tags: string[];
  createdAt: string;
  updatedAt: string;
  lastSynced?: string;
}

type SourceType =
  | "document"
  | "webpage"
  | "faq"
  | "api"
  | "database"
  | "text"
  | "sitemap";

interface KnowledgeCategory {
  id: string;
  name: string;
  description?: string;
  sourceCount: number;
  icon: string;
}

interface Agent {
  id: string;
  name: string;
  status: "active" | "inactive";
}

// Source type configuration
const sourceTypeConfig: Record<
  SourceType,
  { icon: React.ReactNode; label: string; color: string }
> = {
  document: {
    icon: <FileText className="w-5 h-5" />,
    label: "Document",
    color: "blue",
  },
  webpage: {
    icon: <Globe className="w-5 h-5" />,
    label: "Web Page",
    color: "green",
  },
  faq: {
    icon: <HelpCircle className="w-5 h-5" />,
    label: "FAQ",
    color: "purple",
  },
  api: {
    icon: <Zap className="w-5 h-5" />,
    label: "API",
    color: "orange",
  },
  database: {
    icon: <Database className="w-5 h-5" />,
    label: "Database",
    color: "cyan",
  },
  text: {
    icon: <FileCode className="w-5 h-5" />,
    label: "Plain Text",
    color: "gray",
  },
  sitemap: {
    icon: <Globe className="w-5 h-5" />,
    label: "Sitemap",
    color: "pink",
  },
};

// File type icons
const getFileIcon = (mimeType?: string) => {
  if (!mimeType) return <File className="w-5 h-5" />;
  if (mimeType.includes("pdf")) return <FileText className="w-5 h-5 text-red-400" />;
  if (mimeType.includes("json")) return <FileJson className="w-5 h-5 text-yellow-400" />;
  if (mimeType.includes("image")) return <FileImage className="w-5 h-5 text-purple-400" />;
  if (mimeType.includes("spreadsheet") || mimeType.includes("excel"))
    return <FileSpreadsheet className="w-5 h-5 text-green-400" />;
  if (mimeType.includes("zip") || mimeType.includes("archive"))
    return <FileArchive className="w-5 h-5 text-orange-400" />;
  return <FileText className="w-5 h-5 text-blue-400" />;
};

// Mock data
const mockCategories: KnowledgeCategory[] = [
  { id: "cat_1", name: "Products", description: "Product documentation and specs", sourceCount: 12, icon: "box" },
  { id: "cat_2", name: "FAQs", description: "Frequently asked questions", sourceCount: 8, icon: "help" },
  { id: "cat_3", name: "Policies", description: "Company policies and procedures", sourceCount: 5, icon: "shield" },
  { id: "cat_4", name: "Support", description: "Support articles and guides", sourceCount: 15, icon: "headphones" },
  { id: "cat_5", name: "Pricing", description: "Pricing information", sourceCount: 3, icon: "dollar" },
  { id: "cat_6", name: "General", description: "General company information", sourceCount: 7, icon: "info" },
];

const mockSources: KnowledgeSource[] = [
  {
    id: "src_1",
    name: "Product Catalog 2024",
    type: "document",
    status: "active",
    category: "Products",
    description: "Complete product catalog with features and specifications",
    filePath: "/documents/product-catalog-2024.pdf",
    fileSize: 2456000,
    mimeType: "application/pdf",
    metadata: {
      wordCount: 15000,
      pageCount: 45,
      chunkCount: 120,
    },
    agents: ["agent_1", "agent_2"],
    tags: ["products", "catalog", "2024"],
    createdAt: "2024-01-10T10:00:00Z",
    updatedAt: "2024-01-15T14:30:00Z",
    lastSynced: "2024-01-20T08:00:00Z",
  },
  {
    id: "src_2",
    name: "Company Website",
    type: "webpage",
    status: "active",
    category: "General",
    description: "Main company website content",
    url: "https://www.example.com",
    metadata: {
      wordCount: 8500,
      lastCrawled: "2024-01-20T06:00:00Z",
      crawlFrequency: "daily",
      chunkCount: 85,
    },
    agents: ["agent_1", "agent_2", "agent_3"],
    tags: ["website", "general", "auto-sync"],
    createdAt: "2024-01-05T08:00:00Z",
    updatedAt: "2024-01-20T06:00:00Z",
    lastSynced: "2024-01-20T06:00:00Z",
  },
  {
    id: "src_3",
    name: "General FAQs",
    type: "faq",
    status: "active",
    category: "FAQs",
    description: "Most common customer questions and answers",
    metadata: {
      wordCount: 5000,
      chunkCount: 50,
    },
    agents: ["agent_1", "agent_2"],
    tags: ["faq", "support", "common-questions"],
    createdAt: "2024-01-08T09:00:00Z",
    updatedAt: "2024-01-18T11:00:00Z",
    lastSynced: "2024-01-18T11:00:00Z",
  },
  {
    id: "src_4",
    name: "Pricing API",
    type: "api",
    status: "active",
    category: "Pricing",
    description: "Real-time pricing information from internal API",
    url: "https://api.internal.com/pricing",
    metadata: {
      chunkCount: 25,
    },
    agents: ["agent_1"],
    tags: ["pricing", "api", "real-time"],
    createdAt: "2024-01-12T12:00:00Z",
    updatedAt: "2024-01-19T15:00:00Z",
    lastSynced: "2024-01-20T09:00:00Z",
  },
  {
    id: "src_5",
    name: "Return Policy",
    type: "document",
    status: "active",
    category: "Policies",
    description: "Customer return and refund policy",
    filePath: "/documents/return-policy.docx",
    fileSize: 45000,
    mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    metadata: {
      wordCount: 2500,
      pageCount: 5,
      chunkCount: 30,
    },
    agents: ["agent_2"],
    tags: ["policy", "returns", "refunds"],
    createdAt: "2024-01-06T14:00:00Z",
    updatedAt: "2024-01-14T10:00:00Z",
    lastSynced: "2024-01-14T10:00:00Z",
  },
  {
    id: "src_6",
    name: "Support Articles Sitemap",
    type: "sitemap",
    status: "processing",
    category: "Support",
    description: "Auto-crawl of support documentation",
    url: "https://support.example.com/sitemap.xml",
    metadata: {
      lastCrawled: "2024-01-20T02:00:00Z",
      crawlFrequency: "weekly",
    },
    agents: ["agent_2"],
    tags: ["support", "sitemap", "auto-crawl"],
    createdAt: "2024-01-15T16:00:00Z",
    updatedAt: "2024-01-20T02:00:00Z",
  },
  {
    id: "src_7",
    name: "Shipping Information",
    type: "text",
    status: "active",
    category: "Policies",
    description: "Shipping rates and delivery times",
    content: "Standard shipping: 5-7 business days...",
    metadata: {
      wordCount: 800,
      charCount: 4500,
      chunkCount: 8,
    },
    agents: ["agent_1", "agent_2"],
    tags: ["shipping", "delivery"],
    createdAt: "2024-01-09T11:00:00Z",
    updatedAt: "2024-01-17T09:00:00Z",
    lastSynced: "2024-01-17T09:00:00Z",
  },
  {
    id: "src_8",
    name: "Customer Database",
    type: "database",
    status: "error",
    category: "General",
    description: "Customer information database connection",
    url: "postgresql://db.example.com:5432/customers",
    metadata: {},
    agents: [],
    tags: ["database", "customers"],
    createdAt: "2024-01-18T13:00:00Z",
    updatedAt: "2024-01-19T08:00:00Z",
  },
];

const mockAgents: Agent[] = [
  { id: "agent_1", name: "Sales Assistant", status: "active" },
  { id: "agent_2", name: "Support Bot", status: "active" },
  { id: "agent_3", name: "Booking Agent", status: "inactive" },
];

// Format file size
function formatFileSize(bytes?: number): string {
  if (!bytes) return "N/A";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

// Status badge component
function StatusBadge({ status }: { status: KnowledgeSource["status"] }) {
  const config = {
    active: {
      icon: <CheckCircle className="w-3 h-3" />,
      color: "bg-green-500/10 text-green-400 border-green-500/20",
      label: "Active",
    },
    processing: {
      icon: <Loader2 className="w-3 h-3 animate-spin" />,
      color: "bg-blue-500/10 text-blue-400 border-blue-500/20",
      label: "Processing",
    },
    error: {
      icon: <AlertCircle className="w-3 h-3" />,
      color: "bg-red-500/10 text-red-400 border-red-500/20",
      label: "Error",
    },
    disabled: {
      icon: <Pause className="w-3 h-3" />,
      color: "bg-gray-500/10 text-gray-400 border-gray-500/20",
      label: "Disabled",
    },
  };

  const { icon, color, label } = config[status];

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${color}`}
    >
      {icon}
      {label}
    </span>
  );
}

// Source type badge
function TypeBadge({ type }: { type: SourceType }) {
  const { icon, label, color } = sourceTypeConfig[type];
  const colorStyles: Record<string, string> = {
    blue: "bg-blue-500/10 text-blue-400",
    green: "bg-green-500/10 text-green-400",
    purple: "bg-purple-500/10 text-purple-400",
    orange: "bg-orange-500/10 text-orange-400",
    cyan: "bg-cyan-500/10 text-cyan-400",
    gray: "bg-gray-500/10 text-gray-400",
    pink: "bg-pink-500/10 text-pink-400",
  };

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded ${colorStyles[color]}`}
    >
      {React.cloneElement(icon as React.ReactElement, { className: "w-3 h-3" })}
      {label}
    </span>
  );
}

// Stats card component
function StatsCard({
  title,
  value,
  icon,
  color,
  subtext,
}: {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  subtext?: string;
}) {
  const colorStyles: Record<string, string> = {
    purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    blue: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    green: "bg-green-500/10 text-green-400 border-green-500/20",
    orange: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  };

  return (
    <div className="bg-[#1a1a2e]/50 backdrop-blur-sm rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-gray-400">{title}</span>
        <div className={`p-2 rounded-lg border ${colorStyles[color]}`}>
          {icon}
        </div>
      </div>
      <span className="text-2xl font-bold text-white">{value}</span>
      {subtext && <p className="text-xs text-gray-500 mt-1">{subtext}</p>}
    </div>
  );
}

// Knowledge source card component
function SourceCard({
  source,
  onEdit,
  onDelete,
  onSync,
  onToggle,
}: {
  source: KnowledgeSource;
  onEdit: (source: KnowledgeSource) => void;
  onDelete: (id: string) => void;
  onSync: (id: string) => void;
  onToggle: (id: string) => void;
}) {
  const [showMenu, setShowMenu] = useState(false);
  const { icon, color } = sourceTypeConfig[source.type];
  const colorStyles: Record<string, string> = {
    blue: "from-blue-500/20 to-blue-600/20 border-blue-500/20",
    green: "from-green-500/20 to-green-600/20 border-green-500/20",
    purple: "from-purple-500/20 to-purple-600/20 border-purple-500/20",
    orange: "from-orange-500/20 to-orange-600/20 border-orange-500/20",
    cyan: "from-cyan-500/20 to-cyan-600/20 border-cyan-500/20",
    gray: "from-gray-500/20 to-gray-600/20 border-gray-500/20",
    pink: "from-pink-500/20 to-pink-600/20 border-pink-500/20",
  };

  return (
    <div className="group bg-[#1a1a2e]/30 hover:bg-[#1a1a2e]/60 rounded-xl border border-white/5 hover:border-white/10 transition-all overflow-hidden">
      {/* Header with gradient */}
      <div
        className={`bg-gradient-to-r ${colorStyles[color]} px-4 py-3 border-b border-white/5`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
              {source.type === "document"
                ? getFileIcon(source.mimeType)
                : React.cloneElement(icon as React.ReactElement, {
                    className: "w-5 h-5",
                  })}
            </div>
            <div>
              <h3 className="font-semibold text-white">{source.name}</h3>
              <div className="flex items-center gap-2">
                <TypeBadge type={source.type} />
                <StatusBadge status={source.status} />
              </div>
            </div>
          </div>
          <div className="relative">
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <MoreVertical className="w-4 h-4 text-gray-400" />
            </button>
            {showMenu && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setShowMenu(false)}
                />
                <div className="absolute right-0 top-full mt-1 w-44 bg-[#1a1a2e] border border-white/10 rounded-xl shadow-xl z-20 py-1 overflow-hidden">
                  <button
                    onClick={() => {
                      onEdit(source);
                      setShowMenu(false);
                    }}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10"
                  >
                    <Edit className="w-4 h-4" />
                    Edit
                  </button>
                  <button
                    onClick={() => {
                      onSync(source.id);
                      setShowMenu(false);
                    }}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Sync Now
                  </button>
                  <button className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10">
                    <Eye className="w-4 h-4" />
                    Preview
                  </button>
                  <button
                    onClick={() => {
                      onToggle(source.id);
                      setShowMenu(false);
                    }}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10"
                  >
                    {source.status === "disabled" ? (
                      <>
                        <Play className="w-4 h-4" />
                        Enable
                      </>
                    ) : (
                      <>
                        <Pause className="w-4 h-4" />
                        Disable
                      </>
                    )}
                  </button>
                  <div className="border-t border-white/10 my-1" />
                  <button
                    onClick={() => {
                      onDelete(source.id);
                      setShowMenu(false);
                    }}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-red-400 hover:bg-red-500/10"
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

      {/* Content */}
      <div className="p-4">
        {source.description && (
          <p className="text-sm text-gray-400 mb-3">{source.description}</p>
        )}

        {/* Source info */}
        <div className="grid grid-cols-2 gap-2 mb-3">
          {source.url && (
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Link className="w-3 h-3" />
              <span className="truncate">{source.url}</span>
            </div>
          )}
          {source.fileSize && (
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <File className="w-3 h-3" />
              <span>{formatFileSize(source.fileSize)}</span>
            </div>
          )}
          {source.metadata.wordCount && (
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <FileText className="w-3 h-3" />
              <span>{source.metadata.wordCount.toLocaleString()} words</span>
            </div>
          )}
          {source.metadata.chunkCount && (
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Database className="w-3 h-3" />
              <span>{source.metadata.chunkCount} chunks</span>
            </div>
          )}
        </div>

        {/* Tags */}
        {source.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-3">
            {source.tags.map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 text-xs bg-white/5 text-gray-400 rounded"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}

        {/* Agents */}
        <div className="flex items-center justify-between pt-3 border-t border-white/5">
          <div className="flex items-center gap-2">
            <Bot className="w-4 h-4 text-gray-500" />
            <span className="text-xs text-gray-500">
              {source.agents.length} agent{source.agents.length !== 1 ? "s" : ""}
            </span>
          </div>
          {source.lastSynced && (
            <span className="text-xs text-gray-500">
              Synced {new Date(source.lastSynced).toLocaleDateString()}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// Add source dialog
function AddSourceDialog({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const [step, setStep] = useState(1);
  const [sourceType, setSourceType] = useState<SourceType | null>(null);
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [content, setContent] = useState("");
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  if (!isOpen) return null;

  const sourceTypes: {
    type: SourceType;
    icon: React.ReactNode;
    title: string;
    description: string;
  }[] = [
    {
      type: "document",
      icon: <FileText className="w-6 h-6" />,
      title: "Upload Document",
      description: "PDF, Word, Excel, or text files",
    },
    {
      type: "webpage",
      icon: <Globe className="w-6 h-6" />,
      title: "Web Page",
      description: "Crawl content from a URL",
    },
    {
      type: "faq",
      icon: <HelpCircle className="w-6 h-6" />,
      title: "FAQ",
      description: "Question and answer pairs",
    },
    {
      type: "text",
      icon: <FileCode className="w-6 h-6" />,
      title: "Plain Text",
      description: "Enter text content directly",
    },
    {
      type: "sitemap",
      icon: <Globe className="w-6 h-6" />,
      title: "Sitemap",
      description: "Crawl multiple pages from sitemap",
    },
    {
      type: "api",
      icon: <Zap className="w-6 h-6" />,
      title: "API Endpoint",
      description: "Connect to REST API",
    },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-[#0f0f1a] border border-white/10 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <div>
            <h2 className="text-xl font-semibold text-white">
              Add Knowledge Source
            </h2>
            <p className="text-sm text-gray-400">
              {step === 1
                ? "Choose the type of content to add"
                : step === 2
                ? "Configure your source"
                : "Select agents"}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {step === 1 && (
            <div className="grid grid-cols-2 gap-3">
              {sourceTypes.map((source) => (
                <button
                  key={source.type}
                  onClick={() => {
                    setSourceType(source.type);
                    setStep(2);
                  }}
                  className="flex items-start gap-4 p-4 rounded-xl border border-white/10 hover:border-purple-500/50 hover:bg-purple-500/5 transition-all text-left"
                >
                  <div className="w-12 h-12 rounded-lg bg-white/5 flex items-center justify-center text-purple-400">
                    {source.icon}
                  </div>
                  <div>
                    <h3 className="font-medium text-white">{source.title}</h3>
                    <p className="text-sm text-gray-500">{source.description}</p>
                  </div>
                </button>
              ))}
            </div>
          )}

          {step === 2 && sourceType && (
            <div className="space-y-4">
              {/* Name field */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., Product Documentation"
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>

              {/* Type-specific fields */}
              {(sourceType === "document") && (
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                    const files = Array.from(e.dataTransfer.files);
                    setUploadedFiles(files);
                  }}
                  className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all ${
                    isDragging
                      ? "border-purple-500 bg-purple-500/10"
                      : "border-white/10"
                  }`}
                >
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.doc,.docx,.txt,.csv,.xlsx,.json"
                    onChange={(e) => {
                      if (e.target.files) {
                        setUploadedFiles(Array.from(e.target.files));
                      }
                    }}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <Upload className="w-10 h-10 text-gray-400 mx-auto mb-3" />
                  <p className="text-white">
                    Drag & drop files or{" "}
                    <span className="text-purple-400">browse</span>
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    PDF, Word, Excel, CSV, JSON, or plain text
                  </p>
                  {uploadedFiles.length > 0 && (
                    <div className="mt-4 space-y-2">
                      {uploadedFiles.map((file, i) => (
                        <div
                          key={i}
                          className="flex items-center gap-2 p-2 bg-white/5 rounded"
                        >
                          {getFileIcon(file.type)}
                          <span className="text-sm text-gray-300">{file.name}</span>
                          <span className="text-xs text-gray-500">
                            ({formatFileSize(file.size)})
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {(sourceType === "webpage" || sourceType === "sitemap" || sourceType === "api") && (
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    URL
                  </label>
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder={
                      sourceType === "sitemap"
                        ? "https://example.com/sitemap.xml"
                        : sourceType === "api"
                        ? "https://api.example.com/data"
                        : "https://example.com/page"
                    }
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
                  />
                </div>
              )}

              {sourceType === "text" && (
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Content
                  </label>
                  <textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    placeholder="Enter or paste your text content here..."
                    rows={8}
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                  />
                </div>
              )}

              {sourceType === "faq" && (
                <div className="space-y-3">
                  <p className="text-sm text-gray-500">
                    Add question and answer pairs. You can also import from a CSV
                    file.
                  </p>
                  <div className="space-y-2">
                    <div className="flex gap-2">
                      <input
                        type="text"
                        placeholder="Question"
                        className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
                      />
                    </div>
                    <textarea
                      placeholder="Answer"
                      rows={2}
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                    />
                    <button className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300">
                      <Plus className="w-4 h-4" />
                      Add another Q&A pair
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {step === 3 && (
            <div className="space-y-4">
              <p className="text-sm text-gray-400">
                Select which agents can use this knowledge source
              </p>
              <div className="space-y-2">
                {mockAgents.map((agent) => (
                  <label
                    key={agent.id}
                    className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                      selectedAgents.includes(agent.id)
                        ? "border-purple-500 bg-purple-500/10"
                        : "border-white/10 hover:border-white/20"
                    }`}
                  >
                    <div
                      onClick={() => {
                        if (selectedAgents.includes(agent.id)) {
                          setSelectedAgents(
                            selectedAgents.filter((id) => id !== agent.id)
                          );
                        } else {
                          setSelectedAgents([...selectedAgents, agent.id]);
                        }
                      }}
                      className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                        selectedAgents.includes(agent.id)
                          ? "bg-purple-500 border-purple-500"
                          : "border-gray-600"
                      }`}
                    >
                      {selectedAgents.includes(agent.id) && (
                        <Check className="w-3 h-3 text-white" />
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <Bot className="w-5 h-5 text-purple-400" />
                      <span className="text-white">{agent.name}</span>
                    </div>
                    <span
                      className={`ml-auto text-xs ${
                        agent.status === "active"
                          ? "text-green-400"
                          : "text-gray-500"
                      }`}
                    >
                      {agent.status}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-white/10 bg-white/5">
          {step > 1 ? (
            <button
              onClick={() => setStep(step - 1)}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Back
            </button>
          ) : (
            <div />
          )}
          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            {step < 3 ? (
              <button
                onClick={() => setStep(step + 1)}
                disabled={step === 1 ? !sourceType : !name}
                className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Continue
                <ChevronRight className="w-4 h-4" />
              </button>
            ) : (
              <button
                onClick={onClose}
                className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
              >
                <Plus className="w-4 h-4" />
                Add Source
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Main component
export default function KnowledgeBasePage() {
  const [sources, setSources] = useState(mockSources);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<string>("all");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [showAddDialog, setShowAddDialog] = useState(false);

  // Stats
  const stats = useMemo(() => {
    const total = sources.length;
    const active = sources.filter((s) => s.status === "active").length;
    const totalChunks = sources.reduce(
      (sum, s) => sum + (s.metadata.chunkCount || 0),
      0
    );
    const totalWords = sources.reduce(
      (sum, s) => sum + (s.metadata.wordCount || 0),
      0
    );
    return { total, active, totalChunks, totalWords };
  }, [sources]);

  // Filtered sources
  const filteredSources = useMemo(() => {
    return sources.filter((source) => {
      if (
        searchQuery &&
        !source.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !source.description?.toLowerCase().includes(searchQuery.toLowerCase())
      ) {
        return false;
      }
      if (filterType !== "all" && source.type !== filterType) {
        return false;
      }
      if (filterStatus !== "all" && source.status !== filterStatus) {
        return false;
      }
      if (filterCategory !== "all" && source.category !== filterCategory) {
        return false;
      }
      return true;
    });
  }, [sources, searchQuery, filterType, filterStatus, filterCategory]);

  const handleEdit = (source: KnowledgeSource) => {
    // Edit logic
    console.log("Edit source:", source);
  };

  const handleDelete = (id: string) => {
    setSources(sources.filter((s) => s.id !== id));
  };

  const handleSync = (id: string) => {
    setSources(
      sources.map((s) =>
        s.id === id
          ? { ...s, status: "processing" as const, lastSynced: new Date().toISOString() }
          : s
      )
    );
    // Simulate sync completion
    setTimeout(() => {
      setSources((prev) =>
        prev.map((s) => (s.id === id ? { ...s, status: "active" as const } : s))
      );
    }, 2000);
  };

  const handleToggle = (id: string) => {
    setSources(
      sources.map((s) =>
        s.id === id
          ? {
              ...s,
              status: s.status === "disabled" ? ("active" as const) : ("disabled" as const),
            }
          : s
      )
    );
  };

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Knowledge Base</h1>
            <p className="text-gray-400">
              Manage the knowledge sources that power your AI agents
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg border border-white/10 transition-colors">
              <RefreshCw className="w-4 h-4" />
              Sync All
            </button>
            <button
              onClick={() => setShowAddDialog(true)}
              className="flex items-center gap-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              Add Source
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-4">
          <StatsCard
            title="Total Sources"
            value={stats.total}
            icon={<Database className="w-4 h-4" />}
            color="purple"
          />
          <StatsCard
            title="Active Sources"
            value={stats.active}
            icon={<CheckCircle className="w-4 h-4" />}
            color="green"
          />
          <StatsCard
            title="Knowledge Chunks"
            value={stats.totalChunks.toLocaleString()}
            icon={<Sparkles className="w-4 h-4" />}
            color="blue"
            subtext="Embedded for AI retrieval"
          />
          <StatsCard
            title="Total Words"
            value={stats.totalWords.toLocaleString()}
            icon={<FileText className="w-4 h-4" />}
            color="orange"
          />
        </div>

        {/* Filters */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 flex-1">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search knowledge sources..."
                className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>

            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Types</option>
              {Object.entries(sourceTypeConfig).map(([type, config]) => (
                <option key={type} value={type}>
                  {config.label}
                </option>
              ))}
            </select>

            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Status</option>
              <option value="active">Active</option>
              <option value="processing">Processing</option>
              <option value="error">Error</option>
              <option value="disabled">Disabled</option>
            </select>

            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Categories</option>
              {mockCategories.map((cat) => (
                <option key={cat.id} value={cat.name}>
                  {cat.name}
                </option>
              ))}
            </select>
          </div>

          {/* View toggle */}
          <div className="flex items-center gap-1 p-1 bg-white/5 rounded-lg">
            <button
              onClick={() => setViewMode("grid")}
              className={`p-2 rounded-lg transition-colors ${
                viewMode === "grid"
                  ? "bg-purple-500/20 text-purple-400"
                  : "text-gray-400 hover:bg-white/10"
              }`}
            >
              <Grid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode("list")}
              className={`p-2 rounded-lg transition-colors ${
                viewMode === "list"
                  ? "bg-purple-500/20 text-purple-400"
                  : "text-gray-400 hover:bg-white/10"
              }`}
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Sources grid */}
        {filteredSources.length > 0 ? (
          <div
            className={
              viewMode === "grid"
                ? "grid grid-cols-3 gap-4"
                : "space-y-3"
            }
          >
            {filteredSources.map((source) => (
              <SourceCard
                key={source.id}
                source={source}
                onEdit={handleEdit}
                onDelete={handleDelete}
                onSync={handleSync}
                onToggle={handleToggle}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-4">
              <Database className="w-8 h-8 text-gray-600" />
            </div>
            <h3 className="text-lg font-medium text-white mb-2">
              No knowledge sources found
            </h3>
            <p className="text-gray-500 mb-4">
              {searchQuery || filterType !== "all" || filterStatus !== "all"
                ? "Try adjusting your filters"
                : "Add your first knowledge source to get started"}
            </p>
            <button
              onClick={() => setShowAddDialog(true)}
              className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              Add Source
            </button>
          </div>
        )}

        {/* Add dialog */}
        <AddSourceDialog
          isOpen={showAddDialog}
          onClose={() => setShowAddDialog(false)}
        />
      </div>
    </DashboardLayout>
  );
}
