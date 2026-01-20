"use client";

import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import {
  Users,
  User,
  UserPlus,
  UserCheck,
  UserX,
  Search,
  Filter,
  Plus,
  MoreVertical,
  Phone,
  Mail,
  MapPin,
  Building,
  Tag,
  Calendar,
  Clock,
  Star,
  StarOff,
  Edit,
  Trash2,
  Download,
  Upload,
  Grid,
  List,
  ChevronDown,
  ChevronRight,
  ChevronLeft,
  Check,
  X,
  ExternalLink,
  MessageSquare,
  PhoneCall,
  PhoneOff,
  PhoneIncoming,
  PhoneOutgoing,
  History,
  FileText,
  Link,
  Globe,
  Linkedin,
  Twitter,
  Facebook,
  Instagram,
  Send,
  Reply,
  Forward,
  Archive,
  Bookmark,
  Flag,
  AlertCircle,
  CheckCircle,
  XCircle,
  Info,
  Sparkles,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Briefcase,
  Award,
  Zap,
  Activity,
  BarChart3,
  PieChart,
  Settings,
  Copy,
  Eye,
  EyeOff,
  Hash,
  AtSign,
  Layers,
  FolderOpen,
  FolderPlus,
  Share2,
  RefreshCw,
  Import,
  ArrowUpRight,
  ArrowDownRight,
  CircleDot,
  Merge,
  Split,
  GitMerge,
  SlidersHorizontal,
} from "lucide-react";

// Types
type ContactStatus = "active" | "inactive" | "lead" | "customer" | "churned" | "prospect";
type ContactSource = "website" | "referral" | "campaign" | "import" | "manual" | "api" | "social";
type ActivityType = "call" | "email" | "sms" | "meeting" | "note" | "task";

interface ContactActivity {
  id: string;
  type: ActivityType;
  title: string;
  description: string;
  timestamp: Date;
  outcome?: string;
  duration?: number;
  direction?: "inbound" | "outbound";
}

interface ContactDeal {
  id: string;
  name: string;
  value: number;
  stage: string;
  probability: number;
  closeDate: Date;
  status: "open" | "won" | "lost";
}

interface Contact {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  company?: string;
  jobTitle?: string;
  avatar?: string;
  status: ContactStatus;
  source: ContactSource;
  tags: string[];
  isFavorite: boolean;
  address?: {
    street?: string;
    city?: string;
    state?: string;
    country?: string;
    zip?: string;
  };
  socialProfiles?: {
    linkedin?: string;
    twitter?: string;
    facebook?: string;
  };
  customFields?: Record<string, string>;
  activities: ContactActivity[];
  deals: ContactDeal[];
  notes: string;
  assignedTo?: string;
  score: number;
  lastContactedAt?: Date;
  createdAt: Date;
  updatedAt: Date;
  doNotCall: boolean;
  doNotEmail: boolean;
  timezone?: string;
  preferredContactMethod: "phone" | "email" | "sms";
  totalCalls: number;
  totalEmails: number;
  lifetimeValue: number;
}

interface ContactList {
  id: string;
  name: string;
  description: string;
  contactCount: number;
  color: string;
  isSmartList: boolean;
  filters?: Record<string, any>;
  createdAt: Date;
}

// Mock data
const mockContacts: Contact[] = [
  {
    id: "contact-1",
    firstName: "Sarah",
    lastName: "Johnson",
    email: "sarah.johnson@techcorp.com",
    phone: "+1 (555) 123-4567",
    company: "TechCorp Inc.",
    jobTitle: "VP of Sales",
    status: "customer",
    source: "referral",
    tags: ["enterprise", "vip", "tech"],
    isFavorite: true,
    address: {
      city: "San Francisco",
      state: "CA",
      country: "United States",
    },
    socialProfiles: {
      linkedin: "sarah-johnson-tech",
      twitter: "@sarahjtech",
    },
    activities: [
      {
        id: "act-1",
        type: "call",
        title: "Product Demo Call",
        description: "Discussed enterprise features and pricing",
        timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
        outcome: "positive",
        duration: 45,
        direction: "outbound",
      },
      {
        id: "act-2",
        type: "email",
        title: "Follow-up Proposal",
        description: "Sent custom pricing proposal",
        timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
        direction: "outbound",
      },
    ],
    deals: [
      {
        id: "deal-1",
        name: "Enterprise License",
        value: 50000,
        stage: "Negotiation",
        probability: 75,
        closeDate: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000),
        status: "open",
      },
    ],
    notes: "Key decision maker. Prefers morning calls.",
    assignedTo: "John Smith",
    score: 92,
    lastContactedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
    createdAt: new Date("2024-01-15"),
    updatedAt: new Date("2024-03-18"),
    doNotCall: false,
    doNotEmail: false,
    timezone: "America/Los_Angeles",
    preferredContactMethod: "email",
    totalCalls: 12,
    totalEmails: 28,
    lifetimeValue: 125000,
  },
  {
    id: "contact-2",
    firstName: "Michael",
    lastName: "Chen",
    email: "m.chen@globalinc.com",
    phone: "+1 (555) 234-5678",
    company: "Global Inc.",
    jobTitle: "CTO",
    status: "lead",
    source: "website",
    tags: ["enterprise", "tech", "hot-lead"],
    isFavorite: true,
    address: {
      city: "New York",
      state: "NY",
      country: "United States",
    },
    socialProfiles: {
      linkedin: "michaelchen-cto",
    },
    activities: [
      {
        id: "act-3",
        type: "email",
        title: "Whitepaper Download",
        description: "Downloaded AI integration whitepaper",
        timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
        direction: "inbound",
      },
    ],
    deals: [
      {
        id: "deal-2",
        name: "API Integration Package",
        value: 35000,
        stage: "Discovery",
        probability: 40,
        closeDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
        status: "open",
      },
    ],
    notes: "Technical buyer. Interested in API capabilities.",
    assignedTo: "Emily Davis",
    score: 78,
    lastContactedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
    createdAt: new Date("2024-02-20"),
    updatedAt: new Date("2024-03-17"),
    doNotCall: false,
    doNotEmail: false,
    timezone: "America/New_York",
    preferredContactMethod: "phone",
    totalCalls: 5,
    totalEmails: 12,
    lifetimeValue: 0,
  },
  {
    id: "contact-3",
    firstName: "Emma",
    lastName: "Williams",
    email: "emma.w@startup.io",
    phone: "+1 (555) 345-6789",
    company: "Startup.io",
    jobTitle: "Founder & CEO",
    status: "prospect",
    source: "campaign",
    tags: ["startup", "sme", "growth"],
    isFavorite: false,
    address: {
      city: "Austin",
      state: "TX",
      country: "United States",
    },
    activities: [],
    deals: [],
    notes: "Early stage startup. Budget conscious.",
    score: 55,
    createdAt: new Date("2024-03-01"),
    updatedAt: new Date("2024-03-15"),
    doNotCall: false,
    doNotEmail: false,
    preferredContactMethod: "email",
    totalCalls: 0,
    totalEmails: 3,
    lifetimeValue: 0,
  },
  {
    id: "contact-4",
    firstName: "James",
    lastName: "Wilson",
    email: "jwilson@retail.com",
    phone: "+1 (555) 456-7890",
    company: "Retail Solutions",
    jobTitle: "Director of Operations",
    status: "customer",
    source: "referral",
    tags: ["retail", "mid-market", "renewal"],
    isFavorite: false,
    address: {
      city: "Chicago",
      state: "IL",
      country: "United States",
    },
    activities: [
      {
        id: "act-4",
        type: "meeting",
        title: "Quarterly Review",
        description: "Discussed usage and upcoming renewal",
        timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        outcome: "positive",
        duration: 60,
      },
    ],
    deals: [
      {
        id: "deal-3",
        name: "Annual Renewal",
        value: 24000,
        stage: "Proposal",
        probability: 90,
        closeDate: new Date(Date.now() + 21 * 24 * 60 * 60 * 1000),
        status: "open",
      },
    ],
    notes: "Happy customer. Potential upsell opportunity.",
    assignedTo: "John Smith",
    score: 88,
    lastContactedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    createdAt: new Date("2023-06-10"),
    updatedAt: new Date("2024-03-12"),
    doNotCall: false,
    doNotEmail: false,
    preferredContactMethod: "phone",
    totalCalls: 24,
    totalEmails: 45,
    lifetimeValue: 48000,
  },
  {
    id: "contact-5",
    firstName: "Lisa",
    lastName: "Martinez",
    email: "lisa.m@healthcare.org",
    phone: "+1 (555) 567-8901",
    company: "Healthcare Solutions",
    jobTitle: "IT Manager",
    status: "inactive",
    source: "import",
    tags: ["healthcare", "compliance", "cold"],
    isFavorite: false,
    address: {
      city: "Miami",
      state: "FL",
      country: "United States",
    },
    activities: [
      {
        id: "act-5",
        type: "email",
        title: "Re-engagement Email",
        description: "Sent product updates email",
        timestamp: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
        direction: "outbound",
      },
    ],
    deals: [],
    notes: "Previously interested but went silent.",
    score: 25,
    lastContactedAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
    createdAt: new Date("2023-09-15"),
    updatedAt: new Date("2024-02-18"),
    doNotCall: true,
    doNotEmail: false,
    preferredContactMethod: "email",
    totalCalls: 3,
    totalEmails: 15,
    lifetimeValue: 0,
  },
  {
    id: "contact-6",
    firstName: "David",
    lastName: "Brown",
    email: "dbrown@finance.co",
    phone: "+1 (555) 678-9012",
    company: "Finance Co.",
    jobTitle: "CFO",
    status: "churned",
    source: "manual",
    tags: ["finance", "enterprise", "lost"],
    isFavorite: false,
    address: {
      city: "Boston",
      state: "MA",
      country: "United States",
    },
    activities: [
      {
        id: "act-6",
        type: "call",
        title: "Cancellation Call",
        description: "Discussed reasons for cancellation",
        timestamp: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000),
        outcome: "negative",
        duration: 20,
        direction: "inbound",
      },
    ],
    deals: [
      {
        id: "deal-4",
        name: "Enterprise Suite",
        value: 75000,
        stage: "Closed Lost",
        probability: 0,
        closeDate: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000),
        status: "lost",
      },
    ],
    notes: "Lost due to pricing. May revisit in Q3.",
    score: 35,
    lastContactedAt: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000),
    createdAt: new Date("2023-03-20"),
    updatedAt: new Date("2024-02-01"),
    doNotCall: false,
    doNotEmail: false,
    preferredContactMethod: "email",
    totalCalls: 18,
    totalEmails: 32,
    lifetimeValue: 36000,
  },
  {
    id: "contact-7",
    firstName: "Amanda",
    lastName: "Taylor",
    email: "ataylor@media.net",
    phone: "+1 (555) 789-0123",
    company: "Media Networks",
    jobTitle: "Marketing Director",
    status: "lead",
    source: "social",
    tags: ["media", "marketing", "warm-lead"],
    isFavorite: true,
    address: {
      city: "Los Angeles",
      state: "CA",
      country: "United States",
    },
    socialProfiles: {
      linkedin: "amanda-taylor-media",
      twitter: "@amandataylor",
      facebook: "amandataylormedia",
    },
    activities: [
      {
        id: "act-7",
        type: "sms",
        title: "Event RSVP",
        description: "Confirmed attendance to webinar",
        timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
        direction: "inbound",
      },
    ],
    deals: [
      {
        id: "deal-5",
        name: "Marketing Suite",
        value: 15000,
        stage: "Qualification",
        probability: 30,
        closeDate: new Date(Date.now() + 45 * 24 * 60 * 60 * 1000),
        status: "open",
      },
    ],
    notes: "Engaged on social. Interested in marketing automation.",
    assignedTo: "Emily Davis",
    score: 65,
    lastContactedAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
    createdAt: new Date("2024-02-28"),
    updatedAt: new Date("2024-03-14"),
    doNotCall: false,
    doNotEmail: false,
    preferredContactMethod: "sms",
    totalCalls: 2,
    totalEmails: 8,
    lifetimeValue: 0,
  },
  {
    id: "contact-8",
    firstName: "Robert",
    lastName: "Garcia",
    email: "rgarcia@manufacturing.com",
    phone: "+1 (555) 890-1234",
    company: "Manufacturing Plus",
    jobTitle: "Operations Manager",
    status: "customer",
    source: "website",
    tags: ["manufacturing", "mid-market", "upsell"],
    isFavorite: false,
    address: {
      city: "Detroit",
      state: "MI",
      country: "United States",
    },
    activities: [
      {
        id: "act-8",
        type: "call",
        title: "Support Call",
        description: "Helped with integration setup",
        timestamp: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000),
        outcome: "positive",
        duration: 35,
        direction: "inbound",
      },
    ],
    deals: [],
    notes: "Looking to expand usage to other departments.",
    assignedTo: "John Smith",
    score: 72,
    lastContactedAt: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000),
    createdAt: new Date("2023-08-05"),
    updatedAt: new Date("2024-03-09"),
    doNotCall: false,
    doNotEmail: false,
    preferredContactMethod: "phone",
    totalCalls: 15,
    totalEmails: 22,
    lifetimeValue: 32000,
  },
];

const mockLists: ContactList[] = [
  {
    id: "list-1",
    name: "All Contacts",
    description: "All contacts in the system",
    contactCount: 1248,
    color: "blue",
    isSmartList: false,
    createdAt: new Date("2024-01-01"),
  },
  {
    id: "list-2",
    name: "Hot Leads",
    description: "High-value leads to prioritize",
    contactCount: 47,
    color: "red",
    isSmartList: true,
    filters: { score: { min: 70 }, status: ["lead", "prospect"] },
    createdAt: new Date("2024-02-01"),
  },
  {
    id: "list-3",
    name: "Enterprise Customers",
    description: "Enterprise tier customers",
    contactCount: 124,
    color: "purple",
    isSmartList: true,
    filters: { tags: ["enterprise"], status: ["customer"] },
    createdAt: new Date("2024-02-15"),
  },
  {
    id: "list-4",
    name: "Renewal Pipeline",
    description: "Customers up for renewal",
    contactCount: 38,
    color: "green",
    isSmartList: true,
    filters: { status: ["customer"], hasActiveDeal: true },
    createdAt: new Date("2024-03-01"),
  },
  {
    id: "list-5",
    name: "Re-engagement",
    description: "Inactive contacts to re-engage",
    contactCount: 215,
    color: "yellow",
    isSmartList: true,
    filters: { status: ["inactive"], lastContactedDaysAgo: { min: 30 } },
    createdAt: new Date("2024-03-05"),
  },
];

// Helper functions
const getStatusColor = (status: ContactStatus) => {
  switch (status) {
    case "active":
      return "text-green-400 bg-green-500/20";
    case "inactive":
      return "text-gray-400 bg-gray-500/20";
    case "lead":
      return "text-blue-400 bg-blue-500/20";
    case "customer":
      return "text-purple-400 bg-purple-500/20";
    case "churned":
      return "text-red-400 bg-red-500/20";
    case "prospect":
      return "text-yellow-400 bg-yellow-500/20";
  }
};

const getSourceIcon = (source: ContactSource) => {
  switch (source) {
    case "website":
      return <Globe className="w-3.5 h-3.5" />;
    case "referral":
      return <Users className="w-3.5 h-3.5" />;
    case "campaign":
      return <Target className="w-3.5 h-3.5" />;
    case "import":
      return <Import className="w-3.5 h-3.5" />;
    case "manual":
      return <Edit className="w-3.5 h-3.5" />;
    case "api":
      return <Zap className="w-3.5 h-3.5" />;
    case "social":
      return <Share2 className="w-3.5 h-3.5" />;
  }
};

const getActivityIcon = (type: ActivityType) => {
  switch (type) {
    case "call":
      return <Phone className="w-4 h-4" />;
    case "email":
      return <Mail className="w-4 h-4" />;
    case "sms":
      return <MessageSquare className="w-4 h-4" />;
    case "meeting":
      return <Calendar className="w-4 h-4" />;
    case "note":
      return <FileText className="w-4 h-4" />;
    case "task":
      return <CheckCircle className="w-4 h-4" />;
  }
};

const getScoreColor = (score: number) => {
  if (score >= 80) return "text-green-400";
  if (score >= 60) return "text-yellow-400";
  if (score >= 40) return "text-orange-400";
  return "text-red-400";
};

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(amount);
};

const formatDate = (date: Date) => {
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
};

const formatRelativeTime = (date: Date) => {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
  if (days < 365) return `${Math.floor(days / 30)} months ago`;
  return `${Math.floor(days / 365)} years ago`;
};

const getInitials = (firstName: string, lastName: string) => {
  return `${firstName.charAt(0)}${lastName.charAt(0)}`.toUpperCase();
};

// Contact Card Component
const ContactCard: React.FC<{
  contact: Contact;
  isSelected: boolean;
  onSelect: (selected: boolean) => void;
  onClick: () => void;
  onCall: () => void;
  onEmail: () => void;
  onToggleFavorite: () => void;
}> = ({ contact, isSelected, onSelect, onClick, onCall, onEmail, onToggleFavorite }) => {
  return (
    <div
      className={`bg-[#1a1a2e]/80 backdrop-blur-xl border rounded-xl p-4 hover:border-purple-500/30 transition-all duration-300 cursor-pointer ${
        isSelected ? "border-purple-500" : "border-white/10"
      }`}
      onClick={onClick}
    >
      <div className="flex items-start gap-4">
        <div className="flex items-center gap-3">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={(e) => {
              e.stopPropagation();
              onSelect(e.target.checked);
            }}
            className="w-4 h-4 rounded border-white/20 bg-black/30 text-purple-500 focus:ring-purple-500"
          />

          <div className="relative">
            {contact.avatar ? (
              <img
                src={contact.avatar}
                alt={`${contact.firstName} ${contact.lastName}`}
                className="w-12 h-12 rounded-full object-cover"
              />
            ) : (
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white font-semibold">
                {getInitials(contact.firstName, contact.lastName)}
              </div>
            )}
            {contact.isFavorite && (
              <div className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-yellow-500 flex items-center justify-center">
                <Star className="w-3 h-3 text-white fill-current" />
              </div>
            )}
          </div>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="font-semibold text-white truncate">
                {contact.firstName} {contact.lastName}
              </h3>
              {contact.jobTitle && contact.company && (
                <p className="text-sm text-gray-400 truncate">
                  {contact.jobTitle} at {contact.company}
                </p>
              )}
            </div>

            <div className="flex items-center gap-1">
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(contact.status)}`}>
                {contact.status.charAt(0).toUpperCase() + contact.status.slice(1)}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4 mt-2 text-sm">
            <span className="text-gray-400 flex items-center gap-1 truncate">
              <Mail className="w-3.5 h-3.5 flex-shrink-0" />
              <span className="truncate">{contact.email}</span>
            </span>
            <span className="text-gray-400 flex items-center gap-1">
              <Phone className="w-3.5 h-3.5 flex-shrink-0" />
              {contact.phone}
            </span>
          </div>

          <div className="flex items-center gap-2 mt-3 flex-wrap">
            {contact.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 rounded-full text-xs bg-white/5 text-gray-400 border border-white/10"
              >
                {tag}
              </span>
            ))}
            {contact.tags.length > 3 && (
              <span className="text-xs text-gray-500">+{contact.tags.length - 3} more</span>
            )}
          </div>

          <div className="flex items-center justify-between mt-3 pt-3 border-t border-white/5">
            <div className="flex items-center gap-4 text-sm">
              <span className="text-gray-400 flex items-center gap-1">
                <Activity className="w-3.5 h-3.5 text-purple-400" />
                Score: <span className={`font-medium ${getScoreColor(contact.score)}`}>{contact.score}</span>
              </span>
              {contact.lastContactedAt && (
                <span className="text-gray-500">
                  Last: {formatRelativeTime(contact.lastContactedAt)}
                </span>
              )}
            </div>

            <div className="flex items-center gap-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleFavorite();
                }}
                className={`p-1.5 rounded-lg transition-colors ${
                  contact.isFavorite
                    ? "text-yellow-400 hover:bg-yellow-500/10"
                    : "text-gray-500 hover:bg-white/10"
                }`}
              >
                <Star className={`w-4 h-4 ${contact.isFavorite ? "fill-current" : ""}`} />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onCall();
                }}
                className={`p-1.5 rounded-lg transition-colors ${
                  contact.doNotCall
                    ? "text-gray-600 cursor-not-allowed"
                    : "text-green-400 hover:bg-green-500/10"
                }`}
                disabled={contact.doNotCall}
              >
                <Phone className="w-4 h-4" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onEmail();
                }}
                className={`p-1.5 rounded-lg transition-colors ${
                  contact.doNotEmail
                    ? "text-gray-600 cursor-not-allowed"
                    : "text-blue-400 hover:bg-blue-500/10"
                }`}
                disabled={contact.doNotEmail}
              >
                <Mail className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Contact Table Row Component
const ContactRow: React.FC<{
  contact: Contact;
  isSelected: boolean;
  onSelect: (selected: boolean) => void;
  onClick: () => void;
}> = ({ contact, isSelected, onSelect, onClick }) => {
  return (
    <tr
      className={`border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors ${
        isSelected ? "bg-purple-500/10" : ""
      }`}
      onClick={onClick}
    >
      <td className="px-4 py-3">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={(e) => {
            e.stopPropagation();
            onSelect(e.target.checked);
          }}
          className="w-4 h-4 rounded border-white/20 bg-black/30 text-purple-500 focus:ring-purple-500"
        />
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="relative">
            {contact.avatar ? (
              <img
                src={contact.avatar}
                alt={`${contact.firstName} ${contact.lastName}`}
                className="w-8 h-8 rounded-full object-cover"
              />
            ) : (
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white text-xs font-semibold">
                {getInitials(contact.firstName, contact.lastName)}
              </div>
            )}
            {contact.isFavorite && (
              <div className="absolute -top-0.5 -right-0.5 w-3 h-3 rounded-full bg-yellow-500" />
            )}
          </div>
          <div>
            <p className="text-white font-medium">
              {contact.firstName} {contact.lastName}
            </p>
            {contact.company && (
              <p className="text-xs text-gray-500">{contact.company}</p>
            )}
          </div>
        </div>
      </td>
      <td className="px-4 py-3">
        <span className="text-sm text-gray-400">{contact.email}</span>
      </td>
      <td className="px-4 py-3">
        <span className="text-sm text-gray-400">{contact.phone}</span>
      </td>
      <td className="px-4 py-3">
        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(contact.status)}`}>
          {contact.status.charAt(0).toUpperCase() + contact.status.slice(1)}
        </span>
      </td>
      <td className="px-4 py-3">
        <span className={`font-medium ${getScoreColor(contact.score)}`}>{contact.score}</span>
      </td>
      <td className="px-4 py-3">
        <span className="text-sm text-gray-400">
          {contact.lastContactedAt ? formatRelativeTime(contact.lastContactedAt) : "Never"}
        </span>
      </td>
      <td className="px-4 py-3">
        <span className="text-sm text-white">{formatCurrency(contact.lifetimeValue)}</span>
      </td>
    </tr>
  );
};

// Contact Detail Panel
const ContactDetailPanel: React.FC<{
  contact: Contact;
  onClose: () => void;
  onEdit: () => void;
}> = ({ contact, onClose, onEdit }) => {
  const [activeTab, setActiveTab] = useState<"overview" | "activities" | "deals" | "notes">("overview");

  return (
    <div className="w-full lg:w-[450px] bg-[#1a1a2e] border-l border-white/10 h-full overflow-y-auto">
      {/* Header */}
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center justify-between mb-4">
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
          >
            <X className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <button
              onClick={onEdit}
              className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
            >
              <Edit className="w-4 h-4" />
            </button>
            <button className="p-2 rounded-lg hover:bg-white/10 text-gray-400">
              <MoreVertical className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {contact.avatar ? (
            <img
              src={contact.avatar}
              alt={`${contact.firstName} ${contact.lastName}`}
              className="w-16 h-16 rounded-full object-cover"
            />
          ) : (
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white text-xl font-semibold">
              {getInitials(contact.firstName, contact.lastName)}
            </div>
          )}
          <div>
            <h2 className="text-xl font-bold text-white">
              {contact.firstName} {contact.lastName}
            </h2>
            {contact.jobTitle && contact.company && (
              <p className="text-gray-400">
                {contact.jobTitle} at {contact.company}
              </p>
            )}
            <div className="flex items-center gap-2 mt-2">
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(contact.status)}`}>
                {contact.status.charAt(0).toUpperCase() + contact.status.slice(1)}
              </span>
              <span className={`text-sm font-medium ${getScoreColor(contact.score)}`}>
                Score: {contact.score}
              </span>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="flex gap-2 mt-4">
          <button
            className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg transition-colors ${
              contact.doNotCall
                ? "bg-gray-500/20 text-gray-500 cursor-not-allowed"
                : "bg-green-500/20 text-green-400 hover:bg-green-500/30"
            }`}
            disabled={contact.doNotCall}
          >
            <Phone className="w-4 h-4" />
            Call
          </button>
          <button
            className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg transition-colors ${
              contact.doNotEmail
                ? "bg-gray-500/20 text-gray-500 cursor-not-allowed"
                : "bg-blue-500/20 text-blue-400 hover:bg-blue-500/30"
            }`}
            disabled={contact.doNotEmail}
          >
            <Mail className="w-4 h-4" />
            Email
          </button>
          <button className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 transition-colors">
            <MessageSquare className="w-4 h-4" />
            SMS
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/10">
        {(["overview", "activities", "deals", "notes"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${
              activeTab === tab
                ? "text-purple-400 border-b-2 border-purple-400"
                : "text-gray-400 hover:text-white"
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {activeTab === "overview" && (
          <div className="space-y-6">
            {/* Contact Info */}
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-3">CONTACT INFO</h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <Mail className="w-4 h-4 text-gray-500" />
                  <span className="text-white">{contact.email}</span>
                  {contact.doNotEmail && (
                    <span className="px-1.5 py-0.5 rounded text-xs bg-red-500/20 text-red-400">
                      Do Not Email
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <Phone className="w-4 h-4 text-gray-500" />
                  <span className="text-white">{contact.phone}</span>
                  {contact.doNotCall && (
                    <span className="px-1.5 py-0.5 rounded text-xs bg-red-500/20 text-red-400">
                      Do Not Call
                    </span>
                  )}
                </div>
                {contact.address && (
                  <div className="flex items-center gap-3">
                    <MapPin className="w-4 h-4 text-gray-500" />
                    <span className="text-white">
                      {[contact.address.city, contact.address.state, contact.address.country]
                        .filter(Boolean)
                        .join(", ")}
                    </span>
                  </div>
                )}
                {contact.timezone && (
                  <div className="flex items-center gap-3">
                    <Globe className="w-4 h-4 text-gray-500" />
                    <span className="text-white">{contact.timezone}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Stats */}
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-3">ENGAGEMENT</h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-black/30 rounded-lg p-3">
                  <p className="text-sm text-gray-400">Total Calls</p>
                  <p className="text-xl font-bold text-white">{contact.totalCalls}</p>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <p className="text-sm text-gray-400">Total Emails</p>
                  <p className="text-xl font-bold text-white">{contact.totalEmails}</p>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <p className="text-sm text-gray-400">Lifetime Value</p>
                  <p className="text-xl font-bold text-white">{formatCurrency(contact.lifetimeValue)}</p>
                </div>
                <div className="bg-black/30 rounded-lg p-3">
                  <p className="text-sm text-gray-400">Preferred</p>
                  <p className="text-xl font-bold text-white capitalize">{contact.preferredContactMethod}</p>
                </div>
              </div>
            </div>

            {/* Social Profiles */}
            {contact.socialProfiles && Object.keys(contact.socialProfiles).length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-gray-400 mb-3">SOCIAL PROFILES</h3>
                <div className="flex gap-2">
                  {contact.socialProfiles.linkedin && (
                    <a
                      href={`https://linkedin.com/in/${contact.socialProfiles.linkedin}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-2 rounded-lg bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 transition-colors"
                    >
                      <Linkedin className="w-5 h-5" />
                    </a>
                  )}
                  {contact.socialProfiles.twitter && (
                    <a
                      href={`https://twitter.com/${contact.socialProfiles.twitter}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-2 rounded-lg bg-sky-500/20 text-sky-400 hover:bg-sky-500/30 transition-colors"
                    >
                      <Twitter className="w-5 h-5" />
                    </a>
                  )}
                  {contact.socialProfiles.facebook && (
                    <a
                      href={`https://facebook.com/${contact.socialProfiles.facebook}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="p-2 rounded-lg bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-colors"
                    >
                      <Facebook className="w-5 h-5" />
                    </a>
                  )}
                </div>
              </div>
            )}

            {/* Tags */}
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-3">TAGS</h3>
              <div className="flex flex-wrap gap-2">
                {contact.tags.map((tag) => (
                  <span
                    key={tag}
                    className="px-2 py-1 rounded-full text-sm bg-white/5 text-gray-300 border border-white/10"
                  >
                    {tag}
                  </span>
                ))}
                <button className="px-2 py-1 rounded-full text-sm bg-purple-500/20 text-purple-400 border border-purple-500/30 hover:bg-purple-500/30 transition-colors">
                  + Add Tag
                </button>
              </div>
            </div>

            {/* Metadata */}
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-3">DETAILS</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Source</span>
                  <span className="text-white flex items-center gap-1">
                    {getSourceIcon(contact.source)}
                    {contact.source.charAt(0).toUpperCase() + contact.source.slice(1)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Created</span>
                  <span className="text-white">{formatDate(contact.createdAt)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Updated</span>
                  <span className="text-white">{formatDate(contact.updatedAt)}</span>
                </div>
                {contact.assignedTo && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Assigned To</span>
                    <span className="text-white">{contact.assignedTo}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === "activities" && (
          <div className="space-y-4">
            {contact.activities.length > 0 ? (
              contact.activities.map((activity) => (
                <div
                  key={activity.id}
                  className="p-3 bg-black/30 rounded-lg border border-white/5"
                >
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg ${
                      activity.type === "call" ? "bg-green-500/20 text-green-400" :
                      activity.type === "email" ? "bg-blue-500/20 text-blue-400" :
                      activity.type === "sms" ? "bg-purple-500/20 text-purple-400" :
                      activity.type === "meeting" ? "bg-yellow-500/20 text-yellow-400" :
                      "bg-gray-500/20 text-gray-400"
                    }`}>
                      {getActivityIcon(activity.type)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <p className="text-white font-medium">{activity.title}</p>
                        {activity.direction && (
                          <span className={`text-xs ${
                            activity.direction === "inbound" ? "text-blue-400" : "text-green-400"
                          }`}>
                            {activity.direction === "inbound" ? "↓ Inbound" : "↑ Outbound"}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-400 mt-1">{activity.description}</p>
                      <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                        <span>{formatRelativeTime(activity.timestamp)}</span>
                        {activity.duration && <span>{activity.duration} min</span>}
                        {activity.outcome && (
                          <span className={
                            activity.outcome === "positive" ? "text-green-400" :
                            activity.outcome === "negative" ? "text-red-400" :
                            "text-gray-400"
                          }>
                            {activity.outcome.charAt(0).toUpperCase() + activity.outcome.slice(1)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8">
                <Activity className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-400">No activities yet</p>
              </div>
            )}

            <button className="w-full py-2 rounded-lg bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 transition-colors flex items-center justify-center gap-2">
              <Plus className="w-4 h-4" />
              Log Activity
            </button>
          </div>
        )}

        {activeTab === "deals" && (
          <div className="space-y-4">
            {contact.deals.length > 0 ? (
              contact.deals.map((deal) => (
                <div
                  key={deal.id}
                  className="p-4 bg-black/30 rounded-lg border border-white/5"
                >
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="text-white font-medium">{deal.name}</h4>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      deal.status === "won" ? "bg-green-500/20 text-green-400" :
                      deal.status === "lost" ? "bg-red-500/20 text-red-400" :
                      "bg-blue-500/20 text-blue-400"
                    }`}>
                      {deal.status.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-white font-semibold">{formatCurrency(deal.value)}</span>
                    <span className="text-gray-400">Stage: {deal.stage}</span>
                    <span className="text-gray-400">{deal.probability}% prob</span>
                  </div>
                  <div className="mt-2">
                    <div className="w-full bg-white/10 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full ${
                          deal.status === "won" ? "bg-green-500" :
                          deal.status === "lost" ? "bg-red-500" :
                          "bg-purple-500"
                        }`}
                        style={{ width: `${deal.probability}%` }}
                      />
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Close: {formatDate(deal.closeDate)}
                  </p>
                </div>
              ))
            ) : (
              <div className="text-center py-8">
                <Briefcase className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-400">No deals yet</p>
              </div>
            )}

            <button className="w-full py-2 rounded-lg bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 transition-colors flex items-center justify-center gap-2">
              <Plus className="w-4 h-4" />
              Add Deal
            </button>
          </div>
        )}

        {activeTab === "notes" && (
          <div className="space-y-4">
            <textarea
              className="w-full px-4 py-3 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
              rows={6}
              placeholder="Add notes about this contact..."
              defaultValue={contact.notes}
            />
            <button className="w-full py-2 rounded-lg bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 transition-colors">
              Save Notes
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// Create Contact Dialog
const CreateContactDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onSave: (contact: Partial<Contact>) => void;
}> = ({ isOpen, onClose, onSave }) => {
  const [contactData, setContactData] = useState<Partial<Contact>>({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
    company: "",
    jobTitle: "",
    status: "lead",
    source: "manual",
    tags: [],
    preferredContactMethod: "email",
    doNotCall: false,
    doNotEmail: false,
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-2xl bg-[#1a1a2e] border border-white/10 rounded-2xl overflow-hidden">
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">Add New Contact</h2>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="p-6 max-h-[60vh] overflow-y-auto space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">First Name *</label>
              <input
                type="text"
                value={contactData.firstName}
                onChange={(e) => setContactData({ ...contactData, firstName: e.target.value })}
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                placeholder="John"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Last Name *</label>
              <input
                type="text"
                value={contactData.lastName}
                onChange={(e) => setContactData({ ...contactData, lastName: e.target.value })}
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                placeholder="Doe"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Email *</label>
              <input
                type="email"
                value={contactData.email}
                onChange={(e) => setContactData({ ...contactData, email: e.target.value })}
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                placeholder="john@example.com"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Phone</label>
              <input
                type="tel"
                value={contactData.phone}
                onChange={(e) => setContactData({ ...contactData, phone: e.target.value })}
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                placeholder="+1 (555) 123-4567"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Company</label>
              <input
                type="text"
                value={contactData.company}
                onChange={(e) => setContactData({ ...contactData, company: e.target.value })}
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                placeholder="Acme Inc."
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Job Title</label>
              <input
                type="text"
                value={contactData.jobTitle}
                onChange={(e) => setContactData({ ...contactData, jobTitle: e.target.value })}
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                placeholder="CEO"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Status</label>
              <select
                value={contactData.status}
                onChange={(e) => setContactData({ ...contactData, status: e.target.value as ContactStatus })}
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
              >
                <option value="lead">Lead</option>
                <option value="prospect">Prospect</option>
                <option value="customer">Customer</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Source</label>
              <select
                value={contactData.source}
                onChange={(e) => setContactData({ ...contactData, source: e.target.value as ContactSource })}
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
              >
                <option value="manual">Manual Entry</option>
                <option value="website">Website</option>
                <option value="referral">Referral</option>
                <option value="campaign">Campaign</option>
                <option value="import">Import</option>
                <option value="api">API</option>
                <option value="social">Social Media</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Tags</label>
            <input
              type="text"
              value={contactData.tags?.join(", ") || ""}
              onChange={(e) => setContactData({
                ...contactData,
                tags: e.target.value.split(",").map((t) => t.trim()).filter(Boolean),
              })}
              className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              placeholder="Enter tags separated by commas"
            />
          </div>

          <div className="flex items-center gap-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={contactData.doNotCall}
                onChange={(e) => setContactData({ ...contactData, doNotCall: e.target.checked })}
                className="w-4 h-4 rounded border-white/20 bg-black/30 text-purple-500 focus:ring-purple-500"
              />
              <span className="text-sm text-white">Do Not Call</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={contactData.doNotEmail}
                onChange={(e) => setContactData({ ...contactData, doNotEmail: e.target.checked })}
                className="w-4 h-4 rounded border-white/20 bg-black/30 text-purple-500 focus:ring-purple-500"
              />
              <span className="text-sm text-white">Do Not Email</span>
            </label>
          </div>
        </div>

        <div className="p-6 border-t border-white/10 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-gray-400 hover:bg-white/10 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onSave(contactData)}
            className="px-6 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium hover:opacity-90 transition-opacity"
          >
            Add Contact
          </button>
        </div>
      </div>
    </div>
  );
};

// Import Contacts Dialog
const ImportContactsDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
}> = ({ isOpen, onClose }) => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  if (!isOpen) return null;

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-xl bg-[#1a1a2e] border border-white/10 rounded-2xl overflow-hidden">
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">Import Contacts</h2>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="p-6">
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
              dragActive ? "border-purple-500 bg-purple-500/10" : "border-white/20"
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {file ? (
              <div>
                <FileText className="w-12 h-12 text-purple-400 mx-auto mb-3" />
                <p className="text-white font-medium">{file.name}</p>
                <p className="text-sm text-gray-400 mt-1">
                  {(file.size / 1024).toFixed(2)} KB
                </p>
                <button
                  onClick={() => setFile(null)}
                  className="mt-3 text-sm text-red-400 hover:underline"
                >
                  Remove
                </button>
              </div>
            ) : (
              <div>
                <Upload className="w-12 h-12 text-gray-500 mx-auto mb-3" />
                <p className="text-white mb-2">Drag and drop your file here</p>
                <p className="text-sm text-gray-400 mb-4">Supports CSV, XLSX, and JSON files</p>
                <label className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors cursor-pointer">
                  <Upload className="w-4 h-4" />
                  Browse Files
                  <input
                    type="file"
                    className="hidden"
                    accept=".csv,.xlsx,.json"
                    onChange={(e) => e.target.files && setFile(e.target.files[0])}
                  />
                </label>
              </div>
            )}
          </div>

          <div className="mt-6 p-4 bg-black/30 rounded-lg">
            <h4 className="text-sm font-semibold text-white mb-2">File Format Requirements</h4>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• First row should contain column headers</li>
              <li>• Required fields: first_name, last_name, email</li>
              <li>• Optional: phone, company, job_title, tags</li>
              <li>• Maximum 10,000 contacts per import</li>
            </ul>
          </div>
        </div>

        <div className="p-6 border-t border-white/10 flex justify-between">
          <button
            className="flex items-center gap-2 text-sm text-gray-400 hover:text-white"
          >
            <Download className="w-4 h-4" />
            Download Template
          </button>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-gray-400 hover:bg-white/10 transition-colors"
            >
              Cancel
            </button>
            <button
              disabled={!file}
              className="px-6 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Import
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Component
export default function ContactsPage() {
  const [contacts, setContacts] = useState<Contact[]>(mockContacts);
  const [lists, setLists] = useState<ContactList[]>(mockLists);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<ContactStatus | "all">("all");
  const [selectedList, setSelectedList] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [selectedContacts, setSelectedContacts] = useState<Set<string>>(new Set());
  const [selectedContact, setSelectedContact] = useState<Contact | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);

  // Filter contacts
  const filteredContacts = useMemo(() => {
    return contacts.filter((contact) => {
      const matchesSearch =
        contact.firstName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        contact.lastName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        contact.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
        contact.company?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        contact.tags.some((t) => t.toLowerCase().includes(searchQuery.toLowerCase()));

      const matchesStatus = statusFilter === "all" || contact.status === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [contacts, searchQuery, statusFilter]);

  // Stats
  const stats = useMemo(() => {
    const total = contacts.length;
    const leads = contacts.filter((c) => c.status === "lead").length;
    const customers = contacts.filter((c) => c.status === "customer").length;
    const totalValue = contacts.reduce((sum, c) => sum + c.lifetimeValue, 0);

    return { total, leads, customers, totalValue };
  }, [contacts]);

  // Handlers
  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedContacts(new Set(filteredContacts.map((c) => c.id)));
    } else {
      setSelectedContacts(new Set());
    }
  };

  const handleSelectContact = (contactId: string, selected: boolean) => {
    setSelectedContacts((prev) => {
      const next = new Set(prev);
      if (selected) {
        next.add(contactId);
      } else {
        next.delete(contactId);
      }
      return next;
    });
  };

  const handleToggleFavorite = (contactId: string) => {
    setContacts((prev) =>
      prev.map((c) =>
        c.id === contactId ? { ...c, isFavorite: !c.isFavorite } : c
      )
    );
  };

  const handleSaveContact = (contactData: Partial<Contact>) => {
    const newContact: Contact = {
      ...contactData,
      id: `contact-${Date.now()}`,
      activities: [],
      deals: [],
      notes: "",
      score: 50,
      totalCalls: 0,
      totalEmails: 0,
      lifetimeValue: 0,
      isFavorite: false,
      createdAt: new Date(),
      updatedAt: new Date(),
    } as Contact;
    setContacts((prev) => [newContact, ...prev]);
    setShowCreateDialog(false);
  };

  return (
    <DashboardLayout>
      <div className="flex h-[calc(100vh-64px)]">
        {/* Sidebar - Lists */}
        {showSidebar && (
          <div className="w-64 bg-[#1a1a2e]/50 border-r border-white/10 flex-shrink-0 overflow-y-auto">
            <div className="p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Lists</h3>
                <button className="p-1 rounded hover:bg-white/10 text-gray-400">
                  <Plus className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-1">
                {lists.map((list) => (
                  <button
                    key={list.id}
                    onClick={() => setSelectedList(selectedList === list.id ? null : list.id)}
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-left transition-colors ${
                      selectedList === list.id
                        ? "bg-purple-500/20 text-purple-400"
                        : "text-gray-300 hover:bg-white/5"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {list.isSmartList ? (
                        <Sparkles className="w-4 h-4 text-yellow-400" />
                      ) : (
                        <FolderOpen className="w-4 h-4 text-gray-500" />
                      )}
                      <span className="text-sm truncate">{list.name}</span>
                    </div>
                    <span className="text-xs text-gray-500">{list.contactCount}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <div className="p-6 border-b border-white/10">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setShowSidebar(!showSidebar)}
                  className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
                >
                  <SlidersHorizontal className="w-5 h-5" />
                </button>
                <div>
                  <h1 className="text-2xl font-bold text-white">Contacts</h1>
                  <p className="text-gray-400 mt-1">{filteredContacts.length} contacts</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowImportDialog(true)}
                  className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors"
                >
                  <Upload className="w-4 h-4" />
                  Import
                </button>
                <button className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors">
                  <Download className="w-4 h-4" />
                  Export
                </button>
                <button
                  onClick={() => setShowCreateDialog(true)}
                  className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90 transition-opacity"
                >
                  <UserPlus className="w-4 h-4" />
                  Add Contact
                </button>
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400">
                    <Users className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white">{stats.total}</p>
                    <p className="text-sm text-gray-400">Total Contacts</p>
                  </div>
                </div>
              </div>

              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-yellow-500/20 text-yellow-400">
                    <Target className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white">{stats.leads}</p>
                    <p className="text-sm text-gray-400">Active Leads</p>
                  </div>
                </div>
              </div>

              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
                    <UserCheck className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white">{stats.customers}</p>
                    <p className="text-sm text-gray-400">Customers</p>
                  </div>
                </div>
              </div>

              <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-green-500/20 text-green-400">
                    <DollarSign className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white">{formatCurrency(stats.totalValue)}</p>
                    <p className="text-sm text-gray-400">Total Value</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Filters */}
            <div className="flex flex-col md:flex-row gap-4 mt-6">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search contacts by name, email, company..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>

              <div className="flex gap-2">
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value as ContactStatus | "all")}
                  className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Statuses</option>
                  <option value="lead">Leads</option>
                  <option value="prospect">Prospects</option>
                  <option value="customer">Customers</option>
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                  <option value="churned">Churned</option>
                </select>

                <div className="flex border border-white/10 rounded-lg overflow-hidden">
                  <button
                    onClick={() => setViewMode("grid")}
                    className={`p-2 ${viewMode === "grid" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:bg-white/10"}`}
                  >
                    <Grid className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode("list")}
                    className={`p-2 ${viewMode === "list" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:bg-white/10"}`}
                  >
                    <List className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Bulk Actions */}
            {selectedContacts.size > 0 && (
              <div className="flex items-center gap-4 mt-4 p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                <span className="text-sm text-purple-400">
                  {selectedContacts.size} selected
                </span>
                <div className="flex gap-2">
                  <button className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm transition-colors">
                    Add to List
                  </button>
                  <button className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm transition-colors">
                    Update Tags
                  </button>
                  <button className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm transition-colors">
                    Export
                  </button>
                  <button className="px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-400 text-sm transition-colors">
                    Delete
                  </button>
                </div>
                <button
                  onClick={() => setSelectedContacts(new Set())}
                  className="ml-auto text-sm text-gray-400 hover:text-white"
                >
                  Clear selection
                </button>
              </div>
            )}
          </div>

          {/* Content Area */}
          <div className="flex-1 flex overflow-hidden">
            {/* Contact List/Grid */}
            <div className="flex-1 overflow-y-auto p-6">
              {viewMode === "grid" ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {filteredContacts.map((contact) => (
                    <ContactCard
                      key={contact.id}
                      contact={contact}
                      isSelected={selectedContacts.has(contact.id)}
                      onSelect={(selected) => handleSelectContact(contact.id, selected)}
                      onClick={() => setSelectedContact(contact)}
                      onCall={() => console.log("Call", contact.phone)}
                      onEmail={() => console.log("Email", contact.email)}
                      onToggleFavorite={() => handleToggleFavorite(contact.id)}
                    />
                  ))}
                </div>
              ) : (
                <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-white/10">
                        <th className="px-4 py-3 text-left">
                          <input
                            type="checkbox"
                            checked={selectedContacts.size === filteredContacts.length && filteredContacts.length > 0}
                            onChange={(e) => handleSelectAll(e.target.checked)}
                            className="w-4 h-4 rounded border-white/20 bg-black/30 text-purple-500 focus:ring-purple-500"
                          />
                        </th>
                        <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">Name</th>
                        <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">Email</th>
                        <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">Phone</th>
                        <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">Status</th>
                        <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">Score</th>
                        <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">Last Contact</th>
                        <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredContacts.map((contact) => (
                        <ContactRow
                          key={contact.id}
                          contact={contact}
                          isSelected={selectedContacts.has(contact.id)}
                          onSelect={(selected) => handleSelectContact(contact.id, selected)}
                          onClick={() => setSelectedContact(contact)}
                        />
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {filteredContacts.length === 0 && (
                <div className="text-center py-16">
                  <Users className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">No contacts found</h3>
                  <p className="text-gray-400 mb-6">
                    {searchQuery || statusFilter !== "all"
                      ? "No contacts match your current filters"
                      : "Add your first contact to get started"}
                  </p>
                  <button
                    onClick={() => setShowCreateDialog(true)}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors"
                  >
                    <UserPlus className="w-4 h-4" />
                    Add Contact
                  </button>
                </div>
              )}
            </div>

            {/* Contact Detail Panel */}
            {selectedContact && (
              <ContactDetailPanel
                contact={selectedContact}
                onClose={() => setSelectedContact(null)}
                onEdit={() => console.log("Edit contact")}
              />
            )}
          </div>
        </div>

        {/* Create Contact Dialog */}
        <CreateContactDialog
          isOpen={showCreateDialog}
          onClose={() => setShowCreateDialog(false)}
          onSave={handleSaveContact}
        />

        {/* Import Contacts Dialog */}
        <ImportContactsDialog
          isOpen={showImportDialog}
          onClose={() => setShowImportDialog(false)}
        />
      </div>
    </DashboardLayout>
  );
}
