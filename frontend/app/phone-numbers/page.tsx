"use client";

import React, { useState, useMemo } from "react";
import { DashboardLayout } from "@/components/dashboard-layout";
import {
  Phone,
  Plus,
  Search,
  Filter,
  MoreVertical,
  Settings,
  Trash2,
  Copy,
  ExternalLink,
  Globe,
  MapPin,
  Bot,
  PhoneIncoming,
  PhoneOutgoing,
  PhoneMissed,
  Clock,
  DollarSign,
  TrendingUp,
  TrendingDown,
  CheckCircle,
  XCircle,
  AlertCircle,
  RefreshCw,
  Download,
  Upload,
  Zap,
  Shield,
  Headphones,
  MessageSquare,
  Voicemail,
  Forward,
  Volume2,
  Mic,
  Activity,
  BarChart3,
  Calendar,
  ChevronDown,
  ChevronRight,
  Star,
  Eye,
  Edit,
  ArrowUpRight,
  ArrowDownRight,
  Sparkles,
  Building,
  Users,
  Hash,
  Tag,
  Info,
  HelpCircle,
  X,
  Check,
  Loader2,
} from "lucide-react";

// Types
interface PhoneNumber {
  id: string;
  number: string;
  formattedNumber: string;
  friendlyName: string;
  country: string;
  countryCode: string;
  region: string;
  city: string;
  type: "local" | "toll_free" | "mobile" | "national";
  capabilities: {
    voice: boolean;
    sms: boolean;
    mms: boolean;
    fax: boolean;
  };
  status: "active" | "inactive" | "pending" | "suspended";
  assignedAgent: {
    id: string;
    name: string;
    avatar?: string;
  } | null;
  monthlyStats: {
    inbound: number;
    outbound: number;
    missed: number;
    avgDuration: number;
    totalMinutes: number;
    cost: number;
  };
  settings: {
    voicemailEnabled: boolean;
    recordingEnabled: boolean;
    transcriptionEnabled: boolean;
    callForwarding: string | null;
    businessHours: boolean;
  };
  createdAt: string;
  lastCallAt: string | null;
  tags: string[];
  monthlyPrice: number;
  setupPrice: number;
}

interface Country {
  code: string;
  name: string;
  flag: string;
  prefix: string;
  available: boolean;
}

interface NumberType {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  priceRange: string;
}

// Mock data
const mockPhoneNumbers: PhoneNumber[] = [
  {
    id: "pn_1",
    number: "+14155551234",
    formattedNumber: "+1 (415) 555-1234",
    friendlyName: "Main Sales Line",
    country: "United States",
    countryCode: "US",
    region: "California",
    city: "San Francisco",
    type: "local",
    capabilities: { voice: true, sms: true, mms: true, fax: false },
    status: "active",
    assignedAgent: { id: "agent_1", name: "Sales Assistant", avatar: undefined },
    monthlyStats: {
      inbound: 1234,
      outbound: 567,
      missed: 45,
      avgDuration: 4.5,
      totalMinutes: 8100,
      cost: 243.00,
    },
    settings: {
      voicemailEnabled: true,
      recordingEnabled: true,
      transcriptionEnabled: true,
      callForwarding: null,
      businessHours: true,
    },
    createdAt: "2024-01-15T10:00:00Z",
    lastCallAt: "2024-01-20T14:35:00Z",
    tags: ["sales", "primary"],
    monthlyPrice: 2.00,
    setupPrice: 1.00,
  },
  {
    id: "pn_2",
    number: "+18005551234",
    formattedNumber: "+1 (800) 555-1234",
    friendlyName: "Customer Support",
    country: "United States",
    countryCode: "US",
    region: "National",
    city: "Toll-Free",
    type: "toll_free",
    capabilities: { voice: true, sms: true, mms: false, fax: false },
    status: "active",
    assignedAgent: { id: "agent_2", name: "Support Bot", avatar: undefined },
    monthlyStats: {
      inbound: 3456,
      outbound: 123,
      missed: 89,
      avgDuration: 6.2,
      totalMinutes: 22200,
      cost: 666.00,
    },
    settings: {
      voicemailEnabled: true,
      recordingEnabled: true,
      transcriptionEnabled: true,
      callForwarding: "+14155559999",
      businessHours: true,
    },
    createdAt: "2024-01-10T08:00:00Z",
    lastCallAt: "2024-01-20T15:12:00Z",
    tags: ["support", "toll-free", "primary"],
    monthlyPrice: 15.00,
    setupPrice: 0,
  },
  {
    id: "pn_3",
    number: "+442071234567",
    formattedNumber: "+44 20 7123 4567",
    friendlyName: "UK Office",
    country: "United Kingdom",
    countryCode: "GB",
    region: "England",
    city: "London",
    type: "local",
    capabilities: { voice: true, sms: true, mms: false, fax: true },
    status: "active",
    assignedAgent: { id: "agent_3", name: "UK Sales Agent", avatar: undefined },
    monthlyStats: {
      inbound: 456,
      outbound: 234,
      missed: 23,
      avgDuration: 5.1,
      totalMinutes: 3519,
      cost: 105.57,
    },
    settings: {
      voicemailEnabled: true,
      recordingEnabled: true,
      transcriptionEnabled: false,
      callForwarding: null,
      businessHours: true,
    },
    createdAt: "2024-01-12T09:00:00Z",
    lastCallAt: "2024-01-20T11:45:00Z",
    tags: ["uk", "international"],
    monthlyPrice: 4.00,
    setupPrice: 2.00,
  },
  {
    id: "pn_4",
    number: "+14155559999",
    formattedNumber: "+1 (415) 555-9999",
    friendlyName: "After Hours",
    country: "United States",
    countryCode: "US",
    region: "California",
    city: "San Francisco",
    type: "local",
    capabilities: { voice: true, sms: true, mms: true, fax: false },
    status: "inactive",
    assignedAgent: null,
    monthlyStats: {
      inbound: 0,
      outbound: 0,
      missed: 0,
      avgDuration: 0,
      totalMinutes: 0,
      cost: 0,
    },
    settings: {
      voicemailEnabled: true,
      recordingEnabled: false,
      transcriptionEnabled: false,
      callForwarding: null,
      businessHours: false,
    },
    createdAt: "2024-01-18T14:00:00Z",
    lastCallAt: null,
    tags: ["backup"],
    monthlyPrice: 2.00,
    setupPrice: 1.00,
  },
  {
    id: "pn_5",
    number: "+61291234567",
    formattedNumber: "+61 2 9123 4567",
    friendlyName: "Australia Support",
    country: "Australia",
    countryCode: "AU",
    region: "New South Wales",
    city: "Sydney",
    type: "local",
    capabilities: { voice: true, sms: true, mms: false, fax: false },
    status: "pending",
    assignedAgent: null,
    monthlyStats: {
      inbound: 0,
      outbound: 0,
      missed: 0,
      avgDuration: 0,
      totalMinutes: 0,
      cost: 0,
    },
    settings: {
      voicemailEnabled: false,
      recordingEnabled: false,
      transcriptionEnabled: false,
      callForwarding: null,
      businessHours: false,
    },
    createdAt: "2024-01-19T16:00:00Z",
    lastCallAt: null,
    tags: ["australia", "pending-setup"],
    monthlyPrice: 5.00,
    setupPrice: 3.00,
  },
  {
    id: "pn_6",
    number: "+33142123456",
    formattedNumber: "+33 1 42 12 34 56",
    friendlyName: "France Office",
    country: "France",
    countryCode: "FR",
    region: "Île-de-France",
    city: "Paris",
    type: "local",
    capabilities: { voice: true, sms: false, mms: false, fax: true },
    status: "active",
    assignedAgent: { id: "agent_4", name: "French Support", avatar: undefined },
    monthlyStats: {
      inbound: 289,
      outbound: 156,
      missed: 12,
      avgDuration: 4.8,
      totalMinutes: 2136,
      cost: 64.08,
    },
    settings: {
      voicemailEnabled: true,
      recordingEnabled: true,
      transcriptionEnabled: true,
      callForwarding: null,
      businessHours: true,
    },
    createdAt: "2024-01-08T11:00:00Z",
    lastCallAt: "2024-01-20T09:22:00Z",
    tags: ["france", "international", "europe"],
    monthlyPrice: 4.50,
    setupPrice: 2.50,
  },
];

const countries: Country[] = [
  { code: "US", name: "United States", flag: "🇺🇸", prefix: "+1", available: true },
  { code: "GB", name: "United Kingdom", flag: "🇬🇧", prefix: "+44", available: true },
  { code: "CA", name: "Canada", flag: "🇨🇦", prefix: "+1", available: true },
  { code: "AU", name: "Australia", flag: "🇦🇺", prefix: "+61", available: true },
  { code: "DE", name: "Germany", flag: "🇩🇪", prefix: "+49", available: true },
  { code: "FR", name: "France", flag: "🇫🇷", prefix: "+33", available: true },
  { code: "ES", name: "Spain", flag: "🇪🇸", prefix: "+34", available: true },
  { code: "IT", name: "Italy", flag: "🇮🇹", prefix: "+39", available: true },
  { code: "NL", name: "Netherlands", flag: "🇳🇱", prefix: "+31", available: true },
  { code: "JP", name: "Japan", flag: "🇯🇵", prefix: "+81", available: true },
  { code: "SG", name: "Singapore", flag: "🇸🇬", prefix: "+65", available: true },
  { code: "BR", name: "Brazil", flag: "🇧🇷", prefix: "+55", available: false },
  { code: "MX", name: "Mexico", flag: "🇲🇽", prefix: "+52", available: true },
  { code: "IN", name: "India", flag: "🇮🇳", prefix: "+91", available: false },
];

const numberTypes: NumberType[] = [
  {
    id: "local",
    name: "Local Numbers",
    description: "Numbers with local area codes for specific regions",
    icon: <MapPin className="w-5 h-5" />,
    priceRange: "$1-5/mo",
  },
  {
    id: "toll_free",
    name: "Toll-Free Numbers",
    description: "Free to call from anywhere in the country",
    icon: <Phone className="w-5 h-5" />,
    priceRange: "$10-20/mo",
  },
  {
    id: "mobile",
    name: "Mobile Numbers",
    description: "Mobile-enabled numbers with SMS capabilities",
    icon: <MessageSquare className="w-5 h-5" />,
    priceRange: "$2-8/mo",
  },
  {
    id: "national",
    name: "National Numbers",
    description: "Non-geographic national rate numbers",
    icon: <Globe className="w-5 h-5" />,
    priceRange: "$3-10/mo",
  },
];

// Country flag component
function CountryFlag({ code }: { code: string }) {
  const country = countries.find((c) => c.code === code);
  return <span className="text-lg">{country?.flag || "🌍"}</span>;
}

// Status badge component
function StatusBadge({ status }: { status: PhoneNumber["status"] }) {
  const styles = {
    active: "bg-green-500/10 text-green-400 border-green-500/20",
    inactive: "bg-gray-500/10 text-gray-400 border-gray-500/20",
    pending: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
    suspended: "bg-red-500/10 text-red-400 border-red-500/20",
  };

  const icons = {
    active: <CheckCircle className="w-3 h-3" />,
    inactive: <XCircle className="w-3 h-3" />,
    pending: <Clock className="w-3 h-3" />,
    suspended: <AlertCircle className="w-3 h-3" />,
  };

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${styles[status]}`}
    >
      {icons[status]}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

// Type badge component
function TypeBadge({ type }: { type: PhoneNumber["type"] }) {
  const styles = {
    local: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    toll_free: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    mobile: "bg-green-500/10 text-green-400 border-green-500/20",
    national: "bg-orange-500/10 text-orange-400 border-orange-500/20",
  };

  const labels = {
    local: "Local",
    toll_free: "Toll-Free",
    mobile: "Mobile",
    national: "National",
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded border ${styles[type]}`}
    >
      {labels[type]}
    </span>
  );
}

// Capability icons component
function CapabilityIcons({
  capabilities,
}: {
  capabilities: PhoneNumber["capabilities"];
}) {
  return (
    <div className="flex items-center gap-1">
      {capabilities.voice && (
        <span
          className="p-1 rounded bg-green-500/10 text-green-400"
          title="Voice"
        >
          <Phone className="w-3 h-3" />
        </span>
      )}
      {capabilities.sms && (
        <span
          className="p-1 rounded bg-blue-500/10 text-blue-400"
          title="SMS"
        >
          <MessageSquare className="w-3 h-3" />
        </span>
      )}
      {capabilities.mms && (
        <span
          className="p-1 rounded bg-purple-500/10 text-purple-400"
          title="MMS"
        >
          <Sparkles className="w-3 h-3" />
        </span>
      )}
      {capabilities.fax && (
        <span
          className="p-1 rounded bg-gray-500/10 text-gray-400"
          title="Fax"
        >
          <Download className="w-3 h-3" />
        </span>
      )}
    </div>
  );
}

// Stats card component
function StatsCard({
  title,
  value,
  change,
  changeType,
  icon,
  color,
}: {
  title: string;
  value: string;
  change?: string;
  changeType?: "up" | "down" | "neutral";
  icon: React.ReactNode;
  color: string;
}) {
  const colorStyles = {
    blue: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    green: "bg-green-500/10 text-green-400 border-green-500/20",
    purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    orange: "bg-orange-500/10 text-orange-400 border-orange-500/20",
    red: "bg-red-500/10 text-red-400 border-red-500/20",
    cyan: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
  };

  return (
    <div className="bg-[#1a1a2e]/50 backdrop-blur-sm rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-gray-400">{title}</span>
        <div
          className={`p-2 rounded-lg border ${colorStyles[color as keyof typeof colorStyles]}`}
        >
          {icon}
        </div>
      </div>
      <div className="flex items-end justify-between">
        <span className="text-2xl font-bold text-white">{value}</span>
        {change && (
          <div
            className={`flex items-center gap-1 text-xs ${
              changeType === "up"
                ? "text-green-400"
                : changeType === "down"
                ? "text-red-400"
                : "text-gray-400"
            }`}
          >
            {changeType === "up" ? (
              <ArrowUpRight className="w-3 h-3" />
            ) : changeType === "down" ? (
              <ArrowDownRight className="w-3 h-3" />
            ) : null}
            {change}
          </div>
        )}
      </div>
    </div>
  );
}

// Phone number row component
function PhoneNumberRow({
  phoneNumber,
  onSelect,
  onConfigure,
  onDelete,
  isSelected,
}: {
  phoneNumber: PhoneNumber;
  onSelect: (id: string) => void;
  onConfigure: (phone: PhoneNumber) => void;
  onDelete: (id: string) => void;
  isSelected: boolean;
}) {
  const [showMenu, setShowMenu] = useState(false);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div
      className={`group bg-[#1a1a2e]/30 hover:bg-[#1a1a2e]/60 rounded-xl border transition-all duration-200 ${
        isSelected
          ? "border-purple-500/50 bg-purple-500/5"
          : "border-white/5 hover:border-white/10"
      }`}
    >
      <div className="p-4">
        <div className="flex items-center gap-4">
          {/* Checkbox */}
          <div
            onClick={() => onSelect(phoneNumber.id)}
            className={`w-5 h-5 rounded border-2 flex items-center justify-center cursor-pointer transition-all ${
              isSelected
                ? "bg-purple-500 border-purple-500"
                : "border-gray-600 hover:border-gray-500"
            }`}
          >
            {isSelected && <Check className="w-3 h-3 text-white" />}
          </div>

          {/* Country flag and number */}
          <div className="flex items-center gap-3 min-w-[240px]">
            <CountryFlag code={phoneNumber.countryCode} />
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-white font-medium">
                  {phoneNumber.formattedNumber}
                </span>
                <button
                  onClick={() => copyToClipboard(phoneNumber.number)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition-all"
                  title="Copy number"
                >
                  <Copy className="w-3 h-3 text-gray-400" />
                </button>
              </div>
              <span className="text-xs text-gray-500">
                {phoneNumber.friendlyName}
              </span>
            </div>
          </div>

          {/* Type and status */}
          <div className="flex items-center gap-2 min-w-[160px]">
            <TypeBadge type={phoneNumber.type} />
            <StatusBadge status={phoneNumber.status} />
          </div>

          {/* Location */}
          <div className="min-w-[140px]">
            <div className="flex items-center gap-1 text-sm text-gray-400">
              <MapPin className="w-3 h-3" />
              <span>{phoneNumber.city}</span>
            </div>
            <span className="text-xs text-gray-500">{phoneNumber.region}</span>
          </div>

          {/* Capabilities */}
          <div className="min-w-[100px]">
            <CapabilityIcons capabilities={phoneNumber.capabilities} />
          </div>

          {/* Assigned agent */}
          <div className="min-w-[140px]">
            {phoneNumber.assignedAgent ? (
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                  <Bot className="w-3 h-3 text-white" />
                </div>
                <span className="text-sm text-gray-300">
                  {phoneNumber.assignedAgent.name}
                </span>
              </div>
            ) : (
              <span className="text-sm text-gray-500 italic">Unassigned</span>
            )}
          </div>

          {/* Stats */}
          <div className="min-w-[120px] text-right">
            <div className="flex items-center justify-end gap-3 text-xs">
              <span className="flex items-center gap-1 text-green-400">
                <PhoneIncoming className="w-3 h-3" />
                {phoneNumber.monthlyStats.inbound}
              </span>
              <span className="flex items-center gap-1 text-blue-400">
                <PhoneOutgoing className="w-3 h-3" />
                {phoneNumber.monthlyStats.outbound}
              </span>
              <span className="flex items-center gap-1 text-red-400">
                <PhoneMissed className="w-3 h-3" />
                {phoneNumber.monthlyStats.missed}
              </span>
            </div>
            <span className="text-xs text-gray-500">
              {phoneNumber.monthlyStats.totalMinutes.toLocaleString()} min
            </span>
          </div>

          {/* Cost */}
          <div className="min-w-[80px] text-right">
            <span className="text-sm font-medium text-white">
              ${phoneNumber.monthlyStats.cost.toFixed(2)}
            </span>
            <div className="text-xs text-gray-500">this month</div>
          </div>

          {/* Actions */}
          <div className="relative ml-auto">
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
                <div className="absolute right-0 top-full mt-1 w-48 bg-[#1a1a2e] border border-white/10 rounded-xl shadow-xl z-20 py-1 overflow-hidden">
                  <button
                    onClick={() => {
                      onConfigure(phoneNumber);
                      setShowMenu(false);
                    }}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10 transition-colors"
                  >
                    <Settings className="w-4 h-4" />
                    Configure
                  </button>
                  <button className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10 transition-colors">
                    <Eye className="w-4 h-4" />
                    View Details
                  </button>
                  <button className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10 transition-colors">
                    <BarChart3 className="w-4 h-4" />
                    View Analytics
                  </button>
                  <button className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10 transition-colors">
                    <Bot className="w-4 h-4" />
                    Assign Agent
                  </button>
                  <button className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-300 hover:bg-white/10 transition-colors">
                    <Copy className="w-4 h-4" />
                    Copy Number
                  </button>
                  <div className="border-t border-white/10 my-1" />
                  <button
                    onClick={() => {
                      onDelete(phoneNumber.id);
                      setShowMenu(false);
                    }}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                    Release Number
                  </button>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Tags */}
        {phoneNumber.tags.length > 0 && (
          <div className="flex items-center gap-2 mt-3 ml-9">
            {phoneNumber.tags.map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 text-xs bg-white/5 text-gray-400 rounded-full"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Buy number dialog
function BuyNumberDialog({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const [step, setStep] = useState(1);
  const [selectedCountry, setSelectedCountry] = useState<string>("US");
  const [selectedType, setSelectedType] = useState<string>("local");
  const [areaCode, setAreaCode] = useState("");
  const [searchResults, setSearchResults] = useState<
    Array<{
      number: string;
      formatted: string;
      city: string;
      region: string;
      monthlyPrice: number;
      setupPrice: number;
    }>
  >([]);
  const [selectedNumber, setSelectedNumber] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [isPurchasing, setIsPurchasing] = useState(false);

  const searchNumbers = () => {
    setIsSearching(true);
    // Simulate API call
    setTimeout(() => {
      setSearchResults([
        {
          number: "+14155551111",
          formatted: "+1 (415) 555-1111",
          city: "San Francisco",
          region: "California",
          monthlyPrice: 2.0,
          setupPrice: 1.0,
        },
        {
          number: "+14155552222",
          formatted: "+1 (415) 555-2222",
          city: "San Francisco",
          region: "California",
          monthlyPrice: 2.0,
          setupPrice: 1.0,
        },
        {
          number: "+14155553333",
          formatted: "+1 (415) 555-3333",
          city: "Oakland",
          region: "California",
          monthlyPrice: 2.0,
          setupPrice: 1.0,
        },
        {
          number: "+14155554444",
          formatted: "+1 (415) 555-4444",
          city: "San Jose",
          region: "California",
          monthlyPrice: 2.0,
          setupPrice: 1.0,
        },
        {
          number: "+14155555555",
          formatted: "+1 (415) 555-5555",
          city: "Palo Alto",
          region: "California",
          monthlyPrice: 5.0,
          setupPrice: 0,
        },
      ]);
      setIsSearching(false);
      setStep(2);
    }, 1500);
  };

  const purchaseNumber = () => {
    setIsPurchasing(true);
    // Simulate purchase
    setTimeout(() => {
      setIsPurchasing(false);
      setStep(3);
    }, 2000);
  };

  if (!isOpen) return null;

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
              {step === 1
                ? "Get a New Phone Number"
                : step === 2
                ? "Select a Number"
                : "Number Purchased!"}
            </h2>
            <p className="text-sm text-gray-400">
              {step === 1
                ? "Search for available phone numbers"
                : step === 2
                ? "Choose from available numbers"
                : "Your new number is ready to use"}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Progress */}
        <div className="px-6 py-3 bg-white/5 border-b border-white/5">
          <div className="flex items-center gap-4">
            {[1, 2, 3].map((s) => (
              <div key={s} className="flex items-center gap-2">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                    s === step
                      ? "bg-purple-500 text-white"
                      : s < step
                      ? "bg-green-500 text-white"
                      : "bg-white/10 text-gray-400"
                  }`}
                >
                  {s < step ? <Check className="w-4 h-4" /> : s}
                </div>
                <span
                  className={`text-sm ${
                    s === step ? "text-white" : "text-gray-400"
                  }`}
                >
                  {s === 1 ? "Search" : s === 2 ? "Select" : "Complete"}
                </span>
                {s < 3 && (
                  <ChevronRight className="w-4 h-4 text-gray-600 mx-2" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {step === 1 && (
            <div className="space-y-6">
              {/* Country selection */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Country
                </label>
                <div className="grid grid-cols-4 gap-2">
                  {countries
                    .filter((c) => c.available)
                    .slice(0, 8)
                    .map((country) => (
                      <button
                        key={country.code}
                        onClick={() => setSelectedCountry(country.code)}
                        className={`flex items-center gap-2 p-3 rounded-lg border transition-all ${
                          selectedCountry === country.code
                            ? "border-purple-500 bg-purple-500/10"
                            : "border-white/10 hover:border-white/20"
                        }`}
                      >
                        <span className="text-xl">{country.flag}</span>
                        <span className="text-sm text-gray-300">
                          {country.code}
                        </span>
                      </button>
                    ))}
                </div>
              </div>

              {/* Number type */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Number Type
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {numberTypes.map((type) => (
                    <button
                      key={type.id}
                      onClick={() => setSelectedType(type.id)}
                      className={`flex items-start gap-3 p-4 rounded-lg border transition-all text-left ${
                        selectedType === type.id
                          ? "border-purple-500 bg-purple-500/10"
                          : "border-white/10 hover:border-white/20"
                      }`}
                    >
                      <div className="p-2 rounded-lg bg-white/10">
                        {type.icon}
                      </div>
                      <div>
                        <span className="block text-sm font-medium text-white">
                          {type.name}
                        </span>
                        <span className="block text-xs text-gray-400 mt-0.5">
                          {type.description}
                        </span>
                        <span className="block text-xs text-purple-400 mt-1">
                          {type.priceRange}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Area code */}
              {selectedType === "local" && (
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Area Code (optional)
                  </label>
                  <input
                    type="text"
                    value={areaCode}
                    onChange={(e) => setAreaCode(e.target.value)}
                    placeholder="e.g., 415"
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                  />
                </div>
              )}
            </div>
          )}

          {step === 2 && (
            <div className="space-y-3">
              {searchResults.map((result) => (
                <button
                  key={result.number}
                  onClick={() => setSelectedNumber(result.number)}
                  className={`flex items-center justify-between w-full p-4 rounded-lg border transition-all ${
                    selectedNumber === result.number
                      ? "border-purple-500 bg-purple-500/10"
                      : "border-white/10 hover:border-white/20"
                  }`}
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                      <Phone className="w-5 h-5 text-white" />
                    </div>
                    <div className="text-left">
                      <span className="block font-mono text-white font-medium">
                        {result.formatted}
                      </span>
                      <span className="block text-xs text-gray-400">
                        {result.city}, {result.region}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="block text-white font-medium">
                      ${result.monthlyPrice.toFixed(2)}/mo
                    </span>
                    {result.setupPrice > 0 && (
                      <span className="block text-xs text-gray-400">
                        +${result.setupPrice.toFixed(2)} setup
                      </span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          )}

          {step === 3 && (
            <div className="text-center py-8">
              <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="w-8 h-8 text-green-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">
                Number Successfully Purchased!
              </h3>
              <p className="text-gray-400 mb-6">
                Your new phone number is now ready to be configured
              </p>
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/5 rounded-lg">
                <Phone className="w-4 h-4 text-purple-400" />
                <span className="font-mono text-white">
                  {searchResults.find((r) => r.number === selectedNumber)
                    ?.formatted || selectedNumber}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-white/10 bg-white/5">
          {step > 1 && step < 3 ? (
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
            {step < 3 && (
              <button
                onClick={onClose}
                className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
            )}
            {step === 1 && (
              <button
                onClick={searchNumbers}
                disabled={isSearching}
                className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors disabled:opacity-50"
              >
                {isSearching ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    Search Numbers
                  </>
                )}
              </button>
            )}
            {step === 2 && (
              <button
                onClick={purchaseNumber}
                disabled={!selectedNumber || isPurchasing}
                className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors disabled:opacity-50"
              >
                {isPurchasing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Purchasing...
                  </>
                ) : (
                  <>
                    <DollarSign className="w-4 h-4" />
                    Purchase Number
                  </>
                )}
              </button>
            )}
            {step === 3 && (
              <button
                onClick={onClose}
                className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
              >
                <Settings className="w-4 h-4" />
                Configure Number
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Configure number dialog
function ConfigureNumberDialog({
  phoneNumber,
  isOpen,
  onClose,
}: {
  phoneNumber: PhoneNumber | null;
  isOpen: boolean;
  onClose: () => void;
}) {
  const [activeTab, setActiveTab] = useState("general");
  const [settings, setSettings] = useState({
    friendlyName: phoneNumber?.friendlyName || "",
    voicemailEnabled: phoneNumber?.settings.voicemailEnabled || false,
    recordingEnabled: phoneNumber?.settings.recordingEnabled || false,
    transcriptionEnabled: phoneNumber?.settings.transcriptionEnabled || false,
    callForwarding: phoneNumber?.settings.callForwarding || "",
    businessHoursEnabled: phoneNumber?.settings.businessHours || false,
  });

  if (!isOpen || !phoneNumber) return null;

  const tabs = [
    { id: "general", label: "General", icon: <Settings className="w-4 h-4" /> },
    { id: "voice", label: "Voice", icon: <Mic className="w-4 h-4" /> },
    {
      id: "routing",
      label: "Call Routing",
      icon: <Forward className="w-4 h-4" />,
    },
    {
      id: "voicemail",
      label: "Voicemail",
      icon: <Voicemail className="w-4 h-4" />,
    },
    {
      id: "recording",
      label: "Recording",
      icon: <Volume2 className="w-4 h-4" />,
    },
    {
      id: "security",
      label: "Security",
      icon: <Shield className="w-4 h-4" />,
    },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-[#0f0f1a] border border-white/10 rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <Phone className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">
                Configure {phoneNumber.formattedNumber}
              </h2>
              <p className="text-sm text-gray-400">
                {phoneNumber.friendlyName} • {phoneNumber.city},{" "}
                {phoneNumber.country}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        <div className="flex h-[60vh]">
          {/* Sidebar tabs */}
          <div className="w-56 border-r border-white/10 p-4 space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-3 w-full px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                  activeTab === tab.id
                    ? "bg-purple-500/20 text-purple-400"
                    : "text-gray-400 hover:bg-white/5 hover:text-white"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            {activeTab === "general" && (
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Friendly Name
                  </label>
                  <input
                    type="text"
                    value={settings.friendlyName}
                    onChange={(e) =>
                      setSettings({ ...settings, friendlyName: e.target.value })
                    }
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Assigned Agent
                  </label>
                  <select className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500">
                    <option value="">No agent assigned</option>
                    <option value="agent_1">Sales Assistant</option>
                    <option value="agent_2">Support Bot</option>
                    <option value="agent_3">Appointment Scheduler</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Tags
                  </label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {phoneNumber.tags.map((tag) => (
                      <span
                        key={tag}
                        className="flex items-center gap-1 px-3 py-1 bg-white/5 text-gray-300 rounded-full text-sm"
                      >
                        #{tag}
                        <button className="hover:text-white">
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                  <input
                    type="text"
                    placeholder="Add a tag..."
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                  />
                </div>

                <div className="pt-4 border-t border-white/10">
                  <h3 className="text-sm font-medium text-gray-300 mb-4">
                    Number Details
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-white/5 rounded-lg">
                      <span className="text-xs text-gray-500">Number Type</span>
                      <div className="flex items-center gap-2 mt-1">
                        <TypeBadge type={phoneNumber.type} />
                      </div>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg">
                      <span className="text-xs text-gray-500">Status</span>
                      <div className="flex items-center gap-2 mt-1">
                        <StatusBadge status={phoneNumber.status} />
                      </div>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg">
                      <span className="text-xs text-gray-500">Capabilities</span>
                      <div className="mt-1">
                        <CapabilityIcons capabilities={phoneNumber.capabilities} />
                      </div>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg">
                      <span className="text-xs text-gray-500">Monthly Cost</span>
                      <span className="block text-white font-medium mt-1">
                        ${phoneNumber.monthlyPrice.toFixed(2)}/mo
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "voice" && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-white mb-4">
                    Voice Settings
                  </h3>
                  <div className="space-y-4">
                    <div className="p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="block text-sm font-medium text-white">
                            AI Voice
                          </span>
                          <span className="block text-xs text-gray-400">
                            Select the voice for your AI agent
                          </span>
                        </div>
                        <select className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm">
                          <option>Aria (Female)</option>
                          <option>Marcus (Male)</option>
                          <option>Sofia (Female)</option>
                          <option>James (Male)</option>
                        </select>
                      </div>
                    </div>

                    <div className="p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="block text-sm font-medium text-white">
                            Speech Speed
                          </span>
                          <span className="block text-xs text-gray-400">
                            Adjust the speaking pace
                          </span>
                        </div>
                        <input
                          type="range"
                          min="0.5"
                          max="2"
                          step="0.1"
                          defaultValue="1"
                          className="w-32"
                        />
                      </div>
                    </div>

                    <div className="p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="block text-sm font-medium text-white">
                            Language
                          </span>
                          <span className="block text-xs text-gray-400">
                            Primary language for conversations
                          </span>
                        </div>
                        <select className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white text-sm">
                          <option>English (US)</option>
                          <option>English (UK)</option>
                          <option>Spanish</option>
                          <option>French</option>
                          <option>German</option>
                        </select>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "routing" && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-white mb-4">
                    Call Routing
                  </h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                      <div>
                        <span className="block text-sm font-medium text-white">
                          Business Hours Only
                        </span>
                        <span className="block text-xs text-gray-400">
                          Only accept calls during business hours
                        </span>
                      </div>
                      <button
                        onClick={() =>
                          setSettings({
                            ...settings,
                            businessHoursEnabled: !settings.businessHoursEnabled,
                          })
                        }
                        className={`relative w-12 h-6 rounded-full transition-colors ${
                          settings.businessHoursEnabled
                            ? "bg-purple-500"
                            : "bg-gray-600"
                        }`}
                      >
                        <span
                          className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                            settings.businessHoursEnabled
                              ? "translate-x-7"
                              : "translate-x-1"
                          }`}
                        />
                      </button>
                    </div>

                    <div className="p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <span className="block text-sm font-medium text-white">
                            Call Forwarding
                          </span>
                          <span className="block text-xs text-gray-400">
                            Forward calls to another number
                          </span>
                        </div>
                      </div>
                      <input
                        type="text"
                        value={settings.callForwarding}
                        onChange={(e) =>
                          setSettings({
                            ...settings,
                            callForwarding: e.target.value,
                          })
                        }
                        placeholder="+1 (555) 555-5555"
                        className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
                      />
                    </div>

                    <div className="p-4 bg-white/5 rounded-lg">
                      <span className="block text-sm font-medium text-white mb-3">
                        Routing Rules
                      </span>
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 p-2 bg-white/5 rounded">
                          <span className="text-xs text-gray-400">1.</span>
                          <span className="text-sm text-gray-300">
                            If agent busy → Queue call
                          </span>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-white/5 rounded">
                          <span className="text-xs text-gray-400">2.</span>
                          <span className="text-sm text-gray-300">
                            If after hours → Voicemail
                          </span>
                        </div>
                        <div className="flex items-center gap-2 p-2 bg-white/5 rounded">
                          <span className="text-xs text-gray-400">3.</span>
                          <span className="text-sm text-gray-300">
                            If no answer in 30s → Forward
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "voicemail" && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-white mb-4">
                    Voicemail Settings
                  </h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                      <div>
                        <span className="block text-sm font-medium text-white">
                          Enable Voicemail
                        </span>
                        <span className="block text-xs text-gray-400">
                          Allow callers to leave voicemail
                        </span>
                      </div>
                      <button
                        onClick={() =>
                          setSettings({
                            ...settings,
                            voicemailEnabled: !settings.voicemailEnabled,
                          })
                        }
                        className={`relative w-12 h-6 rounded-full transition-colors ${
                          settings.voicemailEnabled
                            ? "bg-purple-500"
                            : "bg-gray-600"
                        }`}
                      >
                        <span
                          className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                            settings.voicemailEnabled
                              ? "translate-x-7"
                              : "translate-x-1"
                          }`}
                        />
                      </button>
                    </div>

                    <div className="p-4 bg-white/5 rounded-lg">
                      <span className="block text-sm font-medium text-white mb-3">
                        Voicemail Greeting
                      </span>
                      <textarea
                        rows={3}
                        placeholder="Enter your voicemail greeting message..."
                        className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                        defaultValue="Hi, you've reached our voicemail. Please leave a message and we'll get back to you as soon as possible."
                      />
                    </div>

                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                      <div>
                        <span className="block text-sm font-medium text-white">
                          Transcription
                        </span>
                        <span className="block text-xs text-gray-400">
                          Transcribe voicemail messages to text
                        </span>
                      </div>
                      <button
                        onClick={() =>
                          setSettings({
                            ...settings,
                            transcriptionEnabled: !settings.transcriptionEnabled,
                          })
                        }
                        className={`relative w-12 h-6 rounded-full transition-colors ${
                          settings.transcriptionEnabled
                            ? "bg-purple-500"
                            : "bg-gray-600"
                        }`}
                      >
                        <span
                          className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                            settings.transcriptionEnabled
                              ? "translate-x-7"
                              : "translate-x-1"
                          }`}
                        />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "recording" && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-white mb-4">
                    Recording Settings
                  </h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                      <div>
                        <span className="block text-sm font-medium text-white">
                          Record Calls
                        </span>
                        <span className="block text-xs text-gray-400">
                          Automatically record all calls
                        </span>
                      </div>
                      <button
                        onClick={() =>
                          setSettings({
                            ...settings,
                            recordingEnabled: !settings.recordingEnabled,
                          })
                        }
                        className={`relative w-12 h-6 rounded-full transition-colors ${
                          settings.recordingEnabled
                            ? "bg-purple-500"
                            : "bg-gray-600"
                        }`}
                      >
                        <span
                          className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                            settings.recordingEnabled
                              ? "translate-x-7"
                              : "translate-x-1"
                          }`}
                        />
                      </button>
                    </div>

                    <div className="p-4 bg-white/5 rounded-lg">
                      <span className="block text-sm font-medium text-white mb-3">
                        Recording Announcement
                      </span>
                      <select className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500">
                        <option>Play announcement before recording</option>
                        <option>No announcement</option>
                        <option>Beep only</option>
                      </select>
                    </div>

                    <div className="p-4 bg-white/5 rounded-lg">
                      <span className="block text-sm font-medium text-white mb-3">
                        Storage Duration
                      </span>
                      <select className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500">
                        <option>30 days</option>
                        <option>60 days</option>
                        <option>90 days</option>
                        <option>1 year</option>
                        <option>Forever</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "security" && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-white mb-4">
                    Security Settings
                  </h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                      <div>
                        <span className="block text-sm font-medium text-white">
                          Spam Protection
                        </span>
                        <span className="block text-xs text-gray-400">
                          Block known spam callers
                        </span>
                      </div>
                      <button className="relative w-12 h-6 rounded-full bg-purple-500 transition-colors">
                        <span className="absolute top-1 translate-x-7 w-4 h-4 bg-white rounded-full transition-transform" />
                      </button>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                      <div>
                        <span className="block text-sm font-medium text-white">
                          Caller ID Verification
                        </span>
                        <span className="block text-xs text-gray-400">
                          Verify caller identities using STIR/SHAKEN
                        </span>
                      </div>
                      <button className="relative w-12 h-6 rounded-full bg-purple-500 transition-colors">
                        <span className="absolute top-1 translate-x-7 w-4 h-4 bg-white rounded-full transition-transform" />
                      </button>
                    </div>

                    <div className="p-4 bg-white/5 rounded-lg">
                      <span className="block text-sm font-medium text-white mb-3">
                        Blocked Numbers
                      </span>
                      <div className="space-y-2 mb-3">
                        <div className="flex items-center justify-between p-2 bg-white/5 rounded">
                          <span className="text-sm text-gray-300 font-mono">
                            +1 (555) 123-4567
                          </span>
                          <button className="text-red-400 hover:text-red-300">
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                      <input
                        type="text"
                        placeholder="Add number to block..."
                        className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-white/10 bg-white/5">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onClose}
            className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
          >
            <Check className="w-4 h-4" />
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}

// Port number dialog
function PortNumberDialog({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    phoneNumber: "",
    currentCarrier: "",
    accountNumber: "",
    pin: "",
    businessName: "",
    authorizedName: "",
    serviceAddress: "",
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-[#0f0f1a] border border-white/10 rounded-2xl shadow-2xl w-full max-w-xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <div>
            <h2 className="text-xl font-semibold text-white">
              Port Existing Number
            </h2>
            <p className="text-sm text-gray-400">
              Transfer your number from another carrier
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
        <div className="p-6 space-y-4 max-h-[60vh] overflow-y-auto">
          <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-400 mt-0.5" />
              <div>
                <span className="block text-sm font-medium text-blue-400">
                  Porting Information
                </span>
                <span className="block text-xs text-blue-300/70 mt-1">
                  Number porting typically takes 1-3 business days. Your current
                  service will continue until the port is complete.
                </span>
              </div>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Phone Number to Port
            </label>
            <input
              type="text"
              value={formData.phoneNumber}
              onChange={(e) =>
                setFormData({ ...formData, phoneNumber: e.target.value })
              }
              placeholder="+1 (555) 555-5555"
              className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Current Carrier
            </label>
            <select
              value={formData.currentCarrier}
              onChange={(e) =>
                setFormData({ ...formData, currentCarrier: e.target.value })
              }
              className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="">Select carrier...</option>
              <option value="att">AT&T</option>
              <option value="verizon">Verizon</option>
              <option value="tmobile">T-Mobile</option>
              <option value="twilio">Twilio</option>
              <option value="vonage">Vonage</option>
              <option value="bandwidth">Bandwidth</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Account Number
              </label>
              <input
                type="text"
                value={formData.accountNumber}
                onChange={(e) =>
                  setFormData({ ...formData, accountNumber: e.target.value })
                }
                placeholder="Your account number"
                className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                PIN / Password
              </label>
              <input
                type="password"
                value={formData.pin}
                onChange={(e) =>
                  setFormData({ ...formData, pin: e.target.value })
                }
                placeholder="Account PIN"
                className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Business / Account Name
            </label>
            <input
              type="text"
              value={formData.businessName}
              onChange={(e) =>
                setFormData({ ...formData, businessName: e.target.value })
              }
              placeholder="Name on the account"
              className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Service Address
            </label>
            <textarea
              rows={2}
              value={formData.serviceAddress}
              onChange={(e) =>
                setFormData({ ...formData, serviceAddress: e.target.value })
              }
              placeholder="Address associated with the number"
              className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500 resize-none"
            />
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-white/10 bg-white/5">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button className="flex items-center gap-2 px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors">
            <Upload className="w-4 h-4" />
            Submit Port Request
          </button>
        </div>
      </div>
    </div>
  );
}

// Main component
export default function PhoneNumbersPage() {
  const [phoneNumbers, setPhoneNumbers] = useState(mockPhoneNumbers);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterType, setFilterType] = useState<string>("all");
  const [filterCountry, setFilterCountry] = useState<string>("all");
  const [sortBy, setSortBy] = useState<string>("number");
  const [showBuyDialog, setShowBuyDialog] = useState(false);
  const [showPortDialog, setShowPortDialog] = useState(false);
  const [configureNumber, setConfigureNumber] = useState<PhoneNumber | null>(
    null
  );

  // Calculate aggregate stats
  const stats = useMemo(() => {
    const activeNumbers = phoneNumbers.filter((p) => p.status === "active");
    const totalInbound = phoneNumbers.reduce(
      (sum, p) => sum + p.monthlyStats.inbound,
      0
    );
    const totalOutbound = phoneNumbers.reduce(
      (sum, p) => sum + p.monthlyStats.outbound,
      0
    );
    const totalMinutes = phoneNumbers.reduce(
      (sum, p) => sum + p.monthlyStats.totalMinutes,
      0
    );
    const totalCost = phoneNumbers.reduce(
      (sum, p) => sum + p.monthlyStats.cost,
      0
    );
    const monthlyCost = phoneNumbers.reduce(
      (sum, p) => sum + p.monthlyPrice,
      0
    );

    return {
      totalNumbers: phoneNumbers.length,
      activeNumbers: activeNumbers.length,
      totalInbound,
      totalOutbound,
      totalMinutes,
      totalCost,
      monthlyCost,
    };
  }, [phoneNumbers]);

  // Filter and sort phone numbers
  const filteredNumbers = useMemo(() => {
    return phoneNumbers
      .filter((phone) => {
        if (
          searchQuery &&
          !phone.number.includes(searchQuery) &&
          !phone.friendlyName.toLowerCase().includes(searchQuery.toLowerCase())
        ) {
          return false;
        }
        if (filterStatus !== "all" && phone.status !== filterStatus) {
          return false;
        }
        if (filterType !== "all" && phone.type !== filterType) {
          return false;
        }
        if (filterCountry !== "all" && phone.countryCode !== filterCountry) {
          return false;
        }
        return true;
      })
      .sort((a, b) => {
        switch (sortBy) {
          case "number":
            return a.number.localeCompare(b.number);
          case "name":
            return a.friendlyName.localeCompare(b.friendlyName);
          case "calls":
            return (
              b.monthlyStats.inbound +
              b.monthlyStats.outbound -
              (a.monthlyStats.inbound + a.monthlyStats.outbound)
            );
          case "cost":
            return b.monthlyStats.cost - a.monthlyStats.cost;
          default:
            return 0;
        }
      });
  }, [phoneNumbers, searchQuery, filterStatus, filterType, filterCountry, sortBy]);

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  const toggleSelectAll = () => {
    if (selectedIds.length === filteredNumbers.length) {
      setSelectedIds([]);
    } else {
      setSelectedIds(filteredNumbers.map((p) => p.id));
    }
  };

  const handleDelete = (id: string) => {
    setPhoneNumbers((prev) => prev.filter((p) => p.id !== id));
    setSelectedIds((prev) => prev.filter((i) => i !== id));
  };

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Phone Numbers</h1>
            <p className="text-gray-400">
              Manage your voice AI phone numbers and configurations
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowPortDialog(true)}
              className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg border border-white/10 transition-colors"
            >
              <Upload className="w-4 h-4" />
              Port Number
            </button>
            <button
              onClick={() => setShowBuyDialog(true)}
              className="flex items-center gap-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              Get New Number
            </button>
          </div>
        </div>

        {/* Stats cards */}
        <div className="grid grid-cols-6 gap-4">
          <StatsCard
            title="Total Numbers"
            value={stats.totalNumbers.toString()}
            icon={<Phone className="w-4 h-4" />}
            color="blue"
          />
          <StatsCard
            title="Active Numbers"
            value={stats.activeNumbers.toString()}
            icon={<CheckCircle className="w-4 h-4" />}
            color="green"
          />
          <StatsCard
            title="Inbound Calls"
            value={stats.totalInbound.toLocaleString()}
            change="+12.5%"
            changeType="up"
            icon={<PhoneIncoming className="w-4 h-4" />}
            color="cyan"
          />
          <StatsCard
            title="Outbound Calls"
            value={stats.totalOutbound.toLocaleString()}
            change="+8.2%"
            changeType="up"
            icon={<PhoneOutgoing className="w-4 h-4" />}
            color="purple"
          />
          <StatsCard
            title="Total Minutes"
            value={stats.totalMinutes.toLocaleString()}
            icon={<Clock className="w-4 h-4" />}
            color="orange"
          />
          <StatsCard
            title="Monthly Cost"
            value={`$${stats.totalCost.toFixed(2)}`}
            change="-2.3%"
            changeType="down"
            icon={<DollarSign className="w-4 h-4" />}
            color="green"
          />
        </div>

        {/* Filters and search */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 flex-1">
            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search phone numbers..."
                className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
              />
            </div>

            {/* Status filter */}
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Status</option>
              <option value="active">Active</option>
              <option value="inactive">Inactive</option>
              <option value="pending">Pending</option>
              <option value="suspended">Suspended</option>
            </select>

            {/* Type filter */}
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Types</option>
              <option value="local">Local</option>
              <option value="toll_free">Toll-Free</option>
              <option value="mobile">Mobile</option>
              <option value="national">National</option>
            </select>

            {/* Country filter */}
            <select
              value={filterCountry}
              onChange={(e) => setFilterCountry(e.target.value)}
              className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Countries</option>
              {countries
                .filter((c) =>
                  phoneNumbers.some((p) => p.countryCode === c.code)
                )
                .map((country) => (
                  <option key={country.code} value={country.code}>
                    {country.flag} {country.name}
                  </option>
                ))}
            </select>
          </div>

          {/* Sort */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="number">Number</option>
              <option value="name">Name</option>
              <option value="calls">Call Volume</option>
              <option value="cost">Cost</option>
            </select>
          </div>
        </div>

        {/* Bulk actions */}
        {selectedIds.length > 0 && (
          <div className="flex items-center gap-4 p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
            <span className="text-sm text-purple-400">
              {selectedIds.length} number{selectedIds.length > 1 ? "s" : ""}{" "}
              selected
            </span>
            <div className="flex items-center gap-2">
              <button className="px-3 py-1.5 text-sm bg-white/10 hover:bg-white/20 text-white rounded transition-colors">
                Assign Agent
              </button>
              <button className="px-3 py-1.5 text-sm bg-white/10 hover:bg-white/20 text-white rounded transition-colors">
                Add Tags
              </button>
              <button className="px-3 py-1.5 text-sm bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded transition-colors">
                Release Numbers
              </button>
            </div>
            <button
              onClick={() => setSelectedIds([])}
              className="ml-auto text-sm text-gray-400 hover:text-white"
            >
              Clear selection
            </button>
          </div>
        )}

        {/* Phone numbers list */}
        <div className="space-y-2">
          {/* Header row */}
          <div className="flex items-center gap-4 px-4 py-2 text-xs text-gray-500 uppercase">
            <div
              onClick={toggleSelectAll}
              className={`w-5 h-5 rounded border-2 flex items-center justify-center cursor-pointer transition-all ${
                selectedIds.length === filteredNumbers.length &&
                filteredNumbers.length > 0
                  ? "bg-purple-500 border-purple-500"
                  : "border-gray-600 hover:border-gray-500"
              }`}
            >
              {selectedIds.length === filteredNumbers.length &&
                filteredNumbers.length > 0 && (
                  <Check className="w-3 h-3 text-white" />
                )}
            </div>
            <span className="min-w-[240px]">Phone Number</span>
            <span className="min-w-[160px]">Type / Status</span>
            <span className="min-w-[140px]">Location</span>
            <span className="min-w-[100px]">Capabilities</span>
            <span className="min-w-[140px]">Agent</span>
            <span className="min-w-[120px] text-right">This Month</span>
            <span className="min-w-[80px] text-right">Cost</span>
            <span className="ml-auto">Actions</span>
          </div>

          {/* Phone number rows */}
          {filteredNumbers.map((phone) => (
            <PhoneNumberRow
              key={phone.id}
              phoneNumber={phone}
              onSelect={toggleSelect}
              onConfigure={setConfigureNumber}
              onDelete={handleDelete}
              isSelected={selectedIds.includes(phone.id)}
            />
          ))}

          {/* Empty state */}
          {filteredNumbers.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-4">
                <Phone className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-medium text-white mb-2">
                No phone numbers found
              </h3>
              <p className="text-gray-400 mb-4">
                {searchQuery || filterStatus !== "all" || filterType !== "all"
                  ? "Try adjusting your filters"
                  : "Get started by purchasing a new phone number"}
              </p>
              <button
                onClick={() => setShowBuyDialog(true)}
                className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
              >
                <Plus className="w-4 h-4" />
                Get New Number
              </button>
            </div>
          )}
        </div>

        {/* Dialogs */}
        <BuyNumberDialog
          isOpen={showBuyDialog}
          onClose={() => setShowBuyDialog(false)}
        />
        <PortNumberDialog
          isOpen={showPortDialog}
          onClose={() => setShowPortDialog(false)}
        />
        <ConfigureNumberDialog
          phoneNumber={configureNumber}
          isOpen={configureNumber !== null}
          onClose={() => setConfigureNumber(null)}
        />
      </div>
    </DashboardLayout>
  );
}
