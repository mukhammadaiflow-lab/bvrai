"use client";

import React, { useState, useEffect, useMemo } from "react";
import DashboardLayout from "../components/DashboardLayout";
import {
  Calendar,
  CalendarDays,
  CalendarClock,
  Clock,
  Plus,
  Search,
  Filter,
  MoreVertical,
  Edit,
  Trash2,
  Copy,
  CheckCircle,
  XCircle,
  AlertCircle,
  AlertTriangle,
  User,
  Users,
  Phone,
  Mail,
  Video,
  MapPin,
  Globe,
  Link,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ArrowRight,
  Settings,
  Bell,
  RefreshCw,
  Download,
  Upload,
  Eye,
  EyeOff,
  Zap,
  Target,
  Activity,
  TrendingUp,
  BarChart3,
  Play,
  Pause,
  X,
  Check,
  Info,
  Send,
  MessageSquare,
  Mic,
  Bot,
  Building,
  Briefcase,
  Tag,
  FileText,
  List,
  Grid,
  Layers,
  Repeat,
  Star,
  Heart,
} from "lucide-react";

// Types
type AppointmentStatus = "scheduled" | "confirmed" | "in_progress" | "completed" | "cancelled" | "no_show" | "rescheduled";
type AppointmentType = "call" | "video" | "in_person" | "demo" | "consultation" | "follow_up";
type ViewMode = "calendar" | "list" | "timeline";
type CalendarView = "day" | "week" | "month";

interface Appointment {
  id: string;
  title: string;
  description?: string;
  type: AppointmentType;
  status: AppointmentStatus;
  startTime: Date;
  endTime: Date;
  duration: number;
  timezone: string;
  customer: {
    id: string;
    name: string;
    email: string;
    phone?: string;
    company?: string;
  };
  agent?: {
    id: string;
    name: string;
    type: "human" | "ai";
  };
  location?: {
    type: "virtual" | "physical";
    address?: string;
    meetingUrl?: string;
    phoneNumber?: string;
  };
  reminders: {
    type: "email" | "sms" | "both";
    time: number;
    sent: boolean;
  }[];
  notes?: string;
  tags: string[];
  recurring?: {
    frequency: "daily" | "weekly" | "biweekly" | "monthly";
    endDate?: Date;
    count?: number;
  };
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  outcome?: {
    status: "successful" | "unsuccessful" | "rescheduled";
    notes?: string;
    nextSteps?: string[];
  };
}

interface TimeSlot {
  time: string;
  available: boolean;
  appointments: Appointment[];
}

// Mock data
const mockAppointments: Appointment[] = [
  {
    id: "apt-1",
    title: "Product Demo - Enterprise Plan",
    description: "Demonstrate enterprise features and discuss pricing",
    type: "demo",
    status: "confirmed",
    startTime: new Date(Date.now() + 2 * 60 * 60 * 1000),
    endTime: new Date(Date.now() + 3 * 60 * 60 * 1000),
    duration: 60,
    timezone: "America/New_York",
    customer: {
      id: "cust-1",
      name: "Sarah Johnson",
      email: "sarah.johnson@techcorp.com",
      phone: "+1 (555) 123-4567",
      company: "TechCorp Inc.",
    },
    agent: {
      id: "agent-1",
      name: "Sales AI",
      type: "ai",
    },
    location: {
      type: "virtual",
      meetingUrl: "https://meet.bvrai.com/demo-12345",
    },
    reminders: [
      { type: "email", time: 60, sent: true },
      { type: "sms", time: 15, sent: false },
    ],
    notes: "Customer interested in API integration capabilities",
    tags: ["enterprise", "priority", "demo"],
    createdAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
    createdBy: "John Smith",
  },
  {
    id: "apt-2",
    title: "Support Follow-up",
    description: "Follow up on billing issue resolution",
    type: "follow_up",
    status: "scheduled",
    startTime: new Date(Date.now() + 26 * 60 * 60 * 1000),
    endTime: new Date(Date.now() + 26.5 * 60 * 60 * 1000),
    duration: 30,
    timezone: "America/Los_Angeles",
    customer: {
      id: "cust-2",
      name: "Michael Chen",
      email: "m.chen@globalinc.com",
      phone: "+1 (555) 234-5678",
      company: "Global Inc.",
    },
    agent: {
      id: "agent-2",
      name: "Support AI",
      type: "ai",
    },
    location: {
      type: "virtual",
      phoneNumber: "+1 (800) 555-0123",
    },
    reminders: [
      { type: "email", time: 24 * 60, sent: false },
    ],
    tags: ["support", "billing"],
    createdAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
    createdBy: "System",
  },
  {
    id: "apt-3",
    title: "Initial Consultation",
    description: "Discuss requirements and potential solutions",
    type: "consultation",
    status: "completed",
    startTime: new Date(Date.now() - 4 * 60 * 60 * 1000),
    endTime: new Date(Date.now() - 3 * 60 * 60 * 1000),
    duration: 60,
    timezone: "America/Chicago",
    customer: {
      id: "cust-3",
      name: "Emily Davis",
      email: "emily.d@startup.io",
      company: "Startup.io",
    },
    agent: {
      id: "agent-1",
      name: "Sales AI",
      type: "ai",
    },
    location: {
      type: "virtual",
      meetingUrl: "https://meet.bvrai.com/consult-67890",
    },
    reminders: [
      { type: "both", time: 30, sent: true },
    ],
    tags: ["new-lead", "startup"],
    createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 4 * 60 * 60 * 1000),
    createdBy: "Emily Davis",
    outcome: {
      status: "successful",
      notes: "Customer interested, requested pricing proposal",
      nextSteps: ["Send pricing proposal", "Schedule follow-up demo"],
    },
  },
  {
    id: "apt-4",
    title: "Quarterly Review",
    description: "Review usage and discuss upcoming features",
    type: "call",
    status: "in_progress",
    startTime: new Date(Date.now() - 15 * 60 * 1000),
    endTime: new Date(Date.now() + 45 * 60 * 1000),
    duration: 60,
    timezone: "America/New_York",
    customer: {
      id: "cust-4",
      name: "James Wilson",
      email: "jwilson@retail.com",
      phone: "+1 (555) 456-7890",
      company: "Retail Solutions",
    },
    agent: {
      id: "human-1",
      name: "John Smith",
      type: "human",
    },
    location: {
      type: "virtual",
      phoneNumber: "+1 (800) 555-0123",
    },
    reminders: [
      { type: "email", time: 60, sent: true },
      { type: "sms", time: 15, sent: true },
    ],
    tags: ["enterprise", "renewal"],
    recurring: {
      frequency: "monthly",
    },
    createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 15 * 60 * 1000),
    createdBy: "John Smith",
  },
  {
    id: "apt-5",
    title: "Onboarding Session",
    description: "Complete account setup and training",
    type: "video",
    status: "cancelled",
    startTime: new Date(Date.now() - 24 * 60 * 60 * 1000),
    endTime: new Date(Date.now() - 23 * 60 * 60 * 1000),
    duration: 60,
    timezone: "America/Denver",
    customer: {
      id: "cust-5",
      name: "Lisa Martinez",
      email: "lisa.m@healthcare.org",
      company: "Healthcare Solutions",
    },
    agent: {
      id: "agent-3",
      name: "Onboarding AI",
      type: "ai",
    },
    location: {
      type: "virtual",
      meetingUrl: "https://meet.bvrai.com/onboard-11111",
    },
    reminders: [
      { type: "email", time: 24 * 60, sent: true },
    ],
    notes: "Cancelled due to customer request - rescheduling",
    tags: ["onboarding", "new-customer"],
    createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000),
    updatedAt: new Date(Date.now() - 26 * 60 * 60 * 1000),
    createdBy: "System",
  },
];

// Helper functions
const getStatusColor = (status: AppointmentStatus) => {
  switch (status) {
    case "scheduled":
      return "text-blue-400 bg-blue-500/20";
    case "confirmed":
      return "text-green-400 bg-green-500/20";
    case "in_progress":
      return "text-purple-400 bg-purple-500/20";
    case "completed":
      return "text-gray-400 bg-gray-500/20";
    case "cancelled":
      return "text-red-400 bg-red-500/20";
    case "no_show":
      return "text-orange-400 bg-orange-500/20";
    case "rescheduled":
      return "text-yellow-400 bg-yellow-500/20";
  }
};

const getStatusIcon = (status: AppointmentStatus) => {
  switch (status) {
    case "scheduled":
      return <Clock className="w-4 h-4" />;
    case "confirmed":
      return <CheckCircle className="w-4 h-4" />;
    case "in_progress":
      return <Play className="w-4 h-4" />;
    case "completed":
      return <CheckCircle className="w-4 h-4" />;
    case "cancelled":
      return <XCircle className="w-4 h-4" />;
    case "no_show":
      return <AlertTriangle className="w-4 h-4" />;
    case "rescheduled":
      return <RefreshCw className="w-4 h-4" />;
  }
};

const getTypeIcon = (type: AppointmentType) => {
  switch (type) {
    case "call":
      return <Phone className="w-4 h-4" />;
    case "video":
      return <Video className="w-4 h-4" />;
    case "in_person":
      return <MapPin className="w-4 h-4" />;
    case "demo":
      return <Play className="w-4 h-4" />;
    case "consultation":
      return <MessageSquare className="w-4 h-4" />;
    case "follow_up":
      return <Repeat className="w-4 h-4" />;
  }
};

const getTypeColor = (type: AppointmentType) => {
  switch (type) {
    case "call":
      return "text-green-400 bg-green-500/20";
    case "video":
      return "text-blue-400 bg-blue-500/20";
    case "in_person":
      return "text-purple-400 bg-purple-500/20";
    case "demo":
      return "text-pink-400 bg-pink-500/20";
    case "consultation":
      return "text-yellow-400 bg-yellow-500/20";
    case "follow_up":
      return "text-orange-400 bg-orange-500/20";
  }
};

const formatTime = (date: Date) => {
  return date.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
};

const formatDate = (date: Date) => {
  return date.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
};

const formatDuration = (minutes: number) => {
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
};

const isToday = (date: Date) => {
  const today = new Date();
  return date.toDateString() === today.toDateString();
};

const isTomorrow = (date: Date) => {
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  return date.toDateString() === tomorrow.toDateString();
};

// Appointment Card Component
const AppointmentCard: React.FC<{
  appointment: Appointment;
  onClick: () => void;
  onAction: (action: string) => void;
  compact?: boolean;
}> = ({ appointment, onClick, onAction, compact = false }) => {
  const [showMenu, setShowMenu] = useState(false);
  const isLive = appointment.status === "in_progress";
  const isPast = new Date(appointment.endTime) < new Date();

  return (
    <div
      onClick={onClick}
      className={`bg-[#1a1a2e]/80 backdrop-blur-xl border rounded-xl overflow-hidden hover:border-purple-500/30 transition-all duration-300 cursor-pointer ${
        isLive ? "border-purple-500 ring-2 ring-purple-500/20" : "border-white/10"
      } ${compact ? "p-3" : "p-4"}`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <div className={`p-2 rounded-lg ${getTypeColor(appointment.type)}`}>
            {getTypeIcon(appointment.type)}
          </div>

          <div>
            <div className="flex items-center gap-2">
              <h3 className={`font-semibold text-white ${compact ? "text-sm" : ""}`}>{appointment.title}</h3>
              {isLive && (
                <span className="px-1.5 py-0.5 text-xs bg-purple-500 text-white rounded animate-pulse">LIVE</span>
              )}
            </div>

            <div className="flex items-center gap-2 mt-1">
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium flex items-center gap-1 ${getStatusColor(appointment.status)}`}>
                {getStatusIcon(appointment.status)}
                {appointment.status.replace("_", " ")}
              </span>
              {appointment.recurring && (
                <span className="px-2 py-0.5 rounded text-xs bg-white/5 text-gray-400 flex items-center gap-1">
                  <Repeat className="w-3 h-3" />
                  {appointment.recurring.frequency}
                </span>
              )}
            </div>

            {!compact && (
              <div className="flex items-center gap-4 mt-2 text-sm text-gray-400">
                <span className="flex items-center gap-1">
                  <User className="w-3.5 h-3.5" />
                  {appointment.customer.name}
                </span>
                {appointment.customer.company && (
                  <span className="flex items-center gap-1">
                    <Building className="w-3.5 h-3.5" />
                    {appointment.customer.company}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-start gap-2">
          <div className="text-right">
            <p className={`text-white font-medium ${compact ? "text-sm" : ""}`}>
              {formatTime(appointment.startTime)}
            </p>
            <p className="text-xs text-gray-500">
              {isToday(appointment.startTime) ? "Today" : isTomorrow(appointment.startTime) ? "Tomorrow" : formatDate(appointment.startTime)}
            </p>
          </div>

          <div className="relative">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              className="p-1.5 rounded-lg hover:bg-white/10 text-gray-400"
            >
              <MoreVertical className="w-4 h-4" />
            </button>

            {showMenu && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setShowMenu(false)} />
                <div className="absolute right-0 top-full mt-1 w-44 bg-[#252540] border border-white/10 rounded-lg shadow-xl z-20 py-1">
                  {!isPast && appointment.status !== "cancelled" && (
                    <>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onAction("join");
                          setShowMenu(false);
                        }}
                        className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                      >
                        <Video className="w-4 h-4" />
                        Join Meeting
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onAction("reschedule");
                          setShowMenu(false);
                        }}
                        className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                      >
                        <CalendarClock className="w-4 h-4" />
                        Reschedule
                      </button>
                    </>
                  )}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onAction("edit");
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/10 flex items-center gap-2"
                  >
                    <Edit className="w-4 h-4" />
                    Edit
                  </button>
                  {!isPast && appointment.status !== "cancelled" && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onAction("cancel");
                        setShowMenu(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-red-400 hover:bg-red-500/10 flex items-center gap-2"
                    >
                      <XCircle className="w-4 h-4" />
                      Cancel
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {!compact && (
        <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <span className="flex items-center gap-1">
              <Clock className="w-3.5 h-3.5" />
              {formatDuration(appointment.duration)}
            </span>
            {appointment.agent && (
              <span className="flex items-center gap-1">
                {appointment.agent.type === "ai" ? <Bot className="w-3.5 h-3.5" /> : <User className="w-3.5 h-3.5" />}
                {appointment.agent.name}
              </span>
            )}
          </div>

          {appointment.tags.length > 0 && (
            <div className="flex gap-1">
              {appointment.tags.slice(0, 2).map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-0.5 rounded text-xs bg-white/5 text-gray-400"
                >
                  {tag}
                </span>
              ))}
              {appointment.tags.length > 2 && (
                <span className="text-xs text-gray-500">+{appointment.tags.length - 2}</span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Create Appointment Dialog
const CreateAppointmentDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onSave: (appointment: Partial<Appointment>) => void;
}> = ({ isOpen, onClose, onSave }) => {
  const [appointmentData, setAppointmentData] = useState<Partial<Appointment>>({
    title: "",
    description: "",
    type: "call",
    duration: 30,
    timezone: "America/New_York",
    customer: {
      id: "",
      name: "",
      email: "",
    },
    reminders: [{ type: "email", time: 60, sent: false }],
    tags: [],
  });

  if (!isOpen) return null;

  const appointmentTypes: { value: AppointmentType; label: string }[] = [
    { value: "call", label: "Phone Call" },
    { value: "video", label: "Video Meeting" },
    { value: "demo", label: "Product Demo" },
    { value: "consultation", label: "Consultation" },
    { value: "follow_up", label: "Follow-up" },
    { value: "in_person", label: "In Person" },
  ];

  const durations = [15, 30, 45, 60, 90, 120];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-lg bg-[#1a1a2e] border border-white/10 rounded-2xl overflow-hidden">
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">Schedule Appointment</h2>
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10 text-gray-400">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="p-6 max-h-[60vh] overflow-y-auto space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Title</label>
            <input
              type="text"
              value={appointmentData.title}
              onChange={(e) => setAppointmentData({ ...appointmentData, title: e.target.value })}
              className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              placeholder="e.g., Product Demo"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Type</label>
            <div className="grid grid-cols-3 gap-2">
              {appointmentTypes.map((type) => (
                <button
                  key={type.value}
                  onClick={() => setAppointmentData({ ...appointmentData, type: type.value })}
                  className={`p-2 rounded-lg border text-sm transition-all flex items-center justify-center gap-2 ${
                    appointmentData.type === type.value
                      ? "border-purple-500 bg-purple-500/20 text-white"
                      : "border-white/10 text-gray-400 hover:border-white/20"
                  }`}
                >
                  {getTypeIcon(type.value)}
                  {type.label}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Date</label>
              <input
                type="date"
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Time</label>
              <input
                type="time"
                className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Duration</label>
            <div className="flex gap-2">
              {durations.map((d) => (
                <button
                  key={d}
                  onClick={() => setAppointmentData({ ...appointmentData, duration: d })}
                  className={`flex-1 py-2 rounded-lg border text-sm transition-all ${
                    appointmentData.duration === d
                      ? "border-purple-500 bg-purple-500/20 text-white"
                      : "border-white/10 text-gray-400 hover:border-white/20"
                  }`}
                >
                  {formatDuration(d)}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Customer Email</label>
            <input
              type="email"
              value={appointmentData.customer?.email || ""}
              onChange={(e) => setAppointmentData({
                ...appointmentData,
                customer: { ...appointmentData.customer!, email: e.target.value },
              })}
              className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              placeholder="customer@example.com"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Description (Optional)</label>
            <textarea
              value={appointmentData.description || ""}
              onChange={(e) => setAppointmentData({ ...appointmentData, description: e.target.value })}
              className="w-full px-4 py-2 bg-black/30 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
              rows={3}
              placeholder="Add any notes or agenda items..."
            />
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
            onClick={() => onSave(appointmentData)}
            className="px-6 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium hover:opacity-90 transition-opacity"
          >
            Schedule
          </button>
        </div>
      </div>
    </div>
  );
};

// Main Component
export default function AppointmentsPage() {
  const [appointments, setAppointments] = useState<Appointment[]>(mockAppointments);
  const [viewMode, setViewMode] = useState<ViewMode>("list");
  const [calendarView, setCalendarView] = useState<CalendarView>("week");
  const [currentDate, setCurrentDate] = useState(new Date());
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<AppointmentStatus | "all">("all");
  const [showCreateDialog, setShowCreateDialog] = useState(false);

  // Filter appointments
  const filteredAppointments = useMemo(() => {
    return appointments.filter((apt) => {
      const matchesSearch =
        apt.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        apt.customer.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        apt.customer.email.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesStatus = statusFilter === "all" || apt.status === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [appointments, searchQuery, statusFilter]);

  // Group appointments by date
  const groupedAppointments = useMemo(() => {
    const groups: Record<string, Appointment[]> = {};

    filteredAppointments
      .sort((a, b) => new Date(a.startTime).getTime() - new Date(b.startTime).getTime())
      .forEach((apt) => {
        const dateKey = new Date(apt.startTime).toDateString();
        if (!groups[dateKey]) {
          groups[dateKey] = [];
        }
        groups[dateKey].push(apt);
      });

    return groups;
  }, [filteredAppointments]);

  // Stats
  const stats = useMemo(() => {
    const total = appointments.length;
    const upcoming = appointments.filter((a) => new Date(a.startTime) > new Date() && a.status !== "cancelled").length;
    const completed = appointments.filter((a) => a.status === "completed").length;
    const today = appointments.filter((a) => isToday(new Date(a.startTime)) && a.status !== "cancelled").length;

    return { total, upcoming, completed, today };
  }, [appointments]);

  const handleAction = (appointmentId: string, action: string) => {
    console.log(`Action ${action} for appointment ${appointmentId}`);
  };

  const handleSave = (data: Partial<Appointment>) => {
    console.log("Save appointment:", data);
    setShowCreateDialog(false);
  };

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Appointments</h1>
            <p className="text-gray-400 mt-1">Manage and schedule appointments with customers</p>
          </div>

          <button
            onClick={() => setShowCreateDialog(true)}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90 transition-opacity"
          >
            <Plus className="w-4 h-4" />
            Schedule Appointment
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400">
                <Calendar className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.total}</p>
                <p className="text-sm text-gray-400">Total</p>
              </div>
            </div>
          </div>

          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
                <CalendarDays className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.today}</p>
                <p className="text-sm text-gray-400">Today</p>
              </div>
            </div>
          </div>

          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-500/20 text-green-400">
                <CalendarClock className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.upcoming}</p>
                <p className="text-sm text-gray-400">Upcoming</p>
              </div>
            </div>
          </div>

          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gray-500/20 text-gray-400">
                <CheckCircle className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{stats.completed}</p>
                <p className="text-sm text-gray-400">Completed</p>
              </div>
            </div>
          </div>
        </div>

        {/* Filters & View Toggle */}
        <div className="flex flex-col md:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search appointments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
            />
          </div>

          <div className="flex gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as AppointmentStatus | "all")}
              className="px-4 py-2 bg-[#1a1a2e]/80 border border-white/10 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value="all">All Statuses</option>
              <option value="scheduled">Scheduled</option>
              <option value="confirmed">Confirmed</option>
              <option value="in_progress">In Progress</option>
              <option value="completed">Completed</option>
              <option value="cancelled">Cancelled</option>
            </select>

            <div className="flex border border-white/10 rounded-lg overflow-hidden">
              <button
                onClick={() => setViewMode("list")}
                className={`p-2 ${viewMode === "list" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:bg-white/10"}`}
              >
                <List className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode("calendar")}
                className={`p-2 ${viewMode === "calendar" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:bg-white/10"}`}
              >
                <Calendar className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Appointments List View */}
        {viewMode === "list" && (
          <div className="space-y-6">
            {Object.entries(groupedAppointments).map(([dateKey, dayAppointments]) => {
              const date = new Date(dateKey);
              return (
                <div key={dateKey}>
                  <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2">
                    {isToday(date) ? (
                      <span className="text-purple-400">Today</span>
                    ) : isTomorrow(date) ? (
                      <span className="text-blue-400">Tomorrow</span>
                    ) : (
                      formatDate(date)
                    )}
                    <span className="text-gray-600">({dayAppointments.length})</span>
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {dayAppointments.map((appointment) => (
                      <AppointmentCard
                        key={appointment.id}
                        appointment={appointment}
                        onClick={() => console.log("View appointment", appointment.id)}
                        onAction={(action) => handleAction(appointment.id, action)}
                      />
                    ))}
                  </div>
                </div>
              );
            })}

            {Object.keys(groupedAppointments).length === 0 && (
              <div className="text-center py-16 bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl">
                <Calendar className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-white mb-2">No appointments found</h3>
                <p className="text-gray-400 mb-6">
                  {searchQuery || statusFilter !== "all"
                    ? "No appointments match your current filters"
                    : "Schedule your first appointment to get started"}
                </p>
                <button
                  onClick={() => setShowCreateDialog(true)}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  Schedule Appointment
                </button>
              </div>
            )}
          </div>
        )}

        {/* Calendar View */}
        {viewMode === "calendar" && (
          <div className="bg-[#1a1a2e]/80 backdrop-blur-xl border border-white/10 rounded-xl p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => {
                    const newDate = new Date(currentDate);
                    newDate.setMonth(newDate.getMonth() - 1);
                    setCurrentDate(newDate);
                  }}
                  className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <h2 className="text-lg font-semibold text-white">
                  {currentDate.toLocaleDateString("en-US", { month: "long", year: "numeric" })}
                </h2>
                <button
                  onClick={() => {
                    const newDate = new Date(currentDate);
                    newDate.setMonth(newDate.getMonth() + 1);
                    setCurrentDate(newDate);
                  }}
                  className="p-2 rounded-lg hover:bg-white/10 text-gray-400"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
              </div>

              <div className="flex gap-2">
                {(["day", "week", "month"] as CalendarView[]).map((view) => (
                  <button
                    key={view}
                    onClick={() => setCalendarView(view)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      calendarView === view
                        ? "bg-purple-500/20 text-purple-400"
                        : "text-gray-400 hover:bg-white/10"
                    }`}
                  >
                    {view.charAt(0).toUpperCase() + view.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Simple Week View */}
            <div className="grid grid-cols-7 gap-2">
              {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((day) => (
                <div key={day} className="text-center text-sm text-gray-500 py-2">
                  {day}
                </div>
              ))}

              {Array.from({ length: 35 }, (_, i) => {
                const firstDay = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
                const startOffset = firstDay.getDay();
                const dayNumber = i - startOffset + 1;
                const date = new Date(currentDate.getFullYear(), currentDate.getMonth(), dayNumber);
                const isCurrentMonth = date.getMonth() === currentDate.getMonth();
                const isTodayDate = isToday(date);
                const dayAppointments = appointments.filter(
                  (a) => new Date(a.startTime).toDateString() === date.toDateString()
                );

                return (
                  <div
                    key={i}
                    className={`min-h-24 p-2 rounded-lg border ${
                      isTodayDate
                        ? "border-purple-500 bg-purple-500/10"
                        : "border-white/5 hover:border-white/10"
                    } ${!isCurrentMonth ? "opacity-40" : ""}`}
                  >
                    <p className={`text-sm ${isTodayDate ? "text-purple-400 font-semibold" : "text-gray-400"}`}>
                      {date.getDate()}
                    </p>
                    <div className="mt-1 space-y-1">
                      {dayAppointments.slice(0, 2).map((apt) => (
                        <div
                          key={apt.id}
                          className={`text-xs px-1.5 py-0.5 rounded truncate ${getTypeColor(apt.type)}`}
                          title={apt.title}
                        >
                          {formatTime(apt.startTime)} {apt.title}
                        </div>
                      ))}
                      {dayAppointments.length > 2 && (
                        <p className="text-xs text-gray-500">+{dayAppointments.length - 2} more</p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Create Dialog */}
        <CreateAppointmentDialog
          isOpen={showCreateDialog}
          onClose={() => setShowCreateDialog(false)}
          onSave={handleSave}
        />
      </div>
    </DashboardLayout>
  );
}
