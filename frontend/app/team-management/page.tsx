"use client";

import React, { useState, useMemo } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import {
  Users,
  User,
  UserPlus,
  UserMinus,
  UserCheck,
  UserX,
  Shield,
  ShieldCheck,
  ShieldAlert,
  Crown,
  Star,
  Mail,
  Phone,
  Calendar,
  Clock,
  MapPin,
  Building2,
  Briefcase,
  Activity,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Settings,
  Edit3,
  Trash2,
  MoreVertical,
  Search,
  Filter,
  Plus,
  X,
  Check,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Copy,
  Send,
  RefreshCw,
  Download,
  Upload,
  Link,
  ExternalLink,
  Globe,
  Zap,
  Target,
  Award,
  MessageSquare,
  Bell,
  Lock,
  Unlock,
  Key,
  History,
  LogIn,
  LogOut,
  Monitor,
  Smartphone,
  Laptop
} from 'lucide-react';

// Types
interface TeamMember {
  id: string;
  name: string;
  email: string;
  phone?: string;
  avatar?: string;
  role: UserRole;
  department: string;
  title: string;
  status: 'active' | 'inactive' | 'pending' | 'suspended';
  joinedAt: string;
  lastActive: string;
  location?: string;
  timezone?: string;
  permissions: string[];
  metrics: MemberMetrics;
  twoFactorEnabled: boolean;
  sessions: UserSession[];
}

interface UserRole {
  id: string;
  name: string;
  level: 'owner' | 'admin' | 'manager' | 'member' | 'viewer';
  color: string;
  permissions: string[];
}

interface MemberMetrics {
  callsHandled: number;
  avgHandleTime: number;
  successRate: number;
  customerSatisfaction: number;
  responseTime: number;
  hoursWorked: number;
}

interface UserSession {
  id: string;
  device: string;
  browser: string;
  location: string;
  ipAddress: string;
  lastActivity: string;
  current: boolean;
}

interface Department {
  id: string;
  name: string;
  description: string;
  memberCount: number;
  lead?: string;
  color: string;
}

interface Invitation {
  id: string;
  email: string;
  role: string;
  department: string;
  sentAt: string;
  expiresAt: string;
  status: 'pending' | 'accepted' | 'expired';
  sentBy: string;
}

interface ActivityLog {
  id: string;
  userId: string;
  userName: string;
  action: string;
  target: string;
  timestamp: string;
  details?: string;
}

// Mock Data
const roles: UserRole[] = [
  { id: 'owner', name: 'Owner', level: 'owner', color: 'yellow', permissions: ['*'] },
  { id: 'admin', name: 'Administrator', level: 'admin', color: 'red', permissions: ['manage_team', 'manage_agents', 'manage_billing', 'view_analytics', 'manage_settings'] },
  { id: 'manager', name: 'Manager', level: 'manager', color: 'purple', permissions: ['manage_team', 'manage_agents', 'view_analytics'] },
  { id: 'member', name: 'Team Member', level: 'member', color: 'blue', permissions: ['manage_agents', 'view_analytics'] },
  { id: 'viewer', name: 'Viewer', level: 'viewer', color: 'gray', permissions: ['view_analytics'] },
];

const departments: Department[] = [
  { id: 'engineering', name: 'Engineering', description: 'Technical development and infrastructure', memberCount: 12, lead: 'John Smith', color: 'blue' },
  { id: 'sales', name: 'Sales', description: 'Revenue generation and client acquisition', memberCount: 8, lead: 'Sarah Johnson', color: 'green' },
  { id: 'support', name: 'Customer Support', description: 'Customer assistance and satisfaction', memberCount: 15, lead: 'Mike Wilson', color: 'purple' },
  { id: 'marketing', name: 'Marketing', description: 'Brand promotion and lead generation', memberCount: 6, lead: 'Emily Chen', color: 'pink' },
  { id: 'operations', name: 'Operations', description: 'Day-to-day business operations', memberCount: 5, lead: 'Alex Turner', color: 'orange' },
];

const mockMembers: TeamMember[] = [
  {
    id: 'user-1',
    name: 'John Smith',
    email: 'john.smith@company.com',
    phone: '+1 (555) 123-4567',
    role: roles[0],
    department: 'Engineering',
    title: 'CEO & Founder',
    status: 'active',
    joinedAt: '2023-01-15T10:00:00Z',
    lastActive: '2024-01-20T14:30:00Z',
    location: 'San Francisco, CA',
    timezone: 'America/Los_Angeles',
    permissions: ['*'],
    metrics: { callsHandled: 0, avgHandleTime: 0, successRate: 0, customerSatisfaction: 0, responseTime: 0, hoursWorked: 160 },
    twoFactorEnabled: true,
    sessions: [
      { id: 's1', device: 'MacBook Pro', browser: 'Chrome 120', location: 'San Francisco, CA', ipAddress: '192.168.1.1', lastActivity: '2024-01-20T14:30:00Z', current: true },
      { id: 's2', device: 'iPhone 15', browser: 'Safari', location: 'San Francisco, CA', ipAddress: '192.168.1.2', lastActivity: '2024-01-20T12:00:00Z', current: false },
    ]
  },
  {
    id: 'user-2',
    name: 'Sarah Johnson',
    email: 'sarah.johnson@company.com',
    phone: '+1 (555) 234-5678',
    role: roles[1],
    department: 'Sales',
    title: 'VP of Sales',
    status: 'active',
    joinedAt: '2023-02-20T09:00:00Z',
    lastActive: '2024-01-20T13:45:00Z',
    location: 'New York, NY',
    timezone: 'America/New_York',
    permissions: ['manage_team', 'manage_agents', 'manage_billing', 'view_analytics', 'manage_settings'],
    metrics: { callsHandled: 245, avgHandleTime: 320, successRate: 92, customerSatisfaction: 4.8, responseTime: 15, hoursWorked: 168 },
    twoFactorEnabled: true,
    sessions: [
      { id: 's3', device: 'Windows PC', browser: 'Edge 120', location: 'New York, NY', ipAddress: '192.168.2.1', lastActivity: '2024-01-20T13:45:00Z', current: true },
    ]
  },
  {
    id: 'user-3',
    name: 'Mike Wilson',
    email: 'mike.wilson@company.com',
    phone: '+1 (555) 345-6789',
    role: roles[2],
    department: 'Customer Support',
    title: 'Support Manager',
    status: 'active',
    joinedAt: '2023-03-10T11:00:00Z',
    lastActive: '2024-01-20T14:15:00Z',
    location: 'Chicago, IL',
    timezone: 'America/Chicago',
    permissions: ['manage_team', 'manage_agents', 'view_analytics'],
    metrics: { callsHandled: 890, avgHandleTime: 280, successRate: 95, customerSatisfaction: 4.9, responseTime: 8, hoursWorked: 172 },
    twoFactorEnabled: true,
    sessions: [
      { id: 's4', device: 'MacBook Air', browser: 'Firefox 121', location: 'Chicago, IL', ipAddress: '192.168.3.1', lastActivity: '2024-01-20T14:15:00Z', current: true },
    ]
  },
  {
    id: 'user-4',
    name: 'Emily Chen',
    email: 'emily.chen@company.com',
    role: roles[2],
    department: 'Marketing',
    title: 'Marketing Manager',
    status: 'active',
    joinedAt: '2023-04-05T10:00:00Z',
    lastActive: '2024-01-20T11:30:00Z',
    location: 'Los Angeles, CA',
    timezone: 'America/Los_Angeles',
    permissions: ['manage_team', 'manage_agents', 'view_analytics'],
    metrics: { callsHandled: 120, avgHandleTime: 450, successRate: 88, customerSatisfaction: 4.6, responseTime: 25, hoursWorked: 158 },
    twoFactorEnabled: false,
    sessions: []
  },
  {
    id: 'user-5',
    name: 'Alex Turner',
    email: 'alex.turner@company.com',
    role: roles[3],
    department: 'Operations',
    title: 'Operations Analyst',
    status: 'active',
    joinedAt: '2023-05-15T09:00:00Z',
    lastActive: '2024-01-20T10:00:00Z',
    location: 'Austin, TX',
    timezone: 'America/Chicago',
    permissions: ['manage_agents', 'view_analytics'],
    metrics: { callsHandled: 456, avgHandleTime: 310, successRate: 91, customerSatisfaction: 4.7, responseTime: 12, hoursWorked: 165 },
    twoFactorEnabled: true,
    sessions: []
  },
  {
    id: 'user-6',
    name: 'Jessica Brown',
    email: 'jessica.brown@company.com',
    role: roles[3],
    department: 'Customer Support',
    title: 'Support Specialist',
    status: 'active',
    joinedAt: '2023-06-20T08:00:00Z',
    lastActive: '2024-01-20T09:45:00Z',
    location: 'Seattle, WA',
    timezone: 'America/Los_Angeles',
    permissions: ['manage_agents', 'view_analytics'],
    metrics: { callsHandled: 678, avgHandleTime: 260, successRate: 94, customerSatisfaction: 4.8, responseTime: 10, hoursWorked: 170 },
    twoFactorEnabled: true,
    sessions: []
  },
  {
    id: 'user-7',
    name: 'David Kim',
    email: 'david.kim@company.com',
    role: roles[3],
    department: 'Engineering',
    title: 'Software Engineer',
    status: 'inactive',
    joinedAt: '2023-07-10T10:00:00Z',
    lastActive: '2024-01-15T16:00:00Z',
    location: 'Boston, MA',
    timezone: 'America/New_York',
    permissions: ['manage_agents', 'view_analytics'],
    metrics: { callsHandled: 0, avgHandleTime: 0, successRate: 0, customerSatisfaction: 0, responseTime: 0, hoursWorked: 140 },
    twoFactorEnabled: false,
    sessions: []
  },
  {
    id: 'user-8',
    name: 'Rachel Green',
    email: 'rachel.green@company.com',
    role: roles[4],
    department: 'Sales',
    title: 'Sales Representative',
    status: 'pending',
    joinedAt: '2024-01-18T09:00:00Z',
    lastActive: '2024-01-18T09:00:00Z',
    location: 'Denver, CO',
    timezone: 'America/Denver',
    permissions: ['view_analytics'],
    metrics: { callsHandled: 0, avgHandleTime: 0, successRate: 0, customerSatisfaction: 0, responseTime: 0, hoursWorked: 0 },
    twoFactorEnabled: false,
    sessions: []
  }
];

const mockInvitations: Invitation[] = [
  { id: 'inv-1', email: 'newuser@company.com', role: 'Team Member', department: 'Engineering', sentAt: '2024-01-19T10:00:00Z', expiresAt: '2024-01-26T10:00:00Z', status: 'pending', sentBy: 'John Smith' },
  { id: 'inv-2', email: 'contractor@external.com', role: 'Viewer', department: 'Marketing', sentAt: '2024-01-18T14:00:00Z', expiresAt: '2024-01-25T14:00:00Z', status: 'pending', sentBy: 'Emily Chen' },
  { id: 'inv-3', email: 'olduser@company.com', role: 'Team Member', department: 'Sales', sentAt: '2024-01-10T09:00:00Z', expiresAt: '2024-01-17T09:00:00Z', status: 'expired', sentBy: 'Sarah Johnson' },
];

const mockActivityLog: ActivityLog[] = [
  { id: 'log-1', userId: 'user-1', userName: 'John Smith', action: 'Invited user', target: 'newuser@company.com', timestamp: '2024-01-19T10:00:00Z' },
  { id: 'log-2', userId: 'user-2', userName: 'Sarah Johnson', action: 'Updated role', target: 'Alex Turner', timestamp: '2024-01-19T09:30:00Z', details: 'Changed from Viewer to Team Member' },
  { id: 'log-3', userId: 'user-3', userName: 'Mike Wilson', action: 'Removed user', target: 'old.employee@company.com', timestamp: '2024-01-18T16:00:00Z' },
  { id: 'log-4', userId: 'user-1', userName: 'John Smith', action: 'Created department', target: 'Operations', timestamp: '2024-01-18T11:00:00Z' },
  { id: 'log-5', userId: 'user-4', userName: 'Emily Chen', action: 'Sent invitation', target: 'contractor@external.com', timestamp: '2024-01-18T14:00:00Z' },
];

const allPermissions = [
  { id: 'manage_team', name: 'Manage Team', description: 'Add, remove, and manage team members', category: 'Team' },
  { id: 'manage_agents', name: 'Manage Agents', description: 'Create and configure AI agents', category: 'Agents' },
  { id: 'manage_billing', name: 'Manage Billing', description: 'Access billing and payment settings', category: 'Billing' },
  { id: 'view_analytics', name: 'View Analytics', description: 'Access analytics and reports', category: 'Analytics' },
  { id: 'manage_settings', name: 'Manage Settings', description: 'Configure account settings', category: 'Settings' },
  { id: 'manage_integrations', name: 'Manage Integrations', description: 'Connect and manage integrations', category: 'Integrations' },
  { id: 'manage_workflows', name: 'Manage Workflows', description: 'Create and edit workflows', category: 'Workflows' },
  { id: 'export_data', name: 'Export Data', description: 'Export data and reports', category: 'Data' },
];

export default function TeamManagementPage() {
  const [members, setMembers] = useState<TeamMember[]>(mockMembers);
  const [invitations, setInvitations] = useState<Invitation[]>(mockInvitations);
  const [activeTab, setActiveTab] = useState<'members' | 'invitations' | 'roles' | 'departments' | 'activity'>('members');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'inactive' | 'pending'>('all');
  const [filterDepartment, setFilterDepartment] = useState<string>('all');
  const [filterRole, setFilterRole] = useState<string>('all');
  const [selectedMember, setSelectedMember] = useState<TeamMember | null>(null);
  const [showMemberDialog, setShowMemberDialog] = useState(false);
  const [showInviteDialog, setShowInviteDialog] = useState(false);
  const [showRoleDialog, setShowRoleDialog] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  // Invite form state
  const [inviteForm, setInviteForm] = useState({
    email: '',
    role: 'member',
    department: 'Engineering',
    message: ''
  });

  const filteredMembers = useMemo(() => {
    return members.filter(member => {
      const matchesSearch = member.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           member.email.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = filterStatus === 'all' || member.status === filterStatus;
      const matchesDepartment = filterDepartment === 'all' || member.department === filterDepartment;
      const matchesRole = filterRole === 'all' || member.role.id === filterRole;
      return matchesSearch && matchesStatus && matchesDepartment && matchesRole;
    });
  }, [members, searchQuery, filterStatus, filterDepartment, filterRole]);

  const stats = useMemo(() => ({
    total: members.length,
    active: members.filter(m => m.status === 'active').length,
    pending: members.filter(m => m.status === 'pending').length + invitations.filter(i => i.status === 'pending').length,
    avgSatisfaction: members.filter(m => m.metrics.customerSatisfaction > 0).reduce((sum, m) => sum + m.metrics.customerSatisfaction, 0) / members.filter(m => m.metrics.customerSatisfaction > 0).length || 0
  }), [members, invitations]);

  const getRoleIcon = (level: string) => {
    switch (level) {
      case 'owner': return Crown;
      case 'admin': return ShieldCheck;
      case 'manager': return Shield;
      case 'member': return User;
      case 'viewer': return Eye;
      default: return User;
    }
  };

  const getRoleColor = (color: string) => {
    const colors: Record<string, string> = {
      yellow: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      red: 'bg-red-500/20 text-red-400 border-red-500/30',
      purple: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
      blue: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
      gray: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
    };
    return colors[color] || colors.gray;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'inactive': return 'bg-gray-500';
      case 'pending': return 'bg-yellow-500';
      case 'suspended': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const formatTimeAgo = (dateStr: string) => {
    const now = new Date();
    const date = new Date(dateStr);
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return formatDate(dateStr);
  };

  const updateMemberStatus = (memberId: string, status: TeamMember['status']) => {
    setMembers(prev => prev.map(m =>
      m.id === memberId ? { ...m, status } : m
    ));
  };

  const updateMemberRole = (memberId: string, roleId: string) => {
    const newRole = roles.find(r => r.id === roleId);
    if (newRole) {
      setMembers(prev => prev.map(m =>
        m.id === memberId ? { ...m, role: newRole } : m
      ));
    }
  };

  const removeMember = (memberId: string) => {
    setMembers(prev => prev.filter(m => m.id !== memberId));
  };

  const sendInvitation = () => {
    if (!inviteForm.email) return;

    const newInvitation: Invitation = {
      id: `inv-${Date.now()}`,
      email: inviteForm.email,
      role: roles.find(r => r.id === inviteForm.role)?.name || 'Team Member',
      department: inviteForm.department,
      sentAt: new Date().toISOString(),
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      status: 'pending',
      sentBy: 'Current User'
    };

    setInvitations(prev => [...prev, newInvitation]);
    setInviteForm({ email: '', role: 'member', department: 'Engineering', message: '' });
    setShowInviteDialog(false);
  };

  const resendInvitation = (invitationId: string) => {
    setInvitations(prev => prev.map(inv =>
      inv.id === invitationId
        ? { ...inv, sentAt: new Date().toISOString(), expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() }
        : inv
    ));
  };

  const cancelInvitation = (invitationId: string) => {
    setInvitations(prev => prev.filter(inv => inv.id !== invitationId));
  };

  // Member Card Component
  const MemberCard = ({ member }: { member: TeamMember }) => {
    const RoleIcon = getRoleIcon(member.role.level);

    return (
      <div className="bg-gray-800/50 rounded-xl border border-gray-700 hover:border-purple-500/50 transition-all p-4 group">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            {/* Avatar */}
            <div className="relative">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white font-bold text-lg">
                {member.name.split(' ').map(n => n[0]).join('')}
              </div>
              <div className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-gray-800 ${getStatusColor(member.status)}`} />
            </div>
            <div>
              <h3 className="font-medium text-white">{member.name}</h3>
              <p className="text-sm text-gray-400">{member.title}</p>
            </div>
          </div>
          <button className="p-1.5 hover:bg-gray-700 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity">
            <MoreVertical className="w-4 h-4 text-gray-400" />
          </button>
        </div>

        {/* Role & Department */}
        <div className="flex items-center gap-2 mb-4">
          <span className={`px-2 py-1 rounded-lg text-xs border flex items-center gap-1 ${getRoleColor(member.role.color)}`}>
            <RoleIcon className="w-3 h-3" />
            {member.role.name}
          </span>
          <span className="px-2 py-1 bg-gray-700/50 rounded-lg text-xs text-gray-400">
            {member.department}
          </span>
        </div>

        {/* Contact */}
        <div className="space-y-2 mb-4 text-sm">
          <div className="flex items-center gap-2 text-gray-400">
            <Mail className="w-4 h-4" />
            <span className="truncate">{member.email}</span>
          </div>
          {member.phone && (
            <div className="flex items-center gap-2 text-gray-400">
              <Phone className="w-4 h-4" />
              <span>{member.phone}</span>
            </div>
          )}
          {member.location && (
            <div className="flex items-center gap-2 text-gray-400">
              <MapPin className="w-4 h-4" />
              <span>{member.location}</span>
            </div>
          )}
        </div>

        {/* Metrics */}
        {member.metrics.callsHandled > 0 && (
          <div className="grid grid-cols-2 gap-2 mb-4 p-3 bg-gray-900/50 rounded-lg">
            <div>
              <p className="text-lg font-semibold text-white">{member.metrics.callsHandled}</p>
              <p className="text-xs text-gray-500">Calls</p>
            </div>
            <div>
              <p className="text-lg font-semibold text-green-400">{member.metrics.successRate}%</p>
              <p className="text-xs text-gray-500">Success</p>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-3 border-t border-gray-700">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <Clock className="w-3 h-3" />
            <span>Active {formatTimeAgo(member.lastActive)}</span>
          </div>
          <div className="flex items-center gap-1">
            {member.twoFactorEnabled && (
              <div className="p-1.5 text-green-400" title="2FA Enabled">
                <ShieldCheck className="w-4 h-4" />
              </div>
            )}
            <button
              onClick={() => { setSelectedMember(member); setShowMemberDialog(true); }}
              className="p-1.5 hover:bg-gray-700 rounded-lg transition-colors"
            >
              <Eye className="w-4 h-4 text-gray-400" />
            </button>
            <button className="p-1.5 hover:bg-gray-700 rounded-lg transition-colors">
              <Edit3 className="w-4 h-4 text-gray-400" />
            </button>
          </div>
        </div>
      </div>
    );
  };

  // Member List Item
  const MemberListItem = ({ member }: { member: TeamMember }) => {
    const RoleIcon = getRoleIcon(member.role.level);

    return (
      <div className="bg-gray-800/50 rounded-xl border border-gray-700 hover:border-purple-500/50 transition-all p-4 flex items-center gap-4 group">
        {/* Avatar */}
        <div className="relative flex-shrink-0">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white font-bold">
            {member.name.split(' ').map(n => n[0]).join('')}
          </div>
          <div className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-gray-800 ${getStatusColor(member.status)}`} />
        </div>

        {/* Info */}
        <div className="flex-grow min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-medium text-white">{member.name}</h3>
            <span className={`px-2 py-0.5 rounded text-xs border flex items-center gap-1 ${getRoleColor(member.role.color)}`}>
              <RoleIcon className="w-3 h-3" />
              {member.role.name}
            </span>
            {member.twoFactorEnabled && (
              <ShieldCheck className="w-4 h-4 text-green-400" title="2FA Enabled" />
            )}
          </div>
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <span>{member.email}</span>
            <span className="text-gray-600">•</span>
            <span>{member.department}</span>
            <span className="text-gray-600">•</span>
            <span>{member.title}</span>
          </div>
        </div>

        {/* Metrics */}
        <div className="flex items-center gap-6 flex-shrink-0">
          {member.metrics.callsHandled > 0 && (
            <>
              <div className="text-center">
                <p className="font-semibold text-white">{member.metrics.callsHandled}</p>
                <p className="text-xs text-gray-500">Calls</p>
              </div>
              <div className="text-center">
                <p className="font-semibold text-green-400">{member.metrics.successRate}%</p>
                <p className="text-xs text-gray-500">Success</p>
              </div>
              <div className="text-center">
                <p className="font-semibold text-white">{member.metrics.customerSatisfaction}</p>
                <p className="text-xs text-gray-500">CSAT</p>
              </div>
            </>
          )}
        </div>

        {/* Last Active */}
        <div className="flex-shrink-0 text-right">
          <p className="text-sm text-gray-400">{formatTimeAgo(member.lastActive)}</p>
          <p className="text-xs text-gray-500">Last active</p>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => { setSelectedMember(member); setShowMemberDialog(true); }}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          >
            <Eye className="w-4 h-4 text-gray-400" />
          </button>
          <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
            <Edit3 className="w-4 h-4 text-gray-400" />
          </button>
          <button
            onClick={() => removeMember(member.id)}
            className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
          >
            <UserMinus className="w-4 h-4 text-red-400" />
          </button>
        </div>
      </div>
    );
  };

  // Invitation Row
  const InvitationRow = ({ invitation }: { invitation: Invitation }) => (
    <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-xl border border-gray-700">
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center">
          <Mail className="w-5 h-5 text-gray-400" />
        </div>
        <div>
          <p className="font-medium text-white">{invitation.email}</p>
          <p className="text-sm text-gray-400">{invitation.role} • {invitation.department}</p>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="text-right">
          <p className="text-sm text-gray-400">Sent by {invitation.sentBy}</p>
          <p className="text-xs text-gray-500">
            {invitation.status === 'expired' ? 'Expired' : `Expires ${formatDate(invitation.expiresAt)}`}
          </p>
        </div>
        <span className={`px-2 py-1 rounded text-xs capitalize ${
          invitation.status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
          invitation.status === 'accepted' ? 'bg-green-500/20 text-green-400' :
          'bg-gray-500/20 text-gray-400'
        }`}>
          {invitation.status}
        </span>
        {invitation.status === 'pending' && (
          <div className="flex items-center gap-1">
            <button
              onClick={() => resendInvitation(invitation.id)}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              title="Resend"
            >
              <RefreshCw className="w-4 h-4 text-gray-400" />
            </button>
            <button
              onClick={() => cancelInvitation(invitation.id)}
              className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
              title="Cancel"
            >
              <X className="w-4 h-4 text-red-400" />
            </button>
          </div>
        )}
      </div>
    </div>
  );

  // Role Card
  const RoleCard = ({ role }: { role: UserRole }) => {
    const Icon = getRoleIcon(role.level);
    const memberCount = members.filter(m => m.role.id === role.id).length;

    return (
      <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getRoleColor(role.color)}`}>
              <Icon className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-medium text-white">{role.name}</h3>
              <p className="text-sm text-gray-400">{memberCount} member{memberCount !== 1 ? 's' : ''}</p>
            </div>
          </div>
          <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
            <Settings className="w-4 h-4 text-gray-400" />
          </button>
        </div>
        <div className="space-y-2">
          <p className="text-xs text-gray-500 uppercase">Permissions</p>
          <div className="flex flex-wrap gap-2">
            {role.permissions.includes('*') ? (
              <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs">Full Access</span>
            ) : (
              role.permissions.slice(0, 4).map(perm => (
                <span key={perm} className="px-2 py-1 bg-gray-700/50 text-gray-400 rounded text-xs">
                  {allPermissions.find(p => p.id === perm)?.name || perm}
                </span>
              ))
            )}
            {role.permissions.length > 4 && (
              <span className="px-2 py-1 text-gray-500 text-xs">+{role.permissions.length - 4} more</span>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Department Card
  const DepartmentCard = ({ department }: { department: Department }) => {
    const deptMembers = members.filter(m => m.department === department.name);

    return (
      <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg bg-${department.color}-500/20 flex items-center justify-center`}>
              <Building2 className={`w-5 h-5 text-${department.color}-400`} />
            </div>
            <div>
              <h3 className="font-medium text-white">{department.name}</h3>
              <p className="text-sm text-gray-400">{department.description}</p>
            </div>
          </div>
          <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
            <Edit3 className="w-4 h-4 text-gray-400" />
          </button>
        </div>
        <div className="flex items-center justify-between pt-4 border-t border-gray-700">
          <div>
            <p className="text-lg font-semibold text-white">{deptMembers.length}</p>
            <p className="text-xs text-gray-500">Members</p>
          </div>
          {department.lead && (
            <div className="text-right">
              <p className="text-sm text-gray-400">Lead</p>
              <p className="text-sm text-white">{department.lead}</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">Team Management</h1>
            <p className="text-gray-400">Manage your team members, roles, and permissions</p>
          </div>
          <button
            onClick={() => setShowInviteDialog(true)}
            className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
          >
            <UserPlus className="w-5 h-5" />
            Invite Member
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Total Members</span>
              <Users className="w-5 h-5 text-purple-400" />
            </div>
            <p className="text-2xl font-bold text-white">{stats.total}</p>
            <p className="text-sm text-gray-500">{departments.length} departments</p>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Active Now</span>
              <Activity className="w-5 h-5 text-green-400" />
            </div>
            <p className="text-2xl font-bold text-white">{stats.active}</p>
            <p className="text-sm text-green-400">{Math.round((stats.active / stats.total) * 100)}% of team</p>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Pending Invites</span>
              <Mail className="w-5 h-5 text-yellow-400" />
            </div>
            <p className="text-2xl font-bold text-white">{stats.pending}</p>
            <p className="text-sm text-gray-500">Awaiting response</p>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Avg Satisfaction</span>
              <Star className="w-5 h-5 text-yellow-400" />
            </div>
            <p className="text-2xl font-bold text-white">{stats.avgSatisfaction.toFixed(1)}</p>
            <p className="text-sm text-green-400">+0.2 from last month</p>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-1 p-1 bg-gray-800/50 rounded-lg w-fit">
          {[
            { id: 'members', label: 'Members', icon: Users },
            { id: 'invitations', label: 'Invitations', icon: Mail },
            { id: 'roles', label: 'Roles', icon: Shield },
            { id: 'departments', label: 'Departments', icon: Building2 },
            { id: 'activity', label: 'Activity', icon: Activity }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                activeTab === tab.id
                  ? 'bg-purple-500 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {tab.id === 'invitations' && invitations.filter(i => i.status === 'pending').length > 0 && (
                <span className="w-5 h-5 rounded-full bg-yellow-500 text-black text-xs flex items-center justify-center">
                  {invitations.filter(i => i.status === 'pending').length}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Members Tab */}
        {activeTab === 'members' && (
          <div className="space-y-6">
            {/* Filters */}
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-4 flex-grow">
                <div className="relative flex-grow max-w-md">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search members..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                  />
                </div>

                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value as any)}
                  className="px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                  <option value="pending">Pending</option>
                </select>

                <select
                  value={filterDepartment}
                  onChange={(e) => setFilterDepartment(e.target.value)}
                  className="px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white"
                >
                  <option value="all">All Departments</option>
                  {departments.map(dept => (
                    <option key={dept.id} value={dept.name}>{dept.name}</option>
                  ))}
                </select>

                <select
                  value={filterRole}
                  onChange={(e) => setFilterRole(e.target.value)}
                  className="px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white"
                >
                  <option value="all">All Roles</option>
                  {roles.map(role => (
                    <option key={role.id} value={role.id}>{role.name}</option>
                  ))}
                </select>
              </div>

              <div className="flex items-center gap-2 bg-gray-800/50 rounded-lg border border-gray-700 p-1">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 rounded transition-colors ${
                    viewMode === 'grid' ? 'bg-purple-500/20 text-purple-400' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <div className="grid grid-cols-2 gap-0.5 w-4 h-4">
                    <div className="bg-current rounded-sm" />
                    <div className="bg-current rounded-sm" />
                    <div className="bg-current rounded-sm" />
                    <div className="bg-current rounded-sm" />
                  </div>
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 rounded transition-colors ${
                    viewMode === 'list' ? 'bg-purple-500/20 text-purple-400' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <div className="flex flex-col gap-0.5 w-4 h-4">
                    <div className="h-1 bg-current rounded" />
                    <div className="h-1 bg-current rounded" />
                    <div className="h-1 bg-current rounded" />
                  </div>
                </button>
              </div>
            </div>

            {/* Members Display */}
            {viewMode === 'grid' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {filteredMembers.map(member => (
                  <MemberCard key={member.id} member={member} />
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                {filteredMembers.map(member => (
                  <MemberListItem key={member.id} member={member} />
                ))}
              </div>
            )}

            {filteredMembers.length === 0 && (
              <div className="text-center py-12">
                <Users className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-white mb-2">No members found</h3>
                <p className="text-gray-400">Try adjusting your filters or invite new team members</p>
              </div>
            )}
          </div>
        )}

        {/* Invitations Tab */}
        {activeTab === 'invitations' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-white">Pending Invitations</h3>
              <button
                onClick={() => setShowInviteDialog(true)}
                className="px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors flex items-center gap-1 text-sm"
              >
                <Plus className="w-4 h-4" />
                New Invitation
              </button>
            </div>
            <div className="space-y-3">
              {invitations.map(invitation => (
                <InvitationRow key={invitation.id} invitation={invitation} />
              ))}
              {invitations.length === 0 && (
                <div className="text-center py-12 bg-gray-800/30 rounded-xl">
                  <Mail className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-white mb-2">No pending invitations</h3>
                  <p className="text-gray-400">Invite new team members to get started</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Roles Tab */}
        {activeTab === 'roles' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-white">Team Roles</h3>
              <button className="px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors flex items-center gap-1 text-sm">
                <Plus className="w-4 h-4" />
                Create Role
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {roles.map(role => (
                <RoleCard key={role.id} role={role} />
              ))}
            </div>

            {/* Permissions Overview */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6">
              <h3 className="text-lg font-medium text-white mb-4">Permissions Overview</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-gray-400 font-medium">Permission</th>
                      {roles.map(role => (
                        <th key={role.id} className="text-center py-3 px-4 text-gray-400 font-medium">
                          {role.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {allPermissions.map(permission => (
                      <tr key={permission.id} className="border-b border-gray-700/50">
                        <td className="py-3 px-4">
                          <div>
                            <p className="text-white">{permission.name}</p>
                            <p className="text-xs text-gray-500">{permission.description}</p>
                          </div>
                        </td>
                        {roles.map(role => (
                          <td key={role.id} className="text-center py-3 px-4">
                            {role.permissions.includes('*') || role.permissions.includes(permission.id) ? (
                              <Check className="w-5 h-5 text-green-400 mx-auto" />
                            ) : (
                              <X className="w-5 h-5 text-gray-600 mx-auto" />
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Departments Tab */}
        {activeTab === 'departments' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-white">Departments</h3>
              <button className="px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors flex items-center gap-1 text-sm">
                <Plus className="w-4 h-4" />
                Create Department
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {departments.map(dept => (
                <DepartmentCard key={dept.id} department={dept} />
              ))}
            </div>
          </div>
        )}

        {/* Activity Tab */}
        {activeTab === 'activity' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-white">Activity Log</h3>
              <button className="px-3 py-1.5 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-1 text-sm">
                <Download className="w-4 h-4" />
                Export
              </button>
            </div>
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 overflow-hidden">
              <div className="divide-y divide-gray-700">
                {mockActivityLog.map(log => (
                  <div key={log.id} className="flex items-center gap-4 p-4 hover:bg-gray-700/30 transition-colors">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white font-bold text-sm">
                      {log.userName.split(' ').map(n => n[0]).join('')}
                    </div>
                    <div className="flex-grow">
                      <p className="text-white">
                        <span className="font-medium">{log.userName}</span>
                        <span className="text-gray-400"> {log.action} </span>
                        <span className="font-medium text-purple-400">{log.target}</span>
                      </p>
                      {log.details && <p className="text-sm text-gray-500">{log.details}</p>}
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-400">{formatTimeAgo(log.timestamp)}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Member Detail Dialog */}
        {showMemberDialog && selectedMember && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white font-bold text-xl">
                        {selectedMember.name.split(' ').map(n => n[0]).join('')}
                      </div>
                      <div className={`absolute -bottom-1 -right-1 w-5 h-5 rounded-full border-2 border-gray-800 ${getStatusColor(selectedMember.status)}`} />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white">{selectedMember.name}</h2>
                      <p className="text-gray-400">{selectedMember.title} • {selectedMember.department}</p>
                    </div>
                  </div>
                  <button onClick={() => setShowMemberDialog(false)} className="p-2 hover:bg-gray-700 rounded-lg">
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              <div className="p-6 overflow-y-auto max-h-[60vh] space-y-6">
                {/* Contact Info */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center gap-3 p-3 bg-gray-700/30 rounded-lg">
                    <Mail className="w-5 h-5 text-gray-400" />
                    <div>
                      <p className="text-xs text-gray-500">Email</p>
                      <p className="text-white">{selectedMember.email}</p>
                    </div>
                  </div>
                  {selectedMember.phone && (
                    <div className="flex items-center gap-3 p-3 bg-gray-700/30 rounded-lg">
                      <Phone className="w-5 h-5 text-gray-400" />
                      <div>
                        <p className="text-xs text-gray-500">Phone</p>
                        <p className="text-white">{selectedMember.phone}</p>
                      </div>
                    </div>
                  )}
                  {selectedMember.location && (
                    <div className="flex items-center gap-3 p-3 bg-gray-700/30 rounded-lg">
                      <MapPin className="w-5 h-5 text-gray-400" />
                      <div>
                        <p className="text-xs text-gray-500">Location</p>
                        <p className="text-white">{selectedMember.location}</p>
                      </div>
                    </div>
                  )}
                  {selectedMember.timezone && (
                    <div className="flex items-center gap-3 p-3 bg-gray-700/30 rounded-lg">
                      <Globe className="w-5 h-5 text-gray-400" />
                      <div>
                        <p className="text-xs text-gray-500">Timezone</p>
                        <p className="text-white">{selectedMember.timezone}</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Role & Permissions */}
                <div>
                  <h3 className="text-lg font-medium text-white mb-3">Role & Permissions</h3>
                  <div className="p-4 bg-gray-700/30 rounded-xl">
                    <div className="flex items-center gap-3 mb-4">
                      {React.createElement(getRoleIcon(selectedMember.role.level), {
                        className: `w-6 h-6 ${getRoleColor(selectedMember.role.color).split(' ')[1]}`
                      })}
                      <div>
                        <p className="font-medium text-white">{selectedMember.role.name}</p>
                        <p className="text-sm text-gray-400">
                          {selectedMember.role.permissions.includes('*') ? 'Full Access' : `${selectedMember.role.permissions.length} permissions`}
                        </p>
                      </div>
                      <button className="ml-auto px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors text-sm">
                        Change Role
                      </button>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {selectedMember.role.permissions.includes('*') ? (
                        <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs">All Permissions</span>
                      ) : (
                        selectedMember.role.permissions.map(perm => (
                          <span key={perm} className="px-2 py-1 bg-gray-600/50 text-gray-300 rounded text-xs">
                            {allPermissions.find(p => p.id === perm)?.name || perm}
                          </span>
                        ))
                      )}
                    </div>
                  </div>
                </div>

                {/* Performance */}
                {selectedMember.metrics.callsHandled > 0 && (
                  <div>
                    <h3 className="text-lg font-medium text-white mb-3">Performance</h3>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-gray-700/30 rounded-xl p-4 text-center">
                        <p className="text-2xl font-bold text-white">{selectedMember.metrics.callsHandled}</p>
                        <p className="text-sm text-gray-400">Calls Handled</p>
                      </div>
                      <div className="bg-gray-700/30 rounded-xl p-4 text-center">
                        <p className="text-2xl font-bold text-green-400">{selectedMember.metrics.successRate}%</p>
                        <p className="text-sm text-gray-400">Success Rate</p>
                      </div>
                      <div className="bg-gray-700/30 rounded-xl p-4 text-center">
                        <p className="text-2xl font-bold text-white">{selectedMember.metrics.customerSatisfaction}</p>
                        <p className="text-sm text-gray-400">CSAT Score</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Sessions */}
                {selectedMember.sessions.length > 0 && (
                  <div>
                    <h3 className="text-lg font-medium text-white mb-3">Active Sessions</h3>
                    <div className="space-y-2">
                      {selectedMember.sessions.map(session => (
                        <div key={session.id} className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                          <div className="flex items-center gap-3">
                            {session.device.includes('iPhone') || session.device.includes('Phone') ? (
                              <Smartphone className="w-5 h-5 text-gray-400" />
                            ) : session.device.includes('MacBook') || session.device.includes('Laptop') ? (
                              <Laptop className="w-5 h-5 text-gray-400" />
                            ) : (
                              <Monitor className="w-5 h-5 text-gray-400" />
                            )}
                            <div>
                              <p className="text-white">{session.device}</p>
                              <p className="text-xs text-gray-500">{session.browser} • {session.location}</p>
                            </div>
                          </div>
                          <div className="flex items-center gap-3">
                            {session.current && (
                              <span className="px-2 py-0.5 bg-green-500/20 text-green-400 rounded text-xs">Current</span>
                            )}
                            <span className="text-sm text-gray-400">{formatTimeAgo(session.lastActivity)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Security */}
                <div>
                  <h3 className="text-lg font-medium text-white mb-3">Security</h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                      <div className="flex items-center gap-3">
                        <ShieldCheck className={`w-5 h-5 ${selectedMember.twoFactorEnabled ? 'text-green-400' : 'text-gray-500'}`} />
                        <div>
                          <p className="text-white">Two-Factor Authentication</p>
                          <p className="text-xs text-gray-500">
                            {selectedMember.twoFactorEnabled ? 'Enabled and active' : 'Not enabled'}
                          </p>
                        </div>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs ${
                        selectedMember.twoFactorEnabled
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-yellow-500/20 text-yellow-400'
                      }`}>
                        {selectedMember.twoFactorEnabled ? 'Enabled' : 'Disabled'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                      <div className="flex items-center gap-3">
                        <Calendar className="w-5 h-5 text-gray-400" />
                        <div>
                          <p className="text-white">Account Created</p>
                          <p className="text-xs text-gray-500">{formatDate(selectedMember.joinedAt)}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-6 border-t border-gray-700 flex justify-between">
                <button
                  onClick={() => {
                    removeMember(selectedMember.id);
                    setShowMemberDialog(false);
                  }}
                  className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors flex items-center gap-2"
                >
                  <UserMinus className="w-4 h-4" />
                  Remove Member
                </button>
                <div className="flex items-center gap-3">
                  <button className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors">
                    Reset Password
                  </button>
                  <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2">
                    <Edit3 className="w-4 h-4" />
                    Edit Profile
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Invite Dialog */}
        {showInviteDialog && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl max-w-lg w-full overflow-hidden">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <UserPlus className="w-5 h-5 text-purple-400" />
                    Invite Team Member
                  </h2>
                  <button onClick={() => setShowInviteDialog(false)} className="p-2 hover:bg-gray-700 rounded-lg">
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              <div className="p-6 space-y-4">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Email Address</label>
                  <input
                    type="email"
                    value={inviteForm.email}
                    onChange={(e) => setInviteForm(prev => ({ ...prev, email: e.target.value }))}
                    placeholder="colleague@company.com"
                    className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Role</label>
                    <select
                      value={inviteForm.role}
                      onChange={(e) => setInviteForm(prev => ({ ...prev, role: e.target.value }))}
                      className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white"
                    >
                      {roles.filter(r => r.level !== 'owner').map(role => (
                        <option key={role.id} value={role.id}>{role.name}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Department</label>
                    <select
                      value={inviteForm.department}
                      onChange={(e) => setInviteForm(prev => ({ ...prev, department: e.target.value }))}
                      className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white"
                    >
                      {departments.map(dept => (
                        <option key={dept.id} value={dept.name}>{dept.name}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Personal Message (Optional)</label>
                  <textarea
                    value={inviteForm.message}
                    onChange={(e) => setInviteForm(prev => ({ ...prev, message: e.target.value }))}
                    placeholder="Add a personal message to the invitation..."
                    rows={3}
                    className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                  />
                </div>

                <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <div className="flex items-start gap-3">
                    <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                    <div className="text-sm text-blue-400">
                      <p className="font-medium mb-1">Invitation Details</p>
                      <p className="text-blue-400/80">
                        The invitation will be valid for 7 days. The invited user will receive an email with a link to create their account.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-6 border-t border-gray-700 flex justify-end gap-3">
                <button
                  onClick={() => setShowInviteDialog(false)}
                  className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={sendInvitation}
                  disabled={!inviteForm.email}
                  className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Send className="w-4 h-4" />
                  Send Invitation
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
