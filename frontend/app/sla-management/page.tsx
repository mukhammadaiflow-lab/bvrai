"use client";

import React, { useState, useMemo } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import {
  Clock,
  Target,
  AlertTriangle,
  CheckCircle,
  XCircle,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Activity,
  Bell,
  Shield,
  Settings,
  Plus,
  Edit3,
  Trash2,
  Eye,
  Search,
  Filter,
  Calendar,
  Users,
  Phone,
  MessageSquare,
  Zap,
  Award,
  Timer,
  Gauge,
  ArrowUpRight,
  ArrowDownRight,
  ChevronDown,
  ChevronRight,
  MoreVertical,
  RefreshCw,
  Download,
  X,
  Info,
  Play,
  Pause,
  AlertCircle,
  Flag
} from 'lucide-react';

// Types
interface SLAPolicy {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  priority: 'critical' | 'high' | 'medium' | 'low';
  targets: SLATarget[];
  conditions: SLACondition[];
  escalations: Escalation[];
  createdAt: string;
  updatedAt: string;
}

interface SLATarget {
  id: string;
  metric: 'response_time' | 'resolution_time' | 'first_response' | 'wait_time' | 'handle_time' | 'callback_time';
  threshold: number;
  unit: 'seconds' | 'minutes' | 'hours';
  warningPercent: number;
}

interface SLACondition {
  field: string;
  operator: 'equals' | 'not_equals' | 'contains' | 'in';
  value: any;
}

interface Escalation {
  id: string;
  level: number;
  triggerPercent: number;
  action: 'notify' | 'reassign' | 'priority_boost';
  recipients: string[];
}

interface SLAMetric {
  id: string;
  policyId: string;
  policyName: string;
  targetMet: number;
  targetMissed: number;
  compliance: number;
  avgTime: number;
  breaches: number;
  trend: number;
}

interface SLABreach {
  id: string;
  policyId: string;
  policyName: string;
  callId: string;
  customerId: string;
  customerName: string;
  target: string;
  threshold: number;
  actual: number;
  breachTime: string;
  status: 'active' | 'resolved' | 'escalated';
  escalationLevel: number;
}

interface SLAAlert {
  id: string;
  type: 'warning' | 'breach' | 'escalation';
  policyName: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

// Mock Data
const mockPolicies: SLAPolicy[] = [
  {
    id: 'sla-1',
    name: 'VIP Customer SLA',
    description: 'Premium support level for VIP customers with expedited response times',
    enabled: true,
    priority: 'critical',
    targets: [
      { id: 't1', metric: 'first_response', threshold: 30, unit: 'seconds', warningPercent: 80 },
      { id: 't2', metric: 'resolution_time', threshold: 5, unit: 'minutes', warningPercent: 75 },
      { id: 't3', metric: 'wait_time', threshold: 15, unit: 'seconds', warningPercent: 70 },
    ],
    conditions: [
      { field: 'customer_tier', operator: 'equals', value: 'vip' }
    ],
    escalations: [
      { id: 'e1', level: 1, triggerPercent: 80, action: 'notify', recipients: ['support-lead@company.com'] },
      { id: 'e2', level: 2, triggerPercent: 100, action: 'reassign', recipients: ['vip-team@company.com'] },
    ],
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-15T10:30:00Z'
  },
  {
    id: 'sla-2',
    name: 'Standard Support SLA',
    description: 'Standard service level for regular customer interactions',
    enabled: true,
    priority: 'medium',
    targets: [
      { id: 't1', metric: 'first_response', threshold: 60, unit: 'seconds', warningPercent: 80 },
      { id: 't2', metric: 'resolution_time', threshold: 15, unit: 'minutes', warningPercent: 75 },
      { id: 't3', metric: 'wait_time', threshold: 45, unit: 'seconds', warningPercent: 70 },
    ],
    conditions: [
      { field: 'customer_tier', operator: 'in', value: ['standard', 'basic'] }
    ],
    escalations: [
      { id: 'e1', level: 1, triggerPercent: 90, action: 'notify', recipients: ['support@company.com'] },
    ],
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-10T14:20:00Z'
  },
  {
    id: 'sla-3',
    name: 'Sales Inquiry SLA',
    description: 'Fast response times for potential sales opportunities',
    enabled: true,
    priority: 'high',
    targets: [
      { id: 't1', metric: 'first_response', threshold: 20, unit: 'seconds', warningPercent: 85 },
      { id: 't2', metric: 'callback_time', threshold: 2, unit: 'hours', warningPercent: 80 },
    ],
    conditions: [
      { field: 'call_type', operator: 'equals', value: 'sales' }
    ],
    escalations: [
      { id: 'e1', level: 1, triggerPercent: 85, action: 'notify', recipients: ['sales-lead@company.com'] },
      { id: 'e2', level: 2, triggerPercent: 100, action: 'priority_boost', recipients: [] },
    ],
    createdAt: '2024-01-05T00:00:00Z',
    updatedAt: '2024-01-18T09:15:00Z'
  },
  {
    id: 'sla-4',
    name: 'Emergency Support SLA',
    description: 'Critical issues requiring immediate attention',
    enabled: true,
    priority: 'critical',
    targets: [
      { id: 't1', metric: 'first_response', threshold: 10, unit: 'seconds', warningPercent: 90 },
      { id: 't2', metric: 'resolution_time', threshold: 30, unit: 'minutes', warningPercent: 85 },
    ],
    conditions: [
      { field: 'priority', operator: 'equals', value: 'emergency' }
    ],
    escalations: [
      { id: 'e1', level: 1, triggerPercent: 70, action: 'notify', recipients: ['emergency-team@company.com'] },
      { id: 'e2', level: 2, triggerPercent: 90, action: 'reassign', recipients: ['senior-support@company.com'] },
      { id: 'e3', level: 3, triggerPercent: 100, action: 'notify', recipients: ['management@company.com'] },
    ],
    createdAt: '2024-01-08T00:00:00Z',
    updatedAt: '2024-01-20T11:00:00Z'
  },
  {
    id: 'sla-5',
    name: 'After Hours SLA',
    description: 'Extended response times for after-hours support',
    enabled: false,
    priority: 'low',
    targets: [
      { id: 't1', metric: 'first_response', threshold: 5, unit: 'minutes', warningPercent: 80 },
      { id: 't2', metric: 'callback_time', threshold: 4, unit: 'hours', warningPercent: 75 },
    ],
    conditions: [
      { field: 'time_of_day', operator: 'in', value: ['after_hours', 'weekend'] }
    ],
    escalations: [],
    createdAt: '2024-01-10T00:00:00Z',
    updatedAt: '2024-01-10T00:00:00Z'
  }
];

const mockMetrics: SLAMetric[] = [
  { id: 'm1', policyId: 'sla-1', policyName: 'VIP Customer SLA', targetMet: 1245, targetMissed: 55, compliance: 95.8, avgTime: 25, breaches: 12, trend: 2.3 },
  { id: 'm2', policyId: 'sla-2', policyName: 'Standard Support SLA', targetMet: 4520, targetMissed: 380, compliance: 92.2, avgTime: 48, breaches: 45, trend: -1.5 },
  { id: 'm3', policyId: 'sla-3', policyName: 'Sales Inquiry SLA', targetMet: 890, targetMissed: 110, compliance: 89.0, avgTime: 18, breaches: 23, trend: 4.2 },
  { id: 'm4', policyId: 'sla-4', policyName: 'Emergency Support SLA', targetMet: 156, targetMissed: 4, compliance: 97.5, avgTime: 8, breaches: 2, trend: 1.8 },
];

const mockBreaches: SLABreach[] = [
  { id: 'b1', policyId: 'sla-1', policyName: 'VIP Customer SLA', callId: 'call-123', customerId: 'cust-456', customerName: 'Acme Corp', target: 'First Response', threshold: 30, actual: 45, breachTime: '2024-01-20T14:30:00Z', status: 'active', escalationLevel: 1 },
  { id: 'b2', policyId: 'sla-2', policyName: 'Standard Support SLA', callId: 'call-124', customerId: 'cust-789', customerName: 'Tech Solutions', target: 'Wait Time', threshold: 45, actual: 72, breachTime: '2024-01-20T14:15:00Z', status: 'escalated', escalationLevel: 2 },
  { id: 'b3', policyId: 'sla-3', policyName: 'Sales Inquiry SLA', callId: 'call-125', customerId: 'cust-101', customerName: 'Global Industries', target: 'First Response', threshold: 20, actual: 35, breachTime: '2024-01-20T13:45:00Z', status: 'resolved', escalationLevel: 1 },
  { id: 'b4', policyId: 'sla-2', policyName: 'Standard Support SLA', callId: 'call-126', customerId: 'cust-102', customerName: 'Smith & Co', target: 'Resolution Time', threshold: 900, actual: 1200, breachTime: '2024-01-20T12:30:00Z', status: 'resolved', escalationLevel: 0 },
  { id: 'b5', policyId: 'sla-1', policyName: 'VIP Customer SLA', callId: 'call-127', customerId: 'cust-103', customerName: 'Premium Partners', target: 'Wait Time', threshold: 15, actual: 28, breachTime: '2024-01-20T11:00:00Z', status: 'active', escalationLevel: 2 },
];

const mockAlerts: SLAAlert[] = [
  { id: 'a1', type: 'breach', policyName: 'VIP Customer SLA', message: 'SLA breach detected for Acme Corp - First Response time exceeded', timestamp: '2024-01-20T14:30:00Z', acknowledged: false },
  { id: 'a2', type: 'escalation', policyName: 'Standard Support SLA', message: 'Call escalated to Level 2 for Tech Solutions', timestamp: '2024-01-20T14:20:00Z', acknowledged: false },
  { id: 'a3', type: 'warning', policyName: 'Sales Inquiry SLA', message: 'SLA at 85% threshold - 3 calls at risk', timestamp: '2024-01-20T14:10:00Z', acknowledged: true },
  { id: 'a4', type: 'breach', policyName: 'VIP Customer SLA', message: 'SLA breach for Premium Partners - Wait time exceeded', timestamp: '2024-01-20T11:00:00Z', acknowledged: true },
];

const metricOptions = [
  { value: 'response_time', label: 'Response Time' },
  { value: 'resolution_time', label: 'Resolution Time' },
  { value: 'first_response', label: 'First Response' },
  { value: 'wait_time', label: 'Wait Time' },
  { value: 'handle_time', label: 'Handle Time' },
  { value: 'callback_time', label: 'Callback Time' },
];

export default function SLAManagementPage() {
  const [policies, setPolicies] = useState<SLAPolicy[]>(mockPolicies);
  const [breaches, setBreaches] = useState<SLABreach[]>(mockBreaches);
  const [alerts, setAlerts] = useState<SLAAlert[]>(mockAlerts);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'policies' | 'breaches' | 'alerts'>('dashboard');
  const [selectedPolicy, setSelectedPolicy] = useState<SLAPolicy | null>(null);
  const [showPolicyDialog, setShowPolicyDialog] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterPriority, setFilterPriority] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const stats = useMemo(() => {
    const totalCompliance = mockMetrics.reduce((sum, m) => sum + m.compliance, 0) / mockMetrics.length;
    const totalBreaches = mockMetrics.reduce((sum, m) => sum + m.breaches, 0);
    const activeBreaches = breaches.filter(b => b.status === 'active').length;
    const unacknowledgedAlerts = alerts.filter(a => !a.acknowledged).length;
    return { totalCompliance, totalBreaches, activeBreaches, unacknowledgedAlerts };
  }, [breaches, alerts]);

  const filteredPolicies = useMemo(() => {
    return policies.filter(policy => {
      const matchesSearch = policy.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesPriority = filterPriority === 'all' || policy.priority === filterPriority;
      return matchesSearch && matchesPriority;
    });
  }, [policies, searchQuery, filterPriority]);

  const filteredBreaches = useMemo(() => {
    return breaches.filter(breach => {
      const matchesSearch = breach.customerName.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           breach.policyName.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = filterStatus === 'all' || breach.status === filterStatus;
      return matchesSearch && matchesStatus;
    });
  }, [breaches, searchQuery, filterStatus]);

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-red-500/20 text-red-400';
      case 'escalated': return 'bg-orange-500/20 text-orange-400';
      case 'resolved': return 'bg-green-500/20 text-green-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'breach': return XCircle;
      case 'escalation': return AlertTriangle;
      case 'warning': return AlertCircle;
      default: return Info;
    }
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'breach': return 'text-red-400 bg-red-500/20';
      case 'escalation': return 'text-orange-400 bg-orange-500/20';
      case 'warning': return 'text-yellow-400 bg-yellow-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const formatDuration = (value: number, unit: string) => {
    if (unit === 'seconds') return `${value}s`;
    if (unit === 'minutes') return `${value}m`;
    if (unit === 'hours') return `${value}h`;
    return `${value}${unit}`;
  };

  const formatTimeAgo = (dateStr: string) => {
    const now = new Date();
    const date = new Date(dateStr);
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);

    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return new Date(dateStr).toLocaleDateString();
  };

  const togglePolicy = (policyId: string) => {
    setPolicies(prev => prev.map(p =>
      p.id === policyId ? { ...p, enabled: !p.enabled } : p
    ));
  };

  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(a =>
      a.id === alertId ? { ...a, acknowledged: true } : a
    ));
  };

  const acknowledgeAllAlerts = () => {
    setAlerts(prev => prev.map(a => ({ ...a, acknowledged: true })));
  };

  // Compliance Gauge Component
  const ComplianceGauge = ({ value, size = 120 }: { value: number; size?: number }) => {
    const radius = (size - 20) / 2;
    const circumference = 2 * Math.PI * radius;
    const progress = (value / 100) * circumference;
    const color = value >= 95 ? '#22C55E' : value >= 85 ? '#EAB308' : '#EF4444';

    return (
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke="#374151"
            strokeWidth="10"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke={color}
            strokeWidth="10"
            strokeDasharray={circumference}
            strokeDashoffset={circumference - progress}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold text-white">{value.toFixed(1)}%</span>
          <span className="text-xs text-gray-500">Compliance</span>
        </div>
      </div>
    );
  };

  // Policy Card Component
  const PolicyCard = ({ policy }: { policy: SLAPolicy }) => {
    const metric = mockMetrics.find(m => m.policyId === policy.id);

    return (
      <div className={`bg-gray-800/50 rounded-xl border transition-all ${
        policy.enabled ? 'border-gray-700 hover:border-purple-500/50' : 'border-gray-700/50 opacity-60'
      }`}>
        <div className="p-4">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getPriorityColor(policy.priority).split(' ')[0]}`}>
                <Target className={`w-5 h-5 ${getPriorityColor(policy.priority).split(' ')[1]}`} />
              </div>
              <div>
                <h3 className="font-medium text-white">{policy.name}</h3>
                <div className="flex items-center gap-2 mt-1">
                  <span className={`px-2 py-0.5 rounded text-xs capitalize ${getPriorityColor(policy.priority)}`}>
                    {policy.priority}
                  </span>
                  {policy.escalations.length > 0 && (
                    <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs">
                      {policy.escalations.length} escalation{policy.escalations.length !== 1 ? 's' : ''}
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => togglePolicy(policy.id)}
                className={`relative w-12 h-6 rounded-full transition-colors ${
                  policy.enabled ? 'bg-green-500' : 'bg-gray-600'
                }`}
              >
                <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                  policy.enabled ? 'left-7' : 'left-1'
                }`} />
              </button>
            </div>
          </div>

          <p className="text-sm text-gray-400 mb-4">{policy.description}</p>

          {/* Targets */}
          <div className="space-y-2 mb-4">
            {policy.targets.slice(0, 3).map(target => (
              <div key={target.id} className="flex items-center justify-between p-2 bg-gray-900/50 rounded-lg">
                <span className="text-sm text-gray-400">
                  {metricOptions.find(m => m.value === target.metric)?.label}
                </span>
                <span className="text-sm font-medium text-white">
                  ≤ {formatDuration(target.threshold, target.unit)}
                </span>
              </div>
            ))}
            {policy.targets.length > 3 && (
              <span className="text-xs text-gray-500">+{policy.targets.length - 3} more targets</span>
            )}
          </div>

          {/* Compliance */}
          {metric && (
            <div className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg mb-4">
              <div>
                <p className="text-sm text-gray-400">Compliance</p>
                <p className={`text-xl font-bold ${
                  metric.compliance >= 95 ? 'text-green-400' :
                  metric.compliance >= 85 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {metric.compliance}%
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-400">Breaches</p>
                <p className="text-xl font-bold text-white">{metric.breaches}</p>
              </div>
              <div className={`flex items-center gap-1 ${metric.trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {metric.trend >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                <span className="text-sm">{Math.abs(metric.trend)}%</span>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-end gap-2">
            <button
              onClick={() => { setSelectedPolicy(policy); setShowPolicyDialog(true); }}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            >
              <Eye className="w-4 h-4 text-gray-400" />
            </button>
            <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
              <Edit3 className="w-4 h-4 text-gray-400" />
            </button>
            <button className="p-2 hover:bg-red-500/20 rounded-lg transition-colors">
              <Trash2 className="w-4 h-4 text-red-400" />
            </button>
          </div>
        </div>
      </div>
    );
  };

  // Breach Row Component
  const BreachRow = ({ breach }: { breach: SLABreach }) => (
    <div className="flex items-center gap-4 p-4 bg-gray-800/50 rounded-xl border border-gray-700 hover:border-purple-500/50 transition-all">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getStatusColor(breach.status).split(' ')[0]}`}>
        {breach.status === 'resolved' ? (
          <CheckCircle className="w-5 h-5 text-green-400" />
        ) : breach.status === 'escalated' ? (
          <AlertTriangle className="w-5 h-5 text-orange-400" />
        ) : (
          <XCircle className="w-5 h-5 text-red-400" />
        )}
      </div>
      <div className="flex-grow">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-white">{breach.customerName}</span>
          <span className={`px-2 py-0.5 rounded text-xs capitalize ${getStatusColor(breach.status)}`}>
            {breach.status}
          </span>
          {breach.escalationLevel > 0 && (
            <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs">
              Level {breach.escalationLevel}
            </span>
          )}
        </div>
        <p className="text-sm text-gray-400">
          {breach.policyName} • {breach.target}
        </p>
      </div>
      <div className="text-right">
        <p className="text-sm text-white">
          <span className="text-red-400">{breach.actual}s</span> / {breach.threshold}s
        </p>
        <p className="text-xs text-gray-500">{formatTimeAgo(breach.breachTime)}</p>
      </div>
      <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
        <Eye className="w-4 h-4 text-gray-400" />
      </button>
    </div>
  );

  // Alert Row Component
  const AlertRow = ({ alert }: { alert: SLAAlert }) => {
    const Icon = getAlertIcon(alert.type);

    return (
      <div className={`flex items-center gap-4 p-4 rounded-xl border transition-all ${
        alert.acknowledged ? 'bg-gray-800/30 border-gray-700/50 opacity-60' : 'bg-gray-800/50 border-gray-700'
      }`}>
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getAlertColor(alert.type).split(' ')[1]}`}>
          <Icon className={`w-5 h-5 ${getAlertColor(alert.type).split(' ')[0]}`} />
        </div>
        <div className="flex-grow">
          <div className="flex items-center gap-2 mb-1">
            <span className={`px-2 py-0.5 rounded text-xs capitalize ${getAlertColor(alert.type)}`}>
              {alert.type}
            </span>
            <span className="text-sm text-gray-400">{alert.policyName}</span>
          </div>
          <p className="text-sm text-white">{alert.message}</p>
        </div>
        <div className="text-right flex-shrink-0">
          <p className="text-xs text-gray-500">{formatTimeAgo(alert.timestamp)}</p>
        </div>
        {!alert.acknowledged && (
          <button
            onClick={() => acknowledgeAlert(alert.id)}
            className="px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors text-sm"
          >
            Acknowledge
          </button>
        )}
      </div>
    );
  };

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">SLA Management</h1>
            <p className="text-gray-400">Monitor and manage service level agreements</p>
          </div>
          <div className="flex items-center gap-3">
            <button className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-2">
              <Download className="w-5 h-5" />
              Export Report
            </button>
            <button
              onClick={() => setShowCreateDialog(true)}
              className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
            >
              <Plus className="w-5 h-5" />
              New SLA Policy
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Overall Compliance</span>
              <Gauge className="w-5 h-5 text-green-400" />
            </div>
            <p className={`text-2xl font-bold ${
              stats.totalCompliance >= 95 ? 'text-green-400' :
              stats.totalCompliance >= 85 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {stats.totalCompliance.toFixed(1)}%
            </p>
            <div className="flex items-center gap-1 text-sm text-green-400">
              <TrendingUp className="w-4 h-4" />
              <span>+1.5% from last week</span>
            </div>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Active Policies</span>
              <Target className="w-5 h-5 text-purple-400" />
            </div>
            <p className="text-2xl font-bold text-white">
              {policies.filter(p => p.enabled).length}
            </p>
            <p className="text-sm text-gray-500">of {policies.length} total</p>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Active Breaches</span>
              <XCircle className="w-5 h-5 text-red-400" />
            </div>
            <p className="text-2xl font-bold text-red-400">{stats.activeBreaches}</p>
            <p className="text-sm text-gray-500">{stats.totalBreaches} total this week</p>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Pending Alerts</span>
              <Bell className="w-5 h-5 text-yellow-400" />
            </div>
            <p className="text-2xl font-bold text-yellow-400">{stats.unacknowledgedAlerts}</p>
            <p className="text-sm text-gray-500">require attention</p>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-1 p-1 bg-gray-800/50 rounded-lg w-fit">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
            { id: 'policies', label: 'Policies', icon: Target },
            { id: 'breaches', label: 'Breaches', icon: XCircle, count: stats.activeBreaches },
            { id: 'alerts', label: 'Alerts', icon: Bell, count: stats.unacknowledgedAlerts }
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
              {tab.count !== undefined && tab.count > 0 && (
                <span className={`w-5 h-5 rounded-full text-xs flex items-center justify-center ${
                  activeTab === tab.id ? 'bg-white/20' : 'bg-red-500'
                }`}>
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Compliance Overview */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6">
              <h3 className="text-lg font-medium text-white mb-6">Compliance Overview</h3>
              <div className="flex justify-center mb-6">
                <ComplianceGauge value={stats.totalCompliance} size={160} />
              </div>
              <div className="space-y-3">
                {mockMetrics.map(metric => (
                  <div key={metric.id} className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">{metric.policyName}</span>
                    <div className="flex items-center gap-2">
                      <span className={`text-sm font-medium ${
                        metric.compliance >= 95 ? 'text-green-400' :
                        metric.compliance >= 85 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {metric.compliance}%
                      </span>
                      {metric.trend >= 0 ? (
                        <TrendingUp className="w-3 h-3 text-green-400" />
                      ) : (
                        <TrendingDown className="w-3 h-3 text-red-400" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Policy Performance */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6 lg:col-span-2">
              <h3 className="text-lg font-medium text-white mb-4">Policy Performance</h3>
              <div className="space-y-4">
                {mockMetrics.map(metric => (
                  <div key={metric.id} className="p-4 bg-gray-900/50 rounded-xl">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <h4 className="font-medium text-white">{metric.policyName}</h4>
                        <p className="text-sm text-gray-400">
                          {metric.targetMet.toLocaleString()} met • {metric.targetMissed.toLocaleString()} missed
                        </p>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <p className="text-sm text-gray-500">Avg Time</p>
                          <p className="text-lg font-semibold text-white">{metric.avgTime}s</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-gray-500">Breaches</p>
                          <p className="text-lg font-semibold text-red-400">{metric.breaches}</p>
                        </div>
                      </div>
                    </div>
                    <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`absolute inset-y-0 left-0 rounded-full ${
                          metric.compliance >= 95 ? 'bg-green-500' :
                          metric.compliance >= 85 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${metric.compliance}%` }}
                      />
                    </div>
                    <div className="flex items-center justify-between mt-2 text-xs">
                      <span className="text-gray-500">0%</span>
                      <span className={`${
                        metric.compliance >= 95 ? 'text-green-400' :
                        metric.compliance >= 85 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {metric.compliance}% compliance
                      </span>
                      <span className="text-gray-500">100%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Breaches */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6 lg:col-span-2">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-white">Recent Breaches</h3>
                <button
                  onClick={() => setActiveTab('breaches')}
                  className="text-sm text-purple-400 hover:text-purple-300"
                >
                  View All →
                </button>
              </div>
              <div className="space-y-3">
                {breaches.slice(0, 3).map(breach => (
                  <BreachRow key={breach.id} breach={breach} />
                ))}
              </div>
            </div>

            {/* Recent Alerts */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-white">Recent Alerts</h3>
                <button
                  onClick={() => setActiveTab('alerts')}
                  className="text-sm text-purple-400 hover:text-purple-300"
                >
                  View All →
                </button>
              </div>
              <div className="space-y-3">
                {alerts.slice(0, 4).map(alert => {
                  const Icon = getAlertIcon(alert.type);
                  return (
                    <div key={alert.id} className={`flex items-center gap-3 p-3 rounded-lg ${
                      alert.acknowledged ? 'bg-gray-900/30 opacity-60' : 'bg-gray-900/50'
                    }`}>
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${getAlertColor(alert.type).split(' ')[1]}`}>
                        <Icon className={`w-4 h-4 ${getAlertColor(alert.type).split(' ')[0]}`} />
                      </div>
                      <div className="flex-grow min-w-0">
                        <p className="text-sm text-white truncate">{alert.message}</p>
                        <p className="text-xs text-gray-500">{formatTimeAgo(alert.timestamp)}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Policies Tab */}
        {activeTab === 'policies' && (
          <div className="space-y-6">
            {/* Filters */}
            <div className="flex items-center gap-4">
              <div className="relative flex-grow max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search policies..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <select
                value={filterPriority}
                onChange={(e) => setFilterPriority(e.target.value)}
                className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white"
              >
                <option value="all">All Priorities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>

            {/* Policies Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredPolicies.map(policy => (
                <PolicyCard key={policy.id} policy={policy} />
              ))}
            </div>
          </div>
        )}

        {/* Breaches Tab */}
        {activeTab === 'breaches' && (
          <div className="space-y-6">
            {/* Filters */}
            <div className="flex items-center gap-4">
              <div className="relative flex-grow max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search breaches..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div className="flex items-center gap-2">
                {['all', 'active', 'escalated', 'resolved'].map(status => (
                  <button
                    key={status}
                    onClick={() => setFilterStatus(status)}
                    className={`px-3 py-1.5 rounded-lg text-sm capitalize transition-all ${
                      filterStatus === status
                        ? 'bg-purple-500 text-white'
                        : 'bg-gray-800/50 text-gray-400 hover:text-white border border-gray-700'
                    }`}
                  >
                    {status}
                  </button>
                ))}
              </div>
            </div>

            {/* Breaches List */}
            <div className="space-y-3">
              {filteredBreaches.map(breach => (
                <BreachRow key={breach.id} breach={breach} />
              ))}
              {filteredBreaches.length === 0 && (
                <div className="text-center py-12 bg-gray-800/30 rounded-xl">
                  <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-white mb-2">No breaches found</h3>
                  <p className="text-gray-400">All SLA targets are being met</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Alerts Tab */}
        {activeTab === 'alerts' && (
          <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <p className="text-gray-400">
                {alerts.filter(a => !a.acknowledged).length} unacknowledged alerts
              </p>
              {alerts.some(a => !a.acknowledged) && (
                <button
                  onClick={acknowledgeAllAlerts}
                  className="px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors"
                >
                  Acknowledge All
                </button>
              )}
            </div>

            {/* Alerts List */}
            <div className="space-y-3">
              {alerts.map(alert => (
                <AlertRow key={alert.id} alert={alert} />
              ))}
              {alerts.length === 0 && (
                <div className="text-center py-12 bg-gray-800/30 rounded-xl">
                  <Bell className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-white mb-2">No alerts</h3>
                  <p className="text-gray-400">All systems operating normally</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Policy Detail Dialog */}
        {showPolicyDialog && selectedPolicy && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${getPriorityColor(selectedPolicy.priority).split(' ')[0]}`}>
                      <Target className={`w-6 h-6 ${getPriorityColor(selectedPolicy.priority).split(' ')[1]}`} />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white">{selectedPolicy.name}</h2>
                      <div className="flex items-center gap-2 mt-1">
                        <span className={`px-2 py-0.5 rounded text-xs capitalize ${getPriorityColor(selectedPolicy.priority)}`}>
                          {selectedPolicy.priority}
                        </span>
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          selectedPolicy.enabled ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'
                        }`}>
                          {selectedPolicy.enabled ? 'Active' : 'Disabled'}
                        </span>
                      </div>
                    </div>
                  </div>
                  <button onClick={() => setShowPolicyDialog(false)} className="p-2 hover:bg-gray-700 rounded-lg">
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              <div className="p-6 overflow-y-auto max-h-[60vh] space-y-6">
                <p className="text-gray-400">{selectedPolicy.description}</p>

                {/* Targets */}
                <div>
                  <h3 className="text-lg font-medium text-white mb-4">SLA Targets</h3>
                  <div className="space-y-3">
                    {selectedPolicy.targets.map(target => (
                      <div key={target.id} className="flex items-center justify-between p-4 bg-gray-700/30 rounded-xl">
                        <div>
                          <p className="font-medium text-white">
                            {metricOptions.find(m => m.value === target.metric)?.label}
                          </p>
                          <p className="text-sm text-gray-400">Warning at {target.warningPercent}%</p>
                        </div>
                        <div className="text-right">
                          <p className="text-xl font-bold text-white">
                            {formatDuration(target.threshold, target.unit)}
                          </p>
                          <p className="text-sm text-gray-500">threshold</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Conditions */}
                {selectedPolicy.conditions.length > 0 && (
                  <div>
                    <h3 className="text-lg font-medium text-white mb-4">Conditions</h3>
                    <div className="space-y-2">
                      {selectedPolicy.conditions.map((cond, index) => (
                        <div key={index} className="flex items-center gap-2 p-3 bg-gray-700/30 rounded-lg">
                          <Filter className="w-4 h-4 text-purple-400" />
                          <span className="text-white">{cond.field}</span>
                          <span className="text-gray-500">{cond.operator.replace('_', ' ')}</span>
                          <span className="text-purple-400">
                            {Array.isArray(cond.value) ? cond.value.join(', ') : cond.value}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Escalations */}
                {selectedPolicy.escalations.length > 0 && (
                  <div>
                    <h3 className="text-lg font-medium text-white mb-4">Escalation Rules</h3>
                    <div className="space-y-3">
                      {selectedPolicy.escalations.map(esc => (
                        <div key={esc.id} className="flex items-center gap-4 p-4 bg-gray-700/30 rounded-xl">
                          <div className="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 font-bold">
                            {esc.level}
                          </div>
                          <div className="flex-grow">
                            <p className="text-white capitalize">{esc.action.replace('_', ' ')}</p>
                            <p className="text-sm text-gray-400">
                              Trigger at {esc.triggerPercent}% of threshold
                            </p>
                          </div>
                          {esc.recipients.length > 0 && (
                            <div className="text-right">
                              <p className="text-sm text-gray-400">Recipients</p>
                              <p className="text-sm text-white">{esc.recipients.length} configured</p>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="p-6 border-t border-gray-700 flex justify-between">
                <button
                  onClick={() => togglePolicy(selectedPolicy.id)}
                  className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
                    selectedPolicy.enabled
                      ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                      : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                  }`}
                >
                  {selectedPolicy.enabled ? (
                    <>
                      <Pause className="w-4 h-4" />
                      Disable Policy
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Enable Policy
                    </>
                  )}
                </button>
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2">
                  <Edit3 className="w-4 h-4" />
                  Edit Policy
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
