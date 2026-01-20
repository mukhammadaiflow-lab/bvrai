"use client";

import React, { useState, useMemo } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import {
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  PhoneForwarded,
  PhoneMissed,
  GitBranch,
  Plus,
  Trash2,
  Edit3,
  Copy,
  Eye,
  Play,
  Pause,
  Settings,
  Clock,
  Calendar,
  Users,
  User,
  Bot,
  Building2,
  Globe,
  MapPin,
  Flag,
  Tag,
  Filter,
  Search,
  MoreVertical,
  ChevronRight,
  ChevronDown,
  Check,
  X,
  AlertTriangle,
  Info,
  HelpCircle,
  ArrowRight,
  ArrowDown,
  Shuffle,
  Target,
  Zap,
  Shield,
  Star,
  StarOff,
  GripVertical,
  Move,
  Layers,
  CircleDot,
  Square,
  Diamond,
  Hexagon,
  Triangle,
  Circle,
  Save,
  RefreshCw,
  TestTube,
  Volume2,
  VolumeX,
  MessageSquare,
  Mail,
  Voicemail,
  Webhook,
  ExternalLink,
  Hash,
  Percent,
  DollarSign,
  Timer,
  Gauge,
  Activity,
  TrendingUp,
  BarChart3
} from 'lucide-react';

// Types
interface RoutingCondition {
  id: string;
  type: 'caller_id' | 'time' | 'day' | 'caller_location' | 'ivr_selection' | 'call_history' | 'vip_status' | 'language' | 'campaign' | 'skill_requirement' | 'queue_status' | 'custom';
  operator: 'equals' | 'not_equals' | 'contains' | 'starts_with' | 'ends_with' | 'greater_than' | 'less_than' | 'between' | 'in' | 'not_in' | 'matches_regex';
  value: any;
  secondaryValue?: any;
}

interface RoutingAction {
  id: string;
  type: 'route_to_agent' | 'route_to_queue' | 'route_to_ivr' | 'route_to_voicemail' | 'route_to_external' | 'play_message' | 'collect_input' | 'set_priority' | 'add_tag' | 'send_webhook' | 'transfer' | 'hangup';
  target?: string;
  config?: Record<string, any>;
}

interface RoutingRule {
  id: string;
  name: string;
  description: string;
  priority: number;
  enabled: boolean;
  conditions: RoutingCondition[];
  conditionLogic: 'all' | 'any' | 'custom';
  customLogic?: string;
  actions: RoutingAction[];
  fallbackAction?: RoutingAction;
  schedule?: RuleSchedule;
  metrics: RuleMetrics;
  tags: string[];
  createdAt: string;
  updatedAt: string;
  createdBy: string;
}

interface RuleSchedule {
  enabled: boolean;
  timezone: string;
  windows: ScheduleWindow[];
}

interface ScheduleWindow {
  days: number[];
  startTime: string;
  endTime: string;
}

interface RuleMetrics {
  totalCalls: number;
  matchedCalls: number;
  successRate: number;
  avgHandleTime: number;
}

interface Queue {
  id: string;
  name: string;
  agentCount: number;
  waitingCalls: number;
  avgWaitTime: number;
}

interface Agent {
  id: string;
  name: string;
  status: 'available' | 'busy' | 'away' | 'offline';
  skills: string[];
  currentCalls: number;
}

// Mock Data
const mockQueues: Queue[] = [
  { id: 'q1', name: 'General Support', agentCount: 12, waitingCalls: 5, avgWaitTime: 45 },
  { id: 'q2', name: 'Sales', agentCount: 8, waitingCalls: 2, avgWaitTime: 30 },
  { id: 'q3', name: 'Technical Support', agentCount: 15, waitingCalls: 8, avgWaitTime: 120 },
  { id: 'q4', name: 'Billing', agentCount: 6, waitingCalls: 3, avgWaitTime: 60 },
  { id: 'q5', name: 'VIP Support', agentCount: 4, waitingCalls: 0, avgWaitTime: 15 },
  { id: 'q6', name: 'Emergency Line', agentCount: 3, waitingCalls: 1, avgWaitTime: 10 },
];

const mockAgents: Agent[] = [
  { id: 'a1', name: 'AI Agent - Sarah', status: 'available', skills: ['sales', 'support', 'billing'], currentCalls: 2 },
  { id: 'a2', name: 'AI Agent - Michael', status: 'busy', skills: ['technical', 'support'], currentCalls: 3 },
  { id: 'a3', name: 'AI Agent - Emma', status: 'available', skills: ['sales', 'vip'], currentCalls: 1 },
  { id: 'a4', name: 'AI Agent - James', status: 'away', skills: ['support', 'billing'], currentCalls: 0 },
  { id: 'a5', name: 'AI Agent - Olivia', status: 'available', skills: ['technical', 'emergency'], currentCalls: 2 },
];

const mockRules: RoutingRule[] = [
  {
    id: 'rule-1',
    name: 'VIP Customer Priority',
    description: 'Route VIP customers directly to dedicated support queue with highest priority',
    priority: 1,
    enabled: true,
    conditions: [
      { id: 'c1', type: 'vip_status', operator: 'equals', value: true },
    ],
    conditionLogic: 'all',
    actions: [
      { id: 'a1', type: 'set_priority', config: { priority: 'high' } },
      { id: 'a2', type: 'route_to_queue', target: 'q5' },
    ],
    metrics: { totalCalls: 1250, matchedCalls: 1180, successRate: 94.4, avgHandleTime: 180 },
    tags: ['vip', 'priority'],
    createdAt: '2024-01-05T10:00:00Z',
    updatedAt: '2024-01-20T14:30:00Z',
    createdBy: 'John Smith'
  },
  {
    id: 'rule-2',
    name: 'After Hours Routing',
    description: 'Route calls outside business hours to voicemail or emergency queue',
    priority: 2,
    enabled: true,
    conditions: [
      { id: 'c1', type: 'time', operator: 'not_in', value: ['09:00', '18:00'] },
    ],
    conditionLogic: 'all',
    actions: [
      { id: 'a1', type: 'play_message', config: { message: 'after_hours_greeting' } },
      { id: 'a2', type: 'route_to_voicemail', target: 'general_voicemail' },
    ],
    fallbackAction: { id: 'fa1', type: 'route_to_queue', target: 'q6' },
    schedule: {
      enabled: true,
      timezone: 'America/New_York',
      windows: [
        { days: [1, 2, 3, 4, 5], startTime: '18:00', endTime: '09:00' },
        { days: [0, 6], startTime: '00:00', endTime: '23:59' },
      ]
    },
    metrics: { totalCalls: 3420, matchedCalls: 2890, successRate: 84.5, avgHandleTime: 45 },
    tags: ['after-hours', 'voicemail'],
    createdAt: '2024-01-08T09:00:00Z',
    updatedAt: '2024-01-19T16:45:00Z',
    createdBy: 'Sarah Johnson'
  },
  {
    id: 'rule-3',
    name: 'Sales Call Routing',
    description: 'Route sales inquiries based on IVR selection to sales queue',
    priority: 3,
    enabled: true,
    conditions: [
      { id: 'c1', type: 'ivr_selection', operator: 'equals', value: '1' },
      { id: 'c2', type: 'campaign', operator: 'in', value: ['summer_promo', 'new_customer'] },
    ],
    conditionLogic: 'any',
    actions: [
      { id: 'a1', type: 'add_tag', config: { tag: 'sales_lead' } },
      { id: 'a2', type: 'route_to_queue', target: 'q2' },
    ],
    metrics: { totalCalls: 5680, matchedCalls: 4920, successRate: 86.6, avgHandleTime: 320 },
    tags: ['sales', 'ivr'],
    createdAt: '2024-01-10T11:30:00Z',
    updatedAt: '2024-01-18T10:15:00Z',
    createdBy: 'Mike Wilson'
  },
  {
    id: 'rule-4',
    name: 'Technical Support Escalation',
    description: 'Route complex technical issues to specialized technical support',
    priority: 4,
    enabled: true,
    conditions: [
      { id: 'c1', type: 'ivr_selection', operator: 'equals', value: '2' },
      { id: 'c2', type: 'skill_requirement', operator: 'contains', value: 'technical' },
    ],
    conditionLogic: 'all',
    actions: [
      { id: 'a1', type: 'route_to_queue', target: 'q3' },
    ],
    metrics: { totalCalls: 4200, matchedCalls: 3780, successRate: 90.0, avgHandleTime: 480 },
    tags: ['technical', 'escalation'],
    createdAt: '2024-01-12T14:00:00Z',
    updatedAt: '2024-01-17T09:30:00Z',
    createdBy: 'Emily Chen'
  },
  {
    id: 'rule-5',
    name: 'Geographic Routing - APAC',
    description: 'Route calls from APAC region to appropriate timezone agents',
    priority: 5,
    enabled: true,
    conditions: [
      { id: 'c1', type: 'caller_location', operator: 'in', value: ['AU', 'NZ', 'JP', 'SG', 'HK'] },
    ],
    conditionLogic: 'all',
    actions: [
      { id: 'a1', type: 'add_tag', config: { tag: 'apac_region' } },
      { id: 'a2', type: 'route_to_agent', target: 'apac_team' },
    ],
    metrics: { totalCalls: 2100, matchedCalls: 1890, successRate: 90.0, avgHandleTime: 240 },
    tags: ['geographic', 'apac'],
    createdAt: '2024-01-14T08:00:00Z',
    updatedAt: '2024-01-16T12:00:00Z',
    createdBy: 'John Smith'
  },
  {
    id: 'rule-6',
    name: 'Billing Inquiry Routing',
    description: 'Route billing-related calls to billing department',
    priority: 6,
    enabled: false,
    conditions: [
      { id: 'c1', type: 'ivr_selection', operator: 'equals', value: '3' },
    ],
    conditionLogic: 'all',
    actions: [
      { id: 'a1', type: 'route_to_queue', target: 'q4' },
    ],
    metrics: { totalCalls: 1800, matchedCalls: 1620, successRate: 90.0, avgHandleTime: 200 },
    tags: ['billing'],
    createdAt: '2024-01-15T10:30:00Z',
    updatedAt: '2024-01-15T10:30:00Z',
    createdBy: 'Sarah Johnson'
  }
];

const conditionTypes = [
  { type: 'caller_id', name: 'Caller ID', icon: Phone, description: 'Match based on caller phone number' },
  { type: 'time', name: 'Time of Day', icon: Clock, description: 'Match based on current time' },
  { type: 'day', name: 'Day of Week', icon: Calendar, description: 'Match based on day of week' },
  { type: 'caller_location', name: 'Caller Location', icon: MapPin, description: 'Match based on geographic location' },
  { type: 'ivr_selection', name: 'IVR Selection', icon: Hash, description: 'Match based on IVR menu selection' },
  { type: 'call_history', name: 'Call History', icon: Activity, description: 'Match based on previous call data' },
  { type: 'vip_status', name: 'VIP Status', icon: Star, description: 'Match VIP/premium customers' },
  { type: 'language', name: 'Language', icon: Globe, description: 'Match based on language preference' },
  { type: 'campaign', name: 'Campaign', icon: Target, description: 'Match based on marketing campaign' },
  { type: 'skill_requirement', name: 'Skill Requirement', icon: Zap, description: 'Match based on required agent skills' },
  { type: 'queue_status', name: 'Queue Status', icon: Users, description: 'Match based on queue conditions' },
  { type: 'custom', name: 'Custom Field', icon: Settings, description: 'Match based on custom data fields' },
];

const actionTypes = [
  { type: 'route_to_queue', name: 'Route to Queue', icon: Users, description: 'Send call to a specific queue' },
  { type: 'route_to_agent', name: 'Route to Agent', icon: Bot, description: 'Send call to a specific agent' },
  { type: 'route_to_ivr', name: 'Route to IVR', icon: GitBranch, description: 'Send call to IVR menu' },
  { type: 'route_to_voicemail', name: 'Route to Voicemail', icon: Voicemail, description: 'Send call to voicemail' },
  { type: 'route_to_external', name: 'External Transfer', icon: ExternalLink, description: 'Transfer to external number' },
  { type: 'play_message', name: 'Play Message', icon: Volume2, description: 'Play audio message to caller' },
  { type: 'collect_input', name: 'Collect Input', icon: MessageSquare, description: 'Collect input from caller' },
  { type: 'set_priority', name: 'Set Priority', icon: Flag, description: 'Set call priority level' },
  { type: 'add_tag', name: 'Add Tag', icon: Tag, description: 'Add tag to the call' },
  { type: 'send_webhook', name: 'Send Webhook', icon: Webhook, description: 'Trigger external webhook' },
  { type: 'transfer', name: 'Transfer Call', icon: PhoneForwarded, description: 'Transfer to another destination' },
  { type: 'hangup', name: 'End Call', icon: PhoneMissed, description: 'Disconnect the call' },
];

export default function CallRoutingPage() {
  const [rules, setRules] = useState<RoutingRule[]>(mockRules);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterEnabled, setFilterEnabled] = useState<'all' | 'enabled' | 'disabled'>('all');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<'list' | 'visual'>('list');
  const [selectedRule, setSelectedRule] = useState<RoutingRule | null>(null);
  const [showRuleDialog, setShowRuleDialog] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showTestDialog, setShowTestDialog] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);

  // Create/Edit state
  const [editingRule, setEditingRule] = useState<Partial<RoutingRule>>({
    name: '',
    description: '',
    priority: rules.length + 1,
    enabled: true,
    conditions: [],
    conditionLogic: 'all',
    actions: [],
    tags: []
  });

  // Test call state
  const [testCallData, setTestCallData] = useState({
    callerId: '+1234567890',
    callerLocation: 'US',
    ivrSelection: '1',
    vipStatus: false,
    language: 'en',
    campaign: '',
    currentTime: '14:30'
  });
  const [testResult, setTestResult] = useState<{ rule: RoutingRule | null; actions: string[] } | null>(null);

  const allTags = useMemo(() => {
    const tags = new Set<string>();
    rules.forEach(rule => rule.tags.forEach(tag => tags.add(tag)));
    return Array.from(tags);
  }, [rules]);

  const filteredRules = useMemo(() => {
    return rules.filter(rule => {
      const matchesSearch = rule.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           rule.description.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesEnabled = filterEnabled === 'all' ||
                            (filterEnabled === 'enabled' && rule.enabled) ||
                            (filterEnabled === 'disabled' && !rule.enabled);
      const matchesTags = selectedTags.length === 0 ||
                         selectedTags.some(tag => rule.tags.includes(tag));
      return matchesSearch && matchesEnabled && matchesTags;
    }).sort((a, b) => a.priority - b.priority);
  }, [rules, searchQuery, filterEnabled, selectedTags]);

  const toggleRule = (ruleId: string) => {
    setRules(prev => prev.map(rule =>
      rule.id === ruleId ? { ...rule, enabled: !rule.enabled } : rule
    ));
  };

  const deleteRule = (ruleId: string) => {
    setRules(prev => prev.filter(r => r.id !== ruleId));
  };

  const duplicateRule = (rule: RoutingRule) => {
    const newRule: RoutingRule = {
      ...rule,
      id: `rule-${Date.now()}`,
      name: `${rule.name} (Copy)`,
      priority: rules.length + 1,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      metrics: { totalCalls: 0, matchedCalls: 0, successRate: 0, avgHandleTime: 0 }
    };
    setRules(prev => [...prev, newRule]);
  };

  const moveRule = (ruleId: string, direction: 'up' | 'down') => {
    setRules(prev => {
      const sorted = [...prev].sort((a, b) => a.priority - b.priority);
      const index = sorted.findIndex(r => r.id === ruleId);
      if ((direction === 'up' && index === 0) || (direction === 'down' && index === sorted.length - 1)) {
        return prev;
      }
      const swapIndex = direction === 'up' ? index - 1 : index + 1;
      const tempPriority = sorted[index].priority;
      sorted[index].priority = sorted[swapIndex].priority;
      sorted[swapIndex].priority = tempPriority;
      return sorted;
    });
  };

  const openCreateDialog = () => {
    setEditingRule({
      name: '',
      description: '',
      priority: rules.length + 1,
      enabled: true,
      conditions: [],
      conditionLogic: 'all',
      actions: [],
      tags: []
    });
    setIsEditMode(false);
    setShowCreateDialog(true);
  };

  const openEditDialog = (rule: RoutingRule) => {
    setEditingRule({ ...rule });
    setIsEditMode(true);
    setShowCreateDialog(true);
  };

  const saveRule = () => {
    if (!editingRule.name) return;

    const now = new Date().toISOString();
    if (isEditMode && editingRule.id) {
      setRules(prev => prev.map(r =>
        r.id === editingRule.id
          ? { ...r, ...editingRule, updatedAt: now } as RoutingRule
          : r
      ));
    } else {
      const newRule: RoutingRule = {
        id: `rule-${Date.now()}`,
        name: editingRule.name,
        description: editingRule.description || '',
        priority: editingRule.priority || rules.length + 1,
        enabled: editingRule.enabled ?? true,
        conditions: editingRule.conditions || [],
        conditionLogic: editingRule.conditionLogic || 'all',
        actions: editingRule.actions || [],
        tags: editingRule.tags || [],
        metrics: { totalCalls: 0, matchedCalls: 0, successRate: 0, avgHandleTime: 0 },
        createdAt: now,
        updatedAt: now,
        createdBy: 'Current User'
      };
      setRules(prev => [...prev, newRule]);
    }
    setShowCreateDialog(false);
  };

  const addCondition = () => {
    const newCondition: RoutingCondition = {
      id: `cond-${Date.now()}`,
      type: 'caller_id',
      operator: 'equals',
      value: ''
    };
    setEditingRule(prev => ({
      ...prev,
      conditions: [...(prev.conditions || []), newCondition]
    }));
  };

  const updateCondition = (condId: string, updates: Partial<RoutingCondition>) => {
    setEditingRule(prev => ({
      ...prev,
      conditions: prev.conditions?.map(c => c.id === condId ? { ...c, ...updates } : c) || []
    }));
  };

  const removeCondition = (condId: string) => {
    setEditingRule(prev => ({
      ...prev,
      conditions: prev.conditions?.filter(c => c.id !== condId) || []
    }));
  };

  const addAction = () => {
    const newAction: RoutingAction = {
      id: `action-${Date.now()}`,
      type: 'route_to_queue',
      target: ''
    };
    setEditingRule(prev => ({
      ...prev,
      actions: [...(prev.actions || []), newAction]
    }));
  };

  const updateAction = (actionId: string, updates: Partial<RoutingAction>) => {
    setEditingRule(prev => ({
      ...prev,
      actions: prev.actions?.map(a => a.id === actionId ? { ...a, ...updates } : a) || []
    }));
  };

  const removeAction = (actionId: string) => {
    setEditingRule(prev => ({
      ...prev,
      actions: prev.actions?.filter(a => a.id !== actionId) || []
    }));
  };

  const runTest = () => {
    // Simulate rule matching
    const matchedRule = rules.find(rule => {
      if (!rule.enabled) return false;
      // Simplified matching logic for demo
      return rule.conditions.some(cond => {
        if (cond.type === 'vip_status') return testCallData.vipStatus === cond.value;
        if (cond.type === 'ivr_selection') return testCallData.ivrSelection === cond.value;
        if (cond.type === 'caller_location') {
          if (Array.isArray(cond.value)) return cond.value.includes(testCallData.callerLocation);
          return testCallData.callerLocation === cond.value;
        }
        return false;
      });
    });

    if (matchedRule) {
      setTestResult({
        rule: matchedRule,
        actions: matchedRule.actions.map(a => {
          const actionType = actionTypes.find(at => at.type === a.type);
          return actionType?.name || a.type;
        })
      });
    } else {
      setTestResult({ rule: null, actions: ['Default routing - General Support Queue'] });
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  // Stats
  const stats = useMemo(() => {
    const enabledRules = rules.filter(r => r.enabled).length;
    const totalCalls = rules.reduce((sum, r) => sum + r.metrics.totalCalls, 0);
    const avgSuccessRate = rules.length > 0
      ? rules.reduce((sum, r) => sum + r.metrics.successRate, 0) / rules.length
      : 0;
    return { enabledRules, totalCalls, avgSuccessRate };
  }, [rules]);

  // Rule Card Component
  const RuleCard = ({ rule, index }: { rule: RoutingRule; index: number }) => (
    <div className={`bg-gray-800/50 rounded-xl border transition-all ${
      rule.enabled ? 'border-gray-700 hover:border-purple-500/50' : 'border-gray-700/50 opacity-60'
    }`}>
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              <button
                onClick={() => moveRule(rule.id, 'up')}
                disabled={index === 0}
                className="p-1 hover:bg-gray-700 rounded disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronDown className="w-4 h-4 text-gray-400 rotate-180" />
              </button>
              <span className="w-6 h-6 rounded-full bg-purple-500/20 text-purple-400 text-xs flex items-center justify-center font-medium">
                {rule.priority}
              </span>
              <button
                onClick={() => moveRule(rule.id, 'down')}
                disabled={index === filteredRules.length - 1}
                className="p-1 hover:bg-gray-700 rounded disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronDown className="w-4 h-4 text-gray-400" />
              </button>
            </div>
            <div>
              <h3 className="font-medium text-white">{rule.name}</h3>
              <p className="text-sm text-gray-400 line-clamp-1">{rule.description}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => toggleRule(rule.id)}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                rule.enabled ? 'bg-green-500' : 'bg-gray-600'
              }`}
            >
              <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                rule.enabled ? 'left-7' : 'left-1'
              }`} />
            </button>
          </div>
        </div>

        {/* Conditions & Actions Summary */}
        <div className="flex items-center gap-4 mb-3 text-sm">
          <div className="flex items-center gap-2 text-gray-400">
            <Filter className="w-4 h-4" />
            <span>{rule.conditions.length} condition{rule.conditions.length !== 1 ? 's' : ''}</span>
            <span className="px-1.5 py-0.5 bg-gray-700 rounded text-xs">
              {rule.conditionLogic === 'all' ? 'AND' : rule.conditionLogic === 'any' ? 'OR' : 'CUSTOM'}
            </span>
          </div>
          <ArrowRight className="w-4 h-4 text-gray-600" />
          <div className="flex items-center gap-2 text-gray-400">
            <Zap className="w-4 h-4" />
            <span>{rule.actions.length} action{rule.actions.length !== 1 ? 's' : ''}</span>
          </div>
        </div>

        {/* Conditions Preview */}
        <div className="flex flex-wrap gap-2 mb-3">
          {rule.conditions.slice(0, 3).map(cond => {
            const condType = conditionTypes.find(ct => ct.type === cond.type);
            const Icon = condType?.icon || Filter;
            return (
              <div key={cond.id} className="flex items-center gap-1.5 px-2 py-1 bg-blue-500/10 border border-blue-500/30 rounded-lg text-xs">
                <Icon className="w-3 h-3 text-blue-400" />
                <span className="text-blue-400">{condType?.name}</span>
              </div>
            );
          })}
          {rule.conditions.length > 3 && (
            <span className="px-2 py-1 text-xs text-gray-500">+{rule.conditions.length - 3} more</span>
          )}
        </div>

        {/* Actions Preview */}
        <div className="flex flex-wrap gap-2 mb-4">
          {rule.actions.map(action => {
            const actionType = actionTypes.find(at => at.type === action.type);
            const Icon = actionType?.icon || Zap;
            let targetName = '';
            if (action.target) {
              if (action.type === 'route_to_queue') {
                targetName = mockQueues.find(q => q.id === action.target)?.name || action.target;
              } else if (action.type === 'route_to_agent') {
                targetName = mockAgents.find(a => a.id === action.target)?.name || action.target;
              } else {
                targetName = action.target;
              }
            }
            return (
              <div key={action.id} className="flex items-center gap-1.5 px-2 py-1 bg-purple-500/10 border border-purple-500/30 rounded-lg text-xs">
                <Icon className="w-3 h-3 text-purple-400" />
                <span className="text-purple-400">{actionType?.name}</span>
                {targetName && (
                  <>
                    <ArrowRight className="w-3 h-3 text-gray-500" />
                    <span className="text-gray-400">{targetName}</span>
                  </>
                )}
              </div>
            );
          })}
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-4 gap-2 mb-4 p-3 bg-gray-900/50 rounded-lg">
          <div className="text-center">
            <p className="text-lg font-semibold text-white">{rule.metrics.totalCalls.toLocaleString()}</p>
            <p className="text-xs text-gray-500">Total Calls</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-semibold text-white">{rule.metrics.matchedCalls.toLocaleString()}</p>
            <p className="text-xs text-gray-500">Matched</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-semibold text-green-400">{rule.metrics.successRate}%</p>
            <p className="text-xs text-gray-500">Success Rate</p>
          </div>
          <div className="text-center">
            <p className="text-lg font-semibold text-white">{rule.metrics.avgHandleTime}s</p>
            <p className="text-xs text-gray-500">Avg Handle</p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {rule.tags.map(tag => (
              <span key={tag} className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-400">
                {tag}
              </span>
            ))}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => { setSelectedRule(rule); setShowRuleDialog(true); }}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              title="View Details"
            >
              <Eye className="w-4 h-4 text-gray-400" />
            </button>
            <button
              onClick={() => openEditDialog(rule)}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              title="Edit Rule"
            >
              <Edit3 className="w-4 h-4 text-gray-400" />
            </button>
            <button
              onClick={() => duplicateRule(rule)}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              title="Duplicate Rule"
            >
              <Copy className="w-4 h-4 text-gray-400" />
            </button>
            <button
              onClick={() => deleteRule(rule.id)}
              className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
              title="Delete Rule"
            >
              <Trash2 className="w-4 h-4 text-red-400" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  // Visual Flow Node
  const FlowNode = ({ rule, index }: { rule: RoutingRule; index: number }) => (
    <div className="relative">
      {/* Connection line */}
      {index > 0 && (
        <div className="absolute -top-8 left-1/2 w-0.5 h-8 bg-gradient-to-b from-gray-700 to-purple-500/50" />
      )}

      <div className={`bg-gray-800/80 rounded-xl border-2 p-4 w-80 ${
        rule.enabled ? 'border-purple-500/50' : 'border-gray-700 opacity-60'
      }`}>
        {/* Priority badge */}
        <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-purple-500 flex items-center justify-center text-white font-bold text-sm">
          {rule.priority}
        </div>

        {/* Status badge */}
        <div className={`absolute -top-2 -right-2 w-4 h-4 rounded-full ${
          rule.enabled ? 'bg-green-500' : 'bg-gray-500'
        }`} />

        <h4 className="font-medium text-white mb-2">{rule.name}</h4>

        {/* Conditions */}
        <div className="mb-3">
          <div className="flex items-center gap-2 mb-2">
            <Diamond className="w-4 h-4 text-blue-400" />
            <span className="text-xs text-gray-400 uppercase">If</span>
          </div>
          <div className="pl-4 space-y-1">
            {rule.conditions.slice(0, 2).map((cond, i) => (
              <div key={cond.id} className="text-xs text-gray-300 flex items-center gap-2">
                {i > 0 && (
                  <span className="text-purple-400">
                    {rule.conditionLogic === 'all' ? 'AND' : 'OR'}
                  </span>
                )}
                <span>{conditionTypes.find(ct => ct.type === cond.type)?.name}</span>
              </div>
            ))}
            {rule.conditions.length > 2 && (
              <span className="text-xs text-gray-500">+{rule.conditions.length - 2} more</span>
            )}
          </div>
        </div>

        {/* Actions */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Hexagon className="w-4 h-4 text-purple-400" />
            <span className="text-xs text-gray-400 uppercase">Then</span>
          </div>
          <div className="pl-4 space-y-1">
            {rule.actions.map(action => (
              <div key={action.id} className="text-xs text-gray-300">
                {actionTypes.find(at => at.type === action.type)?.name}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Arrow to next */}
      {index < filteredRules.length - 1 && (
        <div className="flex flex-col items-center mt-4">
          <ArrowDown className="w-5 h-5 text-gray-500" />
          <span className="text-xs text-gray-500 mt-1">No match</span>
        </div>
      )}
    </div>
  );

  return (
    <DashboardLayout>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">Call Routing Rules</h1>
            <p className="text-gray-400">Configure intelligent call routing based on conditions</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowTestDialog(true)}
              className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-2"
            >
              <TestTube className="w-5 h-5" />
              Test Routing
            </button>
            <button
              onClick={openCreateDialog}
              className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
            >
              <Plus className="w-5 h-5" />
              New Rule
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Total Rules</span>
              <GitBranch className="w-5 h-5 text-purple-400" />
            </div>
            <p className="text-2xl font-bold text-white">{rules.length}</p>
            <p className="text-sm text-green-400">{stats.enabledRules} active</p>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Calls Routed</span>
              <Phone className="w-5 h-5 text-blue-400" />
            </div>
            <p className="text-2xl font-bold text-white">{stats.totalCalls.toLocaleString()}</p>
            <p className="text-sm text-gray-500">Last 30 days</p>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Avg Success Rate</span>
              <TrendingUp className="w-5 h-5 text-green-400" />
            </div>
            <p className="text-2xl font-bold text-white">{stats.avgSuccessRate.toFixed(1)}%</p>
            <p className="text-sm text-green-400">+2.3% from last month</p>
          </div>
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Active Queues</span>
              <Users className="w-5 h-5 text-pink-400" />
            </div>
            <p className="text-2xl font-bold text-white">{mockQueues.length}</p>
            <p className="text-sm text-gray-500">{mockQueues.reduce((sum, q) => sum + q.waitingCalls, 0)} calls waiting</p>
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-4 flex-grow">
            <div className="relative flex-grow max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search rules..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>

            <div className="flex items-center gap-2">
              {['all', 'enabled', 'disabled'].map(filter => (
                <button
                  key={filter}
                  onClick={() => setFilterEnabled(filter as any)}
                  className={`px-3 py-1.5 rounded-lg text-sm capitalize transition-all ${
                    filterEnabled === filter
                      ? 'bg-purple-500 text-white'
                      : 'bg-gray-800/50 text-gray-400 hover:text-white border border-gray-700'
                  }`}
                >
                  {filter}
                </button>
              ))}
            </div>

            {allTags.length > 0 && (
              <div className="flex items-center gap-2">
                {allTags.slice(0, 5).map(tag => (
                  <button
                    key={tag}
                    onClick={() => setSelectedTags(prev =>
                      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
                    )}
                    className={`px-2 py-1 rounded text-xs transition-all ${
                      selectedTags.includes(tag)
                        ? 'bg-purple-500/20 text-purple-400 border border-purple-500/50'
                        : 'bg-gray-700/50 text-gray-400 border border-transparent'
                    }`}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="flex items-center gap-2 bg-gray-800/50 rounded-lg border border-gray-700 p-1">
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded transition-colors ${
                viewMode === 'list' ? 'bg-purple-500/20 text-purple-400' : 'text-gray-400 hover:text-white'
              }`}
            >
              <Layers className="w-5 h-5" />
            </button>
            <button
              onClick={() => setViewMode('visual')}
              className={`p-2 rounded transition-colors ${
                viewMode === 'visual' ? 'bg-purple-500/20 text-purple-400' : 'text-gray-400 hover:text-white'
              }`}
            >
              <GitBranch className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Rules Display */}
        {viewMode === 'list' ? (
          <div className="space-y-4">
            {filteredRules.map((rule, index) => (
              <RuleCard key={rule.id} rule={rule} index={index} />
            ))}
          </div>
        ) : (
          <div className="bg-gray-800/30 rounded-xl border border-gray-700 p-8 min-h-[500px] overflow-auto">
            <div className="flex flex-col items-center gap-8">
              {/* Start node */}
              <div className="bg-green-500/20 border-2 border-green-500/50 rounded-full px-6 py-3 flex items-center gap-2">
                <PhoneIncoming className="w-5 h-5 text-green-400" />
                <span className="text-green-400 font-medium">Incoming Call</span>
              </div>

              <ArrowDown className="w-5 h-5 text-gray-500" />

              {/* Rules */}
              {filteredRules.map((rule, index) => (
                <FlowNode key={rule.id} rule={rule} index={index} />
              ))}

              {/* Default routing */}
              <div className="flex flex-col items-center mt-4">
                <ArrowDown className="w-5 h-5 text-gray-500" />
                <div className="mt-4 bg-gray-700/50 border-2 border-gray-600 rounded-xl px-6 py-3 flex items-center gap-2">
                  <Users className="w-5 h-5 text-gray-400" />
                  <span className="text-gray-400 font-medium">Default Queue</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {filteredRules.length === 0 && (
          <div className="text-center py-12">
            <GitBranch className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">No routing rules found</h3>
            <p className="text-gray-400 mb-4">Create your first routing rule to get started</p>
            <button
              onClick={openCreateDialog}
              className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
            >
              Create Rule
            </button>
          </div>
        )}

        {/* Rule Detail Dialog */}
        {showRuleDialog && selectedRule && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                      selectedRule.enabled ? 'bg-purple-500/20' : 'bg-gray-700'
                    }`}>
                      <GitBranch className={`w-6 h-6 ${selectedRule.enabled ? 'text-purple-400' : 'text-gray-400'}`} />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-white">{selectedRule.name}</h2>
                      <p className="text-gray-400">{selectedRule.description}</p>
                    </div>
                  </div>
                  <button onClick={() => setShowRuleDialog(false)} className="p-2 hover:bg-gray-700 rounded-lg">
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              <div className="p-6 overflow-y-auto max-h-[60vh] space-y-6">
                {/* Conditions */}
                <div>
                  <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                    <Filter className="w-5 h-5 text-blue-400" />
                    Conditions
                    <span className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-400">
                      {selectedRule.conditionLogic.toUpperCase()}
                    </span>
                  </h3>
                  <div className="space-y-3">
                    {selectedRule.conditions.map(cond => {
                      const condType = conditionTypes.find(ct => ct.type === cond.type);
                      const Icon = condType?.icon || Filter;
                      return (
                        <div key={cond.id} className="flex items-center gap-4 p-4 bg-gray-700/30 rounded-xl">
                          <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                            <Icon className="w-5 h-5 text-blue-400" />
                          </div>
                          <div className="flex-grow">
                            <p className="font-medium text-white">{condType?.name}</p>
                            <p className="text-sm text-gray-400">
                              {cond.operator.replace('_', ' ')} {Array.isArray(cond.value) ? cond.value.join(', ') : String(cond.value)}
                            </p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Actions */}
                <div>
                  <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-purple-400" />
                    Actions
                  </h3>
                  <div className="space-y-3">
                    {selectedRule.actions.map((action, index) => {
                      const actionType = actionTypes.find(at => at.type === action.type);
                      const Icon = actionType?.icon || Zap;
                      return (
                        <div key={action.id} className="flex items-center gap-4 p-4 bg-gray-700/30 rounded-xl">
                          <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 font-medium text-sm">
                            {index + 1}
                          </div>
                          <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                            <Icon className="w-5 h-5 text-purple-400" />
                          </div>
                          <div className="flex-grow">
                            <p className="font-medium text-white">{actionType?.name}</p>
                            {action.target && (
                              <p className="text-sm text-gray-400">
                                Target: {action.type === 'route_to_queue'
                                  ? mockQueues.find(q => q.id === action.target)?.name
                                  : action.target}
                              </p>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Metrics */}
                <div>
                  <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-green-400" />
                    Performance Metrics
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gray-700/30 rounded-xl p-4 text-center">
                      <p className="text-2xl font-bold text-white">{selectedRule.metrics.totalCalls.toLocaleString()}</p>
                      <p className="text-sm text-gray-400">Total Calls</p>
                    </div>
                    <div className="bg-gray-700/30 rounded-xl p-4 text-center">
                      <p className="text-2xl font-bold text-white">{selectedRule.metrics.matchedCalls.toLocaleString()}</p>
                      <p className="text-sm text-gray-400">Matched</p>
                    </div>
                    <div className="bg-gray-700/30 rounded-xl p-4 text-center">
                      <p className="text-2xl font-bold text-green-400">{selectedRule.metrics.successRate}%</p>
                      <p className="text-sm text-gray-400">Success Rate</p>
                    </div>
                    <div className="bg-gray-700/30 rounded-xl p-4 text-center">
                      <p className="text-2xl font-bold text-white">{selectedRule.metrics.avgHandleTime}s</p>
                      <p className="text-sm text-gray-400">Avg Handle Time</p>
                    </div>
                  </div>
                </div>

                {/* Info */}
                <div className="flex items-center justify-between text-sm text-gray-400 pt-4 border-t border-gray-700">
                  <div>Created by {selectedRule.createdBy}</div>
                  <div>Updated {formatDate(selectedRule.updatedAt)}</div>
                </div>
              </div>

              <div className="p-6 border-t border-gray-700 flex justify-end gap-3">
                <button
                  onClick={() => { setShowRuleDialog(false); openEditDialog(selectedRule); }}
                  className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-2"
                >
                  <Edit3 className="w-4 h-4" />
                  Edit Rule
                </button>
                <button
                  onClick={() => toggleRule(selectedRule.id)}
                  className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
                    selectedRule.enabled
                      ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                      : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                  }`}
                >
                  {selectedRule.enabled ? (
                    <>
                      <Pause className="w-4 h-4" />
                      Disable
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Enable
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Create/Edit Dialog */}
        {showCreateDialog && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-bold text-white">
                    {isEditMode ? 'Edit Routing Rule' : 'Create Routing Rule'}
                  </h2>
                  <button onClick={() => setShowCreateDialog(false)} className="p-2 hover:bg-gray-700 rounded-lg">
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              <div className="p-6 overflow-y-auto max-h-[60vh] space-y-6">
                {/* Basic Info */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Rule Name</label>
                    <input
                      type="text"
                      value={editingRule.name}
                      onChange={(e) => setEditingRule(prev => ({ ...prev, name: e.target.value }))}
                      placeholder="Enter rule name"
                      className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Priority</label>
                    <input
                      type="number"
                      value={editingRule.priority}
                      onChange={(e) => setEditingRule(prev => ({ ...prev, priority: parseInt(e.target.value) || 1 }))}
                      min={1}
                      className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Description</label>
                  <textarea
                    value={editingRule.description}
                    onChange={(e) => setEditingRule(prev => ({ ...prev, description: e.target.value }))}
                    placeholder="Describe what this rule does"
                    rows={2}
                    className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                  />
                </div>

                {/* Conditions */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-medium text-white flex items-center gap-2">
                      <Filter className="w-5 h-5 text-blue-400" />
                      Conditions
                    </h3>
                    <div className="flex items-center gap-2">
                      <select
                        value={editingRule.conditionLogic}
                        onChange={(e) => setEditingRule(prev => ({ ...prev, conditionLogic: e.target.value as any }))}
                        className="px-3 py-1.5 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                      >
                        <option value="all">Match ALL conditions (AND)</option>
                        <option value="any">Match ANY condition (OR)</option>
                      </select>
                      <button
                        onClick={addCondition}
                        className="px-3 py-1.5 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors flex items-center gap-1 text-sm"
                      >
                        <Plus className="w-4 h-4" />
                        Add Condition
                      </button>
                    </div>
                  </div>

                  {editingRule.conditions && editingRule.conditions.length > 0 ? (
                    <div className="space-y-3">
                      {editingRule.conditions.map((cond, index) => (
                        <div key={cond.id} className="flex items-center gap-3 p-4 bg-gray-700/30 rounded-xl">
                          {index > 0 && (
                            <span className="px-2 py-0.5 bg-gray-600 rounded text-xs text-gray-300">
                              {editingRule.conditionLogic === 'all' ? 'AND' : 'OR'}
                            </span>
                          )}
                          <select
                            value={cond.type}
                            onChange={(e) => updateCondition(cond.id, { type: e.target.value as any })}
                            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm flex-grow"
                          >
                            {conditionTypes.map(ct => (
                              <option key={ct.type} value={ct.type}>{ct.name}</option>
                            ))}
                          </select>
                          <select
                            value={cond.operator}
                            onChange={(e) => updateCondition(cond.id, { operator: e.target.value as any })}
                            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
                          >
                            <option value="equals">equals</option>
                            <option value="not_equals">not equals</option>
                            <option value="contains">contains</option>
                            <option value="in">in list</option>
                            <option value="not_in">not in list</option>
                            <option value="greater_than">greater than</option>
                            <option value="less_than">less than</option>
                          </select>
                          <input
                            type="text"
                            value={Array.isArray(cond.value) ? cond.value.join(', ') : cond.value}
                            onChange={(e) => updateCondition(cond.id, { value: e.target.value })}
                            placeholder="Value"
                            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm flex-grow"
                          />
                          <button
                            onClick={() => removeCondition(cond.id)}
                            className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
                          >
                            <Trash2 className="w-4 h-4 text-red-400" />
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 bg-gray-700/20 rounded-xl border-2 border-dashed border-gray-600">
                      <Filter className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                      <p className="text-gray-500">No conditions added yet</p>
                      <button
                        onClick={addCondition}
                        className="mt-2 text-sm text-purple-400 hover:text-purple-300"
                      >
                        Add your first condition
                      </button>
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-medium text-white flex items-center gap-2">
                      <Zap className="w-5 h-5 text-purple-400" />
                      Actions
                    </h3>
                    <button
                      onClick={addAction}
                      className="px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors flex items-center gap-1 text-sm"
                    >
                      <Plus className="w-4 h-4" />
                      Add Action
                    </button>
                  </div>

                  {editingRule.actions && editingRule.actions.length > 0 ? (
                    <div className="space-y-3">
                      {editingRule.actions.map((action, index) => (
                        <div key={action.id} className="flex items-center gap-3 p-4 bg-gray-700/30 rounded-xl">
                          <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 font-medium text-sm">
                            {index + 1}
                          </div>
                          <select
                            value={action.type}
                            onChange={(e) => updateAction(action.id, { type: e.target.value as any })}
                            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm flex-grow"
                          >
                            {actionTypes.map(at => (
                              <option key={at.type} value={at.type}>{at.name}</option>
                            ))}
                          </select>
                          {['route_to_queue', 'route_to_agent'].includes(action.type) && (
                            <select
                              value={action.target || ''}
                              onChange={(e) => updateAction(action.id, { target: e.target.value })}
                              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm flex-grow"
                            >
                              <option value="">Select target</option>
                              {action.type === 'route_to_queue' && mockQueues.map(q => (
                                <option key={q.id} value={q.id}>{q.name}</option>
                              ))}
                              {action.type === 'route_to_agent' && mockAgents.map(a => (
                                <option key={a.id} value={a.id}>{a.name}</option>
                              ))}
                            </select>
                          )}
                          <button
                            onClick={() => removeAction(action.id)}
                            className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
                          >
                            <Trash2 className="w-4 h-4 text-red-400" />
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 bg-gray-700/20 rounded-xl border-2 border-dashed border-gray-600">
                      <Zap className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                      <p className="text-gray-500">No actions added yet</p>
                      <button
                        onClick={addAction}
                        className="mt-2 text-sm text-purple-400 hover:text-purple-300"
                      >
                        Add your first action
                      </button>
                    </div>
                  )}
                </div>

                {/* Tags */}
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Tags (comma separated)</label>
                  <input
                    type="text"
                    value={editingRule.tags?.join(', ') || ''}
                    onChange={(e) => setEditingRule(prev => ({
                      ...prev,
                      tags: e.target.value.split(',').map(t => t.trim()).filter(Boolean)
                    }))}
                    placeholder="vip, sales, priority"
                    className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                  />
                </div>
              </div>

              <div className="p-6 border-t border-gray-700 flex justify-between">
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={editingRule.enabled}
                      onChange={(e) => setEditingRule(prev => ({ ...prev, enabled: e.target.checked }))}
                      className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-purple-500"
                    />
                    <span className="text-sm text-gray-300">Enable rule immediately</span>
                  </label>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setShowCreateDialog(false)}
                    className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={saveRule}
                    disabled={!editingRule.name}
                    className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    <Save className="w-4 h-4" />
                    {isEditMode ? 'Save Changes' : 'Create Rule'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Test Dialog */}
        {showTestDialog && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl max-w-lg w-full overflow-hidden">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <TestTube className="w-5 h-5 text-purple-400" />
                    Test Call Routing
                  </h2>
                  <button onClick={() => setShowTestDialog(false)} className="p-2 hover:bg-gray-700 rounded-lg">
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              <div className="p-6 space-y-4">
                <p className="text-gray-400 text-sm">Simulate an incoming call to see which routing rules would match.</p>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-gray-400 mb-1 block">Caller ID</label>
                    <input
                      type="text"
                      value={testCallData.callerId}
                      onChange={(e) => setTestCallData(prev => ({ ...prev, callerId: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-1 block">Location</label>
                    <select
                      value={testCallData.callerLocation}
                      onChange={(e) => setTestCallData(prev => ({ ...prev, callerLocation: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm"
                    >
                      <option value="US">United States</option>
                      <option value="GB">United Kingdom</option>
                      <option value="AU">Australia</option>
                      <option value="JP">Japan</option>
                      <option value="SG">Singapore</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-1 block">IVR Selection</label>
                    <select
                      value={testCallData.ivrSelection}
                      onChange={(e) => setTestCallData(prev => ({ ...prev, ivrSelection: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm"
                    >
                      <option value="1">1 - Sales</option>
                      <option value="2">2 - Technical Support</option>
                      <option value="3">3 - Billing</option>
                      <option value="4">4 - General Inquiry</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-1 block">Time</label>
                    <input
                      type="time"
                      value={testCallData.currentTime}
                      onChange={(e) => setTestCallData(prev => ({ ...prev, currentTime: e.target.value }))}
                      className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm"
                    />
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={testCallData.vipStatus}
                      onChange={(e) => setTestCallData(prev => ({ ...prev, vipStatus: e.target.checked }))}
                      className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-purple-500"
                    />
                    <span className="text-sm text-gray-300">VIP Customer</span>
                  </label>
                </div>

                <button
                  onClick={runTest}
                  className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg font-medium hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Run Test
                </button>

                {/* Test Result */}
                {testResult && (
                  <div className="mt-4 p-4 bg-gray-700/30 rounded-xl">
                    <h4 className="font-medium text-white mb-3">Test Result</h4>
                    {testResult.rule ? (
                      <>
                        <div className="flex items-center gap-3 mb-3 p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                          <Check className="w-5 h-5 text-green-400" />
                          <div>
                            <p className="text-green-400 font-medium">Matched Rule: {testResult.rule.name}</p>
                            <p className="text-sm text-gray-400">Priority {testResult.rule.priority}</p>
                          </div>
                        </div>
                        <div>
                          <p className="text-sm text-gray-400 mb-2">Actions that would execute:</p>
                          <div className="space-y-1">
                            {testResult.actions.map((action, i) => (
                              <div key={i} className="flex items-center gap-2 text-sm">
                                <span className="w-5 h-5 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-xs">
                                  {i + 1}
                                </span>
                                <span className="text-gray-300">{action}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="flex items-center gap-3 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                        <AlertTriangle className="w-5 h-5 text-yellow-400" />
                        <div>
                          <p className="text-yellow-400 font-medium">No rule matched</p>
                          <p className="text-sm text-gray-400">Call would go to: {testResult.actions[0]}</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
