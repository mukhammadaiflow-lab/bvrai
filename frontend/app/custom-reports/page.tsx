"use client";

import React, { useState, useMemo } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import {
  BarChart3,
  LineChart,
  PieChart,
  TrendingUp,
  Table2,
  FileText,
  Download,
  Share2,
  Calendar,
  Clock,
  Filter,
  Plus,
  Trash2,
  Edit3,
  Copy,
  Eye,
  Play,
  Pause,
  Settings,
  Database,
  Layers,
  Move,
  ChevronRight,
  ChevronDown,
  Check,
  X,
  RefreshCw,
  Mail,
  Slack,
  Save,
  FolderOpen,
  Search,
  Star,
  StarOff,
  MoreVertical,
  GripVertical,
  ArrowUpDown,
  Maximize2,
  Minimize2,
  Code,
  Palette,
  Layout,
  Image,
  FileSpreadsheet,
  Presentation,
  Send,
  Users,
  Building2,
  Globe,
  Phone,
  MessageSquare,
  Bot,
  Target,
  Zap,
  AlertTriangle,
  Info,
  HelpCircle,
  ArrowLeft,
  ArrowRight,
  RotateCcw
} from 'lucide-react';

// Types
interface DataSource {
  id: string;
  name: string;
  type: 'calls' | 'agents' | 'customers' | 'campaigns' | 'performance' | 'custom';
  icon: React.ElementType;
  fields: DataField[];
  description: string;
}

interface DataField {
  id: string;
  name: string;
  type: 'number' | 'string' | 'date' | 'boolean' | 'currency' | 'percentage' | 'duration';
  aggregations?: ('sum' | 'avg' | 'count' | 'min' | 'max')[];
}

interface ReportWidget {
  id: string;
  type: 'bar' | 'line' | 'pie' | 'donut' | 'area' | 'table' | 'metric' | 'gauge' | 'heatmap' | 'funnel' | 'scatter';
  title: string;
  dataSource: string;
  dimensions: string[];
  measures: string[];
  filters: ReportFilter[];
  position: { x: number; y: number; width: number; height: number };
  style: WidgetStyle;
}

interface ReportFilter {
  field: string;
  operator: 'equals' | 'not_equals' | 'contains' | 'greater_than' | 'less_than' | 'between' | 'in';
  value: any;
}

interface WidgetStyle {
  backgroundColor?: string;
  borderColor?: string;
  chartColors?: string[];
  showLegend?: boolean;
  showLabels?: boolean;
  showGrid?: boolean;
}

interface Report {
  id: string;
  name: string;
  description: string;
  category: string;
  widgets: ReportWidget[];
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  isPublic: boolean;
  isFavorite: boolean;
  schedules: ReportSchedule[];
  thumbnail?: string;
  tags: string[];
}

interface ReportSchedule {
  id: string;
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
  time: string;
  dayOfWeek?: number;
  dayOfMonth?: number;
  recipients: string[];
  format: 'pdf' | 'excel' | 'csv';
  enabled: boolean;
}

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  thumbnail: string;
  widgets: Partial<ReportWidget>[];
  popularity: number;
}

// Mock Data
const dataSources: DataSource[] = [
  {
    id: 'calls',
    name: 'Call Data',
    type: 'calls',
    icon: Phone,
    description: 'All call records and metrics',
    fields: [
      { id: 'call_id', name: 'Call ID', type: 'string' },
      { id: 'duration', name: 'Duration', type: 'duration', aggregations: ['sum', 'avg', 'min', 'max'] },
      { id: 'start_time', name: 'Start Time', type: 'date' },
      { id: 'end_time', name: 'End Time', type: 'date' },
      { id: 'status', name: 'Status', type: 'string' },
      { id: 'direction', name: 'Direction', type: 'string' },
      { id: 'sentiment_score', name: 'Sentiment Score', type: 'number', aggregations: ['avg', 'min', 'max'] },
      { id: 'resolution_status', name: 'Resolution Status', type: 'string' },
      { id: 'wait_time', name: 'Wait Time', type: 'duration', aggregations: ['sum', 'avg', 'min', 'max'] },
      { id: 'transfer_count', name: 'Transfer Count', type: 'number', aggregations: ['sum', 'avg'] },
      { id: 'cost', name: 'Cost', type: 'currency', aggregations: ['sum', 'avg'] },
    ]
  },
  {
    id: 'agents',
    name: 'Agent Data',
    type: 'agents',
    icon: Bot,
    description: 'AI agent performance and configuration',
    fields: [
      { id: 'agent_id', name: 'Agent ID', type: 'string' },
      { id: 'agent_name', name: 'Agent Name', type: 'string' },
      { id: 'calls_handled', name: 'Calls Handled', type: 'number', aggregations: ['sum', 'avg'] },
      { id: 'success_rate', name: 'Success Rate', type: 'percentage', aggregations: ['avg'] },
      { id: 'avg_handle_time', name: 'Avg Handle Time', type: 'duration', aggregations: ['avg'] },
      { id: 'customer_satisfaction', name: 'Customer Satisfaction', type: 'number', aggregations: ['avg'] },
      { id: 'active_status', name: 'Active Status', type: 'boolean' },
      { id: 'escalation_rate', name: 'Escalation Rate', type: 'percentage', aggregations: ['avg'] },
    ]
  },
  {
    id: 'customers',
    name: 'Customer Data',
    type: 'customers',
    icon: Users,
    description: 'Customer information and interactions',
    fields: [
      { id: 'customer_id', name: 'Customer ID', type: 'string' },
      { id: 'customer_name', name: 'Customer Name', type: 'string' },
      { id: 'company', name: 'Company', type: 'string' },
      { id: 'total_calls', name: 'Total Calls', type: 'number', aggregations: ['sum', 'avg'] },
      { id: 'lifetime_value', name: 'Lifetime Value', type: 'currency', aggregations: ['sum', 'avg'] },
      { id: 'satisfaction_score', name: 'Satisfaction Score', type: 'number', aggregations: ['avg'] },
      { id: 'last_interaction', name: 'Last Interaction', type: 'date' },
      { id: 'segment', name: 'Segment', type: 'string' },
    ]
  },
  {
    id: 'campaigns',
    name: 'Campaign Data',
    type: 'campaigns',
    icon: Target,
    description: 'Outbound campaign performance',
    fields: [
      { id: 'campaign_id', name: 'Campaign ID', type: 'string' },
      { id: 'campaign_name', name: 'Campaign Name', type: 'string' },
      { id: 'calls_made', name: 'Calls Made', type: 'number', aggregations: ['sum'] },
      { id: 'connections', name: 'Connections', type: 'number', aggregations: ['sum'] },
      { id: 'conversions', name: 'Conversions', type: 'number', aggregations: ['sum'] },
      { id: 'conversion_rate', name: 'Conversion Rate', type: 'percentage', aggregations: ['avg'] },
      { id: 'revenue', name: 'Revenue', type: 'currency', aggregations: ['sum', 'avg'] },
      { id: 'cost_per_lead', name: 'Cost per Lead', type: 'currency', aggregations: ['avg'] },
      { id: 'start_date', name: 'Start Date', type: 'date' },
      { id: 'end_date', name: 'End Date', type: 'date' },
    ]
  },
  {
    id: 'performance',
    name: 'Performance Metrics',
    type: 'performance',
    icon: TrendingUp,
    description: 'System-wide performance indicators',
    fields: [
      { id: 'date', name: 'Date', type: 'date' },
      { id: 'total_calls', name: 'Total Calls', type: 'number', aggregations: ['sum'] },
      { id: 'answered_calls', name: 'Answered Calls', type: 'number', aggregations: ['sum'] },
      { id: 'abandoned_calls', name: 'Abandoned Calls', type: 'number', aggregations: ['sum'] },
      { id: 'service_level', name: 'Service Level', type: 'percentage', aggregations: ['avg'] },
      { id: 'avg_speed_answer', name: 'Avg Speed to Answer', type: 'duration', aggregations: ['avg'] },
      { id: 'first_call_resolution', name: 'First Call Resolution', type: 'percentage', aggregations: ['avg'] },
      { id: 'occupancy_rate', name: 'Occupancy Rate', type: 'percentage', aggregations: ['avg'] },
    ]
  }
];

const reportTemplates: ReportTemplate[] = [
  {
    id: 'exec-summary',
    name: 'Executive Summary',
    description: 'High-level overview of call center performance for leadership',
    category: 'Executive',
    thumbnail: '📊',
    popularity: 95,
    widgets: [
      { type: 'metric', title: 'Total Calls', dataSource: 'calls' },
      { type: 'metric', title: 'Success Rate', dataSource: 'agents' },
      { type: 'line', title: 'Call Volume Trend', dataSource: 'performance' },
      { type: 'pie', title: 'Call Distribution', dataSource: 'calls' },
    ]
  },
  {
    id: 'agent-performance',
    name: 'Agent Performance Report',
    description: 'Detailed analysis of AI agent effectiveness and metrics',
    category: 'Operations',
    thumbnail: '🤖',
    popularity: 88,
    widgets: [
      { type: 'table', title: 'Agent Leaderboard', dataSource: 'agents' },
      { type: 'bar', title: 'Calls by Agent', dataSource: 'agents' },
      { type: 'gauge', title: 'Average Success Rate', dataSource: 'agents' },
    ]
  },
  {
    id: 'campaign-roi',
    name: 'Campaign ROI Analysis',
    description: 'Track campaign performance and return on investment',
    category: 'Marketing',
    thumbnail: '🎯',
    popularity: 82,
    widgets: [
      { type: 'bar', title: 'Revenue by Campaign', dataSource: 'campaigns' },
      { type: 'funnel', title: 'Conversion Funnel', dataSource: 'campaigns' },
      { type: 'metric', title: 'Total ROI', dataSource: 'campaigns' },
    ]
  },
  {
    id: 'customer-insights',
    name: 'Customer Insights Dashboard',
    description: 'Understand customer behavior and satisfaction trends',
    category: 'Customer Success',
    thumbnail: '👥',
    popularity: 79,
    widgets: [
      { type: 'pie', title: 'Customer Segments', dataSource: 'customers' },
      { type: 'line', title: 'Satisfaction Over Time', dataSource: 'customers' },
      { type: 'heatmap', title: 'Interaction Patterns', dataSource: 'calls' },
    ]
  },
  {
    id: 'quality-metrics',
    name: 'Quality Assurance Report',
    description: 'Monitor call quality and compliance metrics',
    category: 'Quality',
    thumbnail: '✅',
    popularity: 75,
    widgets: [
      { type: 'gauge', title: 'Quality Score', dataSource: 'calls' },
      { type: 'bar', title: 'Issues by Category', dataSource: 'calls' },
      { type: 'line', title: 'Quality Trend', dataSource: 'performance' },
    ]
  },
  {
    id: 'cost-analysis',
    name: 'Cost Analysis Report',
    description: 'Analyze operational costs and efficiency metrics',
    category: 'Finance',
    thumbnail: '💰',
    popularity: 71,
    widgets: [
      { type: 'area', title: 'Cost Over Time', dataSource: 'calls' },
      { type: 'bar', title: 'Cost by Category', dataSource: 'calls' },
      { type: 'metric', title: 'Cost per Call', dataSource: 'calls' },
    ]
  }
];

const mockReports: Report[] = [
  {
    id: 'rpt-1',
    name: 'Weekly Performance Dashboard',
    description: 'Comprehensive weekly overview of all call center metrics',
    category: 'Operations',
    widgets: [],
    createdAt: '2024-01-15T10:30:00Z',
    updatedAt: '2024-01-20T14:45:00Z',
    createdBy: 'John Smith',
    isPublic: true,
    isFavorite: true,
    schedules: [
      {
        id: 'sch-1',
        frequency: 'weekly',
        time: '08:00',
        dayOfWeek: 1,
        recipients: ['team@company.com', 'manager@company.com'],
        format: 'pdf',
        enabled: true
      }
    ],
    tags: ['weekly', 'performance', 'kpi']
  },
  {
    id: 'rpt-2',
    name: 'Agent Efficiency Analysis',
    description: 'Deep dive into AI agent performance and optimization opportunities',
    category: 'Operations',
    widgets: [],
    createdAt: '2024-01-10T09:00:00Z',
    updatedAt: '2024-01-19T11:20:00Z',
    createdBy: 'Sarah Johnson',
    isPublic: false,
    isFavorite: true,
    schedules: [],
    tags: ['agents', 'efficiency', 'optimization']
  },
  {
    id: 'rpt-3',
    name: 'Q1 Campaign Results',
    description: 'Quarterly analysis of outbound campaign performance',
    category: 'Marketing',
    widgets: [],
    createdAt: '2024-01-05T08:00:00Z',
    updatedAt: '2024-01-18T16:30:00Z',
    createdBy: 'Mike Wilson',
    isPublic: true,
    isFavorite: false,
    schedules: [
      {
        id: 'sch-2',
        frequency: 'monthly',
        time: '09:00',
        dayOfMonth: 1,
        recipients: ['marketing@company.com'],
        format: 'excel',
        enabled: true
      }
    ],
    tags: ['campaigns', 'quarterly', 'marketing']
  },
  {
    id: 'rpt-4',
    name: 'Customer Satisfaction Trends',
    description: 'Track and analyze customer satisfaction scores over time',
    category: 'Customer Success',
    widgets: [],
    createdAt: '2024-01-08T11:00:00Z',
    updatedAt: '2024-01-17T13:15:00Z',
    createdBy: 'Emily Chen',
    isPublic: true,
    isFavorite: false,
    schedules: [],
    tags: ['csat', 'customers', 'trends']
  },
  {
    id: 'rpt-5',
    name: 'Daily Operations Summary',
    description: 'Daily snapshot of operational metrics and alerts',
    category: 'Operations',
    widgets: [],
    createdAt: '2024-01-12T07:00:00Z',
    updatedAt: '2024-01-20T07:00:00Z',
    createdBy: 'John Smith',
    isPublic: false,
    isFavorite: true,
    schedules: [
      {
        id: 'sch-3',
        frequency: 'daily',
        time: '07:00',
        recipients: ['ops@company.com'],
        format: 'pdf',
        enabled: true
      }
    ],
    tags: ['daily', 'operations', 'summary']
  }
];

const widgetTypes = [
  { type: 'bar', name: 'Bar Chart', icon: BarChart3, description: 'Compare values across categories' },
  { type: 'line', name: 'Line Chart', icon: LineChart, description: 'Show trends over time' },
  { type: 'pie', name: 'Pie Chart', icon: PieChart, description: 'Show proportions of a whole' },
  { type: 'donut', name: 'Donut Chart', icon: PieChart, description: 'Pie chart with center cut out' },
  { type: 'area', name: 'Area Chart', icon: TrendingUp, description: 'Show cumulative totals over time' },
  { type: 'table', name: 'Data Table', icon: Table2, description: 'Display data in tabular format' },
  { type: 'metric', name: 'Single Metric', icon: FileText, description: 'Highlight a key number' },
  { type: 'gauge', name: 'Gauge', icon: Target, description: 'Show progress toward a goal' },
  { type: 'heatmap', name: 'Heat Map', icon: Layers, description: 'Visualize data density' },
  { type: 'funnel', name: 'Funnel', icon: Filter, description: 'Show conversion stages' },
  { type: 'scatter', name: 'Scatter Plot', icon: Move, description: 'Show correlation between variables' },
];

const categories = ['All', 'Operations', 'Marketing', 'Customer Success', 'Finance', 'Quality', 'Executive'];

export default function CustomReportsPage() {
  const [activeTab, setActiveTab] = useState<'reports' | 'templates' | 'builder'>('reports');
  const [reports, setReports] = useState<Report[]>(mockReports);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [showOnlyFavorites, setShowOnlyFavorites] = useState(false);
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [showReportDialog, setShowReportDialog] = useState(false);
  const [showBuilderDialog, setShowBuilderDialog] = useState(false);
  const [showScheduleDialog, setShowScheduleDialog] = useState(false);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);

  // Builder state
  const [builderReport, setBuilderReport] = useState<Partial<Report>>({
    name: '',
    description: '',
    category: 'Operations',
    widgets: [],
    tags: []
  });
  const [selectedDataSource, setSelectedDataSource] = useState<string>('calls');
  const [selectedWidgetType, setSelectedWidgetType] = useState<string>('bar');
  const [builderWidgets, setBuilderWidgets] = useState<ReportWidget[]>([]);
  const [activeWidget, setActiveWidget] = useState<ReportWidget | null>(null);
  const [draggedField, setDraggedField] = useState<DataField | null>(null);
  const [showDataSourcePanel, setShowDataSourcePanel] = useState(true);
  const [previewMode, setPreviewMode] = useState(false);

  // Schedule state
  const [newSchedule, setNewSchedule] = useState<Partial<ReportSchedule>>({
    frequency: 'weekly',
    time: '08:00',
    dayOfWeek: 1,
    recipients: [],
    format: 'pdf',
    enabled: true
  });
  const [recipientEmail, setRecipientEmail] = useState('');

  const filteredReports = useMemo(() => {
    return reports.filter(report => {
      const matchesCategory = selectedCategory === 'All' || report.category === selectedCategory;
      const matchesSearch = report.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           report.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           report.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
      const matchesFavorite = !showOnlyFavorites || report.isFavorite;
      return matchesCategory && matchesSearch && matchesFavorite;
    });
  }, [reports, selectedCategory, searchQuery, showOnlyFavorites]);

  const filteredTemplates = useMemo(() => {
    return reportTemplates.filter(template => {
      const matchesCategory = selectedCategory === 'All' || template.category === selectedCategory;
      const matchesSearch = template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           template.description.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesCategory && matchesSearch;
    }).sort((a, b) => b.popularity - a.popularity);
  }, [selectedCategory, searchQuery]);

  const currentDataSource = dataSources.find(ds => ds.id === selectedDataSource);

  const toggleFavorite = (reportId: string) => {
    setReports(prev => prev.map(report =>
      report.id === reportId ? { ...report, isFavorite: !report.isFavorite } : report
    ));
  };

  const deleteReport = (reportId: string) => {
    setReports(prev => prev.filter(r => r.id !== reportId));
  };

  const duplicateReport = (report: Report) => {
    const newReport: Report = {
      ...report,
      id: `rpt-${Date.now()}`,
      name: `${report.name} (Copy)`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      schedules: []
    };
    setReports(prev => [...prev, newReport]);
  };

  const startFromTemplate = (template: ReportTemplate) => {
    setBuilderReport({
      name: template.name,
      description: template.description,
      category: template.category,
      widgets: [],
      tags: []
    });
    setActiveTab('builder');
    setIsEditMode(false);
  };

  const editReport = (report: Report) => {
    setBuilderReport(report);
    setBuilderWidgets(report.widgets);
    setActiveTab('builder');
    setIsEditMode(true);
  };

  const addWidget = () => {
    const newWidget: ReportWidget = {
      id: `widget-${Date.now()}`,
      type: selectedWidgetType as any,
      title: `New ${widgetTypes.find(w => w.type === selectedWidgetType)?.name}`,
      dataSource: selectedDataSource,
      dimensions: [],
      measures: [],
      filters: [],
      position: {
        x: (builderWidgets.length % 2) * 6,
        y: Math.floor(builderWidgets.length / 2) * 4,
        width: 6,
        height: 4
      },
      style: {
        showLegend: true,
        showLabels: true,
        showGrid: true,
        chartColors: ['#8B5CF6', '#EC4899', '#3B82F6', '#10B981', '#F59E0B']
      }
    };
    setBuilderWidgets(prev => [...prev, newWidget]);
    setActiveWidget(newWidget);
  };

  const removeWidget = (widgetId: string) => {
    setBuilderWidgets(prev => prev.filter(w => w.id !== widgetId));
    if (activeWidget?.id === widgetId) {
      setActiveWidget(null);
    }
  };

  const updateWidget = (widgetId: string, updates: Partial<ReportWidget>) => {
    setBuilderWidgets(prev => prev.map(w =>
      w.id === widgetId ? { ...w, ...updates } : w
    ));
    if (activeWidget?.id === widgetId) {
      setActiveWidget(prev => prev ? { ...prev, ...updates } : null);
    }
  };

  const saveReport = () => {
    const newReport: Report = {
      id: isEditMode && builderReport.id ? builderReport.id : `rpt-${Date.now()}`,
      name: builderReport.name || 'Untitled Report',
      description: builderReport.description || '',
      category: builderReport.category || 'Operations',
      widgets: builderWidgets,
      createdAt: isEditMode && builderReport.createdAt ? builderReport.createdAt : new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      createdBy: 'Current User',
      isPublic: false,
      isFavorite: false,
      schedules: [],
      tags: builderReport.tags || []
    };

    if (isEditMode) {
      setReports(prev => prev.map(r => r.id === newReport.id ? newReport : r));
    } else {
      setReports(prev => [...prev, newReport]);
    }

    setActiveTab('reports');
    setBuilderReport({ name: '', description: '', category: 'Operations', widgets: [], tags: [] });
    setBuilderWidgets([]);
    setIsEditMode(false);
  };

  const addRecipient = () => {
    if (recipientEmail && !newSchedule.recipients?.includes(recipientEmail)) {
      setNewSchedule(prev => ({
        ...prev,
        recipients: [...(prev.recipients || []), recipientEmail]
      }));
      setRecipientEmail('');
    }
  };

  const removeRecipient = (email: string) => {
    setNewSchedule(prev => ({
      ...prev,
      recipients: prev.recipients?.filter(r => r !== email) || []
    }));
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  // Report Card Component
  const ReportCard = ({ report }: { report: Report }) => (
    <div className="bg-gray-800/50 rounded-xl border border-gray-700 hover:border-purple-500/50 transition-all duration-200 overflow-hidden group">
      {/* Thumbnail */}
      <div className="aspect-video bg-gradient-to-br from-purple-500/20 to-pink-500/20 relative flex items-center justify-center">
        <div className="grid grid-cols-2 gap-2 p-4 opacity-50">
          <div className="h-8 bg-purple-500/30 rounded"></div>
          <div className="h-8 bg-pink-500/30 rounded"></div>
          <div className="h-8 bg-blue-500/30 rounded col-span-2"></div>
        </div>

        {/* Overlay actions */}
        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
          <button
            onClick={() => { setSelectedReport(report); setShowReportDialog(true); }}
            className="p-2 bg-white/20 rounded-lg hover:bg-white/30 transition-colors"
          >
            <Eye className="w-5 h-5 text-white" />
          </button>
          <button
            onClick={() => editReport(report)}
            className="p-2 bg-white/20 rounded-lg hover:bg-white/30 transition-colors"
          >
            <Edit3 className="w-5 h-5 text-white" />
          </button>
          <button
            onClick={() => duplicateReport(report)}
            className="p-2 bg-white/20 rounded-lg hover:bg-white/30 transition-colors"
          >
            <Copy className="w-5 h-5 text-white" />
          </button>
        </div>

        {/* Favorite badge */}
        <button
          onClick={() => toggleFavorite(report.id)}
          className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/30 hover:bg-black/50 transition-colors"
        >
          {report.isFavorite ? (
            <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
          ) : (
            <StarOff className="w-4 h-4 text-gray-400" />
          )}
        </button>

        {/* Schedule indicator */}
        {report.schedules.some(s => s.enabled) && (
          <div className="absolute top-2 left-2 px-2 py-1 rounded-lg bg-green-500/20 border border-green-500/30 flex items-center gap-1">
            <Clock className="w-3 h-3 text-green-400" />
            <span className="text-xs text-green-400">Scheduled</span>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <h3 className="font-medium text-white group-hover:text-purple-400 transition-colors line-clamp-1">
            {report.name}
          </h3>
          <button className="p-1 hover:bg-gray-700 rounded opacity-0 group-hover:opacity-100 transition-opacity">
            <MoreVertical className="w-4 h-4 text-gray-400" />
          </button>
        </div>

        <p className="text-sm text-gray-400 mb-3 line-clamp-2">{report.description}</p>

        <div className="flex items-center justify-between text-xs text-gray-500">
          <span className="px-2 py-1 bg-gray-700/50 rounded">{report.category}</span>
          <span>Updated {formatDate(report.updatedAt)}</span>
        </div>

        {/* Tags */}
        {report.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-3">
            {report.tags.slice(0, 3).map(tag => (
              <span key={tag} className="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs">
                {tag}
              </span>
            ))}
            {report.tags.length > 3 && (
              <span className="px-2 py-0.5 text-gray-500 text-xs">+{report.tags.length - 3}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );

  // Report List Item
  const ReportListItem = ({ report }: { report: Report }) => (
    <div className="bg-gray-800/50 rounded-xl border border-gray-700 hover:border-purple-500/50 transition-all p-4 flex items-center gap-4 group">
      {/* Icon */}
      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center flex-shrink-0">
        <FileText className="w-6 h-6 text-purple-400" />
      </div>

      {/* Content */}
      <div className="flex-grow min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <h3 className="font-medium text-white truncate">{report.name}</h3>
          {report.isFavorite && <Star className="w-4 h-4 text-yellow-400 fill-yellow-400 flex-shrink-0" />}
          {report.schedules.some(s => s.enabled) && (
            <span className="px-2 py-0.5 rounded bg-green-500/20 text-green-400 text-xs flex-shrink-0">
              Scheduled
            </span>
          )}
        </div>
        <p className="text-sm text-gray-400 truncate">{report.description}</p>
        <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
          <span className="px-2 py-0.5 bg-gray-700/50 rounded">{report.category}</span>
          <span>Created by {report.createdBy}</span>
          <span>Updated {formatDate(report.updatedAt)}</span>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => toggleFavorite(report.id)}
          className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
        >
          {report.isFavorite ? (
            <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
          ) : (
            <StarOff className="w-4 h-4 text-gray-400" />
          )}
        </button>
        <button
          onClick={() => { setSelectedReport(report); setShowReportDialog(true); }}
          className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Eye className="w-4 h-4 text-gray-400" />
        </button>
        <button
          onClick={() => editReport(report)}
          className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Edit3 className="w-4 h-4 text-gray-400" />
        </button>
        <button
          onClick={() => duplicateReport(report)}
          className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Copy className="w-4 h-4 text-gray-400" />
        </button>
        <button
          onClick={() => { setSelectedReport(report); setShowExportDialog(true); }}
          className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
        >
          <Download className="w-4 h-4 text-gray-400" />
        </button>
        <button
          onClick={() => deleteReport(report.id)}
          className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
        >
          <Trash2 className="w-4 h-4 text-red-400" />
        </button>
      </div>
    </div>
  );

  // Template Card Component
  const TemplateCard = ({ template }: { template: ReportTemplate }) => (
    <div
      onClick={() => startFromTemplate(template)}
      className="bg-gray-800/50 rounded-xl border border-gray-700 hover:border-purple-500/50 transition-all duration-200 overflow-hidden cursor-pointer group"
    >
      {/* Thumbnail */}
      <div className="aspect-video bg-gradient-to-br from-purple-500/10 to-pink-500/10 relative flex items-center justify-center">
        <span className="text-5xl">{template.thumbnail}</span>

        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
          <span className="px-4 py-2 bg-purple-500 rounded-lg text-white font-medium flex items-center gap-2">
            <Plus className="w-4 h-4" /> Use Template
          </span>
        </div>

        {/* Popularity badge */}
        <div className="absolute top-2 right-2 px-2 py-1 rounded-lg bg-black/30 flex items-center gap-1">
          <TrendingUp className="w-3 h-3 text-green-400" />
          <span className="text-xs text-green-400">{template.popularity}%</span>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        <h3 className="font-medium text-white mb-1 group-hover:text-purple-400 transition-colors">
          {template.name}
        </h3>
        <p className="text-sm text-gray-400 mb-3 line-clamp-2">{template.description}</p>
        <div className="flex items-center justify-between">
          <span className="text-xs px-2 py-1 bg-gray-700/50 rounded text-gray-400">
            {template.category}
          </span>
          <span className="text-xs text-gray-500">
            {template.widgets.length} widgets
          </span>
        </div>
      </div>
    </div>
  );

  // Widget Component for Builder
  const WidgetPreview = ({ widget, isActive }: { widget: ReportWidget; isActive: boolean }) => {
    const widgetInfo = widgetTypes.find(w => w.type === widget.type);
    const Icon = widgetInfo?.icon || BarChart3;

    return (
      <div
        onClick={() => setActiveWidget(widget)}
        className={`bg-gray-800/50 rounded-xl border-2 transition-all cursor-pointer relative group
          ${isActive ? 'border-purple-500' : 'border-gray-700 hover:border-gray-600'}`}
        style={{
          gridColumn: `span ${widget.position.width}`,
          gridRow: `span ${widget.position.height}`,
          minHeight: '150px'
        }}
      >
        {/* Widget Header */}
        <div className="flex items-center justify-between p-3 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <GripVertical className="w-4 h-4 text-gray-500 cursor-move" />
            <span className="font-medium text-white text-sm">{widget.title}</span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={(e) => { e.stopPropagation(); }}
              className="p-1 hover:bg-gray-700 rounded opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <Settings className="w-3 h-3 text-gray-400" />
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); removeWidget(widget.id); }}
              className="p-1 hover:bg-red-500/20 rounded opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <X className="w-3 h-3 text-red-400" />
            </button>
          </div>
        </div>

        {/* Widget Preview */}
        <div className="p-4 flex items-center justify-center h-[calc(100%-50px)]">
          {widget.type === 'bar' && (
            <div className="flex items-end gap-2 h-full w-full">
              {[60, 80, 45, 90, 70, 55].map((h, i) => (
                <div key={i} className="flex-1 bg-gradient-to-t from-purple-500 to-pink-500 rounded-t opacity-50" style={{ height: `${h}%` }} />
              ))}
            </div>
          )}
          {widget.type === 'line' && (
            <svg className="w-full h-full" viewBox="0 0 100 50">
              <polyline
                points="0,40 20,30 40,35 60,20 80,25 100,15"
                fill="none"
                stroke="url(#lineGradient)"
                strokeWidth="2"
              />
              <defs>
                <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#8B5CF6" />
                  <stop offset="100%" stopColor="#EC4899" />
                </linearGradient>
              </defs>
            </svg>
          )}
          {widget.type === 'pie' && (
            <svg className="w-20 h-20" viewBox="0 0 32 32">
              <circle cx="16" cy="16" r="12" fill="transparent" stroke="#8B5CF6" strokeWidth="8" strokeDasharray="25 75" strokeDashoffset="0" />
              <circle cx="16" cy="16" r="12" fill="transparent" stroke="#EC4899" strokeWidth="8" strokeDasharray="35 65" strokeDashoffset="-25" />
              <circle cx="16" cy="16" r="12" fill="transparent" stroke="#3B82F6" strokeWidth="8" strokeDasharray="40 60" strokeDashoffset="-60" />
            </svg>
          )}
          {widget.type === 'metric' && (
            <div className="text-center">
              <div className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 text-transparent bg-clip-text">
                12,847
              </div>
              <div className="text-xs text-gray-500 mt-1">Total Calls</div>
            </div>
          )}
          {widget.type === 'gauge' && (
            <svg className="w-20 h-12" viewBox="0 0 100 50">
              <path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="#374151" strokeWidth="8" strokeLinecap="round" />
              <path d="M10,50 A40,40 0 0,1 70,15" fill="none" stroke="url(#gaugeGradient)" strokeWidth="8" strokeLinecap="round" />
              <defs>
                <linearGradient id="gaugeGradient">
                  <stop offset="0%" stopColor="#8B5CF6" />
                  <stop offset="100%" stopColor="#EC4899" />
                </linearGradient>
              </defs>
            </svg>
          )}
          {widget.type === 'table' && (
            <div className="w-full">
              <div className="flex text-xs text-gray-500 border-b border-gray-700 pb-1 mb-1">
                <span className="flex-1">Name</span>
                <span className="w-16 text-right">Value</span>
              </div>
              {['Item A', 'Item B', 'Item C'].map(item => (
                <div key={item} className="flex text-xs py-1">
                  <span className="flex-1 text-gray-400">{item}</span>
                  <span className="w-16 text-right text-white">{Math.floor(Math.random() * 100)}</span>
                </div>
              ))}
            </div>
          )}
          {!['bar', 'line', 'pie', 'metric', 'gauge', 'table'].includes(widget.type) && (
            <div className="text-center">
              <Icon className="w-8 h-8 text-gray-500 mx-auto mb-2" />
              <span className="text-xs text-gray-500">{widgetInfo?.name}</span>
            </div>
          )}
        </div>

        {/* Data source indicator */}
        <div className="absolute bottom-2 left-2 px-2 py-0.5 bg-gray-700/50 rounded text-xs text-gray-500">
          {dataSources.find(ds => ds.id === widget.dataSource)?.name}
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
            <h1 className="text-2xl font-bold text-white mb-1">Custom Reports</h1>
            <p className="text-gray-400">Build, customize, and schedule powerful reports</p>
          </div>
          <button
            onClick={() => {
              setBuilderReport({ name: '', description: '', category: 'Operations', widgets: [], tags: [] });
              setBuilderWidgets([]);
              setIsEditMode(false);
              setActiveTab('builder');
            }}
            className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
          >
            <Plus className="w-5 h-5" />
            New Report
          </button>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-1 p-1 bg-gray-800/50 rounded-lg w-fit">
          {[
            { id: 'reports', label: 'My Reports', icon: FileText },
            { id: 'templates', label: 'Templates', icon: Layout },
            { id: 'builder', label: 'Report Builder', icon: Layers }
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
            </button>
          ))}
        </div>

        {/* Reports Tab */}
        {activeTab === 'reports' && (
          <div className="space-y-6">
            {/* Filters */}
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-4 flex-grow">
                <div className="relative flex-grow max-w-md">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search reports..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                  />
                </div>

                <div className="flex items-center gap-2">
                  {categories.map(cat => (
                    <button
                      key={cat}
                      onClick={() => setSelectedCategory(cat)}
                      className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                        selectedCategory === cat
                          ? 'bg-purple-500 text-white'
                          : 'bg-gray-800/50 text-gray-400 hover:text-white border border-gray-700'
                      }`}
                    >
                      {cat}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowOnlyFavorites(!showOnlyFavorites)}
                  className={`p-2 rounded-lg transition-colors ${
                    showOnlyFavorites ? 'bg-yellow-500/20 text-yellow-400' : 'bg-gray-800/50 text-gray-400 hover:text-white'
                  }`}
                >
                  <Star className="w-5 h-5" />
                </button>
                <div className="flex items-center bg-gray-800/50 rounded-lg border border-gray-700">
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`p-2 rounded-l-lg transition-colors ${
                      viewMode === 'grid' ? 'bg-purple-500/20 text-purple-400' : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    <Layout className="w-5 h-5" />
                  </button>
                  <button
                    onClick={() => setViewMode('list')}
                    className={`p-2 rounded-r-lg transition-colors ${
                      viewMode === 'list' ? 'bg-purple-500/20 text-purple-400' : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    <Table2 className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Reports Grid/List */}
            {viewMode === 'grid' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {filteredReports.map(report => (
                  <ReportCard key={report.id} report={report} />
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                {filteredReports.map(report => (
                  <ReportListItem key={report.id} report={report} />
                ))}
              </div>
            )}

            {filteredReports.length === 0 && (
              <div className="text-center py-12">
                <FileText className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-white mb-2">No reports found</h3>
                <p className="text-gray-400 mb-4">
                  {searchQuery ? 'Try adjusting your search criteria' : 'Create your first report to get started'}
                </p>
                <button
                  onClick={() => setActiveTab('templates')}
                  className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
                >
                  Browse Templates
                </button>
              </div>
            )}
          </div>
        )}

        {/* Templates Tab */}
        {activeTab === 'templates' && (
          <div className="space-y-6">
            {/* Search */}
            <div className="flex items-center gap-4">
              <div className="relative flex-grow max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search templates..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>

              <div className="flex items-center gap-2">
                {categories.map(cat => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                      selectedCategory === cat
                        ? 'bg-purple-500 text-white'
                        : 'bg-gray-800/50 text-gray-400 hover:text-white border border-gray-700'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            {/* Templates Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredTemplates.map(template => (
                <TemplateCard key={template.id} template={template} />
              ))}
            </div>
          </div>
        )}

        {/* Builder Tab */}
        {activeTab === 'builder' && (
          <div className="flex gap-6 h-[calc(100vh-220px)]">
            {/* Left Panel - Data Sources & Widgets */}
            <div className="w-72 flex-shrink-0 bg-gray-800/50 rounded-xl border border-gray-700 overflow-hidden flex flex-col">
              {/* Panel Header */}
              <div className="p-4 border-b border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-medium text-white">Report Builder</h3>
                  <button
                    onClick={() => setShowDataSourcePanel(!showDataSourcePanel)}
                    className="p-1 hover:bg-gray-700 rounded"
                  >
                    {showDataSourcePanel ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronRight className="w-4 h-4 text-gray-400" />}
                  </button>
                </div>

                {/* Report Name */}
                <input
                  type="text"
                  placeholder="Report Name"
                  value={builderReport.name}
                  onChange={(e) => setBuilderReport(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 text-sm mb-2"
                />
                <textarea
                  placeholder="Description (optional)"
                  value={builderReport.description}
                  onChange={(e) => setBuilderReport(prev => ({ ...prev, description: e.target.value }))}
                  className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 text-sm resize-none"
                  rows={2}
                />
              </div>

              {/* Data Sources */}
              {showDataSourcePanel && (
                <div className="p-4 border-b border-gray-700">
                  <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                    <Database className="w-4 h-4" /> Data Sources
                  </h4>
                  <div className="space-y-2">
                    {dataSources.map(ds => (
                      <button
                        key={ds.id}
                        onClick={() => setSelectedDataSource(ds.id)}
                        className={`w-full flex items-center gap-3 p-2 rounded-lg transition-all ${
                          selectedDataSource === ds.id
                            ? 'bg-purple-500/20 border border-purple-500/50'
                            : 'bg-gray-700/30 border border-transparent hover:bg-gray-700/50'
                        }`}
                      >
                        <ds.icon className={`w-4 h-4 ${selectedDataSource === ds.id ? 'text-purple-400' : 'text-gray-400'}`} />
                        <span className={`text-sm ${selectedDataSource === ds.id ? 'text-white' : 'text-gray-300'}`}>
                          {ds.name}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Available Fields */}
              {showDataSourcePanel && currentDataSource && (
                <div className="p-4 border-b border-gray-700 flex-shrink-0">
                  <h4 className="text-sm font-medium text-gray-400 mb-3">Available Fields</h4>
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {currentDataSource.fields.map(field => (
                      <div
                        key={field.id}
                        draggable
                        onDragStart={() => setDraggedField(field)}
                        onDragEnd={() => setDraggedField(null)}
                        className="flex items-center gap-2 p-2 bg-gray-700/30 rounded-lg cursor-grab hover:bg-gray-700/50 text-sm"
                      >
                        <GripVertical className="w-3 h-3 text-gray-500" />
                        <span className="text-gray-300">{field.name}</span>
                        <span className="text-xs text-gray-500 ml-auto">{field.type}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Widget Types */}
              <div className="p-4 flex-grow overflow-y-auto">
                <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                  <Layers className="w-4 h-4" /> Add Widget
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  {widgetTypes.map(wt => (
                    <button
                      key={wt.type}
                      onClick={() => {
                        setSelectedWidgetType(wt.type);
                        addWidget();
                      }}
                      className="flex flex-col items-center gap-2 p-3 bg-gray-700/30 rounded-lg hover:bg-gray-700/50 transition-colors group"
                    >
                      <wt.icon className="w-5 h-5 text-gray-400 group-hover:text-purple-400 transition-colors" />
                      <span className="text-xs text-gray-400 group-hover:text-white transition-colors">{wt.name}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="p-4 border-t border-gray-700 flex gap-2">
                <button
                  onClick={() => {
                    setActiveTab('reports');
                    setBuilderWidgets([]);
                    setBuilderReport({ name: '', description: '', category: 'Operations', widgets: [], tags: [] });
                  }}
                  className="flex-1 px-3 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors text-sm"
                >
                  Cancel
                </button>
                <button
                  onClick={saveReport}
                  className="flex-1 px-3 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:opacity-90 transition-opacity text-sm flex items-center justify-center gap-2"
                >
                  <Save className="w-4 h-4" />
                  Save
                </button>
              </div>
            </div>

            {/* Center - Canvas */}
            <div className="flex-grow bg-gray-800/30 rounded-xl border border-gray-700 overflow-hidden flex flex-col">
              {/* Canvas Header */}
              <div className="flex items-center justify-between p-4 border-b border-gray-700">
                <div className="flex items-center gap-4">
                  <h3 className="font-medium text-white">
                    {builderReport.name || 'Untitled Report'}
                  </h3>
                  <span className="text-sm text-gray-500">
                    {builderWidgets.length} widgets
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setPreviewMode(!previewMode)}
                    className={`p-2 rounded-lg transition-colors ${
                      previewMode ? 'bg-purple-500/20 text-purple-400' : 'bg-gray-700 text-gray-400 hover:text-white'
                    }`}
                  >
                    {previewMode ? <Edit3 className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                  <button className="p-2 bg-gray-700 text-gray-400 hover:text-white rounded-lg transition-colors">
                    <Maximize2 className="w-4 h-4" />
                  </button>
                  <button className="p-2 bg-gray-700 text-gray-400 hover:text-white rounded-lg transition-colors">
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Canvas */}
              <div className="flex-grow p-4 overflow-auto">
                {builderWidgets.length > 0 ? (
                  <div className="grid grid-cols-12 gap-4 auto-rows-[100px]">
                    {builderWidgets.map(widget => (
                      <WidgetPreview
                        key={widget.id}
                        widget={widget}
                        isActive={activeWidget?.id === widget.id}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <Layers className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-white mb-2">Start Building Your Report</h3>
                      <p className="text-gray-400 mb-4 max-w-md">
                        Add widgets from the panel on the left to create your custom report.
                        Drag and drop to arrange them on the canvas.
                      </p>
                      <button
                        onClick={() => {
                          setSelectedWidgetType('bar');
                          addWidget();
                        }}
                        className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
                      >
                        Add First Widget
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Panel - Widget Properties */}
            {activeWidget && (
              <div className="w-72 flex-shrink-0 bg-gray-800/50 rounded-xl border border-gray-700 overflow-hidden flex flex-col">
                <div className="p-4 border-b border-gray-700">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-medium text-white">Widget Properties</h3>
                    <button
                      onClick={() => setActiveWidget(null)}
                      className="p-1 hover:bg-gray-700 rounded"
                    >
                      <X className="w-4 h-4 text-gray-400" />
                    </button>
                  </div>

                  {/* Widget Title */}
                  <input
                    type="text"
                    value={activeWidget.title}
                    onChange={(e) => updateWidget(activeWidget.id, { title: e.target.value })}
                    className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500 text-sm"
                  />
                </div>

                <div className="p-4 space-y-4 flex-grow overflow-y-auto">
                  {/* Chart Type */}
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Chart Type</label>
                    <select
                      value={activeWidget.type}
                      onChange={(e) => updateWidget(activeWidget.id, { type: e.target.value as any })}
                      className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500 text-sm"
                    >
                      {widgetTypes.map(wt => (
                        <option key={wt.type} value={wt.type}>{wt.name}</option>
                      ))}
                    </select>
                  </div>

                  {/* Data Source */}
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Data Source</label>
                    <select
                      value={activeWidget.dataSource}
                      onChange={(e) => updateWidget(activeWidget.id, { dataSource: e.target.value })}
                      className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500 text-sm"
                    >
                      {dataSources.map(ds => (
                        <option key={ds.id} value={ds.id}>{ds.name}</option>
                      ))}
                    </select>
                  </div>

                  {/* Dimensions */}
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Dimensions (Group By)</label>
                    <div className="p-3 bg-gray-700/30 rounded-lg min-h-[60px] border-2 border-dashed border-gray-600">
                      {activeWidget.dimensions.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {activeWidget.dimensions.map(dim => (
                            <span key={dim} className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-xs flex items-center gap-1">
                              {dim}
                              <button onClick={() => updateWidget(activeWidget.id, {
                                dimensions: activeWidget.dimensions.filter(d => d !== dim)
                              })}>
                                <X className="w-3 h-3" />
                              </button>
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="text-xs text-gray-500 text-center">Drag fields here</p>
                      )}
                    </div>
                  </div>

                  {/* Measures */}
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Measures (Values)</label>
                    <div className="p-3 bg-gray-700/30 rounded-lg min-h-[60px] border-2 border-dashed border-gray-600">
                      {activeWidget.measures.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {activeWidget.measures.map(measure => (
                            <span key={measure} className="px-2 py-1 bg-pink-500/20 text-pink-400 rounded text-xs flex items-center gap-1">
                              {measure}
                              <button onClick={() => updateWidget(activeWidget.id, {
                                measures: activeWidget.measures.filter(m => m !== measure)
                              })}>
                                <X className="w-3 h-3" />
                              </button>
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="text-xs text-gray-500 text-center">Drag fields here</p>
                      )}
                    </div>
                  </div>

                  {/* Style Options */}
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Display Options</label>
                    <div className="space-y-2">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={activeWidget.style.showLegend}
                          onChange={(e) => updateWidget(activeWidget.id, {
                            style: { ...activeWidget.style, showLegend: e.target.checked }
                          })}
                          className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-purple-500 focus:ring-purple-500"
                        />
                        <span className="text-sm text-gray-300">Show Legend</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={activeWidget.style.showLabels}
                          onChange={(e) => updateWidget(activeWidget.id, {
                            style: { ...activeWidget.style, showLabels: e.target.checked }
                          })}
                          className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-purple-500 focus:ring-purple-500"
                        />
                        <span className="text-sm text-gray-300">Show Labels</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={activeWidget.style.showGrid}
                          onChange={(e) => updateWidget(activeWidget.id, {
                            style: { ...activeWidget.style, showGrid: e.target.checked }
                          })}
                          className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-purple-500 focus:ring-purple-500"
                        />
                        <span className="text-sm text-gray-300">Show Grid</span>
                      </label>
                    </div>
                  </div>

                  {/* Size */}
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Size</label>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="text-xs text-gray-500 mb-1 block">Width</label>
                        <select
                          value={activeWidget.position.width}
                          onChange={(e) => updateWidget(activeWidget.id, {
                            position: { ...activeWidget.position, width: parseInt(e.target.value) }
                          })}
                          className="w-full px-2 py-1 bg-gray-700/50 border border-gray-600 rounded text-white text-sm"
                        >
                          {[3, 4, 6, 8, 12].map(w => (
                            <option key={w} value={w}>{w} cols</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 mb-1 block">Height</label>
                        <select
                          value={activeWidget.position.height}
                          onChange={(e) => updateWidget(activeWidget.id, {
                            position: { ...activeWidget.position, height: parseInt(e.target.value) }
                          })}
                          className="w-full px-2 py-1 bg-gray-700/50 border border-gray-600 rounded text-white text-sm"
                        >
                          {[2, 3, 4, 6, 8].map(h => (
                            <option key={h} value={h}>{h} rows</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Widget Actions */}
                <div className="p-4 border-t border-gray-700 flex gap-2">
                  <button
                    onClick={() => {
                      const duplicated: ReportWidget = {
                        ...activeWidget,
                        id: `widget-${Date.now()}`,
                        title: `${activeWidget.title} (Copy)`
                      };
                      setBuilderWidgets(prev => [...prev, duplicated]);
                    }}
                    className="flex-1 px-3 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors text-sm flex items-center justify-center gap-1"
                  >
                    <Copy className="w-4 h-4" />
                    Duplicate
                  </button>
                  <button
                    onClick={() => removeWidget(activeWidget.id)}
                    className="px-3 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors text-sm"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Report Detail Dialog */}
        {showReportDialog && selectedReport && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
              {/* Header */}
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <h2 className="text-xl font-bold text-white">{selectedReport.name}</h2>
                      {selectedReport.isFavorite && (
                        <Star className="w-5 h-5 text-yellow-400 fill-yellow-400" />
                      )}
                    </div>
                    <p className="text-gray-400">{selectedReport.description}</p>
                  </div>
                  <button onClick={() => setShowReportDialog(false)} className="p-2 hover:bg-gray-700 rounded-lg">
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              {/* Content */}
              <div className="p-6 overflow-y-auto max-h-[60vh]">
                {/* Report Preview */}
                <div className="aspect-video bg-gradient-to-br from-purple-500/10 to-pink-500/10 rounded-xl mb-6 flex items-center justify-center">
                  <div className="text-center">
                    <FileText className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                    <p className="text-gray-500">Report Preview</p>
                  </div>
                </div>

                {/* Report Info */}
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm text-gray-400">Category</label>
                      <p className="text-white">{selectedReport.category}</p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Created By</label>
                      <p className="text-white">{selectedReport.createdBy}</p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Created</label>
                      <p className="text-white">{formatDate(selectedReport.createdAt)}</p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Last Updated</label>
                      <p className="text-white">{formatDate(selectedReport.updatedAt)}</p>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm text-gray-400">Visibility</label>
                      <p className="text-white">{selectedReport.isPublic ? 'Public' : 'Private'}</p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Tags</label>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {selectedReport.tags.map(tag => (
                          <span key={tag} className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-sm">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Schedules</label>
                      {selectedReport.schedules.length > 0 ? (
                        <div className="space-y-2 mt-1">
                          {selectedReport.schedules.map(schedule => (
                            <div key={schedule.id} className="flex items-center gap-2 text-sm">
                              <Clock className={`w-4 h-4 ${schedule.enabled ? 'text-green-400' : 'text-gray-500'}`} />
                              <span className={schedule.enabled ? 'text-white' : 'text-gray-500'}>
                                {schedule.frequency.charAt(0).toUpperCase() + schedule.frequency.slice(1)} at {schedule.time}
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-500">No schedules configured</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="p-6 border-t border-gray-700 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => toggleFavorite(selectedReport.id)}
                    className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    {selectedReport.isFavorite ? (
                      <Star className="w-5 h-5 text-yellow-400 fill-yellow-400" />
                    ) : (
                      <StarOff className="w-5 h-5 text-gray-400" />
                    )}
                  </button>
                  <button
                    onClick={() => { setShowReportDialog(false); setShowScheduleDialog(true); }}
                    className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    <Calendar className="w-5 h-5 text-gray-400" />
                  </button>
                  <button
                    onClick={() => { setShowReportDialog(false); setShowExportDialog(true); }}
                    className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    <Download className="w-5 h-5 text-gray-400" />
                  </button>
                  <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
                    <Share2 className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => { setShowReportDialog(false); editReport(selectedReport); }}
                    className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-2"
                  >
                    <Edit3 className="w-4 h-4" />
                    Edit Report
                  </button>
                  <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2">
                    <Play className="w-4 h-4" />
                    Run Report
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Export Dialog */}
        {showExportDialog && selectedReport && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl max-w-md w-full overflow-hidden">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-bold text-white">Export Report</h2>
                  <button onClick={() => setShowExportDialog(false)} className="p-2 hover:bg-gray-700 rounded-lg">
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              <div className="p-6 space-y-4">
                <p className="text-gray-400">Choose an export format for "{selectedReport.name}"</p>

                <div className="space-y-2">
                  {[
                    { id: 'pdf', icon: FileText, name: 'PDF Document', desc: 'Best for printing and sharing' },
                    { id: 'excel', icon: FileSpreadsheet, name: 'Excel Spreadsheet', desc: 'Best for data analysis' },
                    { id: 'csv', icon: Table2, name: 'CSV File', desc: 'Raw data export' },
                    { id: 'image', icon: Image, name: 'PNG Image', desc: 'Best for presentations' },
                  ].map(format => (
                    <button
                      key={format.id}
                      className="w-full flex items-center gap-4 p-4 bg-gray-700/50 rounded-xl hover:bg-gray-700 transition-colors text-left"
                    >
                      <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                        <format.icon className="w-5 h-5 text-purple-400" />
                      </div>
                      <div>
                        <h3 className="font-medium text-white">{format.name}</h3>
                        <p className="text-sm text-gray-400">{format.desc}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Schedule Dialog */}
        {showScheduleDialog && selectedReport && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl max-w-lg w-full overflow-hidden">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-bold text-white">Schedule Report</h2>
                  <button onClick={() => setShowScheduleDialog(false)} className="p-2 hover:bg-gray-700 rounded-lg">
                    <X className="w-5 h-5 text-gray-400" />
                  </button>
                </div>
              </div>

              <div className="p-6 space-y-6">
                {/* Frequency */}
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Frequency</label>
                  <div className="grid grid-cols-4 gap-2">
                    {['daily', 'weekly', 'monthly', 'quarterly'].map(freq => (
                      <button
                        key={freq}
                        onClick={() => setNewSchedule(prev => ({ ...prev, frequency: freq as any }))}
                        className={`px-3 py-2 rounded-lg text-sm transition-colors capitalize ${
                          newSchedule.frequency === freq
                            ? 'bg-purple-500 text-white'
                            : 'bg-gray-700 text-gray-400 hover:text-white'
                        }`}
                      >
                        {freq}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Time */}
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Time</label>
                  <input
                    type="time"
                    value={newSchedule.time}
                    onChange={(e) => setNewSchedule(prev => ({ ...prev, time: e.target.value }))}
                    className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
                  />
                </div>

                {/* Day of Week (for weekly) */}
                {newSchedule.frequency === 'weekly' && (
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Day of Week</label>
                    <select
                      value={newSchedule.dayOfWeek}
                      onChange={(e) => setNewSchedule(prev => ({ ...prev, dayOfWeek: parseInt(e.target.value) }))}
                      className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
                    >
                      {['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'].map((day, i) => (
                        <option key={i} value={i}>{day}</option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Day of Month (for monthly) */}
                {newSchedule.frequency === 'monthly' && (
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Day of Month</label>
                    <select
                      value={newSchedule.dayOfMonth}
                      onChange={(e) => setNewSchedule(prev => ({ ...prev, dayOfMonth: parseInt(e.target.value) }))}
                      className="w-full px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
                    >
                      {Array.from({ length: 28 }, (_, i) => (
                        <option key={i + 1} value={i + 1}>{i + 1}</option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Recipients */}
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Recipients</label>
                  <div className="flex gap-2 mb-2">
                    <input
                      type="email"
                      placeholder="Add email address"
                      value={recipientEmail}
                      onChange={(e) => setRecipientEmail(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && addRecipient()}
                      className="flex-grow px-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                    />
                    <button
                      onClick={addRecipient}
                      className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
                    >
                      Add
                    </button>
                  </div>
                  {newSchedule.recipients && newSchedule.recipients.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {newSchedule.recipients.map(email => (
                        <span key={email} className="px-3 py-1 bg-gray-700 rounded-lg text-sm text-gray-300 flex items-center gap-2">
                          {email}
                          <button onClick={() => removeRecipient(email)}>
                            <X className="w-3 h-3 text-gray-400 hover:text-white" />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Format */}
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Export Format</label>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { id: 'pdf', icon: FileText, name: 'PDF' },
                      { id: 'excel', icon: FileSpreadsheet, name: 'Excel' },
                      { id: 'csv', icon: Table2, name: 'CSV' },
                    ].map(format => (
                      <button
                        key={format.id}
                        onClick={() => setNewSchedule(prev => ({ ...prev, format: format.id as any }))}
                        className={`flex items-center justify-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                          newSchedule.format === format.id
                            ? 'bg-purple-500 text-white'
                            : 'bg-gray-700 text-gray-400 hover:text-white'
                        }`}
                      >
                        <format.icon className="w-4 h-4" />
                        {format.name}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="p-6 border-t border-gray-700 flex justify-end gap-3">
                <button
                  onClick={() => setShowScheduleDialog(false)}
                  className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
                >
                  Cancel
                </button>
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2">
                  <Calendar className="w-4 h-4" />
                  Create Schedule
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
