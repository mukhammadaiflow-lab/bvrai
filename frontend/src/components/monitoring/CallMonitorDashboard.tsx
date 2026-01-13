/**
 * Real-time Call Monitoring Dashboard Component
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';

// Types
interface ActiveCall {
  id: string;
  agent_id: string;
  agent_name: string;
  to_number: string;
  from_number: string;
  direction: 'inbound' | 'outbound';
  status: 'queued' | 'ringing' | 'in_progress' | 'completed' | 'failed';
  started_at: string;
  duration: number;
  current_sentiment: string | null;
  sentiment_score: number;
  transcript_preview: string;
  is_speaking: boolean;
}

interface DashboardMetrics {
  active_calls: number;
  calls_today: number;
  calls_this_hour: number;
  avg_duration: number;
  success_rate: number;
  queued_calls: number;
  agents_in_use: number;
  total_agents: number;
  concurrent_limit: number;
  current_concurrency: number;
  error_rate: number;
  sentiment_breakdown: Record<string, number>;
}

interface Alert {
  id: string;
  rule_name: string;
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  status: 'active' | 'acknowledged' | 'resolved';
  message: string;
  triggered_at: string;
}

interface TranscriptMessage {
  role: 'user' | 'assistant';
  text: string;
  timestamp: string;
}

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
  call_id?: string;
}

// Utility functions
const formatDuration = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const formatPhoneNumber = (number: string): string => {
  if (!number) return '';
  const cleaned = number.replace(/\D/g, '');
  if (cleaned.length === 11) {
    return `+${cleaned[0]} (${cleaned.slice(1, 4)}) ${cleaned.slice(4, 7)}-${cleaned.slice(7)}`;
  }
  return number;
};

const getSentimentColor = (sentiment: string | null): string => {
  switch (sentiment) {
    case 'positive': return '#22c55e';
    case 'negative': return '#ef4444';
    case 'neutral': return '#6b7280';
    default: return '#9ca3af';
  }
};

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'in_progress': return '#22c55e';
    case 'ringing': return '#eab308';
    case 'queued': return '#3b82f6';
    case 'completed': return '#6b7280';
    case 'failed': return '#ef4444';
    default: return '#9ca3af';
  }
};

// Sub-components

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: number;
  color?: string;
}> = ({ title, value, subtitle, trend, color = '#3b82f6' }) => (
  <div className="bg-white rounded-lg shadow p-4 border-l-4" style={{ borderLeftColor: color }}>
    <div className="text-sm text-gray-500 font-medium">{title}</div>
    <div className="text-2xl font-bold mt-1">{value}</div>
    {subtitle && <div className="text-xs text-gray-400 mt-1">{subtitle}</div>}
    {trend !== undefined && (
      <div className={`text-xs mt-1 ${trend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
        {trend >= 0 ? '↑' : '↓'} {Math.abs(trend)}% vs last hour
      </div>
    )}
  </div>
);

const ActiveCallCard: React.FC<{
  call: ActiveCall;
  onClick: () => void;
  isSelected: boolean;
}> = ({ call, onClick, isSelected }) => {
  const [duration, setDuration] = useState(call.duration);

  useEffect(() => {
    if (call.status === 'in_progress') {
      const interval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - new Date(call.started_at).getTime()) / 1000);
        setDuration(elapsed);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [call.started_at, call.status]);

  return (
    <div
      className={`p-3 rounded-lg cursor-pointer transition-all ${
        isSelected ? 'bg-blue-50 border-2 border-blue-500' : 'bg-white border border-gray-200 hover:border-blue-300'
      }`}
      onClick={onClick}
    >
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="font-medium">{call.agent_name}</span>
            {call.is_speaking && (
              <span className="flex items-center text-xs text-green-600">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-1" />
                Speaking
              </span>
            )}
          </div>
          <div className="text-sm text-gray-600 mt-1">
            {call.direction === 'inbound' ? '← ' : '→ '}
            {formatPhoneNumber(call.direction === 'inbound' ? call.from_number : call.to_number)}
          </div>
        </div>
        <div className="text-right">
          <div
            className="px-2 py-1 rounded text-xs font-medium text-white"
            style={{ backgroundColor: getStatusColor(call.status) }}
          >
            {call.status.replace('_', ' ')}
          </div>
          <div className="text-sm font-mono mt-1">{formatDuration(duration)}</div>
        </div>
      </div>

      {call.transcript_preview && (
        <div className="mt-2 text-sm text-gray-500 truncate">
          "{call.transcript_preview}"
        </div>
      )}

      {call.current_sentiment && (
        <div className="mt-2 flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: getSentimentColor(call.current_sentiment) }}
          />
          <span className="text-xs text-gray-500 capitalize">{call.current_sentiment}</span>
          <span className="text-xs text-gray-400">({(call.sentiment_score * 100).toFixed(0)}%)</span>
        </div>
      )}
    </div>
  );
};

const CallDetailsPanel: React.FC<{
  call: ActiveCall;
  transcript: TranscriptMessage[];
  onClose: () => void;
}> = ({ call, transcript, onClose }) => {
  const transcriptEndRef = React.useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript]);

  return (
    <div className="bg-white rounded-lg shadow-lg h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b flex justify-between items-center">
        <div>
          <h3 className="font-semibold text-lg">{call.agent_name}</h3>
          <div className="text-sm text-gray-500">
            {call.direction === 'inbound' ? 'From: ' : 'To: '}
            {formatPhoneNumber(call.direction === 'inbound' ? call.from_number : call.to_number)}
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-100 rounded-full"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Call Info */}
      <div className="p-4 bg-gray-50 border-b grid grid-cols-4 gap-4 text-sm">
        <div>
          <div className="text-gray-500">Status</div>
          <div
            className="font-medium capitalize"
            style={{ color: getStatusColor(call.status) }}
          >
            {call.status.replace('_', ' ')}
          </div>
        </div>
        <div>
          <div className="text-gray-500">Duration</div>
          <div className="font-medium font-mono">{formatDuration(call.duration)}</div>
        </div>
        <div>
          <div className="text-gray-500">Sentiment</div>
          <div className="font-medium flex items-center gap-1">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: getSentimentColor(call.current_sentiment) }}
            />
            {call.current_sentiment || 'Unknown'}
          </div>
        </div>
        <div>
          <div className="text-gray-500">Direction</div>
          <div className="font-medium capitalize">{call.direction}</div>
        </div>
      </div>

      {/* Transcript */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        <h4 className="text-sm font-medium text-gray-700 sticky top-0 bg-white py-2">
          Live Transcript
        </h4>
        {transcript.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            Waiting for conversation...
          </div>
        ) : (
          transcript.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === 'assistant' ? 'justify-start' : 'justify-end'}`}
            >
              <div
                className={`max-w-[80%] p-3 rounded-lg ${
                  msg.role === 'assistant'
                    ? 'bg-blue-100 text-blue-900'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                <div className="text-xs text-gray-500 mb-1">
                  {msg.role === 'assistant' ? 'Agent' : 'Caller'}
                </div>
                <div className="text-sm">{msg.text}</div>
              </div>
            </div>
          ))
        )}
        <div ref={transcriptEndRef} />
      </div>

      {/* Actions */}
      <div className="p-4 border-t flex gap-2">
        <button className="flex-1 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium">
          Inject Message
        </button>
        <button className="flex-1 px-4 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg text-sm font-medium">
          End Call
        </button>
      </div>
    </div>
  );
};

const AlertBanner: React.FC<{ alerts: Alert[] }> = ({ alerts }) => {
  if (alerts.length === 0) return null;

  const criticalAlerts = alerts.filter(a => a.severity === 'critical' || a.severity === 'emergency');

  if (criticalAlerts.length === 0) return null;

  return (
    <div className="bg-red-600 text-white px-4 py-2 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <svg className="w-5 h-5 animate-pulse" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
            clipRule="evenodd"
          />
        </svg>
        <span className="font-medium">
          {criticalAlerts.length} critical alert{criticalAlerts.length > 1 ? 's' : ''}
        </span>
        <span className="text-red-200">
          {criticalAlerts[0].message}
        </span>
      </div>
      <button className="text-red-200 hover:text-white">View all</button>
    </div>
  );
};

// Main Dashboard Component
export const CallMonitorDashboard: React.FC<{
  organizationId: string;
  apiBaseUrl?: string;
  wsBaseUrl?: string;
}> = ({
  organizationId,
  apiBaseUrl = '/api',
  wsBaseUrl = 'ws://localhost:8000',
}) => {
  // State
  const [activeCalls, setActiveCalls] = useState<ActiveCall[]>([]);
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedCall, setSelectedCall] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(`${wsBaseUrl}/monitoring/ws/${organizationId}`);

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
      console.log('Dashboard WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        handleWebSocketMessage(message);
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error');
      setIsConnected(false);
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log('Dashboard WebSocket disconnected');

      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        console.log('Attempting to reconnect...');
      }, 5000);
    };

    return () => {
      ws.close();
    };
  }, [organizationId, wsBaseUrl]);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'initial_state':
        setActiveCalls(message.data.active_calls);
        setMetrics(message.data.metrics);
        break;

      case 'call.started':
        setActiveCalls(prev => [...prev, message.data]);
        break;

      case 'call.ringing':
      case 'call.answered':
        setActiveCalls(prev =>
          prev.map(call =>
            call.id === message.data.call_id
              ? { ...call, status: message.data.status }
              : call
          )
        );
        break;

      case 'call.ended':
        setActiveCalls(prev => prev.filter(call => call.id !== message.data.call_id));
        if (selectedCall === message.data.call_id) {
          setSelectedCall(null);
        }
        break;

      case 'transcription':
        if (message.call_id === selectedCall && message.data.is_final) {
          setTranscript(prev => [
            ...prev,
            {
              role: message.data.role,
              text: message.data.text,
              timestamp: message.timestamp,
            },
          ]);
        }
        // Update transcript preview
        if (message.data.role === 'user') {
          setActiveCalls(prev =>
            prev.map(call =>
              call.id === message.call_id
                ? { ...call, transcript_preview: message.data.text }
                : call
            )
          );
        }
        break;

      case 'agent.speaking':
        setActiveCalls(prev =>
          prev.map(call =>
            call.id === message.call_id
              ? { ...call, is_speaking: message.data.is_speaking }
              : call
          )
        );
        break;

      case 'sentiment.update':
        setActiveCalls(prev =>
          prev.map(call =>
            call.id === message.call_id
              ? {
                  ...call,
                  current_sentiment: message.data.sentiment,
                  sentiment_score: message.data.score,
                }
              : call
          )
        );
        break;

      case 'metrics.update':
        setMetrics(message.data);
        break;

      case 'alert':
        setAlerts(prev => [message.data, ...prev.slice(0, 9)]);
        break;
    }
  }, [selectedCall]);

  // Select call and load transcript
  const handleSelectCall = useCallback((callId: string) => {
    setSelectedCall(callId);
    setTranscript([]);

    // Fetch existing transcript
    fetch(`${apiBaseUrl}/monitoring/active-calls/${callId}/transcript`)
      .then(res => res.json())
      .then(data => {
        if (data.messages) {
          setTranscript(data.messages);
        }
      })
      .catch(err => console.error('Failed to load transcript:', err));
  }, [apiBaseUrl]);

  // Selected call object
  const selectedCallData = useMemo(
    () => activeCalls.find(c => c.id === selectedCall),
    [activeCalls, selectedCall]
  );

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Alert Banner */}
      <AlertBanner alerts={alerts} />

      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Call Monitor</h1>
            <p className="text-sm text-gray-500">Real-time call monitoring dashboard</p>
          </div>
          <div className="flex items-center gap-4">
            <span className={`flex items-center gap-2 text-sm ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Error Message */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 py-2">
          <div className="bg-red-100 text-red-700 px-4 py-2 rounded-lg text-sm">
            {error}
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
          <MetricCard
            title="Active Calls"
            value={metrics?.active_calls ?? 0}
            color="#22c55e"
          />
          <MetricCard
            title="Calls Today"
            value={metrics?.calls_today ?? 0}
            subtitle={`${metrics?.calls_this_hour ?? 0} this hour`}
          />
          <MetricCard
            title="Success Rate"
            value={`${((metrics?.success_rate ?? 0) * 100).toFixed(1)}%`}
            color={metrics?.success_rate ?? 0 >= 0.9 ? '#22c55e' : '#eab308'}
          />
          <MetricCard
            title="Avg Duration"
            value={formatDuration(metrics?.avg_duration ?? 0)}
          />
          <MetricCard
            title="In Queue"
            value={metrics?.queued_calls ?? 0}
            color={metrics?.queued_calls ?? 0 > 5 ? '#ef4444' : '#3b82f6'}
          />
          <MetricCard
            title="Concurrency"
            value={`${metrics?.current_concurrency ?? 0}/${metrics?.concurrent_limit ?? 0}`}
          />
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Active Calls List */}
          <div className={`${selectedCall ? 'lg:col-span-2' : 'lg:col-span-3'}`}>
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b flex justify-between items-center">
                <h2 className="font-semibold">Active Calls ({activeCalls.length})</h2>
                <div className="flex gap-2">
                  <select className="text-sm border rounded-lg px-2 py-1">
                    <option>All Agents</option>
                  </select>
                  <select className="text-sm border rounded-lg px-2 py-1">
                    <option>All Status</option>
                    <option>In Progress</option>
                    <option>Ringing</option>
                    <option>Queued</option>
                  </select>
                </div>
              </div>

              <div className="p-4 space-y-3 max-h-[600px] overflow-y-auto">
                {activeCalls.length === 0 ? (
                  <div className="text-center py-12 text-gray-400">
                    <svg className="w-12 h-12 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                    </svg>
                    <p>No active calls</p>
                  </div>
                ) : (
                  activeCalls.map(call => (
                    <ActiveCallCard
                      key={call.id}
                      call={call}
                      onClick={() => handleSelectCall(call.id)}
                      isSelected={call.id === selectedCall}
                    />
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Call Details Panel */}
          {selectedCall && selectedCallData && (
            <div className="lg:col-span-1">
              <CallDetailsPanel
                call={selectedCallData}
                transcript={transcript}
                onClose={() => setSelectedCall(null)}
              />
            </div>
          )}
        </div>

        {/* Sentiment Breakdown */}
        {metrics?.sentiment_breakdown && Object.keys(metrics.sentiment_breakdown).length > 0 && (
          <div className="mt-6 bg-white rounded-lg shadow p-4">
            <h3 className="font-semibold mb-4">Current Sentiment Distribution</h3>
            <div className="flex gap-4">
              {Object.entries(metrics.sentiment_breakdown).map(([sentiment, count]) => (
                <div key={sentiment} className="flex items-center gap-2">
                  <div
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: getSentimentColor(sentiment) }}
                  />
                  <span className="capitalize">{sentiment}</span>
                  <span className="font-bold">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default CallMonitorDashboard;
