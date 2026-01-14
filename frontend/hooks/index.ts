/**
 * Frontend Hooks
 *
 * Central export point for all custom React hooks.
 * Includes auth, data fetching, and real-time hooks.
 */

// Authentication
export { useAuth } from './useAuth';

// Agents
export {
  useAgents,
  useAgent,
  useAgentVersions,
  useCreateAgent,
  useUpdateAgent,
  useDeleteAgent,
  useDuplicateAgent,
  useRollbackAgent,
  useTestAgent,
  agentKeys,
} from './useAgents';

// Calls
export {
  useCalls,
  useCall,
  useCallConversation,
  useCallEvents,
  useInitiateCall,
  useHangupCall,
  callKeys,
} from './useCalls';

// Analytics
export {
  useAnalyticsSummary,
  useCallsByDay,
  useAgentPerformance,
  useDashboardStats,
  useDateRange,
  analyticsKeys,
} from './useAnalytics';

// Real-time
export {
  useWebSocket,
  useCallUpdates,
  useRealTimeMetrics,
} from './useWebSocket';
