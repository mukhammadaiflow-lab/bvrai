/**
 * Custom React Query Hooks
 *
 * Export all hooks for easy imports:
 * import { useAgents, useCalls, useCurrentUser } from '@/lib/hooks';
 */

// Agent hooks
export {
  useAgents,
  useAgent,
  useCreateAgent,
  useUpdateAgent,
  useDeleteAgent,
  useDeployAgent,
  agentKeys,
} from "./use-agents";

// Call hooks
export {
  useCalls,
  useCall,
  useInitiateCall,
  useEndCall,
  useCallAnalytics,
  useCallTranscript,
  useCallRecording,
  callKeys,
} from "./use-calls";

// Auth hooks
export {
  useCurrentUser,
  useLogin,
  useRegister,
  useLogout,
  useIsAuthenticated,
  getStoredToken,
  setStoredToken,
  clearStoredToken,
  authKeys,
} from "./use-auth";
