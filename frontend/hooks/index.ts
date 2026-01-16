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
  useCallTranscript,
  useCallRecording,
  useInitiateCall,
  useEndCall,
  useAddCallNote,
  useHangupCall,
  useCallConversation,
  useCallEvents,
  callKeys,
} from './useCalls';

// Analytics
export {
  useAnalyticsSummary,
  useCallsByDay,
  useAgentPerformance,
  useDashboardStats,
  useCallQuality,
  useSentimentAnalysis,
  useDateRange,
  analyticsKeys,
} from './useAnalytics';

// Real-time
export {
  useWebSocket,
  useCallUpdates,
  useRealTimeMetrics,
} from './useWebSocket';

// Voice Configuration
export {
  useVoiceConfigs,
  useVoiceConfigById,
  useVoiceConfigById as useVoiceConfig, // Alias for backwards compatibility
  useVoices,
  useVoiceProviders,
  useCreateVoiceConfig,
  useUpdateVoiceConfig,
  useDeleteVoiceConfig,
  useSetDefaultVoiceConfig,
  usePreviewVoice,
  voiceConfigKeys,
} from './useVoiceConfig';

// Webhooks
export {
  useWebhooks,
  useWebhook,
  useWebhookDeliveries,
  useWebhookEvents,
  useCreateWebhook,
  useUpdateWebhook,
  useDeleteWebhook,
  useTestWebhook,
  useEnableWebhook,
  useDisableWebhook,
  useRotateWebhookSecret,
  webhookKeys,
} from './useWebhooks';

// Billing
export {
  useSubscription,
  useCurrentPlan,
  useInvoices,
  useUsage,
  useCreateCheckout,
  useCancelSubscription,
  useResumeSubscription,
  useCreatePortalSession,
  billingKeys,
} from './useBilling';

// Organization & Team
export {
  useOrganization,
  useUpdateOrganization,
  useTeamMembers,
  useInviteMember,
  useRemoveMember,
  useUpdateMemberRole,
  useApiKeys,
  useCreateApiKey,
  useRevokeApiKey,
  useRegenerateApiKey,
  organizationKeys,
  apiKeyKeys,
} from './useOrganization';
