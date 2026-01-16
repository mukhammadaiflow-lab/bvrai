/**
 * Voice Configuration Hook
 *
 * Provides voice configuration management using TanStack Query.
 * Handles STT/TTS providers, voice library, and voice preview.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { voiceConfigApi, VoiceConfig, Voice, CreateVoiceConfigRequest } from '@/lib/api';

// Query keys for cache management
export const voiceConfigKeys = {
  all: ['voiceConfig'] as const,
  lists: () => [...voiceConfigKeys.all, 'list'] as const,
  details: () => [...voiceConfigKeys.all, 'detail'] as const,
  detail: (id: string) => [...voiceConfigKeys.details(), id] as const,
  voices: (provider?: string) => [...voiceConfigKeys.all, 'voices', provider] as const,
  providers: () => [...voiceConfigKeys.all, 'providers'] as const,
};

/**
 * Hook for fetching all voice configurations
 */
export function useVoiceConfigs() {
  return useQuery({
    queryKey: voiceConfigKeys.lists(),
    queryFn: async () => {
      const response = await voiceConfigApi.list();
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for fetching a single voice configuration
 */
export function useVoiceConfigById(id: string) {
  return useQuery({
    queryKey: voiceConfigKeys.detail(id),
    queryFn: async () => {
      const response = await voiceConfigApi.get(id);
      return response.data;
    },
    enabled: !!id,
  });
}

/**
 * Hook for fetching available voices
 */
export function useVoices(provider?: string) {
  return useQuery({
    queryKey: voiceConfigKeys.voices(provider),
    queryFn: async () => {
      const response = await voiceConfigApi.listVoices(provider);
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes - voices don't change often
  });
}

/**
 * Hook for fetching available providers
 */
export function useVoiceProviders() {
  return useQuery({
    queryKey: voiceConfigKeys.providers(),
    queryFn: async () => {
      const response = await voiceConfigApi.listProviders();
      return response.data;
    },
    staleTime: 10 * 60 * 1000, // 10 minutes - providers rarely change
  });
}

/**
 * Hook for creating a voice configuration
 */
export function useCreateVoiceConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: CreateVoiceConfigRequest) => {
      const response = await voiceConfigApi.create(data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: voiceConfigKeys.lists() });
    },
  });
}

/**
 * Hook for updating a voice configuration
 */
export function useUpdateVoiceConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ id, data }: { id: string; data: Partial<CreateVoiceConfigRequest> }) => {
      const response = await voiceConfigApi.update(id, data);
      return response.data;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: voiceConfigKeys.detail(variables.id) });
      queryClient.invalidateQueries({ queryKey: voiceConfigKeys.lists() });
    },
  });
}

/**
 * Hook for deleting a voice configuration
 */
export function useDeleteVoiceConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      await voiceConfigApi.delete(id);
    },
    onSuccess: (_, id) => {
      queryClient.removeQueries({ queryKey: voiceConfigKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: voiceConfigKeys.lists() });
    },
  });
}

/**
 * Hook for setting a voice config as default
 */
export function useSetDefaultVoiceConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      await voiceConfigApi.setDefault(id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: voiceConfigKeys.lists() });
    },
  });
}

/**
 * Hook for previewing a voice
 */
export function usePreviewVoice() {
  return useMutation({
    mutationFn: async ({ configId, text }: { configId: string; text: string }) => {
      const response = await voiceConfigApi.preview(configId, text);
      return response.data;
    },
    onSuccess: (audioBlob) => {
      // Create audio element and play
      if (audioBlob instanceof Blob) {
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
        // Clean up URL after playback
        audio.onended = () => URL.revokeObjectURL(audioUrl);
      }
    },
  });
}
