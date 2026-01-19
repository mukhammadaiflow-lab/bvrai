/**
 * Voice Configuration Hook
 *
 * Provides voice configuration management using TanStack Query.
 * Handles STT/TTS providers, voice library, and voice preview.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { voiceConfigApi } from '@/lib/api';
import { VoiceConfiguration, Voice } from '@/types';

// Query keys for cache management
export const voiceConfigKeys = {
  all: ['voiceConfig'] as const,
  lists: () => [...voiceConfigKeys.all, 'list'] as const,
  details: () => [...voiceConfigKeys.all, 'detail'] as const,
  detail: (id: string) => [...voiceConfigKeys.details(), id] as const,
  voices: (provider?: string) => [...voiceConfigKeys.all, 'voices', provider] as const,
};

/**
 * Hook for fetching all voice configurations
 */
export function useVoiceConfigs() {
  return useQuery({
    queryKey: voiceConfigKeys.lists(),
    queryFn: () => voiceConfigApi.list(),
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook for fetching a single voice configuration
 */
export function useVoiceConfig(id: string) {
  return useQuery({
    queryKey: voiceConfigKeys.detail(id),
    queryFn: () => voiceConfigApi.get(id),
    enabled: !!id,
  });
}

/**
 * Hook for fetching available voices
 */
export function useVoices(provider?: string) {
  return useQuery({
    queryKey: voiceConfigKeys.voices(provider),
    queryFn: () => voiceConfigApi.listVoices(provider),
    staleTime: 5 * 60 * 1000, // 5 minutes - voices don't change often
  });
}

/**
 * Hook for creating a voice configuration
 */
export function useCreateVoiceConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: Partial<VoiceConfiguration>) => voiceConfigApi.create(data),
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
    mutationFn: ({ id, data }: { id: string; data: Partial<VoiceConfiguration> }) =>
      voiceConfigApi.update(id, data),
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
    mutationFn: (id: string) => voiceConfigApi.delete(id),
    onSuccess: (_, id) => {
      queryClient.removeQueries({ queryKey: voiceConfigKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: voiceConfigKeys.lists() });
    },
  });
}

/**
 * Hook for previewing a voice
 */
export function usePreviewVoice() {
  return useMutation({
    mutationFn: ({ voiceId, text }: { voiceId: string; text: string }) =>
      voiceConfigApi.previewVoice(voiceId, text),
    onSuccess: (audioBlob) => {
      // Create audio element and play
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
      // Clean up URL after playback
      audio.onended = () => URL.revokeObjectURL(audioUrl);
    },
  });
}
