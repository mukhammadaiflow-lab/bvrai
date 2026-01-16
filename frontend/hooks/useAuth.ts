/**
 * Authentication Hook
 *
 * Provides authentication state management and methods using Zustand.
 * Handles login, logout, and user session persistence.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authApi, User } from '@/lib/api';

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  error: string | null;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  register: (email: string, password: string, name: string, organizationName?: string) => Promise<void>;
  fetchCurrentUser: () => Promise<void>;
  clearError: () => void;
}

export const useAuth = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isLoading: false,
      isAuthenticated: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await authApi.login({ email, password });
          const { access_token, user } = response.data;

          // Store token in localStorage for API client
          localStorage.setItem('auth_token', access_token);

          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: unknown) {
          const err = error as { message?: string; response?: { data?: { detail?: string } } };
          const message = err.response?.data?.detail || err.message || 'Login failed';
          set({ error: message, isLoading: false });
          throw error;
        }
      },

      logout: async () => {
        set({ isLoading: true });
        try {
          await authApi.logout();
        } catch {
          // Ignore logout errors, clear state anyway
        } finally {
          localStorage.removeItem('auth_token');
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
            error: null,
          });
        }
      },

      register: async (email: string, password: string, name: string, organizationName?: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await authApi.register({
            email,
            password,
            name,
            organization_name: organizationName,
          });
          const { access_token, user } = response.data;

          localStorage.setItem('auth_token', access_token);

          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: unknown) {
          const err = error as { message?: string; response?: { data?: { detail?: string } } };
          const message = err.response?.data?.detail || err.message || 'Registration failed';
          set({ error: message, isLoading: false });
          throw error;
        }
      },

      fetchCurrentUser: async () => {
        const token = get().token || localStorage.getItem('auth_token');
        if (!token) {
          set({ isAuthenticated: false });
          return;
        }

        set({ isLoading: true });
        try {
          const response = await authApi.me();
          set({
            user: response.data,
            token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch {
          // Token invalid, clear state
          localStorage.removeItem('auth_token');
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          });
        }
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
