/**
 * HTTP client utilities for Builder Engine SDK
 */

import type { ClientConfig, RequestOptions } from '../types';
import {
  BvraiError,
  TimeoutError,
  ConnectionError,
  createErrorFromResponse,
} from './errors';

export interface HttpRequestOptions extends RequestOptions {
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  path: string;
  body?: Record<string, unknown>;
  params?: Record<string, string | number | boolean | undefined>;
  headers?: Record<string, string>;
}

export class HttpClient {
  private readonly config: Required<ClientConfig>;

  constructor(config: ClientConfig) {
    this.config = {
      apiKey: config.apiKey,
      baseUrl: config.baseUrl || 'https://api.bvrai.com',
      timeout: config.timeout || 30000,
      maxRetries: config.maxRetries || 3,
      retryDelay: config.retryDelay || 1000,
    };
  }

  /**
   * Make an HTTP request
   */
  async request<T = Record<string, unknown>>(options: HttpRequestOptions): Promise<T> {
    const { method, path, body, params, headers: customHeaders, signal, timeout } = options;

    // Build URL with query params
    const url = new URL(path, this.config.baseUrl);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          url.searchParams.append(key, String(value));
        }
      });
    }

    // Build headers
    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.config.apiKey}`,
      'Content-Type': 'application/json',
      'User-Agent': '@bvrai/sdk/1.0.0',
      ...customHeaders,
    };

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutMs = timeout || this.config.timeout;
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.config.maxRetries; attempt++) {
      try {
        const response = await fetch(url.toString(), {
          method,
          headers,
          body: body ? JSON.stringify(body) : undefined,
          signal: signal || controller.signal,
        });

        clearTimeout(timeoutId);

        // Handle successful responses
        if (response.ok) {
          if (response.status === 204) {
            return {} as T;
          }
          return (await response.json()) as T;
        }

        // Handle error responses
        let errorBody: Record<string, unknown>;
        try {
          errorBody = await response.json();
        } catch {
          errorBody = { message: response.statusText };
        }

        const error = createErrorFromResponse(response.status, errorBody);

        // Don't retry client errors (except rate limit)
        if (response.status < 500 && response.status !== 429) {
          throw error;
        }

        lastError = error;

        // Wait before retrying
        if (attempt < this.config.maxRetries - 1) {
          await this.sleep(this.config.retryDelay * Math.pow(2, attempt));
        }
      } catch (err) {
        clearTimeout(timeoutId);

        if (err instanceof BvraiError) {
          throw err;
        }

        if (err instanceof Error) {
          if (err.name === 'AbortError') {
            throw new TimeoutError(`Request timed out after ${timeoutMs}ms`);
          }
          lastError = err;
        }

        // Wait before retrying network errors
        if (attempt < this.config.maxRetries - 1) {
          await this.sleep(this.config.retryDelay * Math.pow(2, attempt));
        }
      }
    }

    throw lastError || new ConnectionError('Request failed after max retries');
  }

  /**
   * GET request
   */
  async get<T>(
    path: string,
    params?: Record<string, string | number | boolean | undefined>,
    options?: RequestOptions
  ): Promise<T> {
    return this.request<T>({ method: 'GET', path, params, ...options });
  }

  /**
   * POST request
   */
  async post<T>(
    path: string,
    body?: Record<string, unknown>,
    options?: RequestOptions
  ): Promise<T> {
    return this.request<T>({ method: 'POST', path, body, ...options });
  }

  /**
   * PUT request
   */
  async put<T>(
    path: string,
    body?: Record<string, unknown>,
    options?: RequestOptions
  ): Promise<T> {
    return this.request<T>({ method: 'PUT', path, body, ...options });
  }

  /**
   * PATCH request
   */
  async patch<T>(
    path: string,
    body?: Record<string, unknown>,
    options?: RequestOptions
  ): Promise<T> {
    return this.request<T>({ method: 'PATCH', path, body, ...options });
  }

  /**
   * DELETE request
   */
  async delete<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>({ method: 'DELETE', path, ...options });
  }

  /**
   * GET request returning raw bytes
   */
  async getRaw(
    path: string,
    params?: Record<string, string | number | boolean | undefined>,
    options?: RequestOptions
  ): Promise<ArrayBuffer> {
    const url = new URL(path, this.config.baseUrl);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          url.searchParams.append(key, String(value));
        }
      });
    }

    const headers: Record<string, string> = {
      'Authorization': `Bearer ${this.config.apiKey}`,
      'User-Agent': '@bvrai/sdk/1.0.0',
    };

    const controller = new AbortController();
    const timeoutMs = options?.timeout || this.config.timeout;
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers,
        signal: options?.signal || controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorBody: Record<string, unknown>;
        try {
          errorBody = await response.json();
        } catch {
          errorBody = { message: response.statusText };
        }
        throw createErrorFromResponse(response.status, errorBody);
      }

      return response.arrayBuffer();
    } catch (err) {
      clearTimeout(timeoutId);

      if (err instanceof BvraiError) {
        throw err;
      }

      if (err instanceof Error && err.name === 'AbortError') {
        throw new TimeoutError(`Request timed out after ${timeoutMs}ms`);
      }

      throw new ConnectionError('Failed to fetch raw data');
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
