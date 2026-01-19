/**
 * Builder Engine TypeScript SDK - Streaming Client
 *
 * WebSocket client for real-time events.
 */

import { WebSocketError } from './exceptions';

export type StreamEventType =
  | 'connected'
  | 'disconnected'
  | 'error'
  | 'call.started'
  | 'call.ringing'
  | 'call.answered'
  | 'call.ended'
  | 'call.failed'
  | 'audio.start'
  | 'audio.data'
  | 'audio.end'
  | 'transcription.partial'
  | 'transcription.final'
  | 'user.speech.start'
  | 'user.speech.end'
  | 'agent.speech.start'
  | 'agent.speech.end'
  | 'function.call'
  | 'function.result'
  | 'dtmf.received';

export interface StreamEvent {
  type: StreamEventType;
  data: Record<string, any>;
  callId?: string;
  timestamp?: string;
}

export type EventHandler = (event: StreamEvent) => void | Promise<void>;

export interface StreamingClientConfig {
  apiKey: string;
  baseUrl?: string;
}

/**
 * WebSocket client for real-time streaming events.
 *
 * @example
 * ```typescript
 * const streaming = client.streaming();
 *
 * streaming.on('transcription.final', (event) => {
 *   console.log('User said:', event.data.text);
 * });
 *
 * await streaming.connect({ callId: 'call_abc123' });
 * ```
 */
export class StreamingClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private ws: WebSocket | null = null;
  private handlers: Map<StreamEventType | '*', EventHandler[]> = new Map();
  private running = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  constructor(config: StreamingClientConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'wss://api.builderengine.io';
  }

  /**
   * Register an event handler.
   */
  on(eventType: StreamEventType | '*', handler: EventHandler): this {
    const handlers = this.handlers.get(eventType) || [];
    handlers.push(handler);
    this.handlers.set(eventType, handlers);
    return this;
  }

  /**
   * Remove an event handler.
   */
  off(eventType: StreamEventType | '*', handler: EventHandler): this {
    const handlers = this.handlers.get(eventType);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index !== -1) {
        handlers.splice(index, 1);
      }
    }
    return this;
  }

  /**
   * Connect to the WebSocket server.
   */
  async connect(options: { callId?: string; subscribeAll?: boolean } = {}): Promise<void> {
    const { callId, subscribeAll } = options;

    let url = `${this.baseUrl}/ws/v1/stream`;
    const params = new URLSearchParams();
    if (callId) params.append('call_id', callId);
    if (subscribeAll) params.append('subscribe_all', 'true');

    const queryString = params.toString();
    if (queryString) url += `?${queryString}`;

    return new Promise((resolve, reject) => {
      try {
        // For browser environment
        if (typeof WebSocket !== 'undefined') {
          this.ws = new WebSocket(url, ['builderengine', this.apiKey]);
        } else {
          // For Node.js environment, would need ws package
          throw new Error('WebSocket not available. Install ws package for Node.js.');
        }

        this.ws.onopen = () => {
          this.running = true;
          this.reconnectAttempts = 0;
          this.dispatchEvent({ type: 'connected', data: { url, callId } });
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.dispatchEvent(data as StreamEvent);
          } catch (e) {
            console.warn('Invalid message:', e);
          }
        };

        this.ws.onerror = (error) => {
          this.dispatchEvent({ type: 'error', data: { error } });
        };

        this.ws.onclose = (event) => {
          this.dispatchEvent({ type: 'disconnected', data: { code: event.code, reason: event.reason } });
          if (this.running) {
            this.handleReconnect();
          }
        };
      } catch (error) {
        reject(new WebSocketError(`Failed to connect: ${error}`));
      }
    });
  }

  /**
   * Disconnect from the WebSocket server.
   */
  disconnect(): void {
    this.running = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Send a message to the server.
   */
  send(message: Record<string, any>): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new WebSocketError('Not connected');
    }
    this.ws.send(JSON.stringify(message));
  }

  /**
   * Subscribe to events for a specific call.
   */
  subscribe(callId: string): void {
    this.send({ action: 'subscribe', call_id: callId });
  }

  /**
   * Unsubscribe from events for a specific call.
   */
  unsubscribe(callId: string): void {
    this.send({ action: 'unsubscribe', call_id: callId });
  }

  /**
   * Send audio data to a call.
   */
  sendAudio(callId: string, audioData: ArrayBuffer): void {
    const base64 = btoa(String.fromCharCode(...new Uint8Array(audioData)));
    this.send({ action: 'audio', call_id: callId, data: base64 });
  }

  /**
   * Inject text for the agent to speak.
   */
  injectText(callId: string, text: string, interrupt: boolean = false): void {
    this.send({ action: 'inject', call_id: callId, text, interrupt });
  }

  /**
   * Dispatch an event to all handlers.
   */
  private dispatchEvent(event: StreamEvent): void {
    // Call specific handlers
    const handlers = this.handlers.get(event.type) || [];
    handlers.forEach(handler => {
      try {
        handler(event);
      } catch (e) {
        console.error('Error in event handler:', e);
      }
    });

    // Call wildcard handlers
    const wildcardHandlers = this.handlers.get('*') || [];
    wildcardHandlers.forEach(handler => {
      try {
        handler(event);
      } catch (e) {
        console.error('Error in wildcard handler:', e);
      }
    });
  }

  /**
   * Handle reconnection after disconnect.
   */
  private async handleReconnect(): Promise<void> {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.pow(2, this.reconnectAttempts) * 1000;

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    await new Promise(resolve => setTimeout(resolve, delay));

    try {
      await this.connect();
    } catch (e) {
      console.error('Reconnection failed:', e);
    }
  }
}
