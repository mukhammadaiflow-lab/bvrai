/**
 * Streaming module for real-time events via WebSocket
 */

import EventEmitter from 'eventemitter3';
import type {
  StreamEvent,
  StreamEventType,
  TranscriptEvent,
  AudioFrame,
  ClientConfig,
} from './types';

// Check if we're in Node.js and if ws is available
let WebSocketImpl: typeof WebSocket;
try {
  if (typeof WebSocket !== 'undefined') {
    WebSocketImpl = WebSocket;
  } else {
    // In Node.js, try to use the ws package
    WebSocketImpl = require('ws');
  }
} catch {
  // WebSocket not available
}

export interface StreamingConnectionOptions {
  apiKey: string;
  callId?: string;
  sessionId?: string;
  baseUrl?: string;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export type StreamEventHandler = (event: StreamEvent) => void | Promise<void>;

export class StreamingConnection extends EventEmitter {
  private ws: WebSocket | null = null;
  private connected = false;
  private shouldStop = false;
  private reconnectCount = 0;
  private options: Required<StreamingConnectionOptions>;

  constructor(options: StreamingConnectionOptions) {
    super();

    if (!WebSocketImpl) {
      throw new Error(
        'WebSocket not available. In Node.js, install the "ws" package.'
      );
    }

    this.options = {
      apiKey: options.apiKey,
      callId: options.callId || '',
      sessionId: options.sessionId || '',
      baseUrl: options.baseUrl || 'wss://api.bvrai.com',
      autoReconnect: options.autoReconnect ?? true,
      reconnectInterval: options.reconnectInterval ?? 1000,
      maxReconnectAttempts: options.maxReconnectAttempts ?? 5,
    };
  }

  /**
   * Check if connected
   */
  get isConnected(): boolean {
    return this.connected && this.ws !== null;
  }

  /**
   * Connect to the streaming endpoint
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = this.buildUrl();

      this.ws = new WebSocketImpl(url, {
        headers: {
          Authorization: `Bearer ${this.options.apiKey}`,
        },
      } as any);

      this.ws.onopen = () => {
        this.connected = true;
        this.reconnectCount = 0;
        this.emit('connected', {
          type: 'connected' as StreamEventType,
          timestamp: new Date().toISOString(),
          data: { url },
          callId: this.options.callId,
          sessionId: this.options.sessionId,
        });
        resolve();
      };

      this.ws.onerror = (error) => {
        this.emit('error', {
          type: 'error' as StreamEventType,
          timestamp: new Date().toISOString(),
          data: { error: error.toString() },
          callId: this.options.callId,
          sessionId: this.options.sessionId,
        });
        if (!this.connected) {
          reject(new Error('WebSocket connection failed'));
        }
      };

      this.ws.onclose = () => {
        this.connected = false;
        this.emit('disconnected', {
          type: 'disconnected' as StreamEventType,
          timestamp: new Date().toISOString(),
          data: {},
          callId: this.options.callId,
          sessionId: this.options.sessionId,
        });

        if (!this.shouldStop && this.options.autoReconnect) {
          this.attemptReconnect();
        }
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
    });
  }

  /**
   * Disconnect from the streaming endpoint
   */
  disconnect(): void {
    this.shouldStop = true;
    this.connected = false;

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Send a message to the server
   */
  send(message: Record<string, unknown>): void {
    if (!this.ws || !this.connected) {
      throw new Error('Not connected');
    }

    this.ws.send(JSON.stringify(message));
  }

  /**
   * Send an audio frame
   */
  sendAudio(frame: AudioFrame): void {
    let base64Audio: string;

    if (frame.data instanceof ArrayBuffer) {
      base64Audio = this.arrayBufferToBase64(frame.data);
    } else if (Buffer.isBuffer(frame.data)) {
      base64Audio = frame.data.toString('base64');
    } else {
      throw new Error('Invalid audio data type');
    }

    this.send({
      type: 'audio.input',
      data: {
        audio: base64Audio,
        sample_rate: frame.sampleRate || 16000,
        channels: frame.channels || 1,
        encoding: frame.encoding || 'pcm_s16le',
        timestamp_ms: frame.timestampMs || 0,
      },
    });
  }

  /**
   * Send text input (simulated speech)
   */
  sendText(text: string): void {
    this.send({
      type: 'text.input',
      data: { text },
    });
  }

  /**
   * Send DTMF tones
   */
  sendDtmf(digits: string): void {
    this.send({
      type: 'dtmf.input',
      data: { digits },
    });
  }

  /**
   * Interrupt the current agent response
   */
  interrupt(): void {
    this.send({ type: 'interrupt' });
  }

  /**
   * Signal end of user turn
   */
  endTurn(): void {
    this.send({ type: 'turn.end' });
  }

  private buildUrl(): string {
    let url = `${this.options.baseUrl}/v1/stream`;
    if (this.options.callId) {
      url += `?call_id=${this.options.callId}`;
    } else if (this.options.sessionId) {
      url += `?session_id=${this.options.sessionId}`;
    }
    return url;
  }

  private handleMessage(data: string | ArrayBuffer): void {
    if (typeof data !== 'string') {
      // Binary message (audio)
      const event: StreamEvent = {
        type: 'audio.output',
        timestamp: new Date().toISOString(),
        data: { audio: data },
        callId: this.options.callId,
        sessionId: this.options.sessionId,
      };
      this.emit('audio.output', event);
      this.emit('event', event);
      return;
    }

    try {
      const parsed = JSON.parse(data);
      const event: StreamEvent = {
        type: parsed.type,
        timestamp: parsed.timestamp || new Date().toISOString(),
        data: parsed.data || {},
        callId: parsed.call_id || this.options.callId,
        sessionId: parsed.session_id || this.options.sessionId,
      };

      // Emit specific event type
      this.emit(event.type, event);

      // Emit generic event
      this.emit('event', event);
    } catch (err) {
      this.emit('error', {
        type: 'error' as StreamEventType,
        timestamp: new Date().toISOString(),
        data: { error: `Failed to parse message: ${err}` },
      });
    }
  }

  private async attemptReconnect(): Promise<void> {
    if (this.reconnectCount >= this.options.maxReconnectAttempts) {
      this.emit('error', {
        type: 'error' as StreamEventType,
        timestamp: new Date().toISOString(),
        data: { error: 'Max reconnect attempts reached' },
      });
      return;
    }

    this.reconnectCount++;
    const delay = this.options.reconnectInterval * Math.pow(2, this.reconnectCount - 1);

    await new Promise((resolve) => setTimeout(resolve, delay));

    if (!this.shouldStop) {
      try {
        await this.connect();
      } catch {
        // Will trigger another reconnect attempt via onclose
      }
    }
  }

  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }
}

/**
 * High-level streaming session for voice conversations
 */
export class StreamingSession extends EventEmitter {
  private connection: StreamingConnection | null = null;
  private _callId: string | null = null;
  private ended = false;

  private transcriptHandler?: (text: string, isFinal: boolean, speaker: string) => void | Promise<void>;
  private functionHandler?: (
    name: string,
    args: Record<string, unknown>
  ) => Record<string, unknown> | Promise<Record<string, unknown>>;
  private stateHandler?: (state: string) => void | Promise<void>;
  private audioHandler?: (data: ArrayBuffer | Buffer) => void | Promise<void>;

  constructor(
    private readonly apiKey: string,
    private readonly agentId: string,
    private readonly baseUrl = 'wss://api.bvrai.com'
  ) {
    super();
  }

  /**
   * Get the current call ID
   */
  get callId(): string | null {
    return this._callId;
  }

  /**
   * Check if connected
   */
  get connected(): boolean {
    return this.connection?.isConnected ?? false;
  }

  /**
   * Connect to streaming endpoint
   */
  async connect(): Promise<void> {
    this.connection = new StreamingConnection({
      apiKey: this.apiKey,
      baseUrl: this.baseUrl,
    });

    // Forward events
    this.connection.on('event', (event: StreamEvent) => {
      this.handleEvent(event);
    });

    await this.connection.connect();
  }

  /**
   * Disconnect from streaming endpoint
   */
  disconnect(): void {
    if (this.connection) {
      this.connection.disconnect();
      this.connection = null;
    }
  }

  /**
   * Start a new call
   */
  async startCall(params?: {
    toNumber?: string;
    fromNumber?: string;
    metadata?: Record<string, unknown>;
  }): Promise<string> {
    if (!this.connection) {
      throw new Error('Not connected');
    }

    this.connection.send({
      type: 'call.start',
      data: {
        agent_id: this.agentId,
        to_number: params?.toNumber,
        from_number: params?.fromNumber,
        metadata: params?.metadata || {},
      },
    });

    // Wait for call_id
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Timeout waiting for call to start'));
      }, 30000);

      const handler = (event: StreamEvent) => {
        if (event.type === 'call.started') {
          clearTimeout(timeout);
          this._callId = (event.data.call_id as string) || null;
          if (this._callId) {
            resolve(this._callId);
          } else {
            reject(new Error('No call_id received'));
          }
          this.connection?.off('event', handler);
        } else if (event.type === 'call.failed' || event.type === 'error') {
          clearTimeout(timeout);
          reject(new Error((event.data.error as string) || 'Call failed to start'));
          this.connection?.off('event', handler);
        }
      };

      this.connection!.on('event', handler);
    });
  }

  /**
   * End the current call
   */
  endCall(): void {
    if (this.connection && this._callId) {
      this.connection.send({
        type: 'call.end',
        data: { call_id: this._callId },
      });
    }
  }

  /**
   * Wait for the call to end
   */
  waitForEnd(): Promise<void> {
    if (this.ended) {
      return Promise.resolve();
    }

    return new Promise((resolve) => {
      this.once('ended', resolve);
    });
  }

  /**
   * Register transcript handler
   */
  onTranscript(
    handler: (text: string, isFinal: boolean, speaker: string) => void | Promise<void>
  ): this {
    this.transcriptHandler = handler;
    return this;
  }

  /**
   * Register function call handler
   */
  onFunctionCall(
    handler: (
      name: string,
      args: Record<string, unknown>
    ) => Record<string, unknown> | Promise<Record<string, unknown>>
  ): this {
    this.functionHandler = handler;
    return this;
  }

  /**
   * Register state change handler
   */
  onStateChange(handler: (state: string) => void | Promise<void>): this {
    this.stateHandler = handler;
    return this;
  }

  /**
   * Register audio handler
   */
  onAudio(handler: (data: ArrayBuffer | Buffer) => void | Promise<void>): this {
    this.audioHandler = handler;
    return this;
  }

  /**
   * Send audio data
   */
  sendAudio(data: ArrayBuffer | Buffer): void {
    if (this.connection) {
      this.connection.sendAudio({ data });
    }
  }

  /**
   * Send text input
   */
  sendText(text: string): void {
    if (this.connection) {
      this.connection.sendText(text);
    }
  }

  /**
   * Interrupt the agent
   */
  interrupt(): void {
    if (this.connection) {
      this.connection.interrupt();
    }
  }

  private async handleEvent(event: StreamEvent): Promise<void> {
    switch (event.type) {
      case 'call.started':
        this._callId = (event.data.call_id as string) || null;
        break;

      case 'call.ended':
      case 'call.failed':
        this.ended = true;
        this.emit('ended');
        break;

      case 'transcript.partial':
      case 'transcript.final':
        if (this.transcriptHandler) {
          const text = (event.data.text as string) || '';
          const isFinal = event.type === 'transcript.final';
          const speaker = (event.data.speaker as string) || 'user';
          await this.transcriptHandler(text, isFinal, speaker);
        }
        break;

      case 'function.call':
        if (this.functionHandler) {
          const name = (event.data.name as string) || '';
          const args = (event.data.arguments as Record<string, unknown>) || {};
          const funcCallId = event.data.call_id as string;

          try {
            const result = await this.functionHandler(name, args);
            this.connection?.send({
              type: 'function.result',
              data: {
                call_id: funcCallId,
                result,
              },
            });
          } catch (err) {
            this.connection?.send({
              type: 'function.result',
              data: {
                call_id: funcCallId,
                error: err instanceof Error ? err.message : String(err),
              },
            });
          }
        }
        break;

      case 'state.changed':
        if (this.stateHandler) {
          const state = (event.data.state as string) || '';
          await this.stateHandler(state);
        }
        break;

      case 'audio.output':
        if (this.audioHandler) {
          const audio = event.data.audio as ArrayBuffer | Buffer;
          await this.audioHandler(audio);
        }
        break;
    }

    // Forward all events
    this.emit(event.type, event);
    this.emit('event', event);
  }
}

/**
 * Parse a transcript event from a stream event
 */
export function parseTranscriptEvent(event: StreamEvent): TranscriptEvent | null {
  if (event.type !== 'transcript.partial' && event.type !== 'transcript.final') {
    return null;
  }

  return {
    text: (event.data.text as string) || '',
    isFinal: event.type === 'transcript.final',
    confidence: (event.data.confidence as number) || 0,
    speaker: ((event.data.speaker as string) || 'user') as 'user' | 'agent',
    startTime: (event.data.start_time as number) || 0,
    endTime: (event.data.end_time as number) || 0,
    words: event.data.words as TranscriptEvent['words'],
  };
}
