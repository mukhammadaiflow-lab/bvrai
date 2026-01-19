import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { StreamingClient, StreamEvent, StreamEventType } from '../streaming';

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  onopen: ((event: any) => void) | null = null;
  onmessage: ((event: any) => void) | null = null;
  onerror: ((event: any) => void) | null = null;
  onclose: ((event: any) => void) | null = null;

  private url: string;
  private protocols: string | string[] | undefined;
  private sentMessages: string[] = [];

  constructor(url: string, protocols?: string | string[]) {
    this.url = url;
    this.protocols = protocols;

    // Simulate connection after a tick
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen({});
      }
    }, 0);
  }

  send(data: string): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    this.sentMessages.push(data);
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose({ code: 1000, reason: 'Normal closure' });
    }
  }

  // Test helpers
  simulateMessage(data: StreamEvent): void {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(data) });
    }
  }

  simulateError(error: any): void {
    if (this.onerror) {
      this.onerror({ error });
    }
  }

  simulateClose(code: number, reason: string): void {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose({ code, reason });
    }
  }

  getSentMessages(): string[] {
    return this.sentMessages;
  }

  getUrl(): string {
    return this.url;
  }
}

describe('StreamingClient', () => {
  let mockWs: MockWebSocket | null = null;

  beforeEach(() => {
    vi.stubGlobal(
      'WebSocket',
      class extends MockWebSocket {
        constructor(url: string, protocols?: string | string[]) {
          super(url, protocols);
          mockWs = this;
        }
      }
    );
  });

  afterEach(() => {
    mockWs = null;
    vi.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should create client with config', () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      expect(client).toBeInstanceOf(StreamingClient);
    });

    it('should use custom base URL', () => {
      const client = new StreamingClient({
        apiKey: 'test-key',
        baseUrl: 'wss://custom.example.com',
      });
      expect(client).toBeInstanceOf(StreamingClient);
    });
  });

  describe('event handling', () => {
    it('should register event handler', () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      const handler = vi.fn();

      const result = client.on('call.started', handler);

      expect(result).toBe(client); // Chainable
    });

    it('should call handler when event is received', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      const handler = vi.fn();

      client.on('transcription.final', handler);
      await client.connect();

      const event: StreamEvent = {
        type: 'transcription.final',
        data: { text: 'Hello world' },
        callId: 'call_123',
      };
      mockWs?.simulateMessage(event);

      expect(handler).toHaveBeenCalledWith(event);
    });

    it('should call wildcard handler for all events', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      const handler = vi.fn();

      client.on('*', handler);
      await client.connect();

      const event1: StreamEvent = { type: 'call.started', data: {} };
      const event2: StreamEvent = { type: 'call.ended', data: {} };

      mockWs?.simulateMessage(event1);
      mockWs?.simulateMessage(event2);

      expect(handler).toHaveBeenCalledTimes(2);
    });

    it('should remove event handler with off()', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      const handler = vi.fn();

      client.on('call.started', handler);
      client.off('call.started', handler);

      await client.connect();

      mockWs?.simulateMessage({ type: 'call.started', data: {} });

      expect(handler).not.toHaveBeenCalled();
    });

    it('should handle multiple handlers for same event', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      client.on('call.started', handler1);
      client.on('call.started', handler2);

      await client.connect();

      mockWs?.simulateMessage({ type: 'call.started', data: {} });

      expect(handler1).toHaveBeenCalled();
      expect(handler2).toHaveBeenCalled();
    });
  });

  describe('connect', () => {
    it('should connect to WebSocket server', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      await client.connect();

      expect(mockWs).not.toBeNull();
      expect(mockWs?.getUrl()).toContain('/ws/v1/stream');
    });

    it('should include callId in connection URL', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      await client.connect({ callId: 'call_123' });

      expect(mockWs?.getUrl()).toContain('call_id=call_123');
    });

    it('should include subscribeAll in connection URL', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      await client.connect({ subscribeAll: true });

      expect(mockWs?.getUrl()).toContain('subscribe_all=true');
    });

    it('should dispatch connected event', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      const handler = vi.fn();

      client.on('connected', handler);
      await client.connect();

      expect(handler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'connected',
          data: expect.objectContaining({ url: expect.any(String) }),
        })
      );
    });
  });

  describe('disconnect', () => {
    it('should close WebSocket connection', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      await client.connect();
      client.disconnect();

      expect(mockWs?.readyState).toBe(MockWebSocket.CLOSED);
    });

    it('should dispatch disconnected event', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      const handler = vi.fn();

      client.on('disconnected', handler);
      await client.connect();
      client.disconnect();

      expect(handler).toHaveBeenCalled();
    });
  });

  describe('send', () => {
    it('should send JSON message', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      await client.connect();
      client.send({ action: 'test', data: 'value' });

      const sent = mockWs?.getSentMessages();
      expect(sent).toContain(JSON.stringify({ action: 'test', data: 'value' }));
    });

    it('should throw if not connected', () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      expect(() => client.send({ action: 'test' })).toThrow('Not connected');
    });
  });

  describe('subscribe', () => {
    it('should send subscribe message', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      await client.connect();
      client.subscribe('call_123');

      const sent = mockWs?.getSentMessages();
      expect(sent).toContain(JSON.stringify({ action: 'subscribe', call_id: 'call_123' }));
    });
  });

  describe('unsubscribe', () => {
    it('should send unsubscribe message', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      await client.connect();
      client.unsubscribe('call_123');

      const sent = mockWs?.getSentMessages();
      expect(sent).toContain(JSON.stringify({ action: 'unsubscribe', call_id: 'call_123' }));
    });
  });

  describe('sendAudio', () => {
    it('should send base64 encoded audio', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      await client.connect();

      // Create a simple ArrayBuffer
      const audioData = new Uint8Array([72, 101, 108, 108, 111]).buffer; // "Hello" in ASCII
      client.sendAudio('call_123', audioData);

      const sent = mockWs?.getSentMessages();
      const lastMessage = sent?.[sent.length - 1];
      const parsed = JSON.parse(lastMessage || '{}');

      expect(parsed.action).toBe('audio');
      expect(parsed.call_id).toBe('call_123');
      expect(parsed.data).toBe('SGVsbG8='); // "Hello" base64 encoded
    });
  });

  describe('injectText', () => {
    it('should send inject message', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      await client.connect();
      client.injectText('call_123', 'Please hold', false);

      const sent = mockWs?.getSentMessages();
      expect(sent).toContain(
        JSON.stringify({
          action: 'inject',
          call_id: 'call_123',
          text: 'Please hold',
          interrupt: false,
        })
      );
    });

    it('should send inject message with interrupt', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });

      await client.connect();
      client.injectText('call_123', 'Important update', true);

      const sent = mockWs?.getSentMessages();
      expect(sent).toContain(
        JSON.stringify({
          action: 'inject',
          call_id: 'call_123',
          text: 'Important update',
          interrupt: true,
        })
      );
    });
  });

  describe('error handling', () => {
    it('should dispatch error event on WebSocket error', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      const handler = vi.fn();

      client.on('error', handler);
      await client.connect();

      mockWs?.simulateError(new Error('Connection failed'));

      expect(handler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'error',
          data: expect.objectContaining({ error: expect.any(Error) }),
        })
      );
    });

    it('should handle invalid JSON messages gracefully', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      const handler = vi.fn();

      client.on('*', handler);
      await client.connect();

      // Simulate invalid JSON
      if (mockWs?.onmessage) {
        mockWs.onmessage({ data: 'not valid json' });
      }

      // Should not throw, just log warning
      expect(handler).not.toHaveBeenCalled();
    });

    it('should catch errors in event handlers', async () => {
      const client = new StreamingClient({ apiKey: 'test-key' });
      const errorHandler = vi.fn(() => {
        throw new Error('Handler error');
      });
      const successHandler = vi.fn();

      client.on('call.started', errorHandler);
      client.on('call.started', successHandler);

      await client.connect();

      // Should not throw
      expect(() => {
        mockWs?.simulateMessage({ type: 'call.started', data: {} });
      }).not.toThrow();

      // Second handler should still be called
      expect(successHandler).toHaveBeenCalled();
    });
  });
});

describe('StreamEvent types', () => {
  it('should have correct event types', () => {
    const eventTypes: StreamEventType[] = [
      'connected',
      'disconnected',
      'error',
      'call.started',
      'call.ringing',
      'call.answered',
      'call.ended',
      'call.failed',
      'audio.start',
      'audio.data',
      'audio.end',
      'transcription.partial',
      'transcription.final',
      'user.speech.start',
      'user.speech.end',
      'agent.speech.start',
      'agent.speech.end',
      'function.call',
      'function.result',
      'dtmf.received',
    ];

    // Type check - if this compiles, the types are correct
    eventTypes.forEach((type) => {
      const event: StreamEvent = { type, data: {} };
      expect(event.type).toBe(type);
    });
  });
});
