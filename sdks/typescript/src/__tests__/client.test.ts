import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { BuilderEngine } from '../client';
import {
  AuthenticationError,
  RateLimitError,
  ValidationError,
  NotFoundError,
  BuilderEngineError,
} from '../exceptions';

describe('BuilderEngine Client', () => {
  let client: BuilderEngine;
  const mockFetch = vi.fn();

  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch);
    client = new BuilderEngine({ apiKey: 'test-api-key' });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should create client with default options', () => {
      const c = new BuilderEngine({ apiKey: 'test-key' });
      expect(c).toBeInstanceOf(BuilderEngine);
    });

    it('should create client with custom options', () => {
      const c = new BuilderEngine({
        apiKey: 'test-key',
        baseUrl: 'https://custom.api.com',
        timeout: 60000,
        maxRetries: 5,
      });
      expect(c).toBeInstanceOf(BuilderEngine);
    });

    it('should throw error without API key', () => {
      expect(() => new BuilderEngine({ apiKey: '' })).toThrow('API key is required');
    });
  });

  describe('request', () => {
    it('should make successful GET request', async () => {
      const mockResponse = { id: 'agent_123', name: 'Test Agent' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockResponse),
        headers: new Headers(),
      });

      const result = await client.request({
        method: 'GET',
        path: '/api/v1/agents/agent_123',
      });

      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.builderengine.io/api/v1/agents/agent_123',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-api-key',
            'Content-Type': 'application/json',
          }),
        })
      );
    });

    it('should make successful POST request with body', async () => {
      const mockResponse = { id: 'agent_new', name: 'New Agent' };
      const requestBody = { name: 'New Agent', voice: 'nova' };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 201,
        json: () => Promise.resolve(mockResponse),
        headers: new Headers(),
      });

      const result = await client.request({
        method: 'POST',
        path: '/api/v1/agents',
        body: requestBody,
      });

      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.builderengine.io/api/v1/agents',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(requestBody),
        })
      );
    });

    it('should include query params in GET request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ data: [] }),
        headers: new Headers(),
      });

      await client.request({
        method: 'GET',
        path: '/api/v1/agents',
        params: { limit: 10, offset: 0, status: 'active' },
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.builderengine.io/api/v1/agents?limit=10&offset=0&status=active',
        expect.any(Object)
      );
    });

    it('should handle 401 authentication error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ error: { message: 'Invalid API key' } }),
        headers: new Headers(),
      });

      await expect(
        client.request({ method: 'GET', path: '/api/v1/agents' })
      ).rejects.toThrow(AuthenticationError);
    });

    it('should handle 429 rate limit error', async () => {
      const headers = new Headers();
      headers.set('Retry-After', '30');

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () => Promise.resolve({ error: { message: 'Rate limit exceeded' } }),
        headers,
      });

      try {
        await client.request({ method: 'GET', path: '/api/v1/agents' });
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(RateLimitError);
        expect((error as RateLimitError).retryAfter).toBe(30);
      }
    });

    it('should handle 422 validation error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 422,
        json: () =>
          Promise.resolve({
            error: {
              message: 'Validation failed',
              details: [{ field: 'name', message: 'Name is required' }],
            },
          }),
        headers: new Headers(),
      });

      try {
        await client.request({ method: 'POST', path: '/api/v1/agents', body: {} });
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(ValidationError);
        expect((error as ValidationError).errors).toBeDefined();
      }
    });

    it('should handle 404 not found error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ error: { message: 'Agent not found' } }),
        headers: new Headers(),
      });

      await expect(
        client.request({ method: 'GET', path: '/api/v1/agents/nonexistent' })
      ).rejects.toThrow(NotFoundError);
    });

    it('should retry on server errors', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: () => Promise.resolve({ error: { message: 'Internal error' } }),
          headers: new Headers(),
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ id: 'agent_123' }),
          headers: new Headers(),
        });

      const result = await client.request({ method: 'GET', path: '/api/v1/agents/agent_123' });

      expect(result).toEqual({ id: 'agent_123' });
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it('should give up after max retries', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: { message: 'Internal error' } }),
        headers: new Headers(),
      });

      await expect(
        client.request({ method: 'GET', path: '/api/v1/agents/agent_123' })
      ).rejects.toThrow(BuilderEngineError);

      expect(mockFetch).toHaveBeenCalledTimes(4); // Initial + 3 retries
    });
  });

  describe('resources', () => {
    it('should have agents resource', () => {
      expect(client.agents).toBeDefined();
    });

    it('should have calls resource', () => {
      expect(client.calls).toBeDefined();
    });

    it('should have conversations resource', () => {
      expect(client.conversations).toBeDefined();
    });

    it('should have phoneNumbers resource', () => {
      expect(client.phoneNumbers).toBeDefined();
    });

    it('should have voices resource', () => {
      expect(client.voices).toBeDefined();
    });

    it('should have webhooks resource', () => {
      expect(client.webhooks).toBeDefined();
    });

    it('should have knowledgeBase resource', () => {
      expect(client.knowledgeBase).toBeDefined();
    });

    it('should have workflows resource', () => {
      expect(client.workflows).toBeDefined();
    });

    it('should have campaigns resource', () => {
      expect(client.campaigns).toBeDefined();
    });

    it('should have analytics resource', () => {
      expect(client.analytics).toBeDefined();
    });

    it('should have organizations resource', () => {
      expect(client.organizations).toBeDefined();
    });

    it('should have users resource', () => {
      expect(client.users).toBeDefined();
    });

    it('should have apiKeys resource', () => {
      expect(client.apiKeys).toBeDefined();
    });

    it('should have billing resource', () => {
      expect(client.billing).toBeDefined();
    });
  });

  describe('streaming', () => {
    it('should return streaming client', () => {
      const streaming = client.streaming();
      expect(streaming).toBeDefined();
      expect(typeof streaming.on).toBe('function');
      expect(typeof streaming.connect).toBe('function');
    });
  });
});

describe('Agents Resource', () => {
  let client: BuilderEngine;
  const mockFetch = vi.fn();

  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch);
    client = new BuilderEngine({ apiKey: 'test-api-key' });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should list agents', async () => {
    const mockResponse = {
      data: [{ id: 'agent_1' }, { id: 'agent_2' }],
      total: 2,
      limit: 10,
      offset: 0,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockResponse),
      headers: new Headers(),
    });

    const result = await client.agents.list({ limit: 10 });

    expect(result.data).toHaveLength(2);
    expect(result.total).toBe(2);
  });

  it('should create agent', async () => {
    const mockAgent = { id: 'agent_new', name: 'Test', voice: 'nova' };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: () => Promise.resolve(mockAgent),
      headers: new Headers(),
    });

    const result = await client.agents.create({
      name: 'Test',
      voice: 'nova',
      systemPrompt: 'You are helpful.',
    });

    expect(result.id).toBe('agent_new');
  });

  it('should get agent by id', async () => {
    const mockAgent = { id: 'agent_123', name: 'Test' };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockAgent),
      headers: new Headers(),
    });

    const result = await client.agents.get('agent_123');

    expect(result.id).toBe('agent_123');
  });

  it('should update agent', async () => {
    const mockAgent = { id: 'agent_123', name: 'Updated' };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockAgent),
      headers: new Headers(),
    });

    const result = await client.agents.update('agent_123', { name: 'Updated' });

    expect(result.name).toBe('Updated');
  });

  it('should delete agent', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 204,
      json: () => Promise.resolve(undefined),
      headers: new Headers(),
    });

    await expect(client.agents.delete('agent_123')).resolves.not.toThrow();
  });
});

describe('Calls Resource', () => {
  let client: BuilderEngine;
  const mockFetch = vi.fn();

  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch);
    client = new BuilderEngine({ apiKey: 'test-api-key' });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should create outbound call', async () => {
    const mockCall = { id: 'call_123', status: 'queued' };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: () => Promise.resolve(mockCall),
      headers: new Headers(),
    });

    const result = await client.calls.create({
      agentId: 'agent_123',
      toNumber: '+14155551234',
    });

    expect(result.id).toBe('call_123');
    expect(result.status).toBe('queued');
  });

  it('should get call status', async () => {
    const mockCall = { id: 'call_123', status: 'in_progress', duration: 45 };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockCall),
      headers: new Headers(),
    });

    const result = await client.calls.get('call_123');

    expect(result.status).toBe('in_progress');
    expect(result.duration).toBe(45);
  });

  it('should end call', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ success: true }),
      headers: new Headers(),
    });

    await expect(client.calls.end('call_123')).resolves.not.toThrow();
  });

  it('should get call transcript', async () => {
    const mockTranscript = {
      callId: 'call_123',
      messages: [
        { role: 'assistant', content: 'Hello!' },
        { role: 'user', content: 'Hi there' },
      ],
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockTranscript),
      headers: new Headers(),
    });

    const result = await client.calls.getTranscript('call_123');

    expect(result.messages).toHaveLength(2);
  });

  it('should get call recording', async () => {
    const mockRecording = { url: 'https://cdn.example.com/recording.mp3' };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockRecording),
      headers: new Headers(),
    });

    const result = await client.calls.getRecording('call_123');

    expect(result.url).toContain('recording.mp3');
  });
});

describe('Campaigns Resource', () => {
  let client: BuilderEngine;
  const mockFetch = vi.fn();

  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch);
    client = new BuilderEngine({ apiKey: 'test-api-key' });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should create campaign', async () => {
    const mockCampaign = { id: 'camp_123', name: 'Test Campaign', status: 'draft' };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: () => Promise.resolve(mockCampaign),
      headers: new Headers(),
    });

    const result = await client.campaigns.create({
      name: 'Test Campaign',
      agentId: 'agent_123',
    });

    expect(result.id).toBe('camp_123');
    expect(result.status).toBe('draft');
  });

  it('should start campaign', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ success: true }),
      headers: new Headers(),
    });

    await expect(client.campaigns.start('camp_123')).resolves.not.toThrow();
  });

  it('should pause campaign', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ success: true }),
      headers: new Headers(),
    });

    await expect(client.campaigns.pause('camp_123')).resolves.not.toThrow();
  });

  it('should get campaign progress', async () => {
    const mockProgress = {
      total: 100,
      completed: 45,
      failed: 5,
      pending: 50,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockProgress),
      headers: new Headers(),
    });

    const result = await client.campaigns.getProgress('camp_123');

    expect(result.total).toBe(100);
    expect(result.completed).toBe(45);
  });
});
