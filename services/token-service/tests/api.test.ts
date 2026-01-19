/**
 * API integration tests for Token Service
 */
import request from 'supertest';
import { app } from '../src/index';

describe('Token Service API', () => {
  describe('GET /health', () => {
    it('should return healthy status', async () => {
      const response = await request(app).get('/health');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('status', 'healthy');
      expect(response.body).toHaveProperty('service', 'token-service');
      expect(response.body).toHaveProperty('timestamp');
    });
  });

  describe('GET /ready', () => {
    it('should return ready status', async () => {
      const response = await request(app).get('/ready');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('status', 'ready');
    });
  });

  describe('POST /token', () => {
    const validRequest = {
      room: 'test-room',
      identity: 'user-123',
      ttl_seconds: 3600,
    };

    it('should generate a token for valid request', async () => {
      const response = await request(app)
        .post('/token')
        .send(validRequest)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('token');
      expect(response.body).toHaveProperty('wsUrl');
      expect(response.body).toHaveProperty('expiresAt');
      expect(response.body).toHaveProperty('identity', 'user-123');
      expect(response.body).toHaveProperty('room', 'test-room');
    });

    it('should return 400 for missing room', async () => {
      const response = await request(app)
        .post('/token')
        .send({ identity: 'user-123' })
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error', 'Validation failed');
      expect(response.body).toHaveProperty('code', 'VALIDATION_ERROR');
    });

    it('should return 400 for missing identity', async () => {
      const response = await request(app)
        .post('/token')
        .send({ room: 'test-room' })
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error', 'Validation failed');
    });

    it('should return 400 for invalid room name', async () => {
      const response = await request(app)
        .post('/token')
        .send({
          room: 'room with spaces!@#',
          identity: 'user-123',
        })
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('code', 'VALIDATION_ERROR');
    });

    it('should accept optional grants', async () => {
      const response = await request(app)
        .post('/token')
        .send({
          ...validRequest,
          grants: {
            canPublish: false,
            canSubscribe: true,
          },
        })
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('token');
    });

    it('should accept metadata as string', async () => {
      const response = await request(app)
        .post('/token')
        .send({
          ...validRequest,
          metadata: '{"role": "admin"}',
        })
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
    });
  });

  describe('POST /token/agent', () => {
    it('should generate agent token', async () => {
      const response = await request(app)
        .post('/token/agent')
        .send({
          room: 'agent-room',
          agent_id: 'bridge-001',
        })
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('token');
      expect(response.body.identity).toBe('agent-bridge-001');
    });

    it('should return 400 for missing agent_id', async () => {
      const response = await request(app)
        .post('/token/agent')
        .send({ room: 'agent-room' })
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(400);
    });
  });

  describe('POST /token/verify', () => {
    it('should verify a valid token', async () => {
      // First generate a token
      const tokenResponse = await request(app)
        .post('/token')
        .send({
          room: 'test-room',
          identity: 'user-123',
        })
        .set('Content-Type', 'application/json');

      const { token } = tokenResponse.body;

      // Then verify it
      const verifyResponse = await request(app)
        .post('/token/verify')
        .send({ token })
        .set('Content-Type', 'application/json');

      expect(verifyResponse.status).toBe(200);
      expect(verifyResponse.body).toHaveProperty('valid', true);
      expect(verifyResponse.body.claims).toHaveProperty('identity', 'user-123');
      expect(verifyResponse.body.claims).toHaveProperty('room', 'test-room');
    });

    it('should return 401 for invalid token', async () => {
      const response = await request(app)
        .post('/token/verify')
        .send({ token: 'invalid-token' })
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(401);
      expect(response.body).toHaveProperty('code', 'INVALID_TOKEN');
    });

    it('should return 400 for missing token', async () => {
      const response = await request(app)
        .post('/token/verify')
        .send({})
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('code', 'MISSING_TOKEN');
    });
  });

  describe('404 handling', () => {
    it('should return 404 for unknown routes', async () => {
      const response = await request(app).get('/unknown-route');

      expect(response.status).toBe(404);
      expect(response.body).toHaveProperty('code', 'NOT_FOUND');
    });
  });
});
