/**
 * Unit tests for Token Generator
 */
import jwt from 'jsonwebtoken';
import {
  generateToken,
  generateAgentToken,
  verifyToken,
  tokenRequestSchema,
  MediaTokenClaims,
} from '../src/token-generator';

describe('Token Generator', () => {
  const validRequest = {
    room: 'test-room',
    identity: 'user-123',
    ttl_seconds: 3600,
    name: 'Test User',
    metadata: '{"role": "customer"}',
    grants: {
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    },
  };

  describe('generateToken', () => {
    it('should generate a valid JWT token', () => {
      const response = generateToken(validRequest);

      expect(response).toHaveProperty('token');
      expect(response).toHaveProperty('wsUrl');
      expect(response).toHaveProperty('expiresAt');
      expect(response).toHaveProperty('identity', 'user-123');
      expect(response).toHaveProperty('room', 'test-room');
    });

    it('should include correct claims in the token', () => {
      const response = generateToken(validRequest);
      const decoded = jwt.decode(response.token) as MediaTokenClaims;

      expect(decoded).toBeTruthy();
      expect(decoded.sub).toBe('user-123');
      expect(decoded.iss).toBe('bvrai-token-service');
      expect(decoded.room).toBe('test-room');
      expect(decoded.grants).toBeDefined();
      expect(decoded.grants.canPublish).toBe(true);
      expect(decoded.grants.canSubscribe).toBe(true);
      expect(decoded.name).toBe('Test User');
      expect(decoded.metadata).toBe('{"role": "customer"}');
    });

    it('should set expiration based on ttl_seconds', () => {
      const response = generateToken({
        room: 'test-room',
        identity: 'user-123',
        ttl_seconds: 7200,
      });

      const decoded = jwt.decode(response.token) as MediaTokenClaims;
      const now = Math.floor(Date.now() / 1000);

      expect(decoded.exp).toBeGreaterThan(now);
      expect(decoded.exp).toBeLessThanOrEqual(now + 7200 + 1);
    });

    it('should use default TTL when not specified', () => {
      const response = generateToken({
        room: 'test-room',
        identity: 'user-123',
      });

      expect(response.expiresAt).toBeGreaterThan(Math.floor(Date.now() / 1000));
    });

    it('should cap TTL at maximum allowed value', () => {
      const response = generateToken({
        room: 'test-room',
        identity: 'user-123',
        ttl_seconds: 999999, // Very large TTL
      });

      const decoded = jwt.decode(response.token) as MediaTokenClaims;
      const now = Math.floor(Date.now() / 1000);

      // Should be capped at max (86400 by default)
      expect(decoded.exp).toBeLessThanOrEqual(now + 86400 + 1);
    });

    it('should set default grants when not specified', () => {
      const response = generateToken({
        room: 'test-room',
        identity: 'user-123',
      });

      const decoded = jwt.decode(response.token) as MediaTokenClaims;

      expect(decoded.grants.canPublish).toBe(true);
      expect(decoded.grants.canSubscribe).toBe(true);
      expect(decoded.grants.canPublishData).toBe(true);
      expect(decoded.grants.hidden).toBe(false);
      expect(decoded.grants.isAgent).toBe(false);
    });
  });

  describe('generateAgentToken', () => {
    it('should generate a token with agent grants', () => {
      const response = generateAgentToken('agent-room', 'bridge-001');

      expect(response.identity).toBe('agent-bridge-001');
      expect(response.room).toBe('agent-room');

      const decoded = jwt.decode(response.token) as MediaTokenClaims;
      expect(decoded.grants.isAgent).toBe(true);
      expect(decoded.grants.hidden).toBe(true);
    });
  });

  describe('verifyToken', () => {
    it('should verify a valid token', () => {
      const response = generateToken(validRequest);
      const claims = verifyToken(response.token);

      expect(claims).toBeTruthy();
      expect(claims?.sub).toBe('user-123');
      expect(claims?.room).toBe('test-room');
    });

    it('should return null for invalid token', () => {
      const claims = verifyToken('invalid-token');
      expect(claims).toBeNull();
    });

    it('should return null for token with wrong secret', () => {
      const wrongToken = jwt.sign(
        { sub: 'user-123' },
        'wrong-secret-that-is-very-different',
        { algorithm: 'HS256' }
      );
      const claims = verifyToken(wrongToken);
      expect(claims).toBeNull();
    });
  });

  describe('tokenRequestSchema', () => {
    it('should validate a valid request', () => {
      const result = tokenRequestSchema.safeParse(validRequest);
      expect(result.success).toBe(true);
    });

    it('should reject empty room name', () => {
      const result = tokenRequestSchema.safeParse({
        ...validRequest,
        room: '',
      });
      expect(result.success).toBe(false);
    });

    it('should reject invalid room name characters', () => {
      const result = tokenRequestSchema.safeParse({
        ...validRequest,
        room: 'room with spaces',
      });
      expect(result.success).toBe(false);
    });

    it('should reject empty identity', () => {
      const result = tokenRequestSchema.safeParse({
        ...validRequest,
        identity: '',
      });
      expect(result.success).toBe(false);
    });

    it('should allow optional fields to be omitted', () => {
      const result = tokenRequestSchema.safeParse({
        room: 'test-room',
        identity: 'user-123',
      });
      expect(result.success).toBe(true);
    });
  });
});
