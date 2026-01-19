/**
 * Token Generator for Self-Hosted Media Plane
 *
 * Original implementation of JWT token signing for our WebSocket-based media plane.
 * Tokens are used to authenticate participants joining media rooms.
 *
 * Token Structure:
 * - Header: { alg: "HS256", typ: "JWT" }
 * - Payload: Contains room, identity, and permission grants
 * - Signature: HMAC-SHA256 with JWT_SECRET
 *
 * TODO: For production, consider:
 * - Migrating to asymmetric keys (RS256) for better key management
 * - KMS integration for key storage and rotation
 * - Token refresh mechanism for long-running sessions
 */
import jwt from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';
import { z } from 'zod';
import { config } from './config';
import { logger } from './logger';

/**
 * Permission grants for media room participants
 */
export interface MediaGrants {
  /** Can publish audio/video tracks */
  canPublish?: boolean;
  /** Can subscribe to other participants' tracks */
  canSubscribe?: boolean;
  /** Can publish data messages */
  canPublishData?: boolean;
  /** Hidden from participant list (for server-side agents) */
  hidden?: boolean;
  /** Marks this participant as an agent (server-side) */
  isAgent?: boolean;
  /** Can manage room (kick participants, etc.) */
  canManageRoom?: boolean;
}

/**
 * JWT claims structure for media plane tokens
 */
export interface MediaTokenClaims {
  /** Issuer identifier */
  iss: string;
  /** Subject (participant identity) */
  sub: string;
  /** Room name */
  room: string;
  /** Not before (Unix timestamp) */
  nbf: number;
  /** Expiration (Unix timestamp) */
  exp: number;
  /** Unique token ID */
  jti: string;
  /** Permission grants */
  grants: MediaGrants;
  /** Display name */
  name?: string;
  /** Custom metadata (JSON string) */
  metadata?: string;
}

/**
 * Token request validation schema
 */
export const tokenRequestSchema = z.object({
  room: z.string().min(1).max(128).regex(/^[a-zA-Z0-9_-]+$/,
    'Room name must be alphanumeric with underscores and hyphens'),
  identity: z.string().min(1).max(128),
  ttl_seconds: z.number().int().positive().optional(),
  name: z.string().max(256).optional(),
  metadata: z.string().max(4096).optional(),
  grants: z.object({
    canPublish: z.boolean().optional(),
    canSubscribe: z.boolean().optional(),
    canPublishData: z.boolean().optional(),
    hidden: z.boolean().optional(),
    isAgent: z.boolean().optional(),
    canManageRoom: z.boolean().optional(),
  }).optional(),
});

export type TokenRequest = z.infer<typeof tokenRequestSchema>;

/**
 * Token response structure
 */
export interface TokenResponse {
  token: string;
  wsUrl: string;
  expiresAt: number;
  identity: string;
  room: string;
}

/**
 * Generates a media plane access token with the specified grants
 *
 * @param request - Token request parameters
 * @returns TokenResponse with JWT and connection details
 */
export function generateToken(request: TokenRequest): TokenResponse {
  const {
    room,
    identity,
    ttl_seconds,
    name,
    metadata,
    grants = {},
  } = request;

  // Calculate TTL with bounds checking
  const defaultTtl = parseInt(config.DEFAULT_TOKEN_TTL_SECONDS, 10);
  const maxTtl = parseInt(config.MAX_TOKEN_TTL_SECONDS, 10);
  const effectiveTtl = Math.min(ttl_seconds || defaultTtl, maxTtl);

  const now = Math.floor(Date.now() / 1000);
  const expiresAt = now + effectiveTtl;

  // Build permission grants with defaults
  const mediaGrants: MediaGrants = {
    canPublish: grants.canPublish ?? true,
    canSubscribe: grants.canSubscribe ?? true,
    canPublishData: grants.canPublishData ?? true,
    hidden: grants.hidden ?? false,
    isAgent: grants.isAgent ?? false,
    canManageRoom: grants.canManageRoom ?? false,
  };

  // Build JWT claims
  const claims: MediaTokenClaims = {
    iss: 'bvrai-token-service',
    sub: identity,
    room: room,
    nbf: now,
    exp: expiresAt,
    jti: uuidv4(),
    grants: mediaGrants,
  };

  // Add optional claims
  if (name) {
    claims.name = name;
  }
  if (metadata) {
    claims.metadata = metadata;
  }

  // Sign the token with HMAC-SHA256
  // TODO: For production, consider RS256 with KMS-managed keys
  const token = jwt.sign(claims, config.JWT_SECRET, {
    algorithm: 'HS256',
    header: { alg: 'HS256', typ: 'JWT' },
  });

  logger.debug({
    msg: 'Token generated',
    identity: identity,
    room: room,
    ttl: effectiveTtl,
    grants: Object.keys(mediaGrants).filter(k => mediaGrants[k as keyof MediaGrants]),
  });

  return {
    token,
    wsUrl: config.MEDIA_PLANE_WS_URL,
    expiresAt,
    identity,
    room,
  };
}

/**
 * Generates a token specifically for server-side agents (Media Bridge bots)
 * These tokens have elevated permissions and are marked as hidden agents
 */
export function generateAgentToken(room: string, agentId: string): TokenResponse {
  return generateToken({
    room,
    identity: `agent-${agentId}`,
    name: 'Voice Agent',
    grants: {
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
      hidden: true,
      isAgent: true,
      canManageRoom: false,
    },
  });
}

/**
 * Verifies and decodes a media plane token
 * Used by Media Bridge to authenticate WebSocket connections
 *
 * @param token - JWT token string
 * @returns Decoded claims if valid, null if invalid/expired
 */
export function verifyToken(token: string): MediaTokenClaims | null {
  try {
    const decoded = jwt.verify(token, config.JWT_SECRET, {
      algorithms: ['HS256'],
    }) as MediaTokenClaims;
    return decoded;
  } catch (error) {
    logger.warn({
      msg: 'Token verification failed',
      error: error instanceof Error ? error.message : 'Unknown error',
    });
    return null;
  }
}

/**
 * Export the JWT secret for use by other services (Media Bridge)
 * In production, this should use a shared secret store or asymmetric keys
 */
export function getJwtSecret(): string {
  return config.JWT_SECRET;
}
