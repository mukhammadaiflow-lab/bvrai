/**
 * Express routes for Token Service
 * Handles token generation and health endpoints
 */
import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import {
  generateToken,
  generateAgentToken,
  verifyToken,
  tokenRequestSchema,
} from './token-generator';
import { logger } from './logger';

export const router = Router();

/**
 * Error response structure
 */
interface ErrorResponse {
  error: string;
  code: string;
  details?: unknown;
}

/**
 * Async handler wrapper to catch errors
 */
const asyncHandler = (fn: (req: Request, res: Response, next: NextFunction) => Promise<void>) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

/**
 * Health check endpoint
 * GET /health
 */
router.get('/health', (_req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    service: 'token-service',
    timestamp: new Date().toISOString(),
  });
});

/**
 * Readiness check endpoint
 * GET /ready
 */
router.get('/ready', (_req: Request, res: Response) => {
  // TODO: Add checks for downstream dependencies if needed
  res.json({
    status: 'ready',
    service: 'token-service',
    timestamp: new Date().toISOString(),
  });
});

/**
 * Generate access token for a room participant
 * POST /token
 *
 * Request body:
 * {
 *   "room": "room-name",
 *   "identity": "user-123",
 *   "ttl_seconds": 3600,
 *   "name": "Display Name",
 *   "metadata": "{\"role\": \"customer\"}",
 *   "grants": {
 *     "canPublish": true,
 *     "canSubscribe": true
 *   }
 * }
 *
 * Response:
 * {
 *   "token": "eyJ...",
 *   "wsUrl": "wss://livekit.example.com",
 *   "expiresAt": 1704067200,
 *   "identity": "user-123",
 *   "room": "room-name"
 * }
 */
router.post(
  '/token',
  asyncHandler(async (req: Request, res: Response) => {
    try {
      // Validate request body
      const validatedRequest = tokenRequestSchema.parse(req.body);

      // Generate token
      const tokenResponse = generateToken(validatedRequest);

      logger.info({
        msg: 'Token generated successfully',
        room: validatedRequest.room,
        identity: validatedRequest.identity,
        expiresAt: tokenResponse.expiresAt,
      });

      res.json(tokenResponse);
    } catch (error) {
      if (error instanceof z.ZodError) {
        const errorResponse: ErrorResponse = {
          error: 'Validation failed',
          code: 'VALIDATION_ERROR',
          details: error.errors.map(e => ({
            field: e.path.join('.'),
            message: e.message,
          })),
        };
        res.status(400).json(errorResponse);
        return;
      }
      throw error;
    }
  })
);

/**
 * Generate agent token for server-side participants
 * POST /token/agent
 *
 * Request body:
 * {
 *   "room": "room-name",
 *   "agent_id": "agent-uuid"
 * }
 */
const agentTokenSchema = z.object({
  room: z.string().min(1).max(128).regex(/^[a-zA-Z0-9_-]+$/),
  agent_id: z.string().min(1).max(128),
});

router.post(
  '/token/agent',
  asyncHandler(async (req: Request, res: Response) => {
    try {
      const { room, agent_id } = agentTokenSchema.parse(req.body);
      const tokenResponse = generateAgentToken(room, agent_id);

      logger.info({
        msg: 'Agent token generated',
        room,
        agentId: agent_id,
      });

      res.json(tokenResponse);
    } catch (error) {
      if (error instanceof z.ZodError) {
        const errorResponse: ErrorResponse = {
          error: 'Validation failed',
          code: 'VALIDATION_ERROR',
          details: error.errors,
        };
        res.status(400).json(errorResponse);
        return;
      }
      throw error;
    }
  })
);

/**
 * Verify a token (for debugging)
 * POST /token/verify
 *
 * Request body:
 * {
 *   "token": "eyJ..."
 * }
 */
router.post(
  '/token/verify',
  asyncHandler(async (req: Request, res: Response) => {
    const { token } = req.body;

    if (!token || typeof token !== 'string') {
      res.status(400).json({
        error: 'Token is required',
        code: 'MISSING_TOKEN',
      });
      return;
    }

    const claims = verifyToken(token);

    if (!claims) {
      res.status(401).json({
        error: 'Invalid or expired token',
        code: 'INVALID_TOKEN',
      });
      return;
    }

    res.json({
      valid: true,
      claims: {
        identity: claims.sub,
        room: claims.room,
        expiresAt: claims.exp,
        grants: claims.grants,
      },
    });
  })
);
