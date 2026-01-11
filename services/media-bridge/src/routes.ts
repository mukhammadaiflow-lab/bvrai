/**
 * Express routes for Media Bridge / Media Plane Service
 *
 * HTTP API for:
 * - Health/readiness checks
 * - Room management
 * - Bridge (server-side agent) management
 * - Statistics
 */
import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { v4 as uuidv4 } from 'uuid';
import { roomManager } from './room-manager';
import { getMediaServerStats } from './media-server';
import { logger } from './logger';

export const router = Router();

/**
 * Async handler wrapper
 */
const asyncHandler = (fn: (req: Request, res: Response, next: NextFunction) => Promise<void>) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

/**
 * Health check
 * GET /health
 */
router.get('/health', (_req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    service: 'media-bridge',
    timestamp: new Date().toISOString(),
  });
});

/**
 * Readiness check with statistics
 * GET /ready
 */
router.get('/ready', (_req: Request, res: Response) => {
  const stats = getMediaServerStats();
  res.json({
    status: 'ready',
    service: 'media-bridge',
    stats,
    timestamp: new Date().toISOString(),
  });
});

/**
 * Get server statistics
 * GET /stats
 */
router.get('/stats', (_req: Request, res: Response) => {
  const stats = getMediaServerStats();
  const rooms = roomManager.listRooms();

  res.json({
    ...stats,
    roomDetails: rooms,
  });
});

/**
 * List all rooms
 * GET /rooms
 */
router.get('/rooms', (_req: Request, res: Response) => {
  const rooms = roomManager.listRooms();
  res.json({
    rooms,
    count: rooms.length,
  });
});

/**
 * Get room details
 * GET /rooms/:roomName
 */
router.get('/rooms/:roomName', (req: Request, res: Response) => {
  const { roomName } = req.params;
  const room = roomManager.getRoom(roomName);

  if (!room) {
    res.status(404).json({
      error: 'Room not found',
      code: 'ROOM_NOT_FOUND',
    });
    return;
  }

  // Get participant list (excluding hidden participants)
  const participants = Array.from(room.participants.values())
    .filter(p => !p.grants.hidden)
    .map(p => ({
      id: p.id,
      identity: p.identity,
      name: p.name,
      isAgent: p.isAgent,
      joinedAt: p.joinedAt.toISOString(),
    }));

  res.json({
    name: room.name,
    participantCount: room.participants.size,
    participants,
    createdAt: room.createdAt.toISOString(),
    lastActivityAt: room.lastActivityAt.toISOString(),
  });
});

/**
 * Close a room (admin operation)
 * DELETE /rooms/:roomName
 */
router.delete('/rooms/:roomName', (req: Request, res: Response) => {
  const { roomName } = req.params;
  const room = roomManager.getRoom(roomName);

  if (!room) {
    res.status(404).json({
      error: 'Room not found',
      code: 'ROOM_NOT_FOUND',
    });
    return;
  }

  roomManager.closeRoom(roomName);

  res.json({
    status: 'closed',
    room: roomName,
  });
});

/**
 * Start a server-side agent/bridge in a room
 * POST /bridge/start
 *
 * This creates a virtual participant that can:
 * - Receive transcripts from human participants
 * - Inject TTS audio responses
 * - Forward to/from Dialog Manager
 *
 * Request:
 * {
 *   "room": "room-name",
 *   "agent_id": "optional-custom-id"
 * }
 *
 * Response:
 * {
 *   "bridge_id": "uuid"
 * }
 */
const bridgeStartSchema = z.object({
  room: z.string().min(1).max(128).regex(/^[a-zA-Z0-9_-]+$/),
  agent_id: z.string().max(128).optional(),
});

// Track active bridges
const activeBridges = new Map<string, {
  id: string;
  room: string;
  agentId: string;
  createdAt: Date;
}>();

router.post(
  '/bridge/start',
  asyncHandler(async (req: Request, res: Response) => {
    try {
      const validated = bridgeStartSchema.parse(req.body);

      const bridgeId = uuidv4();
      const agentId = validated.agent_id || `agent-${bridgeId.slice(0, 8)}`;

      // Ensure room exists
      roomManager.getOrCreateRoom(validated.room);

      // Track the bridge
      activeBridges.set(bridgeId, {
        id: bridgeId,
        room: validated.room,
        agentId,
        createdAt: new Date(),
      });

      logger.info({
        msg: 'Bridge started',
        bridgeId,
        room: validated.room,
        agentId,
      });

      res.status(201).json({
        bridge_id: bridgeId,
        room: validated.room,
        agent_id: agentId,
        ws_url: `/media/${validated.room}`,
      });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({
          error: 'Validation failed',
          code: 'VALIDATION_ERROR',
          details: error.errors,
        });
        return;
      }
      throw error;
    }
  })
);

/**
 * Stop a bridge
 * POST /bridge/:bridgeId/stop
 */
router.post('/bridge/:bridgeId/stop', (req: Request, res: Response) => {
  const { bridgeId } = req.params;

  const bridge = activeBridges.get(bridgeId);
  if (!bridge) {
    res.status(404).json({
      error: 'Bridge not found',
      code: 'BRIDGE_NOT_FOUND',
    });
    return;
  }

  activeBridges.delete(bridgeId);

  logger.info({
    msg: 'Bridge stopped',
    bridgeId,
    room: bridge.room,
  });

  res.json({
    status: 'stopped',
    bridge_id: bridgeId,
  });
});

/**
 * Get bridge status
 * GET /bridge/:bridgeId
 */
router.get('/bridge/:bridgeId', (req: Request, res: Response) => {
  const { bridgeId } = req.params;

  const bridge = activeBridges.get(bridgeId);
  if (!bridge) {
    res.status(404).json({
      error: 'Bridge not found',
      code: 'BRIDGE_NOT_FOUND',
    });
    return;
  }

  res.json({
    bridge_id: bridge.id,
    room: bridge.room,
    agent_id: bridge.agentId,
    created_at: bridge.createdAt.toISOString(),
  });
});

/**
 * List all bridges
 * GET /bridges
 */
router.get('/bridges', (_req: Request, res: Response) => {
  const bridges = Array.from(activeBridges.values()).map(b => ({
    bridge_id: b.id,
    room: b.room,
    agent_id: b.agentId,
    created_at: b.createdAt.toISOString(),
  }));

  res.json({
    bridges,
    count: bridges.length,
  });
});

/**
 * Send a dialog response to a room (for testing/debugging)
 * POST /rooms/:roomName/dialog
 */
const dialogMessageSchema = z.object({
  speak_text: z.string().min(1),
  action_object: z.record(z.unknown()).nullable().optional(),
  confidence: z.number().min(0).max(1).optional(),
});

router.post(
  '/rooms/:roomName/dialog',
  asyncHandler(async (req: Request, res: Response) => {
    const { roomName } = req.params;

    const room = roomManager.getRoom(roomName);
    if (!room) {
      res.status(404).json({
        error: 'Room not found',
        code: 'ROOM_NOT_FOUND',
      });
      return;
    }

    try {
      const validated = dialogMessageSchema.parse(req.body);

      roomManager.sendDialogResponse(roomName, {
        speakText: validated.speak_text,
        actionObject: validated.action_object || null,
        confidence: validated.confidence || 0.9,
        sessionId: 'api-inject',
      });

      res.json({
        status: 'sent',
        room: roomName,
      });
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({
          error: 'Validation failed',
          code: 'VALIDATION_ERROR',
          details: error.errors,
        });
        return;
      }
      throw error;
    }
  })
);
