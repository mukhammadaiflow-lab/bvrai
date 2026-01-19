/**
 * Media Server - WebSocket server for self-hosted media plane
 *
 * Handles WebSocket connections at /media/{room}?token=JWT
 * - Authenticates connections using JWT tokens
 * - Routes participants to rooms
 * - Forwards messages between participants and Dialog Manager
 *
 * Protocol:
 * - Connect: WS /media/{room}?token=JWT
 * - Send: JSON text frames or binary audio frames
 * - Receive: JSON messages (transcripts, dialog responses, events)
 *
 * TODO: For production:
 * - Add TLS termination (or use a reverse proxy)
 * - Implement proper audio codec support (Opus/PCM)
 * - Add server-side audio mixing for multi-party calls
 * - Add WebRTC support for browser clients
 */
import { Server as HttpServer } from 'http';
import { WebSocket, WebSocketServer } from 'ws';
import { URL } from 'url';
import jwt from 'jsonwebtoken';
import axios from 'axios';
import { config } from './config';
import { logger } from './logger';
import {
  roomManager,
  ParticipantGrants,
  TranscriptMessage,
} from './room-manager';

/**
 * JWT claims structure (must match Token Service)
 */
interface MediaTokenClaims {
  iss: string;
  sub: string;
  room: string;
  nbf: number;
  exp: number;
  jti: string;
  grants: ParticipantGrants;
  name?: string;
  metadata?: string;
}

/**
 * Dialog Manager response structure
 */
interface DialogTurnResponse {
  speak_text: string;
  action_object: Record<string, unknown> | null;
  confidence: number;
}

/**
 * Active WebSocket connections mapped by participant ID
 */
const connections = new Map<string, { roomName: string; participantId: string }>();

/**
 * Session state for dialog context
 */
const sessions = new Map<string, {
  tenantId: string;
  sessionId: string;
  lastActivity: Date;
}>();

/**
 * Verify JWT token
 */
function verifyToken(token: string): MediaTokenClaims | null {
  try {
    const decoded = jwt.verify(token, config.JWT_SECRET, {
      algorithms: ['HS256'],
    }) as MediaTokenClaims;
    return decoded;
  } catch (error) {
    logger.warn({
      msg: 'Token verification failed',
      error: error instanceof Error ? error.message : 'Unknown',
    });
    return null;
  }
}

/**
 * Send transcript to Dialog Manager
 */
async function sendToDialogManager(
  tenantId: string,
  sessionId: string,
  transcript: string,
  isFinal: boolean
): Promise<DialogTurnResponse | null> {
  try {
    const response = await axios.post<DialogTurnResponse>(
      `${config.DIALOG_MANAGER_URL}/dialog/turn`,
      {
        tenant_id: tenantId,
        session_id: sessionId,
        transcript,
        is_final: isFinal,
      },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 10000,
      }
    );
    return response.data;
  } catch (error) {
    logger.error({
      msg: 'Dialog Manager request failed',
      error: error instanceof Error ? error.message : 'Unknown',
      tenantId,
      sessionId,
    });
    return null;
  }
}

/**
 * Initialize WebSocket server and attach to HTTP server
 */
export function initializeMediaServer(server: HttpServer): WebSocketServer {
  const wss = new WebSocketServer({
    server,
    path: undefined, // Handle routing manually
  });

  // Set up room manager event handlers
  setupRoomManagerHandlers();

  wss.on('connection', (ws: WebSocket, req) => {
    const url = new URL(req.url || '', `http://${req.headers.host}`);
    const pathParts = url.pathname.split('/').filter(Boolean);

    // Validate path: /media/{room}
    if (pathParts.length !== 2 || pathParts[0] !== 'media') {
      logger.warn({ msg: 'Invalid WebSocket path', path: url.pathname });
      ws.close(4000, 'Invalid path. Use /media/{room}?token=...');
      return;
    }

    const roomName = pathParts[1];
    const token = url.searchParams.get('token');

    if (!token) {
      logger.warn({ msg: 'Missing token', room: roomName });
      ws.close(4001, 'Missing token');
      return;
    }

    // Verify token
    const claims = verifyToken(token);
    if (!claims) {
      logger.warn({ msg: 'Invalid token', room: roomName });
      ws.close(4002, 'Invalid or expired token');
      return;
    }

    // Verify room matches token
    if (claims.room !== roomName) {
      logger.warn({
        msg: 'Room mismatch',
        tokenRoom: claims.room,
        requestRoom: roomName,
      });
      ws.close(4003, 'Token not valid for this room');
      return;
    }

    // Add participant to room
    handleConnection(ws, roomName, claims);
  });

  logger.info({ msg: 'Media server initialized' });
  return wss;
}

/**
 * Handle a new authenticated WebSocket connection
 */
function handleConnection(
  ws: WebSocket,
  roomName: string,
  claims: MediaTokenClaims
): void {
  try {
    const participant = roomManager.addParticipant(
      roomName,
      claims.sub,
      ws,
      claims.grants,
      claims.name,
      claims.metadata
    );

    // Track connection
    connections.set(participant.id, { roomName, participantId: participant.id });

    // Initialize or get session
    const sessionKey = `${roomName}:${claims.sub}`;
    if (!sessions.has(sessionKey)) {
      sessions.set(sessionKey, {
        tenantId: 'default', // TODO: Extract tenant from token or room
        sessionId: participant.id,
        lastActivity: new Date(),
      });
    }

    // Send welcome message
    roomManager.send(ws, {
      type: 'connected',
      participantId: participant.id,
      room: roomName,
      identity: claims.sub,
      grants: claims.grants,
      timestamp: Date.now(),
    });

    logger.info({
      msg: 'Client connected to media server',
      room: roomName,
      identity: claims.sub,
      participantId: participant.id,
    });

    // Handle incoming messages
    ws.on('message', (data: Buffer | ArrayBuffer | Buffer[], isBinary: boolean) => {
      let messageData: Buffer | string;

      if (isBinary) {
        // Binary audio data
        if (Buffer.isBuffer(data)) {
          messageData = data;
        } else if (data instanceof ArrayBuffer) {
          messageData = Buffer.from(data);
        } else {
          messageData = Buffer.concat(data);
        }
      } else {
        // Text message
        messageData = data.toString();
      }

      roomManager.handleMessage(roomName, participant.id, messageData);
    });

    // Handle connection close
    ws.on('close', (code, reason) => {
      logger.info({
        msg: 'Client disconnected',
        room: roomName,
        participantId: participant.id,
        code,
        reason: reason.toString(),
      });

      roomManager.removeParticipant(roomName, participant.id);
      connections.delete(participant.id);
    });

    // Handle errors
    ws.on('error', (error) => {
      logger.error({
        msg: 'WebSocket error',
        room: roomName,
        participantId: participant.id,
        error: error.message,
      });
    });
  } catch (error) {
    logger.error({
      msg: 'Failed to add participant',
      room: roomName,
      identity: claims.sub,
      error: error instanceof Error ? error.message : 'Unknown',
    });
    ws.close(4004, error instanceof Error ? error.message : 'Failed to join room');
  }
}

/**
 * Set up event handlers for room manager
 */
function setupRoomManagerHandlers(): void {
  // Handle transcript events - forward to Dialog Manager
  roomManager.on('transcript', async (roomName, participant, message: TranscriptMessage) => {
    // Only process final transcripts for dialog
    if (!message.isFinal) return;

    const sessionKey = `${roomName}:${participant.identity}`;
    const session = sessions.get(sessionKey);
    if (!session) {
      logger.warn({ msg: 'No session found', room: roomName, identity: participant.identity });
      return;
    }

    session.lastActivity = new Date();

    // Send to Dialog Manager
    const response = await sendToDialogManager(
      session.tenantId,
      session.sessionId,
      message.text,
      true
    );

    if (response) {
      // Broadcast dialog response to room
      roomManager.sendDialogResponse(roomName, {
        speakText: response.speak_text,
        actionObject: response.action_object,
        confidence: response.confidence,
        sessionId: session.sessionId,
      });

      // TODO: If TTS is enabled, convert speak_text to audio and inject
      // This would involve calling a TTS service and then:
      // roomManager.injectTTSAudio(roomName, audioBuffer, 'opus');
    }
  });

  // Handle audio frames - for future ASR integration
  roomManager.on('audioFrame', (roomName, participant, data) => {
    // TODO: Implement streaming ASR integration
    // This would involve:
    // 1. Creating an ASR session for the participant
    // 2. Streaming audio frames to ASR service
    // 3. Receiving transcripts and emitting them as transcript events
    logger.debug({
      msg: 'Audio frame received',
      room: roomName,
      identity: participant.identity,
      bytes: data.length,
    });
  });
}

/**
 * Get media server statistics
 */
export function getMediaServerStats(): {
  rooms: number;
  connections: number;
  sessions: number;
} {
  return {
    rooms: roomManager.listRooms().length,
    connections: connections.size,
    sessions: sessions.size,
  };
}
