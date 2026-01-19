/**
 * Room Manager for Self-Hosted Media Plane
 *
 * Manages rooms and participants in-memory.
 * Handles:
 * - Room creation and cleanup
 * - Participant join/leave
 * - Message routing between participants
 * - Agent injection for server-side bots
 *
 * TODO: For production scaling, consider:
 * - Redis-based room state for multi-instance deployment
 * - Room persistence for reconnection handling
 * - Server-side audio mixing (currently just forwards frames)
 */
import { WebSocket } from 'ws';
import EventEmitter from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';
import { config } from './config';
import { logger } from './logger';

/**
 * Permission grants from JWT token
 */
export interface ParticipantGrants {
  canPublish: boolean;
  canSubscribe: boolean;
  canPublishData: boolean;
  hidden: boolean;
  isAgent: boolean;
  canManageRoom: boolean;
}

/**
 * Participant in a room
 */
export interface Participant {
  id: string;
  identity: string;
  name?: string;
  ws: WebSocket;
  grants: ParticipantGrants;
  joinedAt: Date;
  metadata?: string;
  isAgent: boolean;
}

/**
 * Room state
 */
export interface Room {
  name: string;
  participants: Map<string, Participant>;
  createdAt: Date;
  lastActivityAt: Date;
  tenantId?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Message types for WebSocket communication
 */
export type IncomingMessageType =
  | 'transcript'
  | 'audio'
  | 'data'
  | 'ping';

export type OutgoingMessageType =
  | 'transcript'
  | 'dialog_response'
  | 'tts_audio'
  | 'participant_joined'
  | 'participant_left'
  | 'error'
  | 'pong';

/**
 * Base message structure
 */
export interface BaseMessage {
  type: string;
  timestamp?: number;
}

/**
 * Incoming transcript message from participant
 */
export interface TranscriptMessage extends BaseMessage {
  type: 'transcript';
  text: string;
  isFinal?: boolean;
}

/**
 * Incoming audio frame (binary will be converted)
 */
export interface AudioMessage extends BaseMessage {
  type: 'audio';
  data: string; // base64 encoded audio
  encoding?: string;
  sampleRate?: number;
}

/**
 * Outgoing dialog response
 */
export interface DialogResponseMessage extends BaseMessage {
  type: 'dialog_response';
  speakText: string;
  actionObject: Record<string, unknown> | null;
  confidence: number;
  sessionId: string;
}

/**
 * Participant event
 */
export interface ParticipantEventMessage extends BaseMessage {
  type: 'participant_joined' | 'participant_left';
  participantId: string;
  identity: string;
  name?: string;
  isAgent: boolean;
}

/**
 * Error message
 */
export interface ErrorMessage extends BaseMessage {
  type: 'error';
  code: string;
  message: string;
}

/**
 * Events emitted by RoomManager
 */
interface RoomManagerEvents {
  roomCreated: (room: Room) => void;
  roomClosed: (roomName: string) => void;
  participantJoined: (roomName: string, participant: Participant) => void;
  participantLeft: (roomName: string, participantId: string) => void;
  transcript: (roomName: string, participant: Participant, message: TranscriptMessage) => void;
  audioFrame: (roomName: string, participant: Participant, data: Buffer) => void;
}

/**
 * RoomManager - Singleton manager for all rooms
 */
export class RoomManager extends EventEmitter<RoomManagerEvents> {
  private rooms: Map<string, Room> = new Map();
  private cleanupInterval: NodeJS.Timeout | null = null;
  private readonly maxRooms: number;
  private readonly maxParticipantsPerRoom: number;
  private readonly roomIdleTimeout: number;

  constructor() {
    super();
    this.maxRooms = parseInt(config.MAX_ROOMS, 10);
    this.maxParticipantsPerRoom = parseInt(config.MAX_PARTICIPANTS_PER_ROOM, 10);
    this.roomIdleTimeout = parseInt(config.ROOM_IDLE_TIMEOUT_MS, 10);

    // Start cleanup interval
    this.cleanupInterval = setInterval(
      () => this.cleanupIdleRooms(),
      60000 // Check every minute
    );
  }

  /**
   * Get or create a room
   */
  getOrCreateRoom(roomName: string, tenantId?: string): Room {
    let room = this.rooms.get(roomName);

    if (!room) {
      if (this.rooms.size >= this.maxRooms) {
        throw new Error('Maximum room capacity reached');
      }

      room = {
        name: roomName,
        participants: new Map(),
        createdAt: new Date(),
        lastActivityAt: new Date(),
        tenantId,
      };

      this.rooms.set(roomName, room);
      this.emit('roomCreated', room);

      logger.info({
        msg: 'Room created',
        room: roomName,
        totalRooms: this.rooms.size,
      });
    }

    return room;
  }

  /**
   * Add a participant to a room
   */
  addParticipant(
    roomName: string,
    identity: string,
    ws: WebSocket,
    grants: ParticipantGrants,
    name?: string,
    metadata?: string
  ): Participant {
    const room = this.getOrCreateRoom(roomName);

    if (room.participants.size >= this.maxParticipantsPerRoom) {
      throw new Error('Room is full');
    }

    // Check if participant with same identity already exists
    for (const [, p] of room.participants) {
      if (p.identity === identity) {
        throw new Error('Participant with this identity already in room');
      }
    }

    const participant: Participant = {
      id: uuidv4(),
      identity,
      name,
      ws,
      grants,
      joinedAt: new Date(),
      metadata,
      isAgent: grants.isAgent,
    };

    room.participants.set(participant.id, participant);
    room.lastActivityAt = new Date();

    this.emit('participantJoined', roomName, participant);

    // Broadcast to other participants (unless hidden)
    if (!grants.hidden) {
      this.broadcastToRoom(roomName, {
        type: 'participant_joined',
        participantId: participant.id,
        identity: participant.identity,
        name: participant.name,
        isAgent: participant.isAgent,
        timestamp: Date.now(),
      }, participant.id);
    }

    logger.info({
      msg: 'Participant joined',
      room: roomName,
      participantId: participant.id,
      identity,
      isAgent: grants.isAgent,
      totalParticipants: room.participants.size,
    });

    return participant;
  }

  /**
   * Remove a participant from a room
   */
  removeParticipant(roomName: string, participantId: string): void {
    const room = this.rooms.get(roomName);
    if (!room) return;

    const participant = room.participants.get(participantId);
    if (!participant) return;

    room.participants.delete(participantId);
    room.lastActivityAt = new Date();

    this.emit('participantLeft', roomName, participantId);

    // Broadcast to other participants (unless hidden)
    if (!participant.grants.hidden) {
      this.broadcastToRoom(roomName, {
        type: 'participant_left',
        participantId,
        identity: participant.identity,
        isAgent: participant.isAgent,
        timestamp: Date.now(),
      });
    }

    logger.info({
      msg: 'Participant left',
      room: roomName,
      participantId,
      identity: participant.identity,
      remainingParticipants: room.participants.size,
    });

    // Clean up empty rooms
    if (room.participants.size === 0) {
      this.closeRoom(roomName);
    }
  }

  /**
   * Handle incoming message from a participant
   */
  handleMessage(
    roomName: string,
    participantId: string,
    data: Buffer | string
  ): void {
    const room = this.rooms.get(roomName);
    if (!room) return;

    const participant = room.participants.get(participantId);
    if (!participant) return;

    room.lastActivityAt = new Date();

    // Handle binary data (audio frames)
    if (Buffer.isBuffer(data)) {
      if (!participant.grants.canPublish) {
        this.sendError(participant.ws, 'PERMISSION_DENIED', 'Cannot publish audio');
        return;
      }
      // TODO: Implement real audio processing here
      // For now, emit event for potential ASR processing
      this.emit('audioFrame', roomName, participant, data);
      return;
    }

    // Handle text messages (JSON)
    try {
      const message = JSON.parse(data) as BaseMessage;

      switch (message.type) {
        case 'transcript':
          this.handleTranscript(room, participant, message as TranscriptMessage);
          break;

        case 'audio':
          // Base64 encoded audio
          if (!participant.grants.canPublish) {
            this.sendError(participant.ws, 'PERMISSION_DENIED', 'Cannot publish audio');
            return;
          }
          const audioMsg = message as AudioMessage;
          const audioBuffer = Buffer.from(audioMsg.data, 'base64');
          this.emit('audioFrame', roomName, participant, audioBuffer);
          break;

        case 'data':
          // Data channel message - broadcast to room
          if (!participant.grants.canPublishData) {
            this.sendError(participant.ws, 'PERMISSION_DENIED', 'Cannot publish data');
            return;
          }
          this.broadcastToRoom(roomName, message, participantId);
          break;

        case 'ping':
          this.send(participant.ws, { type: 'pong', timestamp: Date.now() });
          break;

        default:
          logger.debug({
            msg: 'Unknown message type',
            type: message.type,
            room: roomName,
          });
      }
    } catch (error) {
      logger.warn({
        msg: 'Failed to parse message',
        error: error instanceof Error ? error.message : 'Unknown',
      });
      this.sendError(participant.ws, 'INVALID_MESSAGE', 'Invalid JSON message');
    }
  }

  /**
   * Handle transcript message
   */
  private handleTranscript(
    room: Room,
    participant: Participant,
    message: TranscriptMessage
  ): void {
    if (!participant.grants.canPublish) {
      this.sendError(participant.ws, 'PERMISSION_DENIED', 'Cannot publish');
      return;
    }

    logger.debug({
      msg: 'Transcript received',
      room: room.name,
      identity: participant.identity,
      text: message.text.substring(0, 50),
      isFinal: message.isFinal,
    });

    // Emit for Dialog Manager integration
    this.emit('transcript', room.name, participant, message);

    // Broadcast transcript to subscribers (for real-time display)
    this.broadcastToRoom(room.name, {
      type: 'transcript',
      text: message.text,
      isFinal: message.isFinal,
      participantId: participant.id,
      identity: participant.identity,
      timestamp: Date.now(),
    }, participant.id);
  }

  /**
   * Send dialog response to room
   */
  sendDialogResponse(
    roomName: string,
    response: {
      speakText: string;
      actionObject: Record<string, unknown> | null;
      confidence: number;
      sessionId: string;
    }
  ): void {
    const room = this.rooms.get(roomName);
    if (!room) return;

    const message: DialogResponseMessage = {
      type: 'dialog_response',
      speakText: response.speakText,
      actionObject: response.actionObject,
      confidence: response.confidence,
      sessionId: response.sessionId,
      timestamp: Date.now(),
    };

    // Broadcast to all participants
    this.broadcastToRoom(roomName, message);

    logger.info({
      msg: 'Dialog response sent',
      room: roomName,
      speakTextLength: response.speakText.length,
      hasAction: !!response.actionObject,
    });
  }

  /**
   * Inject TTS audio into room (from agent)
   * TODO: Implement real audio encoding/streaming
   */
  injectTTSAudio(roomName: string, audioData: Buffer, encoding: string): void {
    const room = this.rooms.get(roomName);
    if (!room) return;

    // For now, send as base64 - in production use proper audio streaming
    const message = {
      type: 'tts_audio',
      data: audioData.toString('base64'),
      encoding,
      timestamp: Date.now(),
    };

    this.broadcastToRoom(roomName, message);
  }

  /**
   * Broadcast message to all participants in a room
   */
  broadcastToRoom(
    roomName: string,
    message: object,
    excludeParticipantId?: string
  ): void {
    const room = this.rooms.get(roomName);
    if (!room) return;

    const messageStr = JSON.stringify(message);

    for (const [id, participant] of room.participants) {
      if (id === excludeParticipantId) continue;
      if (!participant.grants.canSubscribe) continue;

      if (participant.ws.readyState === WebSocket.OPEN) {
        participant.ws.send(messageStr);
      }
    }
  }

  /**
   * Send message to a specific participant
   */
  send(ws: WebSocket, message: object): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  /**
   * Send error message
   */
  sendError(ws: WebSocket, code: string, message: string): void {
    this.send(ws, {
      type: 'error',
      code,
      message,
      timestamp: Date.now(),
    });
  }

  /**
   * Get room info
   */
  getRoom(roomName: string): Room | undefined {
    return this.rooms.get(roomName);
  }

  /**
   * List all rooms
   */
  listRooms(): Array<{
    name: string;
    participantCount: number;
    createdAt: Date;
  }> {
    return Array.from(this.rooms.values()).map((room) => ({
      name: room.name,
      participantCount: room.participants.size,
      createdAt: room.createdAt,
    }));
  }

  /**
   * Close a room
   */
  closeRoom(roomName: string): void {
    const room = this.rooms.get(roomName);
    if (!room) return;

    // Close all participant connections
    for (const [, participant] of room.participants) {
      if (participant.ws.readyState === WebSocket.OPEN) {
        participant.ws.close(1000, 'Room closed');
      }
    }

    this.rooms.delete(roomName);
    this.emit('roomClosed', roomName);

    logger.info({
      msg: 'Room closed',
      room: roomName,
      totalRooms: this.rooms.size,
    });
  }

  /**
   * Clean up idle rooms
   */
  private cleanupIdleRooms(): void {
    const now = Date.now();
    const roomsToClose: string[] = [];

    for (const [name, room] of this.rooms) {
      const idleTime = now - room.lastActivityAt.getTime();
      if (idleTime > this.roomIdleTimeout && room.participants.size === 0) {
        roomsToClose.push(name);
      }
    }

    for (const name of roomsToClose) {
      logger.info({ msg: 'Cleaning up idle room', room: name });
      this.closeRoom(name);
    }
  }

  /**
   * Shutdown manager
   */
  shutdown(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }

    // Close all rooms
    for (const roomName of this.rooms.keys()) {
      this.closeRoom(roomName);
    }

    logger.info({ msg: 'Room manager shutdown complete' });
  }
}

// Singleton instance
export const roomManager = new RoomManager();
