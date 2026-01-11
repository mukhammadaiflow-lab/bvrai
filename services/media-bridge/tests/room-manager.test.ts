/**
 * Tests for Room Manager
 */
import { WebSocket } from 'ws';
import { RoomManager } from '../src/room-manager';

// Mock WebSocket
const createMockWs = (): WebSocket => {
  return {
    readyState: WebSocket.OPEN,
    send: jest.fn(),
    close: jest.fn(),
    on: jest.fn(),
  } as unknown as WebSocket;
};

describe('RoomManager', () => {
  let roomManager: RoomManager;

  beforeEach(() => {
    roomManager = new RoomManager();
  });

  afterEach(() => {
    roomManager.shutdown();
  });

  const defaultGrants = {
    canPublish: true,
    canSubscribe: true,
    canPublishData: true,
    hidden: false,
    isAgent: false,
    canManageRoom: false,
  };

  describe('getOrCreateRoom', () => {
    it('should create a new room', () => {
      const room = roomManager.getOrCreateRoom('test-room');

      expect(room).toBeDefined();
      expect(room.name).toBe('test-room');
      expect(room.participants.size).toBe(0);
    });

    it('should return existing room if already created', () => {
      const room1 = roomManager.getOrCreateRoom('test-room');
      const room2 = roomManager.getOrCreateRoom('test-room');

      expect(room1).toBe(room2);
    });

    it('should set tenantId if provided', () => {
      const room = roomManager.getOrCreateRoom('test-room', 'tenant-123');

      expect(room.tenantId).toBe('tenant-123');
    });
  });

  describe('addParticipant', () => {
    it('should add a participant to a room', () => {
      const ws = createMockWs();

      const participant = roomManager.addParticipant(
        'test-room',
        'user-123',
        ws,
        defaultGrants,
        'Test User'
      );

      expect(participant).toBeDefined();
      expect(participant.identity).toBe('user-123');
      expect(participant.name).toBe('Test User');
      expect(participant.isAgent).toBe(false);
    });

    it('should create room if it does not exist', () => {
      const ws = createMockWs();

      roomManager.addParticipant('new-room', 'user-123', ws, defaultGrants);

      const room = roomManager.getRoom('new-room');
      expect(room).toBeDefined();
      expect(room?.participants.size).toBe(1);
    });

    it('should reject duplicate identity in same room', () => {
      const ws1 = createMockWs();
      const ws2 = createMockWs();

      roomManager.addParticipant('test-room', 'user-123', ws1, defaultGrants);

      expect(() => {
        roomManager.addParticipant('test-room', 'user-123', ws2, defaultGrants);
      }).toThrow('Participant with this identity already in room');
    });

    it('should mark participant as agent based on grants', () => {
      const ws = createMockWs();
      const agentGrants = { ...defaultGrants, isAgent: true, hidden: true };

      const participant = roomManager.addParticipant(
        'test-room',
        'agent-001',
        ws,
        agentGrants
      );

      expect(participant.isAgent).toBe(true);
      expect(participant.grants.hidden).toBe(true);
    });
  });

  describe('removeParticipant', () => {
    it('should remove a participant from a room', () => {
      const ws = createMockWs();

      const participant = roomManager.addParticipant(
        'test-room',
        'user-123',
        ws,
        defaultGrants
      );

      roomManager.removeParticipant('test-room', participant.id);

      const room = roomManager.getRoom('test-room');
      expect(room?.participants.has(participant.id)).toBe(false);
    });

    it('should close room when last participant leaves', () => {
      const ws = createMockWs();

      const participant = roomManager.addParticipant(
        'test-room',
        'user-123',
        ws,
        defaultGrants
      );

      roomManager.removeParticipant('test-room', participant.id);

      const room = roomManager.getRoom('test-room');
      expect(room).toBeUndefined();
    });

    it('should not throw for non-existent participant', () => {
      roomManager.getOrCreateRoom('test-room');

      expect(() => {
        roomManager.removeParticipant('test-room', 'non-existent-id');
      }).not.toThrow();
    });
  });

  describe('handleMessage', () => {
    it('should handle transcript messages', () => {
      const ws = createMockWs();
      const transcriptHandler = jest.fn();

      roomManager.on('transcript', transcriptHandler);

      const participant = roomManager.addParticipant(
        'test-room',
        'user-123',
        ws,
        defaultGrants
      );

      const message = JSON.stringify({
        type: 'transcript',
        text: 'Hello, world',
        isFinal: true,
      });

      roomManager.handleMessage('test-room', participant.id, message);

      expect(transcriptHandler).toHaveBeenCalled();
    });

    it('should handle ping messages', () => {
      const ws = createMockWs();

      const participant = roomManager.addParticipant(
        'test-room',
        'user-123',
        ws,
        defaultGrants
      );

      const message = JSON.stringify({ type: 'ping' });
      roomManager.handleMessage('test-room', participant.id, message);

      expect(ws.send).toHaveBeenCalledWith(
        expect.stringContaining('"type":"pong"')
      );
    });

    it('should reject publish from participant without permission', () => {
      const ws = createMockWs();
      const restrictedGrants = { ...defaultGrants, canPublish: false };

      const participant = roomManager.addParticipant(
        'test-room',
        'user-123',
        ws,
        restrictedGrants
      );

      const message = JSON.stringify({
        type: 'transcript',
        text: 'Test',
        isFinal: true,
      });

      roomManager.handleMessage('test-room', participant.id, message);

      expect(ws.send).toHaveBeenCalledWith(
        expect.stringContaining('PERMISSION_DENIED')
      );
    });
  });

  describe('sendDialogResponse', () => {
    it('should broadcast dialog response to room', () => {
      const ws1 = createMockWs();
      const ws2 = createMockWs();

      roomManager.addParticipant('test-room', 'user-1', ws1, defaultGrants);
      roomManager.addParticipant('test-room', 'user-2', ws2, defaultGrants);

      roomManager.sendDialogResponse('test-room', {
        speakText: 'Hello there!',
        actionObject: null,
        confidence: 0.9,
        sessionId: 'session-123',
      });

      expect(ws1.send).toHaveBeenCalledWith(
        expect.stringContaining('dialog_response')
      );
      expect(ws2.send).toHaveBeenCalledWith(
        expect.stringContaining('dialog_response')
      );
    });
  });

  describe('listRooms', () => {
    it('should list all rooms', () => {
      roomManager.getOrCreateRoom('room-1');
      roomManager.getOrCreateRoom('room-2');

      const rooms = roomManager.listRooms();

      expect(rooms).toHaveLength(2);
      expect(rooms.map((r) => r.name)).toContain('room-1');
      expect(rooms.map((r) => r.name)).toContain('room-2');
    });
  });

  describe('closeRoom', () => {
    it('should close room and disconnect participants', () => {
      const ws = createMockWs();

      roomManager.addParticipant('test-room', 'user-123', ws, defaultGrants);
      roomManager.closeRoom('test-room');

      expect(ws.close).toHaveBeenCalled();
      expect(roomManager.getRoom('test-room')).toBeUndefined();
    });
  });
});
