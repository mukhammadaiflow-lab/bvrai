# Media Bridge Service

Server-side LiveKit participant that forwards audio frames to streaming ASR and routes transcripts to the Dialog Manager.

## Overview

The Media Bridge acts as an intermediary between:
- **LiveKit rooms** - Receives audio from participants
- **ASR service** - Streams audio for transcription
- **Dialog Manager** - Receives transcripts for response generation

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   LiveKit    │────▶│ Media Bridge │────▶│     ASR      │
│    Room      │     │              │     │   Service    │
└──────────────┘     │              │     └──────────────┘
                     │              │            │
                     │              │◀───────────┘
                     │              │      transcripts
                     │              │
                     │              │────▶┌──────────────┐
                     └──────────────┘     │   Dialog     │
                                          │   Manager    │
                                          └──────────────┘
```

## API Endpoints

### Health & Readiness

```bash
GET /health
GET /ready
```

### Bridge Management

#### POST /bridge/start

Start a new media bridge session.

**Request:**
```json
{
  "room": "room-name",
  "agent_id": "optional-custom-agent-id",
  "tenant_id": "tenant-123",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "bridge_id": "uuid",
  "room": "room-name",
  "agent_id": "agent-xxx",
  "session_id": "uuid",
  "ws_url": "/bridge/{bridge_id}/audio"
}
```

#### GET /bridge/:bridgeId

Get bridge status.

#### POST /bridge/:bridgeId/stop

Stop a bridge session.

#### GET /bridges

List all active bridges.

### Audio Streaming

#### WebSocket /bridge/:bridgeId/audio

Connect via WebSocket for real-time audio streaming.

**Outgoing Messages:**
- Binary frames: Raw audio data (PCM16 or Opus)
- JSON `{"type": "end_stream"}`: Signal end of audio

**Incoming Messages:**
```json
{
  "type": "transcript",
  "text": "Hello, how can I help?",
  "is_final": true,
  "confidence": 0.95
}
```

```json
{
  "type": "dialog_response",
  "speak_text": "I'd be happy to help with that.",
  "action_object": {"action": "lookup", "params": {}},
  "confidence": 0.9
}
```

#### POST /bridge/:bridgeId/audio

HTTP fallback for sending audio frames.

**With JSON:**
```bash
curl -X POST http://localhost:3002/bridge/{id}/audio \
  -H "Content-Type: application/json" \
  -d '{"data": "base64-encoded-audio"}'
```

**With binary:**
```bash
curl -X POST http://localhost:3002/bridge/{id}/audio \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.raw
```

## Quick Start

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Run in development
npm run dev

# Run tests
npm test
```

## Docker

```bash
docker build -t bvrai/media-bridge .
docker run -p 3002:3002 \
  -e DIALOG_MANAGER_URL=http://dialog-manager:3003 \
  bvrai/media-bridge
```

## ASR Adapter

The service uses an adapter pattern for ASR integration. Currently includes:

- **MockASRAdapter**: For testing without a real ASR service

To add a real ASR provider, implement the `ASRAdapter` interface:

```typescript
interface ASRAdapter {
  connect(config: ASRConfig, events: ASRAdapterEvents): Promise<void>;
  sendAudioFrame(frame: Buffer): void;
  endAudioStream(): void;
  disconnect(): void;
  getState(): ASRConnectionState;
}
```

Supported providers (TODO):
- Google Cloud Speech-to-Text
- Azure Speech Services
- Deepgram
- AssemblyAI

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `3002` |
| `NODE_ENV` | Environment | `development` |
| `TOKEN_SERVICE_URL` | Token service URL | `http://localhost:3001` |
| `DIALOG_MANAGER_URL` | Dialog Manager URL | `http://localhost:3003` |
| `ASR_WS_URL` | ASR WebSocket URL | `ws://localhost:8080/asr` |
| `ASR_SAMPLE_RATE` | Audio sample rate | `16000` |
| `ASR_ENCODING` | Audio encoding | `linear16` |
| `MAX_BRIDGES` | Max concurrent bridges | `100` |
| `BRIDGE_TIMEOUT_MS` | Bridge idle timeout | `300000` |

## License

MIT
