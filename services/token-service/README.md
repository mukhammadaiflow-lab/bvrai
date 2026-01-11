# LiveKit Token Service

JWT token generation service for LiveKit room access. Part of the Builder Engine voice AI platform.

## Overview

This service generates access tokens for:
- **Room participants**: Users joining voice/video calls
- **Server-side agents**: Media Bridge and other backend services

## API Endpoints

### Health & Readiness

```bash
# Health check
GET /health
Response: { "status": "healthy", "service": "token-service", "timestamp": "..." }

# Readiness check
GET /ready
Response: { "status": "ready", "service": "token-service", "timestamp": "..." }
```

### Token Generation

#### POST /token

Generate access token for a room participant.

**Request:**
```json
{
  "room": "room-name",
  "identity": "user-123",
  "ttl_seconds": 3600,
  "name": "Display Name",
  "metadata": "{\"role\": \"customer\"}",
  "grants": {
    "canPublish": true,
    "canSubscribe": true,
    "canPublishData": true,
    "canUpdateOwnMetadata": false,
    "hidden": false,
    "agent": false
  }
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "wsUrl": "wss://livekit.example.com",
  "expiresAt": 1704067200,
  "identity": "user-123",
  "room": "room-name"
}
```

#### POST /token/agent

Generate token for server-side agent (Media Bridge).

**Request:**
```json
{
  "room": "room-name",
  "agent_id": "bridge-001"
}
```

#### POST /token/verify

Verify an existing token (for debugging).

**Request:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

## Quick Start

### Local Development

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env
# Edit .env with your LiveKit credentials

# Run in development mode
npm run dev

# Run tests
npm test
```

### Docker

```bash
# Build image
docker build -t bvrai/token-service .

# Run container
docker run -p 3001:3001 \
  -e LIVEKIT_API_KEY=your-key \
  -e LIVEKIT_API_SECRET=your-secret \
  -e LIVEKIT_WS_URL=wss://your-livekit.com \
  bvrai/token-service
```

## Example curl Commands

```bash
# Generate participant token
curl -X POST http://localhost:3001/token \
  -H "Content-Type: application/json" \
  -d '{
    "room": "test-room",
    "identity": "user-123",
    "ttl_seconds": 3600
  }'

# Generate agent token
curl -X POST http://localhost:3001/token/agent \
  -H "Content-Type: application/json" \
  -d '{
    "room": "test-room",
    "agent_id": "bridge-001"
  }'

# Verify token
curl -X POST http://localhost:3001/token/verify \
  -H "Content-Type: application/json" \
  -d '{"token": "eyJ..."}'
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `3001` |
| `NODE_ENV` | Environment (development/production/test) | `development` |
| `LIVEKIT_API_KEY` | LiveKit API key | Required |
| `LIVEKIT_API_SECRET` | LiveKit API secret | Required |
| `LIVEKIT_WS_URL` | LiveKit WebSocket URL | Required |
| `DEFAULT_TOKEN_TTL_SECONDS` | Default token TTL | `3600` |
| `MAX_TOKEN_TTL_SECONDS` | Maximum token TTL | `86400` |
| `RATE_LIMIT_WINDOW_MS` | Rate limit window | `60000` |
| `RATE_LIMIT_MAX_REQUESTS` | Max requests per window | `100` |
| `CORS_ORIGINS` | Allowed CORS origins | `*` |

## Security Considerations

- All secrets are read from environment variables
- Rate limiting is enabled by default
- CORS can be configured to restrict origins
- Tokens use HS256 signing algorithm
- Sensitive fields are redacted from logs

## License

MIT
