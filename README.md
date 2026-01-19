# Builder Engine

Backend scaffold for an AI voice-agent SaaS platform. Self-hosted WebSocket-based media plane with RAG-powered dialog management.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Builder Engine                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │    Token     │    │    Media     │    │      Dialog          │  │
│  │   Service    │───▶│    Bridge    │───▶│      Manager         │  │
│  │  (JWT Gen)   │    │  (WS Media)  │    │  (RAG + LLM)         │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      │                │
│         │                   │                      │                │
│         ▼                   ▼                      ▼                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     Ingestion Service                         │  │
│  │              (Scraping, Chunking, Persona Gen)                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Vector Database                             │  │
│  │              (Local SQLite / Pinecone)                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Token Service | 3001 | JWT token generation for media plane auth |
| Media Bridge | 3002 | WebSocket-based media plane (rooms/participants) |
| Dialog Manager | 3003 | RAG orchestration and LLM response generation |
| Ingestion Service | 3004 | Content scraping and persona generation |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### Run with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd bvrai

# Start all services
docker-compose up --build

# Services will be available at:
# - Token Service:     http://localhost:3001
# - Media Bridge:      http://localhost:3002
# - Dialog Manager:    http://localhost:3003
# - Ingestion Service: http://localhost:3004
```

### Run Integration Test

```bash
# Install test dependencies
pip install httpx websockets

# Run integration test (with services running)
python integration/simulate_call.py
```

## API Contracts

### Token Service

**POST /token** - Generate access token

```bash
curl -X POST http://localhost:3001/token \
  -H "Content-Type: application/json" \
  -d '{
    "room": "test-room",
    "identity": "user-123",
    "ttl_seconds": 3600
  }'
```

Response:
```json
{
  "token": "eyJ...",
  "wsUrl": "ws://localhost:3002",
  "expiresAt": 1704067200,
  "identity": "user-123",
  "room": "test-room"
}
```

### Media Bridge

**WebSocket /media/{room}?token=JWT** - Connect to room

```javascript
const ws = new WebSocket(`ws://localhost:3002/media/test-room?token=${token}`);

// Send transcript
ws.send(JSON.stringify({
  type: "transcript",
  text: "Hello, I need help",
  isFinal: true
}));

// Receive dialog response
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "dialog_response") {
    console.log("AI:", data.speakText);
  }
};
```

**POST /bridge/start** - Start server-side agent

```bash
curl -X POST http://localhost:3002/bridge/start \
  -H "Content-Type: application/json" \
  -d '{
    "room": "test-room",
    "agent_id": "agent-001"
  }'
```

### Dialog Manager

**POST /dialog/turn** - Process dialog turn

```bash
curl -X POST http://localhost:3003/dialog/turn \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant-123",
    "session_id": "session-abc",
    "transcript": "I want to book an appointment",
    "is_final": true
  }'
```

Response:
```json
{
  "speak_text": "I'd be happy to help you book an appointment...",
  "action_object": {
    "action_type": "initiate_booking",
    "parameters": {"intent": "booking"},
    "confidence": 0.92
  },
  "confidence": 0.85,
  "session_id": "session-abc",
  "context_used": ["doc-1", "doc-2"]
}
```

### Ingestion Service

**POST /ingest** - Ingest content

```bash
curl -X POST http://localhost:3004/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant-123",
    "urls": ["https://example.com/about"]
  }'
```

Response:
```json
{
  "status": "accepted",
  "tasks": [{"task_id": "uuid", "status": "pending"}]
}
```

## Environment Variables

### Token Service
| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET` | Secret for signing tokens (min 32 chars) | Required |
| `MEDIA_PLANE_WS_URL` | Media plane WebSocket URL | `ws://localhost:3002` |

### Media Bridge
| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET` | Must match Token Service | Required |
| `DIALOG_MANAGER_URL` | Dialog Manager URL | `http://localhost:3003` |

### Dialog Manager
| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (mock/openai/anthropic) | `mock` |
| `VECTOR_DB_PROVIDER` | Vector DB (local/pinecone) | `local` |
| `OPENAI_API_KEY` | OpenAI API key (if using) | - |
| `ANTHROPIC_API_KEY` | Anthropic API key (if using) | - |

See `.env.example` in each service directory for complete list.

## Local Development

### Token Service
```bash
cd services/token-service
npm install
cp .env.example .env
npm run dev
npm test
```

### Media Bridge
```bash
cd services/media-bridge
npm install
cp .env.example .env
npm run dev
npm test
```

### Dialog Manager
```bash
cd services/dialog-manager
pip install -e ".[dev]"
cp .env.example .env
python -m app.main
pytest
```

### Ingestion Service
```bash
cd services/ingestion-service
pip install -e ".[dev]"
cp .env.example .env
python -m app.main
```

## Testing

### Unit Tests
```bash
# Node services
cd services/token-service && npm test
cd services/media-bridge && npm test

# Python services
cd services/dialog-manager && pytest
cd services/ingestion-service && pytest
```

### Integration Test
```bash
# Start all services
docker-compose up -d

# Run integration test
python integration/simulate_call.py
```

## TODO: Production Integration Points

Marked with `TODO` comments in code:

1. **LLM Integration**
   - OpenAI GPT-4/ChatGPT
   - Anthropic Claude
   - See `dialog-manager/app/adapters/llm_adapter.py`

2. **Vector Database**
   - Pinecone
   - See `dialog-manager/app/adapters/vector_adapter.py`

3. **ASR/TTS Integration**
   - Streaming ASR (Google, Azure, Deepgram)
   - TTS (ElevenLabs, Google, Azure)
   - See `media-bridge/src/media-server.ts`

4. **Security**
   - KMS integration for key management
   - TLS termination
   - See `token-service/src/token-generator.ts`

5. **Monitoring**
   - Prometheus metrics
   - Distributed tracing

## Project Structure

```
bvrai/
├── services/
│   ├── token-service/       # JWT token generation
│   │   ├── src/
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── package.json
│   ├── media-bridge/        # WebSocket media plane
│   │   ├── src/
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── package.json
│   ├── dialog-manager/      # RAG + LLM orchestration
│   │   ├── app/
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   └── ingestion-service/   # Content ingestion
│       ├── app/
│       ├── tests/
│       ├── Dockerfile
│       └── pyproject.toml
├── integration/             # Integration tests
│   └── simulate_call.py
├── infra/
│   └── k8s/                # Kubernetes manifests
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI
├── docker-compose.yml
├── LICENSE
├── COMPATIBILITY_AND_LICENSES.md
└── README.md
```

## License

MIT License - See [LICENSE](./LICENSE)

## Compatibility

This project uses only permissively licensed dependencies. No GPL code is included. See [COMPATIBILITY_AND_LICENSES.md](./COMPATIBILITY_AND_LICENSES.md).
