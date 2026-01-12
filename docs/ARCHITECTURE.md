# Builder Engine - Architecture Document

## Overview

Builder Engine is a self-hosted AI voice agent platform that enables businesses to deploy conversational AI agents for phone calls, web calls, and voice interfaces.

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL INTERFACES                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                    │
│   │    Phone     │    │   WebRTC     │    │   WebSocket  │                    │
│   │  (PSTN/SIP)  │    │   Browser    │    │    Client    │                    │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                    │
│          │                   │                   │                            │
└──────────┼───────────────────┼───────────────────┼────────────────────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                           TELEPHONY GATEWAY                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Twilio / Telnyx Adapter                          │   │
│  │  • Webhook handlers for incoming calls                                   │   │
│  │  • Media stream WebSocket connection                                     │   │
│  │  • Call control (answer, hangup, transfer)                              │   │
│  │  • DTMF handling                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         WebRTC Gateway                                   │   │
│  │  • STUN/TURN coordination                                               │   │
│  │  • Peer connection management                                           │   │
│  │  • Browser audio capture                                                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                              MEDIA PIPELINE                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Audio Router                                     │   │
│  │  • Demux incoming audio streams                                         │   │
│  │  • Route to ASR engine                                                  │   │
│  │  • Receive from TTS engine                                              │   │
│  │  • Mux outgoing audio streams                                           │   │
│  │  • Handle codec conversion (mulaw ↔ PCM ↔ Opus)                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌──────────────────────┐              ┌──────────────────────┐               │
│  │     Audio Buffer     │              │   Recording Store    │               │
│  │  • Jitter buffer     │              │  • Full call audio   │               │
│  │  • Resampling        │              │  • Chunked storage   │               │
│  │  • VAD preprocessing │              │  • S3/Local FS       │               │
│  └──────────────────────┘              └──────────────────────┘               │
└────────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌─────────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│       ASR ENGINE        │ │ CONVERSATION    │ │       TTS ENGINE        │
│  ┌───────────────────┐  │ │    ENGINE       │ │  ┌───────────────────┐  │
│  │ Deepgram Adapter  │  │ │                 │ │  │ ElevenLabs Adapter│  │
│  │ • Streaming ASR   │  │ │ See detail below│ │  │ • Streaming TTS   │  │
│  │ • Interim results │  │ │                 │ │  │ • Voice selection │  │
│  │ • Final results   │  │ │                 │ │  │ • SSML support    │  │
│  └───────────────────┘  │ │                 │ │  └───────────────────┘  │
│  ┌───────────────────┐  │ │                 │ │  ┌───────────────────┐  │
│  │ Whisper Adapter   │  │ │                 │ │  │ PlayHT Adapter    │  │
│  │ • Self-hosted     │  │ │                 │ │  │ • Alternative     │  │
│  │ • Batch/streaming │  │ │                 │ │  └───────────────────┘  │
│  └───────────────────┘  │ │                 │ │  ┌───────────────────┐  │
│  ┌───────────────────┐  │ │                 │ │  │ Piper Adapter     │  │
│  │ VAD (Silero)      │  │ │                 │ │  │ • Self-hosted     │  │
│  │ • Voice activity  │  │ │                 │ │  │ • Low latency     │  │
│  │ • Endpointing     │  │ │                 │ │  └───────────────────┘  │
│  └───────────────────┘  │ │                 │ │                         │
└─────────────────────────┘ └─────────────────┘ └─────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                           CONVERSATION ENGINE                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Turn Manager                                     │   │
│  │  • Detect user turn completion (silence + VAD + semantic)               │   │
│  │  • Handle barge-in (user interrupts agent)                              │   │
│  │  • Manage turn queue                                                    │   │
│  │  • Backpressure control                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         State Machine                                    │   │
│  │  States: IDLE → LISTENING → PROCESSING → SPEAKING → LISTENING          │   │
│  │                    ↑                         │                          │   │
│  │                    └─────────────────────────┘                          │   │
│  │  • IDLE: Waiting for call                                               │   │
│  │  • LISTENING: Receiving user audio, streaming to ASR                    │   │
│  │  • PROCESSING: User done speaking, waiting for LLM                      │   │
│  │  • SPEAKING: Playing TTS audio to user                                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Interrupt Handler                                │   │
│  │  • Detect user speech during SPEAKING state                             │   │
│  │  • Stop TTS playback immediately                                        │   │
│  │  • Transition to LISTENING                                              │   │
│  │  • Preserve partial context                                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Context Manager                                  │   │
│  │  • Conversation history                                                 │   │
│  │  • Extracted entities/slots                                             │   │
│  │  • Function call results                                                │   │
│  │  • Session metadata                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                              AI LAYER                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         LLM Orchestrator                                 │   │
│  │  • Multi-provider support (OpenAI, Anthropic, Groq, local)             │   │
│  │  • Streaming response handling                                          │   │
│  │  • Function/tool calling                                                │   │
│  │  • Retry with fallback                                                  │   │
│  │  • Token counting and limits                                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         RAG Pipeline                                     │   │
│  │  • Query embedding                                                      │   │
│  │  • Vector similarity search                                             │   │
│  │  • Reranking                                                            │   │
│  │  • Context injection                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Prompt Engine                                    │   │
│  │  • System prompt templates                                              │   │
│  │  • Persona injection                                                    │   │
│  │  • Few-shot examples                                                    │   │
│  │  • Dynamic prompt assembly                                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Function Registry                                │   │
│  │  • Built-in functions (transfer, hangup, hold, book_appointment)       │   │
│  │  • Webhook functions (call external APIs)                               │   │
│  │  • Function schema validation                                           │   │
│  │  • Async execution                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                           PLATFORM SERVICES                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │   Agent API    │  │   Call API     │  │  Analytics     │  │   Webhooks   │  │
│  │  • CRUD agents │  │  • Call logs   │  │  • Metrics     │  │  • Events    │  │
│  │  • Versioning  │  │  • Recordings  │  │  • Dashboards  │  │  • Callbacks │  │
│  │  • Deployment  │  │  • Transcripts │  │  • Alerts      │  │  • Retries   │  │
│  └────────────────┘  └────────────────┘  └────────────────┘  └──────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘

## Data Flow: Incoming Call

```
1. Phone call comes in via Twilio
   └─▶ Twilio sends webhook to /telephony/incoming

2. Telephony Gateway answers call
   └─▶ Creates call session
   └─▶ Opens media stream WebSocket with Twilio
   └─▶ Loads agent configuration

3. Audio flows in (mulaw 8kHz from Twilio)
   └─▶ Audio Router converts to PCM 16kHz
   └─▶ Buffers and denoises
   └─▶ Streams to ASR Engine

4. ASR produces transcripts
   └─▶ Interim results for UI/monitoring
   └─▶ Final results to Conversation Engine

5. Conversation Engine detects turn complete
   └─▶ Gathers context (history, RAG results)
   └─▶ Sends to LLM Orchestrator

6. LLM streams response
   └─▶ Chunks sent to TTS Engine in real-time
   └─▶ Function calls executed in parallel

7. TTS produces audio
   └─▶ Audio Router converts to mulaw 8kHz
   └─▶ Streams back to Twilio
   └─▶ User hears response

8. If user interrupts (barge-in)
   └─▶ VAD detects speech during SPEAKING
   └─▶ TTS playback stopped immediately
   └─▶ Return to step 3
```

## Latency Budget

Target: **< 800ms** from user stops speaking to agent starts responding

| Component | Target | Notes |
|-----------|--------|-------|
| Turn detection | 200ms | Silence + VAD endpointing |
| ASR final | 100ms | Deepgram streaming |
| RAG retrieval | 50ms | Cached embeddings |
| LLM first token | 200ms | GPT-4o-mini or Groq |
| TTS first audio | 150ms | ElevenLabs streaming |
| Network overhead | 100ms | Internal + Twilio |
| **Total** | **800ms** | Acceptable for voice |

## Technology Stack

### Core Services (Rust/Go for performance-critical, Python for AI)

| Service | Language | Framework | Why |
|---------|----------|-----------|-----|
| Telephony Gateway | Go | net/http + gorilla/websocket | Concurrent connections, low memory |
| Media Pipeline | Go | Custom | Audio processing, low latency |
| ASR Service | Python | FastAPI + asyncio | Deepgram SDK, async streaming |
| TTS Service | Python | FastAPI + asyncio | ElevenLabs SDK, async streaming |
| Conversation Engine | Python | FastAPI + asyncio | Complex logic, LLM integration |
| Platform API | Python | FastAPI | Rapid development, good enough perf |

### Data Stores

| Store | Technology | Purpose |
|-------|------------|---------|
| Primary DB | PostgreSQL | Agents, calls, users, config |
| Vector DB | Qdrant | Knowledge base embeddings |
| Cache | Redis | Session state, rate limiting |
| Queue | Redis Streams | Async tasks, events |
| Object Store | S3/MinIO | Call recordings, documents |

### External Services (Phase 1)

| Service | Provider | Fallback |
|---------|----------|----------|
| Telephony | Twilio | Telnyx |
| ASR | Deepgram | AssemblyAI |
| TTS | ElevenLabs | PlayHT |
| LLM | OpenAI GPT-4o | Anthropic Claude |
| Embeddings | OpenAI | Cohere |

## Directory Structure

```
bvrai/
├── services/
│   ├── telephony-gateway/     # Go - Twilio/WebRTC handling
│   │   ├── cmd/
│   │   ├── internal/
│   │   │   ├── twilio/        # Twilio adapter
│   │   │   ├── webrtc/        # WebRTC adapter
│   │   │   └── session/       # Call session management
│   │   └── pkg/
│   │
│   ├── media-pipeline/        # Go - Audio routing and processing
│   │   ├── cmd/
│   │   └── internal/
│   │       ├── router/        # Audio routing
│   │       ├── buffer/        # Jitter buffer
│   │       ├── codec/         # Codec conversion
│   │       └── recorder/      # Call recording
│   │
│   ├── asr-service/           # Python - Speech-to-text
│   │   ├── app/
│   │   │   ├── adapters/      # Deepgram, Whisper, etc.
│   │   │   ├── vad/           # Voice activity detection
│   │   │   └── streaming/     # WebSocket handlers
│   │   └── tests/
│   │
│   ├── tts-service/           # Python - Text-to-speech
│   │   ├── app/
│   │   │   ├── adapters/      # ElevenLabs, PlayHT, etc.
│   │   │   ├── ssml/          # SSML processing
│   │   │   └── streaming/     # Audio streaming
│   │   └── tests/
│   │
│   ├── conversation-engine/   # Python - Turn management & state
│   │   ├── app/
│   │   │   ├── turn/          # Turn detection
│   │   │   ├── state/         # State machine
│   │   │   ├── interrupt/     # Barge-in handling
│   │   │   └── context/       # Context management
│   │   └── tests/
│   │
│   ├── ai-orchestrator/       # Python - LLM & RAG
│   │   ├── app/
│   │   │   ├── llm/           # LLM adapters
│   │   │   ├── rag/           # RAG pipeline
│   │   │   ├── prompt/        # Prompt templates
│   │   │   └── functions/     # Function calling
│   │   └── tests/
│   │
│   └── platform-api/          # Python - REST API
│       ├── app/
│       │   ├── agents/        # Agent CRUD
│       │   ├── calls/         # Call management
│       │   ├── analytics/     # Metrics & reporting
│       │   └── auth/          # Authentication
│       └── tests/
│
├── shared/
│   ├── proto/                 # gRPC/protobuf definitions
│   ├── schemas/               # JSON schemas
│   └── contracts/             # API contracts
│
├── infra/
│   ├── docker/                # Dockerfiles
│   ├── k8s/                   # Kubernetes manifests
│   └── terraform/             # Infrastructure as code
│
├── web/                       # Frontend (Next.js) - Phase 3
│
├── docs/
│   ├── ARCHITECTURE.md        # This file
│   ├── API.md                 # API documentation
│   └── DEPLOYMENT.md          # Deployment guide
│
├── docker-compose.yml
├── docker-compose.dev.yml
└── Makefile
```

## API Contracts

### Internal Service Communication

Services communicate via:
1. **gRPC** - For streaming audio and real-time data
2. **REST** - For configuration and management
3. **Redis Streams** - For async events

### Telephony Gateway → Media Pipeline

```protobuf
service MediaPipeline {
  rpc StreamAudio(stream AudioChunk) returns (stream AudioChunk);
}

message AudioChunk {
  string call_id = 1;
  bytes audio = 2;           // Raw PCM 16kHz mono
  int64 timestamp_ms = 3;
  AudioDirection direction = 4;
}
```

### Media Pipeline → ASR Service

```protobuf
service ASRService {
  rpc StreamingRecognize(stream AudioChunk) returns (stream Transcript);
}

message Transcript {
  string call_id = 1;
  string text = 2;
  bool is_final = 3;
  float confidence = 4;
  int64 start_ms = 5;
  int64 end_ms = 6;
}
```

### Conversation Engine → AI Orchestrator

```json
// POST /ai/complete
{
  "call_id": "uuid",
  "agent_id": "uuid",
  "transcript": "I'd like to book an appointment",
  "context": {
    "history": [...],
    "entities": {...},
    "rag_results": [...]
  }
}

// Response (streamed)
{
  "text": "I'd be happy to help you book an appointment...",
  "function_calls": [
    {
      "name": "check_availability",
      "arguments": {"date": "2024-01-15"}
    }
  ],
  "done": false
}
```

## Security Considerations

1. **API Authentication**: JWT tokens for all external APIs
2. **Service-to-Service**: mTLS between internal services
3. **Secrets Management**: HashiCorp Vault or AWS Secrets Manager
4. **Audio Encryption**: TLS for all audio streams
5. **PII Handling**: Redaction in logs, encryption at rest
6. **Rate Limiting**: Per-tenant limits on all endpoints

## Monitoring & Observability

1. **Metrics**: Prometheus + Grafana
   - Call volume, duration, success rate
   - Latency percentiles per component
   - ASR/TTS accuracy metrics

2. **Logging**: Structured JSON → ELK Stack
   - Correlation IDs across services
   - Call-level log aggregation

3. **Tracing**: Jaeger/Zipkin
   - End-to-end call tracing
   - Latency breakdown

4. **Alerting**: PagerDuty/Opsgenie
   - SLA violations
   - Error rate spikes
   - Infrastructure issues
