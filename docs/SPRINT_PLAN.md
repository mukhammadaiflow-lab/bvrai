# Builder Engine - 1 Month Sprint Plan

## Timeline Overview

| Week | Focus | Deliverable |
|------|-------|-------------|
| Week 1 | Telephony + Media Pipeline | Twilio integration, audio streaming working |
| Week 2 | ASR + TTS Integration | Real-time speech-to-text and text-to-speech |
| Week 3 | Conversation Engine + LLM | Turn management, interrupts, LLM streaming |
| Week 4 | RAG + Polish + Deploy | Knowledge base, testing, production deploy |

---

## Week 1: Telephony & Media Pipeline

### Day 1-2: Telephony Gateway (Go)

**Goal**: Accept incoming Twilio calls and establish media stream

- [ ] Project setup (Go modules, structure)
- [ ] Twilio webhook handler (`/incoming`)
- [ ] TwiML response generation
- [ ] Media stream WebSocket handler
- [ ] Call session management
- [ ] Basic call control (answer, hangup)

**Files**:
```
services/telephony-gateway/
├── cmd/server/main.go
├── internal/
│   ├── config/config.go
│   ├── twilio/
│   │   ├── webhook.go       # HTTP webhook handlers
│   │   ├── media.go         # WebSocket media stream
│   │   └── twiml.go         # TwiML generation
│   ├── session/
│   │   ├── manager.go       # Session lifecycle
│   │   └── session.go       # Call session struct
│   └── api/
│       └── routes.go        # HTTP routes
├── pkg/
│   └── audio/
│       └── mulaw.go         # μ-law codec
├── go.mod
├── go.sum
└── Dockerfile
```

**Test**: Make a phone call → See logs showing audio packets received

### Day 3-4: Media Pipeline (Go)

**Goal**: Route audio between services with proper buffering

- [ ] Audio router (demux/mux streams)
- [ ] Jitter buffer implementation
- [ ] Codec conversion (mulaw ↔ PCM16)
- [ ] gRPC service for audio streaming
- [ ] Call recording to file

**Files**:
```
services/media-pipeline/
├── cmd/server/main.go
├── internal/
│   ├── router/
│   │   ├── router.go        # Main audio router
│   │   └── stream.go        # Stream management
│   ├── buffer/
│   │   ├── jitter.go        # Jitter buffer
│   │   └── ring.go          # Ring buffer
│   ├── codec/
│   │   ├── mulaw.go         # μ-law encode/decode
│   │   ├── pcm.go           # PCM utilities
│   │   └── resample.go      # Sample rate conversion
│   └── recorder/
│       └── recorder.go      # Call recording
├── pkg/
│   └── proto/
│       └── media.proto      # gRPC definitions
├── go.mod
└── Dockerfile
```

**Test**: Audio flows from Twilio → Media Pipeline → recorded to file

### Day 5: Integration & Testing

- [ ] Docker Compose for Week 1 services
- [ ] End-to-end test: call → audio captured
- [ ] Latency measurement
- [ ] Bug fixes

---

## Week 2: ASR & TTS Integration

### Day 1-2: ASR Service (Python)

**Goal**: Real-time speech-to-text with Deepgram

- [ ] Deepgram WebSocket client
- [ ] Streaming audio input
- [ ] Interim + final transcript handling
- [ ] Voice Activity Detection (VAD)
- [ ] Endpointing (detect when user stops)

**Files**:
```
services/asr-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py          # ASR adapter interface
│   │   ├── deepgram.py      # Deepgram implementation
│   │   └── mock.py          # Mock for testing
│   ├── vad/
│   │   ├── __init__.py
│   │   ├── silero.py        # Silero VAD
│   │   └── energy.py        # Energy-based VAD
│   └── streaming/
│       ├── __init__.py
│       └── handler.py       # WebSocket handler
├── tests/
├── pyproject.toml
└── Dockerfile
```

**Test**: Speak into phone → See transcript in logs

### Day 3-4: TTS Service (Python)

**Goal**: Stream text-to-speech with ElevenLabs

- [ ] ElevenLabs streaming client
- [ ] Chunk text for lower latency
- [ ] Audio format conversion
- [ ] Voice selection/caching
- [ ] Interruption support (cancel mid-stream)

**Files**:
```
services/tts-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py          # TTS adapter interface
│   │   ├── elevenlabs.py    # ElevenLabs implementation
│   │   ├── playht.py        # PlayHT alternative
│   │   └── mock.py          # Mock for testing
│   └── streaming/
│       ├── __init__.py
│       └── handler.py       # Streaming handler
├── tests/
├── pyproject.toml
└── Dockerfile
```

**Test**: Send text → Hear speech on phone

### Day 5: Integration

- [ ] Connect ASR → TTS (echo bot)
- [ ] Measure end-to-end latency
- [ ] Docker Compose update
- [ ] Bug fixes

**Milestone**: Working echo bot - speak and hear your words back!

---

## Week 3: Conversation Engine & LLM

### Day 1-2: Conversation Engine (Python)

**Goal**: Turn management and state machine

- [ ] Turn detector (silence + VAD + timing)
- [ ] State machine (IDLE, LISTENING, PROCESSING, SPEAKING)
- [ ] Barge-in handler (interrupt detection)
- [ ] Context manager (history, entities)
- [ ] Session state persistence

**Files**:
```
services/conversation-engine/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI/gRPC server
│   ├── config.py
│   ├── turn/
│   │   ├── __init__.py
│   │   ├── detector.py      # Turn detection logic
│   │   └── timing.py        # Timing parameters
│   ├── state/
│   │   ├── __init__.py
│   │   ├── machine.py       # State machine
│   │   └── transitions.py   # State transitions
│   ├── interrupt/
│   │   ├── __init__.py
│   │   └── handler.py       # Barge-in handling
│   ├── context/
│   │   ├── __init__.py
│   │   ├── manager.py       # Context management
│   │   └── history.py       # Conversation history
│   └── session/
│       ├── __init__.py
│       └── store.py         # Session storage
├── tests/
├── pyproject.toml
└── Dockerfile
```

### Day 3-4: AI Orchestrator (Python)

**Goal**: LLM integration with streaming and function calling

- [ ] LLM adapter interface
- [ ] OpenAI streaming implementation
- [ ] Anthropic streaming implementation
- [ ] Function/tool calling
- [ ] Prompt template engine
- [ ] Response streaming to TTS

**Files**:
```
services/ai-orchestrator/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py          # LLM adapter interface
│   │   ├── openai.py        # OpenAI implementation
│   │   ├── anthropic.py     # Anthropic implementation
│   │   ├── groq.py          # Groq implementation
│   │   └── mock.py          # Mock for testing
│   ├── prompt/
│   │   ├── __init__.py
│   │   ├── engine.py        # Prompt assembly
│   │   └── templates/       # Prompt templates
│   ├── functions/
│   │   ├── __init__.py
│   │   ├── registry.py      # Function registry
│   │   ├── executor.py      # Function execution
│   │   └── builtin/         # Built-in functions
│   └── streaming/
│       ├── __init__.py
│       └── handler.py       # Streaming handler
├── tests/
├── pyproject.toml
└── Dockerfile
```

### Day 5: Integration

- [ ] Full conversation loop working
- [ ] Latency optimization
- [ ] Error handling
- [ ] Bug fixes

**Milestone**: Have a conversation with AI on the phone!

---

## Week 4: RAG, Platform & Polish

### Day 1-2: RAG Pipeline

**Goal**: Knowledge base integration

- [ ] Qdrant vector store setup
- [ ] Embedding generation (OpenAI)
- [ ] Document ingestion API
- [ ] Retrieval pipeline
- [ ] Context injection into prompts

**Add to ai-orchestrator**:
```
├── app/
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py    # Embedding generation
│   │   ├── retriever.py     # Vector search
│   │   ├── reranker.py      # Result reranking
│   │   └── pipeline.py      # RAG pipeline
│   └── ingestion/
│       ├── __init__.py
│       ├── chunker.py       # Text chunking
│       ├── loader.py        # Document loading
│       └── processor.py     # Processing pipeline
```

### Day 3: Platform API

**Goal**: Agent management and call logs

- [ ] Agent CRUD endpoints
- [ ] Call log storage
- [ ] Transcript storage
- [ ] Basic analytics

**Files**:
```
services/platform-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── service.py
│   ├── calls/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── service.py
│   └── database/
│       ├── __init__.py
│       ├── models.py
│       └── session.py
├── tests/
├── pyproject.toml
└── Dockerfile
```

### Day 4: Testing & Polish

- [ ] End-to-end integration tests
- [ ] Load testing
- [ ] Error handling review
- [ ] Logging & monitoring
- [ ] Documentation

### Day 5: Deployment

- [ ] Production Docker builds
- [ ] Kubernetes manifests
- [ ] CI/CD pipeline
- [ ] Staging deployment
- [ ] Production deployment

---

## Success Criteria

### Week 1 Complete ✓
- [ ] Can receive phone calls via Twilio
- [ ] Audio is captured and recorded
- [ ] < 50ms internal audio routing latency

### Week 2 Complete ✓
- [ ] Real-time speech-to-text working
- [ ] Text-to-speech playing on phone
- [ ] Echo bot demo functional
- [ ] < 500ms ASR latency

### Week 3 Complete ✓
- [ ] Full conversation with AI
- [ ] Interruption (barge-in) working
- [ ] Function calling working
- [ ] < 800ms response latency

### Week 4 Complete ✓
- [ ] Knowledge base queries working
- [ ] Multiple agents configurable
- [ ] Call logs and transcripts stored
- [ ] Production deployment ready

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Deepgram latency issues | Have AssemblyAI as backup |
| ElevenLabs rate limits | Use PlayHT or local Piper |
| Twilio costs during dev | Use ngrok + test numbers |
| Complexity creep | Strict scope per week |
| Integration bugs | Daily integration testing |

---

## Resources Needed

### API Keys (get these now)
- [ ] Twilio Account SID + Auth Token
- [ ] Twilio Phone Number
- [ ] Deepgram API Key
- [ ] ElevenLabs API Key
- [ ] OpenAI API Key
- [ ] Anthropic API Key (optional)

### Infrastructure
- [ ] PostgreSQL database
- [ ] Redis instance
- [ ] Qdrant instance (can run in Docker)
- [ ] S3/MinIO for recordings

### Development
- [ ] Go 1.21+
- [ ] Python 3.11+
- [ ] Docker & Docker Compose
- [ ] ngrok (for local Twilio testing)
