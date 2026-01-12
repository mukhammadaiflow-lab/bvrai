package twilio

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/bvrai/telephony-gateway/internal/config"
	"github.com/bvrai/telephony-gateway/internal/session"
	"github.com/bvrai/telephony-gateway/pkg/audio"
	"github.com/go-chi/chi/v5"
	"github.com/gorilla/websocket"
	"github.com/rs/zerolog"
)

// MediaHandler handles Twilio media stream WebSocket connections
type MediaHandler struct {
	cfg            *config.Config
	sessionManager *session.Manager
	logger         zerolog.Logger
	upgrader       websocket.Upgrader

	// Audio processor callback - sends audio to media pipeline
	onAudioReceived func(sessionID string, audio []byte, timestamp int64)
	// Called when session needs audio output
	getAudioOutput func(sessionID string) <-chan []byte

	mu          sync.RWMutex
	connections map[string]*MediaConnection
}

// MediaConnection represents an active media stream connection
type MediaConnection struct {
	SessionID   string
	StreamSID   string
	Conn        *websocket.Conn
	Session     *session.Session
	StartTime   time.Time
	SequenceNum int

	// Audio buffers
	inboundSeq  int
	outboundSeq int

	ctx    context.Context
	cancel context.CancelFunc
	logger zerolog.Logger
}

// NewMediaHandler creates a new media handler
func NewMediaHandler(cfg *config.Config, sm *session.Manager, logger zerolog.Logger) *MediaHandler {
	return &MediaHandler{
		cfg:            cfg,
		sessionManager: sm,
		logger:         logger.With().Str("component", "twilio_media").Logger(),
		upgrader: websocket.Upgrader{
			ReadBufferSize:  cfg.WSReadBufferSize,
			WriteBufferSize: cfg.WSWriteBufferSize,
			CheckOrigin: func(r *http.Request) bool {
				return true // Twilio's origin
			},
		},
		connections: make(map[string]*MediaConnection),
	}
}

// RegisterRoutes registers media stream routes
func (h *MediaHandler) RegisterRoutes(r chi.Router) {
	r.HandleFunc("/stream/{sessionID}", h.HandleMediaStream)
}

// SetAudioCallback sets the callback for received audio
func (h *MediaHandler) SetAudioCallback(fn func(sessionID string, audio []byte, timestamp int64)) {
	h.onAudioReceived = fn
}

// SetAudioOutputProvider sets the function to get audio output channel
func (h *MediaHandler) SetAudioOutputProvider(fn func(sessionID string) <-chan []byte) {
	h.getAudioOutput = fn
}

// Twilio Media Stream message types
type MediaMessage struct {
	Event          string          `json:"event"`
	SequenceNumber string          `json:"sequenceNumber,omitempty"`
	StreamSID      string          `json:"streamSid,omitempty"`
	Start          *StartMessage   `json:"start,omitempty"`
	Media          *MediaPayload   `json:"media,omitempty"`
	Stop           *StopMessage    `json:"stop,omitempty"`
	Mark           *MarkMessage    `json:"mark,omitempty"`
	DTMF           *DTMFMessage    `json:"dtmf,omitempty"`
}

type StartMessage struct {
	AccountSID    string            `json:"accountSid"`
	StreamSID     string            `json:"streamSid"`
	CallSID       string            `json:"callSid"`
	MediaFormat   MediaFormat       `json:"mediaFormat"`
	CustomParams  map[string]string `json:"customParameters"`
	Tracks        []string          `json:"tracks"`
}

type MediaFormat struct {
	Encoding   string `json:"encoding"`   // audio/x-mulaw
	SampleRate int    `json:"sampleRate"` // 8000
	Channels   int    `json:"channels"`   // 1
}

type MediaPayload struct {
	Track     string `json:"track"`     // inbound or outbound
	Chunk     string `json:"chunk"`     // sequence number
	Timestamp string `json:"timestamp"` // milliseconds
	Payload   string `json:"payload"`   // base64 encoded audio
}

type StopMessage struct {
	AccountSID string `json:"accountSid"`
	CallSID    string `json:"callSid"`
}

type MarkMessage struct {
	Name string `json:"name"`
}

type DTMFMessage struct {
	Track string `json:"track"`
	Digit string `json:"digit"`
}

// Outbound message types
type OutboundMediaMessage struct {
	Event     string                `json:"event"`
	StreamSID string                `json:"streamSid"`
	Media     *OutboundMediaPayload `json:"media,omitempty"`
	Mark      *OutboundMark         `json:"mark,omitempty"`
	Clear     *ClearMessage         `json:"clear,omitempty"`
}

type OutboundMediaPayload struct {
	Payload string `json:"payload"` // base64 encoded mulaw audio
}

type OutboundMark struct {
	Name string `json:"name"`
}

type ClearMessage struct {
	// Empty - just clears the audio buffer
}

// HandleMediaStream handles WebSocket connections for media streaming
func (h *MediaHandler) HandleMediaStream(w http.ResponseWriter, r *http.Request) {
	sessionID := chi.URLParam(r, "sessionID")
	if sessionID == "" {
		h.logger.Error().Msg("Missing session ID in media stream request")
		http.Error(w, "Missing session ID", http.StatusBadRequest)
		return
	}

	// Get session
	sess, found := h.sessionManager.GetSession(sessionID)
	if !found {
		h.logger.Error().Str("session_id", sessionID).Msg("Session not found")
		http.Error(w, "Session not found", http.StatusNotFound)
		return
	}

	// Upgrade to WebSocket
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.Error().Err(err).Msg("Failed to upgrade WebSocket")
		return
	}

	ctx, cancel := context.WithCancel(sess.Context())

	mc := &MediaConnection{
		SessionID: sessionID,
		Conn:      conn,
		Session:   sess,
		StartTime: time.Now(),
		ctx:       ctx,
		cancel:    cancel,
		logger:    h.logger.With().Str("session_id", sessionID).Logger(),
	}

	h.mu.Lock()
	h.connections[sessionID] = mc
	h.mu.Unlock()

	mc.logger.Info().Msg("Media stream WebSocket connected")

	// Handle connection
	go h.handleConnection(mc)
}

// handleConnection manages a single media stream connection
func (h *MediaHandler) handleConnection(mc *MediaConnection) {
	defer func() {
		mc.Conn.Close()
		mc.cancel()

		h.mu.Lock()
		delete(h.connections, mc.SessionID)
		h.mu.Unlock()

		mc.logger.Info().
			Dur("duration", time.Since(mc.StartTime)).
			Msg("Media stream WebSocket closed")
	}()

	// Start writer goroutine for sending audio back
	go h.writeLoop(mc)

	// Read loop for incoming audio
	h.readLoop(mc)
}

// readLoop reads messages from Twilio
func (h *MediaHandler) readLoop(mc *MediaConnection) {
	for {
		select {
		case <-mc.ctx.Done():
			return
		default:
		}

		_, message, err := mc.Conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				mc.logger.Error().Err(err).Msg("WebSocket read error")
			}
			return
		}

		var msg MediaMessage
		if err := json.Unmarshal(message, &msg); err != nil {
			mc.logger.Error().Err(err).Msg("Failed to parse media message")
			continue
		}

		h.handleMessage(mc, &msg)
	}
}

// handleMessage processes incoming media messages
func (h *MediaHandler) handleMessage(mc *MediaConnection, msg *MediaMessage) {
	switch msg.Event {
	case "connected":
		mc.logger.Debug().Msg("Media stream connected event")

	case "start":
		h.handleStart(mc, msg)

	case "media":
		h.handleMedia(mc, msg)

	case "stop":
		h.handleStop(mc, msg)

	case "mark":
		mc.logger.Debug().
			Str("mark", msg.Mark.Name).
			Msg("Mark received")

	case "dtmf":
		h.handleDTMF(mc, msg)

	default:
		mc.logger.Warn().Str("event", msg.Event).Msg("Unknown media event")
	}
}

// handleStart processes stream start message
func (h *MediaHandler) handleStart(mc *MediaConnection, msg *MediaMessage) {
	if msg.Start == nil {
		return
	}

	mc.StreamSID = msg.Start.StreamSID
	mc.Session.SetStreamSID(mc.StreamSID)
	mc.Session.SetState(session.CallStateMediaReady, "media stream started")

	// Register stream SID with session manager
	h.sessionManager.SetStreamSID(mc.SessionID, mc.StreamSID)

	mc.logger.Info().
		Str("stream_sid", mc.StreamSID).
		Str("call_sid", msg.Start.CallSID).
		Str("encoding", msg.Start.MediaFormat.Encoding).
		Int("sample_rate", msg.Start.MediaFormat.SampleRate).
		Strs("tracks", msg.Start.Tracks).
		Msg("Media stream started")
}

// handleMedia processes incoming audio
func (h *MediaHandler) handleMedia(mc *MediaConnection, msg *MediaMessage) {
	if msg.Media == nil {
		return
	}

	// Only process inbound audio (from caller)
	if msg.Media.Track != "inbound" {
		return
	}

	// Decode base64 audio
	audioData, err := base64.StdEncoding.DecodeString(msg.Media.Payload)
	if err != nil {
		mc.logger.Error().Err(err).Msg("Failed to decode audio payload")
		return
	}

	mc.inboundSeq++

	// Forward to session
	mc.Session.ReceiveAudio(audioData)

	// Call audio callback if set
	if h.onAudioReceived != nil {
		// Parse timestamp
		var timestamp int64
		fmt.Sscanf(msg.Media.Timestamp, "%d", &timestamp)
		h.onAudioReceived(mc.SessionID, audioData, timestamp)
	}
}

// handleStop processes stream stop message
func (h *MediaHandler) handleStop(mc *MediaConnection, msg *MediaMessage) {
	mc.logger.Info().Msg("Media stream stopped")
	mc.Session.End("media stream stopped")
}

// handleDTMF processes DTMF tones
func (h *MediaHandler) handleDTMF(mc *MediaConnection, msg *MediaMessage) {
	if msg.DTMF == nil {
		return
	}

	mc.logger.Info().
		Str("digit", msg.DTMF.Digit).
		Str("track", msg.DTMF.Track).
		Msg("DTMF received")

	// TODO: Forward DTMF to conversation engine
}

// writeLoop sends audio back to Twilio
func (h *MediaHandler) writeLoop(mc *MediaConnection) {
	// Get audio output channel
	var audioChan <-chan []byte
	if h.getAudioOutput != nil {
		audioChan = h.getAudioOutput(mc.SessionID)
	}

	// Also listen to session's AudioOut channel
	sessionAudio := mc.Session.AudioOut

	for {
		select {
		case <-mc.ctx.Done():
			return

		case audioData := <-sessionAudio:
			if err := h.sendAudio(mc, audioData); err != nil {
				mc.logger.Error().Err(err).Msg("Failed to send audio")
				return
			}

		case audioData, ok := <-audioChan:
			if !ok {
				continue
			}
			if err := h.sendAudio(mc, audioData); err != nil {
				mc.logger.Error().Err(err).Msg("Failed to send audio")
				return
			}
		}
	}
}

// sendAudio sends audio data to Twilio
func (h *MediaHandler) sendAudio(mc *MediaConnection, audioData []byte) error {
	if mc.StreamSID == "" {
		return fmt.Errorf("stream not started")
	}

	// Audio should already be in mulaw format, but convert if needed
	// Twilio expects 8kHz mulaw audio
	mulawData := audio.EnsureMulaw(audioData)

	// Encode to base64
	payload := base64.StdEncoding.EncodeToString(mulawData)

	msg := OutboundMediaMessage{
		Event:     "media",
		StreamSID: mc.StreamSID,
		Media: &OutboundMediaPayload{
			Payload: payload,
		},
	}

	mc.outboundSeq++

	return mc.Conn.WriteJSON(msg)
}

// SendMark sends a mark message for synchronization
func (h *MediaHandler) SendMark(sessionID, markName string) error {
	h.mu.RLock()
	mc, found := h.connections[sessionID]
	h.mu.RUnlock()

	if !found {
		return fmt.Errorf("connection not found for session: %s", sessionID)
	}

	msg := OutboundMediaMessage{
		Event:     "mark",
		StreamSID: mc.StreamSID,
		Mark: &OutboundMark{
			Name: markName,
		},
	}

	return mc.Conn.WriteJSON(msg)
}

// ClearAudioBuffer clears the pending audio buffer (for interruption)
func (h *MediaHandler) ClearAudioBuffer(sessionID string) error {
	h.mu.RLock()
	mc, found := h.connections[sessionID]
	h.mu.RUnlock()

	if !found {
		return fmt.Errorf("connection not found for session: %s", sessionID)
	}

	msg := OutboundMediaMessage{
		Event:     "clear",
		StreamSID: mc.StreamSID,
	}

	mc.logger.Debug().Msg("Clearing audio buffer")

	return mc.Conn.WriteJSON(msg)
}

// GetConnection returns a media connection by session ID
func (h *MediaHandler) GetConnection(sessionID string) (*MediaConnection, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	mc, found := h.connections[sessionID]
	return mc, found
}

// ActiveConnections returns the number of active connections
func (h *MediaHandler) ActiveConnections() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.connections)
}
