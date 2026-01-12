package session

import (
	"context"
	"sync"
	"time"

	"github.com/google/uuid"
)

// CallDirection indicates whether the call is inbound or outbound
type CallDirection string

const (
	CallDirectionInbound  CallDirection = "inbound"
	CallDirectionOutbound CallDirection = "outbound"
)

// CallState represents the current state of the call
type CallState string

const (
	CallStateInitiated  CallState = "initiated"   // Call started, not yet connected
	CallStateRinging    CallState = "ringing"     // Ringing on the other end
	CallStateConnected  CallState = "connected"   // Both parties connected
	CallStateMediaReady CallState = "media_ready" // Media stream established
	CallStateActive     CallState = "active"      // Conversation in progress
	CallStateHold       CallState = "hold"        // Call on hold
	CallStateEnding     CallState = "ending"      // Call is being terminated
	CallStateCompleted  CallState = "completed"   // Call finished
	CallStateFailed     CallState = "failed"      // Call failed
)

// Session represents an active call session
type Session struct {
	mu sync.RWMutex

	// Identifiers
	ID        string `json:"id"`         // Our internal session ID
	CallSID   string `json:"call_sid"`   // Twilio Call SID
	StreamSID string `json:"stream_sid"` // Twilio Media Stream SID

	// Call info
	Direction   CallDirection `json:"direction"`
	From        string        `json:"from"`         // Caller phone number
	To          string        `json:"to"`           // Called phone number
	CallerName  string        `json:"caller_name"`  // Caller ID name if available
	AccountSID  string        `json:"account_sid"`  // Twilio account

	// Agent info
	AgentID     string            `json:"agent_id"`
	AgentConfig map[string]string `json:"agent_config"`

	// State
	State         CallState `json:"state"`
	StateHistory  []StateChange `json:"state_history"`

	// Timestamps
	CreatedAt     time.Time  `json:"created_at"`
	ConnectedAt   *time.Time `json:"connected_at,omitempty"`
	MediaReadyAt  *time.Time `json:"media_ready_at,omitempty"`
	EndedAt       *time.Time `json:"ended_at,omitempty"`

	// Audio stats
	AudioPacketsIn  int64 `json:"audio_packets_in"`
	AudioPacketsOut int64 `json:"audio_packets_out"`
	AudioBytesIn    int64 `json:"audio_bytes_in"`
	AudioBytesOut   int64 `json:"audio_bytes_out"`

	// Media stream channels
	AudioIn  chan []byte `json:"-"` // Incoming audio from caller
	AudioOut chan []byte `json:"-"` // Outgoing audio to caller

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}

	// Callbacks
	onStateChange func(s *Session, from, to CallState)
	onAudioIn     func(s *Session, audio []byte)
	onEnded       func(s *Session)
}

// StateChange records a state transition
type StateChange struct {
	From      CallState `json:"from"`
	To        CallState `json:"to"`
	Timestamp time.Time `json:"timestamp"`
	Reason    string    `json:"reason,omitempty"`
}

// NewSession creates a new call session
func NewSession(callSID, from, to string, direction CallDirection) *Session {
	ctx, cancel := context.WithCancel(context.Background())

	s := &Session{
		ID:          uuid.New().String(),
		CallSID:     callSID,
		Direction:   direction,
		From:        from,
		To:          to,
		State:       CallStateInitiated,
		CreatedAt:   time.Now(),
		AudioIn:     make(chan []byte, 100),  // Buffer ~2 seconds of audio at 50 packets/sec
		AudioOut:    make(chan []byte, 100),
		ctx:         ctx,
		cancel:      cancel,
		done:        make(chan struct{}),
		AgentConfig: make(map[string]string),
	}

	s.StateHistory = append(s.StateHistory, StateChange{
		From:      "",
		To:        CallStateInitiated,
		Timestamp: s.CreatedAt,
	})

	return s
}

// SetState transitions the session to a new state
func (s *Session) SetState(newState CallState, reason string) {
	s.mu.Lock()
	oldState := s.State
	s.State = newState

	change := StateChange{
		From:      oldState,
		To:        newState,
		Timestamp: time.Now(),
		Reason:    reason,
	}
	s.StateHistory = append(s.StateHistory, change)

	// Update timestamps
	now := time.Now()
	switch newState {
	case CallStateConnected:
		s.ConnectedAt = &now
	case CallStateMediaReady:
		s.MediaReadyAt = &now
	case CallStateCompleted, CallStateFailed:
		s.EndedAt = &now
	}

	callback := s.onStateChange
	s.mu.Unlock()

	if callback != nil {
		callback(s, oldState, newState)
	}
}

// GetState returns the current call state
func (s *Session) GetState() CallState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.State
}

// SetStreamSID sets the Twilio media stream SID
func (s *Session) SetStreamSID(streamSID string) {
	s.mu.Lock()
	s.StreamSID = streamSID
	s.mu.Unlock()
}

// SetAgentID sets the agent handling this call
func (s *Session) SetAgentID(agentID string) {
	s.mu.Lock()
	s.AgentID = agentID
	s.mu.Unlock()
}

// SetAgentConfig sets agent configuration
func (s *Session) SetAgentConfig(key, value string) {
	s.mu.Lock()
	s.AgentConfig[key] = value
	s.mu.Unlock()
}

// GetAgentConfig gets agent configuration
func (s *Session) GetAgentConfig(key string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.AgentConfig[key]
}

// OnStateChange sets a callback for state changes
func (s *Session) OnStateChange(fn func(s *Session, from, to CallState)) {
	s.mu.Lock()
	s.onStateChange = fn
	s.mu.Unlock()
}

// OnAudioIn sets a callback for incoming audio
func (s *Session) OnAudioIn(fn func(s *Session, audio []byte)) {
	s.mu.Lock()
	s.onAudioIn = fn
	s.mu.Unlock()
}

// OnEnded sets a callback for when the session ends
func (s *Session) OnEnded(fn func(s *Session)) {
	s.mu.Lock()
	s.onEnded = fn
	s.mu.Unlock()
}

// SendAudio sends audio to the caller
func (s *Session) SendAudio(audio []byte) bool {
	select {
	case s.AudioOut <- audio:
		s.mu.Lock()
		s.AudioPacketsOut++
		s.AudioBytesOut += int64(len(audio))
		s.mu.Unlock()
		return true
	case <-s.ctx.Done():
		return false
	default:
		// Channel full, drop packet
		return false
	}
}

// ReceiveAudio processes incoming audio from the caller
func (s *Session) ReceiveAudio(audio []byte) {
	s.mu.Lock()
	s.AudioPacketsIn++
	s.AudioBytesIn += int64(len(audio))
	callback := s.onAudioIn
	s.mu.Unlock()

	select {
	case s.AudioIn <- audio:
	case <-s.ctx.Done():
		return
	default:
		// Channel full, drop packet
	}

	if callback != nil {
		callback(s, audio)
	}
}

// Context returns the session context
func (s *Session) Context() context.Context {
	return s.ctx
}

// Done returns a channel that's closed when the session ends
func (s *Session) Done() <-chan struct{} {
	return s.done
}

// End terminates the session
func (s *Session) End(reason string) {
	s.mu.Lock()
	if s.State == CallStateCompleted || s.State == CallStateFailed {
		s.mu.Unlock()
		return
	}
	callback := s.onEnded
	s.mu.Unlock()

	s.SetState(CallStateCompleted, reason)
	s.cancel()
	close(s.done)

	if callback != nil {
		callback(s)
	}
}

// Fail marks the session as failed
func (s *Session) Fail(reason string) {
	s.mu.Lock()
	if s.State == CallStateCompleted || s.State == CallStateFailed {
		s.mu.Unlock()
		return
	}
	callback := s.onEnded
	s.mu.Unlock()

	s.SetState(CallStateFailed, reason)
	s.cancel()
	close(s.done)

	if callback != nil {
		callback(s)
	}
}

// Duration returns the call duration
func (s *Session) Duration() time.Duration {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.ConnectedAt == nil {
		return 0
	}

	end := time.Now()
	if s.EndedAt != nil {
		end = *s.EndedAt
	}

	return end.Sub(*s.ConnectedAt)
}

// Stats returns session statistics
func (s *Session) Stats() SessionStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return SessionStats{
		SessionID:       s.ID,
		CallSID:         s.CallSID,
		State:           s.State,
		Duration:        s.Duration(),
		AudioPacketsIn:  s.AudioPacketsIn,
		AudioPacketsOut: s.AudioPacketsOut,
		AudioBytesIn:    s.AudioBytesIn,
		AudioBytesOut:   s.AudioBytesOut,
	}
}

// SessionStats holds session statistics
type SessionStats struct {
	SessionID       string        `json:"session_id"`
	CallSID         string        `json:"call_sid"`
	State           CallState     `json:"state"`
	Duration        time.Duration `json:"duration"`
	AudioPacketsIn  int64         `json:"audio_packets_in"`
	AudioPacketsOut int64         `json:"audio_packets_out"`
	AudioBytesIn    int64         `json:"audio_bytes_in"`
	AudioBytesOut   int64         `json:"audio_bytes_out"`
}
