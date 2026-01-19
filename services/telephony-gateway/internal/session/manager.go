package session

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/rs/zerolog"
)

// Manager handles all active call sessions
type Manager struct {
	mu       sync.RWMutex
	sessions map[string]*Session  // keyed by session ID
	byCallSID map[string]*Session // keyed by Twilio Call SID
	byStreamSID map[string]*Session // keyed by Twilio Stream SID

	maxConcurrent int
	timeout       time.Duration
	logger        zerolog.Logger

	// Callbacks
	onSessionCreated func(*Session)
	onSessionEnded   func(*Session)
}

// NewManager creates a new session manager
func NewManager(maxConcurrent int, timeout time.Duration, logger zerolog.Logger) *Manager {
	m := &Manager{
		sessions:      make(map[string]*Session),
		byCallSID:     make(map[string]*Session),
		byStreamSID:   make(map[string]*Session),
		maxConcurrent: maxConcurrent,
		timeout:       timeout,
		logger:        logger.With().Str("component", "session_manager").Logger(),
	}

	// Start cleanup goroutine
	go m.cleanupLoop()

	return m
}

// CreateSession creates a new call session
func (m *Manager) CreateSession(callSID, from, to string, direction CallDirection) (*Session, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check for existing session with same call SID
	if existing, ok := m.byCallSID[callSID]; ok {
		return existing, nil
	}

	// Check concurrent limit
	if len(m.sessions) >= m.maxConcurrent {
		return nil, fmt.Errorf("maximum concurrent calls (%d) reached", m.maxConcurrent)
	}

	// Create session
	session := NewSession(callSID, from, to, direction)

	// Set up session callbacks
	session.OnStateChange(func(s *Session, from, to CallState) {
		m.logger.Info().
			Str("session_id", s.ID).
			Str("call_sid", s.CallSID).
			Str("from_state", string(from)).
			Str("to_state", string(to)).
			Msg("Session state changed")
	})

	session.OnEnded(func(s *Session) {
		m.removeSession(s)
		if m.onSessionEnded != nil {
			m.onSessionEnded(s)
		}
	})

	// Store session
	m.sessions[session.ID] = session
	m.byCallSID[callSID] = session

	m.logger.Info().
		Str("session_id", session.ID).
		Str("call_sid", callSID).
		Str("from", from).
		Str("to", to).
		Str("direction", string(direction)).
		Int("active_sessions", len(m.sessions)).
		Msg("Session created")

	if m.onSessionCreated != nil {
		go m.onSessionCreated(session)
	}

	return session, nil
}

// GetSession retrieves a session by ID
func (m *Manager) GetSession(id string) (*Session, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	s, ok := m.sessions[id]
	return s, ok
}

// GetSessionByCallSID retrieves a session by Twilio Call SID
func (m *Manager) GetSessionByCallSID(callSID string) (*Session, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	s, ok := m.byCallSID[callSID]
	return s, ok
}

// GetSessionByStreamSID retrieves a session by Twilio Stream SID
func (m *Manager) GetSessionByStreamSID(streamSID string) (*Session, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	s, ok := m.byStreamSID[streamSID]
	return s, ok
}

// SetStreamSID associates a stream SID with a session
func (m *Manager) SetStreamSID(sessionID, streamSID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	session, ok := m.sessions[sessionID]
	if !ok {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	session.SetStreamSID(streamSID)
	m.byStreamSID[streamSID] = session

	m.logger.Debug().
		Str("session_id", sessionID).
		Str("stream_sid", streamSID).
		Msg("Stream SID associated with session")

	return nil
}

// EndSession ends a session by ID
func (m *Manager) EndSession(id, reason string) error {
	m.mu.RLock()
	session, ok := m.sessions[id]
	m.mu.RUnlock()

	if !ok {
		return fmt.Errorf("session not found: %s", id)
	}

	session.End(reason)
	return nil
}

// EndSessionByCallSID ends a session by Call SID
func (m *Manager) EndSessionByCallSID(callSID, reason string) error {
	m.mu.RLock()
	session, ok := m.byCallSID[callSID]
	m.mu.RUnlock()

	if !ok {
		return fmt.Errorf("session not found for call SID: %s", callSID)
	}

	session.End(reason)
	return nil
}

// removeSession removes a session from all maps
func (m *Manager) removeSession(s *Session) {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.sessions, s.ID)
	delete(m.byCallSID, s.CallSID)
	if s.StreamSID != "" {
		delete(m.byStreamSID, s.StreamSID)
	}

	m.logger.Info().
		Str("session_id", s.ID).
		Str("call_sid", s.CallSID).
		Dur("duration", s.Duration()).
		Int("active_sessions", len(m.sessions)).
		Msg("Session removed")
}

// ListSessions returns all active sessions
func (m *Manager) ListSessions() []*Session {
	m.mu.RLock()
	defer m.mu.RUnlock()

	sessions := make([]*Session, 0, len(m.sessions))
	for _, s := range m.sessions {
		sessions = append(sessions, s)
	}
	return sessions
}

// Count returns the number of active sessions
func (m *Manager) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.sessions)
}

// OnSessionCreated sets a callback for new sessions
func (m *Manager) OnSessionCreated(fn func(*Session)) {
	m.mu.Lock()
	m.onSessionCreated = fn
	m.mu.Unlock()
}

// OnSessionEnded sets a callback for ended sessions
func (m *Manager) OnSessionEnded(fn func(*Session)) {
	m.mu.Lock()
	m.onSessionEnded = fn
	m.mu.Unlock()
}

// cleanupLoop periodically cleans up stale sessions
func (m *Manager) cleanupLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		m.cleanup()
	}
}

// cleanup removes sessions that have exceeded the timeout
func (m *Manager) cleanup() {
	m.mu.Lock()
	var stale []*Session

	for _, s := range m.sessions {
		if time.Since(s.CreatedAt) > m.timeout {
			stale = append(stale, s)
		}
	}
	m.mu.Unlock()

	for _, s := range stale {
		m.logger.Warn().
			Str("session_id", s.ID).
			Str("call_sid", s.CallSID).
			Dur("age", time.Since(s.CreatedAt)).
			Msg("Cleaning up stale session")
		s.Fail("session timeout")
	}
}

// Shutdown gracefully shuts down all sessions
func (m *Manager) Shutdown(ctx context.Context) error {
	m.mu.Lock()
	sessions := make([]*Session, 0, len(m.sessions))
	for _, s := range m.sessions {
		sessions = append(sessions, s)
	}
	m.mu.Unlock()

	m.logger.Info().
		Int("active_sessions", len(sessions)).
		Msg("Shutting down session manager")

	// End all sessions
	for _, s := range sessions {
		s.End("server shutdown")
	}

	return nil
}

// Stats returns manager statistics
type ManagerStats struct {
	ActiveSessions  int `json:"active_sessions"`
	MaxConcurrent   int `json:"max_concurrent"`
	TotalCreated    int `json:"total_created"`
	TotalCompleted  int `json:"total_completed"`
	TotalFailed     int `json:"total_failed"`
}

func (m *Manager) Stats() ManagerStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return ManagerStats{
		ActiveSessions: len(m.sessions),
		MaxConcurrent:  m.maxConcurrent,
	}
}
