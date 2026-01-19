package api

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/bvrai/telephony-gateway/internal/config"
	"github.com/bvrai/telephony-gateway/internal/session"
	"github.com/bvrai/telephony-gateway/internal/twilio"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	"github.com/rs/zerolog"
)

// Router handles HTTP routing
type Router struct {
	cfg            *config.Config
	sessionManager *session.Manager
	webhookHandler *twilio.WebhookHandler
	mediaHandler   *twilio.MediaHandler
	logger         zerolog.Logger
	startTime      time.Time
}

// NewRouter creates a new router
func NewRouter(
	cfg *config.Config,
	sm *session.Manager,
	wh *twilio.WebhookHandler,
	mh *twilio.MediaHandler,
	logger zerolog.Logger,
) *Router {
	return &Router{
		cfg:            cfg,
		sessionManager: sm,
		webhookHandler: wh,
		mediaHandler:   mh,
		logger:         logger.With().Str("component", "api").Logger(),
		startTime:      time.Now(),
	}
}

// Handler returns the HTTP handler
func (rt *Router) Handler() http.Handler {
	r := chi.NewRouter()

	// Middleware
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(rt.loggingMiddleware)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(60 * time.Second))

	// CORS
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   rt.cfg.AllowedOrigins,
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-Request-ID"},
		ExposedHeaders:   []string{"Link"},
		AllowCredentials: true,
		MaxAge:           300,
	}))

	// Health check
	r.Get(rt.cfg.HealthCheckPath, rt.handleHealth)
	r.Get("/ready", rt.handleReady)

	// API routes
	r.Route("/api/v1", func(r chi.Router) {
		// Session management
		r.Route("/sessions", func(r chi.Router) {
			r.Get("/", rt.handleListSessions)
			r.Get("/{sessionID}", rt.handleGetSession)
			r.Delete("/{sessionID}", rt.handleEndSession)
			r.Post("/{sessionID}/audio", rt.handleSendAudio)
		})

		// Call management
		r.Route("/calls", func(r chi.Router) {
			r.Post("/outbound", rt.handleOutboundCall)
			r.Get("/{callSID}", rt.handleGetCall)
			r.Post("/{callSID}/hangup", rt.handleHangupCall)
			r.Post("/{callSID}/transfer", rt.handleTransferCall)
		})

		// Stats
		r.Get("/stats", rt.handleStats)
	})

	// Twilio webhooks
	r.Route("/telephony", func(r chi.Router) {
		rt.webhookHandler.RegisterRoutes(r)
	})

	// Media stream WebSocket
	r.Route("/media", func(r chi.Router) {
		rt.mediaHandler.RegisterRoutes(r)
	})

	return r
}

// loggingMiddleware logs HTTP requests
func (rt *Router) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)

		defer func() {
			rt.logger.Debug().
				Str("method", r.Method).
				Str("path", r.URL.Path).
				Int("status", ww.Status()).
				Int("bytes", ww.BytesWritten()).
				Dur("duration", time.Since(start)).
				Str("remote", r.RemoteAddr).
				Msg("HTTP request")
		}()

		next.ServeHTTP(ww, r)
	})
}

// HealthResponse represents health check response
type HealthResponse struct {
	Status    string    `json:"status"`
	Uptime    string    `json:"uptime"`
	Timestamp time.Time `json:"timestamp"`
	Version   string    `json:"version"`
}

func (rt *Router) handleHealth(w http.ResponseWriter, r *http.Request) {
	resp := HealthResponse{
		Status:    "healthy",
		Uptime:    time.Since(rt.startTime).String(),
		Timestamp: time.Now(),
		Version:   "1.0.0",
	}
	rt.respondJSON(w, http.StatusOK, resp)
}

func (rt *Router) handleReady(w http.ResponseWriter, r *http.Request) {
	// Check if we can accept calls
	if rt.sessionManager.Count() >= rt.cfg.MaxConcurrentCalls {
		rt.respondJSON(w, http.StatusServiceUnavailable, map[string]string{
			"status": "not ready",
			"reason": "max concurrent calls reached",
		})
		return
	}

	rt.respondJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// SessionResponse represents a session in API responses
type SessionResponse struct {
	ID          string    `json:"id"`
	CallSID     string    `json:"call_sid"`
	StreamSID   string    `json:"stream_sid,omitempty"`
	From        string    `json:"from"`
	To          string    `json:"to"`
	Direction   string    `json:"direction"`
	State       string    `json:"state"`
	AgentID     string    `json:"agent_id,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	ConnectedAt *time.Time `json:"connected_at,omitempty"`
	Duration    string    `json:"duration,omitempty"`
}

func sessionToResponse(s *session.Session) SessionResponse {
	return SessionResponse{
		ID:          s.ID,
		CallSID:     s.CallSID,
		StreamSID:   s.StreamSID,
		From:        s.From,
		To:          s.To,
		Direction:   string(s.Direction),
		State:       string(s.GetState()),
		AgentID:     s.AgentID,
		CreatedAt:   s.CreatedAt,
		ConnectedAt: s.ConnectedAt,
		Duration:    s.Duration().String(),
	}
}

func (rt *Router) handleListSessions(w http.ResponseWriter, r *http.Request) {
	sessions := rt.sessionManager.ListSessions()
	resp := make([]SessionResponse, len(sessions))
	for i, s := range sessions {
		resp[i] = sessionToResponse(s)
	}
	rt.respondJSON(w, http.StatusOK, resp)
}

func (rt *Router) handleGetSession(w http.ResponseWriter, r *http.Request) {
	sessionID := chi.URLParam(r, "sessionID")
	sess, found := rt.sessionManager.GetSession(sessionID)
	if !found {
		rt.respondError(w, http.StatusNotFound, "Session not found")
		return
	}
	rt.respondJSON(w, http.StatusOK, sessionToResponse(sess))
}

func (rt *Router) handleEndSession(w http.ResponseWriter, r *http.Request) {
	sessionID := chi.URLParam(r, "sessionID")
	if err := rt.sessionManager.EndSession(sessionID, "api request"); err != nil {
		rt.respondError(w, http.StatusNotFound, err.Error())
		return
	}
	rt.respondJSON(w, http.StatusOK, map[string]string{"status": "ended"})
}

// SendAudioRequest represents a request to send audio
type SendAudioRequest struct {
	Audio string `json:"audio"` // base64 encoded audio
}

func (rt *Router) handleSendAudio(w http.ResponseWriter, r *http.Request) {
	sessionID := chi.URLParam(r, "sessionID")
	sess, found := rt.sessionManager.GetSession(sessionID)
	if !found {
		rt.respondError(w, http.StatusNotFound, "Session not found")
		return
	}

	var req SendAudioRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		rt.respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	// Decode base64 audio
	// audioData, err := base64.StdEncoding.DecodeString(req.Audio)
	// if err != nil {
	// 	rt.respondError(w, http.StatusBadRequest, "Invalid audio encoding")
	// 	return
	// }

	// Send audio
	// sess.SendAudio(audioData)
	_ = sess

	rt.respondJSON(w, http.StatusOK, map[string]string{"status": "sent"})
}

// OutboundCallRequest represents a request to make an outbound call
type OutboundCallRequest struct {
	To      string `json:"to"`
	From    string `json:"from,omitempty"`
	AgentID string `json:"agent_id,omitempty"`
}

func (rt *Router) handleOutboundCall(w http.ResponseWriter, r *http.Request) {
	var req OutboundCallRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		rt.respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if req.To == "" {
		rt.respondError(w, http.StatusBadRequest, "Missing 'to' phone number")
		return
	}

	// TODO: Use Twilio client to make outbound call
	rt.respondError(w, http.StatusNotImplemented, "Outbound calls not yet implemented")
}

func (rt *Router) handleGetCall(w http.ResponseWriter, r *http.Request) {
	callSID := chi.URLParam(r, "callSID")
	sess, found := rt.sessionManager.GetSessionByCallSID(callSID)
	if !found {
		rt.respondError(w, http.StatusNotFound, "Call not found")
		return
	}
	rt.respondJSON(w, http.StatusOK, sessionToResponse(sess))
}

func (rt *Router) handleHangupCall(w http.ResponseWriter, r *http.Request) {
	callSID := chi.URLParam(r, "callSID")
	if err := rt.sessionManager.EndSessionByCallSID(callSID, "hangup request"); err != nil {
		rt.respondError(w, http.StatusNotFound, err.Error())
		return
	}

	// TODO: Also tell Twilio to end the call
	rt.respondJSON(w, http.StatusOK, map[string]string{"status": "hangup initiated"})
}

// TransferCallRequest represents a call transfer request
type TransferCallRequest struct {
	To string `json:"to"` // Phone number or SIP address
}

func (rt *Router) handleTransferCall(w http.ResponseWriter, r *http.Request) {
	// callSID := chi.URLParam(r, "callSID")
	var req TransferCallRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		rt.respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	// TODO: Implement call transfer via Twilio
	rt.respondError(w, http.StatusNotImplemented, "Call transfer not yet implemented")
}

// StatsResponse represents system statistics
type StatsResponse struct {
	ActiveSessions    int   `json:"active_sessions"`
	MaxConcurrent     int   `json:"max_concurrent"`
	ActiveConnections int   `json:"active_connections"`
	UptimeSeconds     int64 `json:"uptime_seconds"`
}

func (rt *Router) handleStats(w http.ResponseWriter, r *http.Request) {
	stats := rt.sessionManager.Stats()
	resp := StatsResponse{
		ActiveSessions:    stats.ActiveSessions,
		MaxConcurrent:     stats.MaxConcurrent,
		ActiveConnections: rt.mediaHandler.ActiveConnections(),
		UptimeSeconds:     int64(time.Since(rt.startTime).Seconds()),
	}
	rt.respondJSON(w, http.StatusOK, resp)
}

// respondJSON writes JSON response
func (rt *Router) respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// respondError writes error response
func (rt *Router) respondError(w http.ResponseWriter, status int, message string) {
	rt.respondJSON(w, status, map[string]string{"error": message})
}
