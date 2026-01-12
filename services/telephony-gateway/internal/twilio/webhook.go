package twilio

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/bvrai/telephony-gateway/internal/config"
	"github.com/bvrai/telephony-gateway/internal/session"
	"github.com/go-chi/chi/v5"
	"github.com/rs/zerolog"
)

// WebhookHandler handles Twilio webhook requests
type WebhookHandler struct {
	cfg            *config.Config
	sessionManager *session.Manager
	logger         zerolog.Logger
}

// NewWebhookHandler creates a new webhook handler
func NewWebhookHandler(cfg *config.Config, sm *session.Manager, logger zerolog.Logger) *WebhookHandler {
	return &WebhookHandler{
		cfg:            cfg,
		sessionManager: sm,
		logger:         logger.With().Str("component", "twilio_webhook").Logger(),
	}
}

// RegisterRoutes registers webhook routes
func (h *WebhookHandler) RegisterRoutes(r chi.Router) {
	r.Post("/incoming", h.HandleIncomingCall)
	r.Post("/status", h.HandleStatusCallback)
	r.Post("/recording", h.HandleRecordingCallback)
}

// IncomingCallRequest represents the Twilio incoming call webhook payload
type IncomingCallRequest struct {
	AccountSID    string `form:"AccountSid"`
	APIVersion    string `form:"ApiVersion"`
	CallSID       string `form:"CallSid"`
	CallStatus    string `form:"CallStatus"`
	Called        string `form:"Called"`
	CalledCity    string `form:"CalledCity"`
	CalledCountry string `form:"CalledCountry"`
	CalledState   string `form:"CalledState"`
	CalledZip     string `form:"CalledZip"`
	Caller        string `form:"Caller"`
	CallerCity    string `form:"CallerCity"`
	CallerCountry string `form:"CallerCountry"`
	CallerName    string `form:"CallerName"`
	CallerState   string `form:"CallerState"`
	CallerZip     string `form:"CallerZip"`
	Direction     string `form:"Direction"`
	From          string `form:"From"`
	FromCity      string `form:"FromCity"`
	FromCountry   string `form:"FromCountry"`
	FromState     string `form:"FromState"`
	FromZip       string `form:"FromZip"`
	To            string `form:"To"`
	ToCity        string `form:"ToCity"`
	ToCountry     string `form:"ToCountry"`
	ToState       string `form:"ToState"`
	ToZip         string `form:"ToZip"`
}

// HandleIncomingCall handles incoming call webhooks from Twilio
func (h *WebhookHandler) HandleIncomingCall(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		h.logger.Error().Err(err).Msg("Failed to parse form")
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	req := IncomingCallRequest{
		AccountSID:  r.FormValue("AccountSid"),
		CallSID:     r.FormValue("CallSid"),
		CallStatus:  r.FormValue("CallStatus"),
		Called:      r.FormValue("Called"),
		Caller:      r.FormValue("Caller"),
		CallerName:  r.FormValue("CallerName"),
		Direction:   r.FormValue("Direction"),
		From:        r.FormValue("From"),
		To:          r.FormValue("To"),
		FromCity:    r.FormValue("FromCity"),
		FromCountry: r.FormValue("FromCountry"),
	}

	h.logger.Info().
		Str("call_sid", req.CallSID).
		Str("from", req.From).
		Str("to", req.To).
		Str("direction", req.Direction).
		Str("status", req.CallStatus).
		Str("caller_name", req.CallerName).
		Msg("Incoming call webhook received")

	// Create session for this call
	sess, err := h.sessionManager.CreateSession(
		req.CallSID,
		req.From,
		req.To,
		session.CallDirectionInbound,
	)
	if err != nil {
		h.logger.Error().Err(err).Msg("Failed to create session")
		// Return TwiML to reject call
		h.respondWithTwiML(w, GenerateRejectTwiML("Service unavailable"))
		return
	}

	sess.SetAgentConfig("caller_name", req.CallerName)
	sess.SetAgentConfig("from_city", req.FromCity)
	sess.SetAgentConfig("from_country", req.FromCountry)

	// Determine the WebSocket URL for media stream
	wsURL := h.getMediaStreamURL(sess.ID)

	// Generate TwiML response
	twiml := GenerateStreamTwiML(StreamConfig{
		URL:                  wsURL,
		Track:                "both_tracks", // Receive both inbound and outbound audio
		StatusCallback:       fmt.Sprintf("%s/telephony/status", h.cfg.TwilioWebhookURL),
		StatusCallbackEvents: []string{"initiated", "ringing", "answered", "completed"},
	})

	h.logger.Debug().
		Str("session_id", sess.ID).
		Str("ws_url", wsURL).
		Msg("Responding with stream TwiML")

	h.respondWithTwiML(w, twiml)
}

// StatusCallbackRequest represents status callback from Twilio
type StatusCallbackRequest struct {
	AccountSID      string `form:"AccountSid"`
	CallSID         string `form:"CallSid"`
	CallStatus      string `form:"CallStatus"`
	CallDuration    string `form:"CallDuration"`
	SequenceNumber  string `form:"SequenceNumber"`
	Timestamp       string `form:"Timestamp"`
}

// HandleStatusCallback handles call status callbacks
func (h *WebhookHandler) HandleStatusCallback(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		h.logger.Error().Err(err).Msg("Failed to parse form")
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	callSID := r.FormValue("CallSid")
	status := r.FormValue("CallStatus")
	duration := r.FormValue("CallDuration")

	h.logger.Info().
		Str("call_sid", callSID).
		Str("status", status).
		Str("duration", duration).
		Msg("Status callback received")

	sess, found := h.sessionManager.GetSessionByCallSID(callSID)
	if !found {
		h.logger.Warn().Str("call_sid", callSID).Msg("Session not found for status callback")
		w.WriteHeader(http.StatusOK)
		return
	}

	// Update session state based on Twilio status
	switch status {
	case "ringing":
		sess.SetState(session.CallStateRinging, "twilio status")
	case "in-progress", "answered":
		sess.SetState(session.CallStateConnected, "twilio status")
	case "completed":
		sess.End("call completed")
	case "busy", "no-answer", "canceled", "failed":
		sess.Fail(fmt.Sprintf("call %s", status))
	}

	w.WriteHeader(http.StatusOK)
}

// HandleRecordingCallback handles recording callbacks
func (h *WebhookHandler) HandleRecordingCallback(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		h.logger.Error().Err(err).Msg("Failed to parse form")
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	recordingSID := r.FormValue("RecordingSid")
	callSID := r.FormValue("CallSid")
	recordingURL := r.FormValue("RecordingUrl")
	recordingStatus := r.FormValue("RecordingStatus")
	recordingDuration := r.FormValue("RecordingDuration")

	h.logger.Info().
		Str("call_sid", callSID).
		Str("recording_sid", recordingSID).
		Str("recording_url", recordingURL).
		Str("status", recordingStatus).
		Str("duration", recordingDuration).
		Msg("Recording callback received")

	// TODO: Store recording info in database

	w.WriteHeader(http.StatusOK)
}

// getMediaStreamURL returns the WebSocket URL for media streaming
func (h *WebhookHandler) getMediaStreamURL(sessionID string) string {
	// Replace http(s) with ws(s) in the webhook URL
	baseURL := h.cfg.TwilioWebhookURL
	if len(baseURL) > 0 {
		if baseURL[:5] == "https" {
			baseURL = "wss" + baseURL[5:]
		} else if baseURL[:4] == "http" {
			baseURL = "ws" + baseURL[4:]
		}
	}
	return fmt.Sprintf("%s/media/stream/%s", baseURL, sessionID)
}

// respondWithTwiML writes TwiML response
func (h *WebhookHandler) respondWithTwiML(w http.ResponseWriter, twiml string) {
	w.Header().Set("Content-Type", "application/xml")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(twiml))
}

// respondWithJSON writes JSON response
func (h *WebhookHandler) respondWithJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
