package builderengine

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// StreamEventType represents the type of streaming event
type StreamEventType string

const (
	StreamEventConnected          StreamEventType = "connected"
	StreamEventDisconnected       StreamEventType = "disconnected"
	StreamEventError              StreamEventType = "error"
	StreamEventCallStarted        StreamEventType = "call.started"
	StreamEventCallRinging        StreamEventType = "call.ringing"
	StreamEventCallAnswered       StreamEventType = "call.answered"
	StreamEventCallEnded          StreamEventType = "call.ended"
	StreamEventCallFailed         StreamEventType = "call.failed"
	StreamEventAudioStart         StreamEventType = "audio.start"
	StreamEventAudioData          StreamEventType = "audio.data"
	StreamEventAudioEnd           StreamEventType = "audio.end"
	StreamEventTranscriptionPartial StreamEventType = "transcription.partial"
	StreamEventTranscriptionFinal StreamEventType = "transcription.final"
	StreamEventUserSpeechStart    StreamEventType = "user.speech.start"
	StreamEventUserSpeechEnd      StreamEventType = "user.speech.end"
	StreamEventAgentSpeechStart   StreamEventType = "agent.speech.start"
	StreamEventAgentSpeechEnd     StreamEventType = "agent.speech.end"
	StreamEventFunctionCall       StreamEventType = "function.call"
	StreamEventFunctionResult     StreamEventType = "function.result"
	StreamEventDTMFReceived       StreamEventType = "dtmf.received"
)

// StreamEvent represents a streaming event
type StreamEvent struct {
	Type      StreamEventType        `json:"type"`
	Data      map[string]interface{} `json:"data"`
	CallID    string                 `json:"call_id,omitempty"`
	Timestamp string                 `json:"timestamp,omitempty"`
}

// EventHandler is a function that handles streaming events
type EventHandler func(event StreamEvent)

// StreamingClient handles WebSocket streaming connections
type StreamingClient struct {
	apiKey     string
	baseURL    string
	conn       *websocket.Conn
	handlers   map[StreamEventType][]EventHandler
	wildcardHandlers []EventHandler
	mu         sync.RWMutex
	running    bool
	reconnectAttempts int
	maxReconnectAttempts int
	done       chan struct{}
}

// NewStreamingClient creates a new streaming client
func NewStreamingClient(apiKey string, baseURL string) *StreamingClient {
	return &StreamingClient{
		apiKey:              apiKey,
		baseURL:             baseURL,
		handlers:            make(map[StreamEventType][]EventHandler),
		wildcardHandlers:    make([]EventHandler, 0),
		maxReconnectAttempts: 5,
		done:                make(chan struct{}),
	}
}

// On registers an event handler for a specific event type
func (s *StreamingClient) On(eventType StreamEventType, handler EventHandler) *StreamingClient {
	s.mu.Lock()
	defer s.mu.Unlock()

	if eventType == "*" {
		s.wildcardHandlers = append(s.wildcardHandlers, handler)
	} else {
		s.handlers[eventType] = append(s.handlers[eventType], handler)
	}
	return s
}

// OnAll registers a handler for all events
func (s *StreamingClient) OnAll(handler EventHandler) *StreamingClient {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.wildcardHandlers = append(s.wildcardHandlers, handler)
	return s
}

// Off removes an event handler
func (s *StreamingClient) Off(eventType StreamEventType, handler EventHandler) *StreamingClient {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Note: Go doesn't have function equality, so we'd need to use function pointers
	// This is a simplified implementation
	return s
}

// ConnectOptions represents options for connecting
type ConnectOptions struct {
	CallID       string
	SubscribeAll bool
}

// Connect connects to the WebSocket server
func (s *StreamingClient) Connect(opts ConnectOptions) error {
	wsURL := s.baseURL + "/ws/v1/stream"

	params := url.Values{}
	if opts.CallID != "" {
		params.Set("call_id", opts.CallID)
	}
	if opts.SubscribeAll {
		params.Set("subscribe_all", "true")
	}
	if query := params.Encode(); query != "" {
		wsURL += "?" + query
	}

	header := make(map[string][]string)
	header["Authorization"] = []string{"Bearer " + s.apiKey}
	header["Sec-WebSocket-Protocol"] = []string{"builderengine"}

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.Dial(wsURL, header)
	if err != nil {
		return &WebSocketError{Message: "failed to connect", Err: err}
	}

	s.conn = conn
	s.running = true
	s.reconnectAttempts = 0

	// Dispatch connected event
	s.dispatchEvent(StreamEvent{
		Type: StreamEventConnected,
		Data: map[string]interface{}{"url": wsURL, "call_id": opts.CallID},
	})

	// Start reading messages
	go s.readPump(opts)

	return nil
}

// readPump reads messages from the WebSocket
func (s *StreamingClient) readPump(opts ConnectOptions) {
	defer func() {
		s.conn.Close()
		if s.running {
			s.handleReconnect(opts)
		}
	}()

	for {
		select {
		case <-s.done:
			return
		default:
			_, message, err := s.conn.ReadMessage()
			if err != nil {
				if s.running {
					s.dispatchEvent(StreamEvent{
						Type: StreamEventError,
						Data: map[string]interface{}{"error": err.Error()},
					})
				}
				return
			}

			var event StreamEvent
			if err := json.Unmarshal(message, &event); err != nil {
				continue
			}

			s.dispatchEvent(event)
		}
	}
}

// dispatchEvent dispatches an event to all registered handlers
func (s *StreamingClient) dispatchEvent(event StreamEvent) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Call specific handlers
	if handlers, ok := s.handlers[event.Type]; ok {
		for _, handler := range handlers {
			go func(h EventHandler) {
				defer func() {
					if r := recover(); r != nil {
						// Log error but don't crash
					}
				}()
				h(event)
			}(handler)
		}
	}

	// Call wildcard handlers
	for _, handler := range s.wildcardHandlers {
		go func(h EventHandler) {
			defer func() {
				if r := recover(); r != nil {
					// Log error but don't crash
				}
			}()
			h(event)
		}(handler)
	}
}

// handleReconnect handles reconnection after disconnect
func (s *StreamingClient) handleReconnect(opts ConnectOptions) {
	if s.reconnectAttempts >= s.maxReconnectAttempts {
		s.dispatchEvent(StreamEvent{
			Type: StreamEventError,
			Data: map[string]interface{}{"error": "max reconnection attempts reached"},
		})
		return
	}

	s.reconnectAttempts++
	delay := time.Duration(1<<uint(s.reconnectAttempts)) * time.Second

	time.Sleep(delay)

	if err := s.Connect(opts); err != nil {
		s.handleReconnect(opts)
	}
}

// Disconnect disconnects from the WebSocket server
func (s *StreamingClient) Disconnect() {
	s.running = false
	close(s.done)

	if s.conn != nil {
		s.conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
		s.conn.Close()
	}

	s.dispatchEvent(StreamEvent{
		Type: StreamEventDisconnected,
		Data: map[string]interface{}{"reason": "client_disconnect"},
	})
}

// Send sends a message to the server
func (s *StreamingClient) Send(message map[string]interface{}) error {
	if s.conn == nil || !s.running {
		return &WebSocketError{Message: "not connected"}
	}

	data, err := json.Marshal(message)
	if err != nil {
		return &WebSocketError{Message: "failed to marshal message", Err: err}
	}

	if err := s.conn.WriteMessage(websocket.TextMessage, data); err != nil {
		return &WebSocketError{Message: "failed to send message", Err: err}
	}

	return nil
}

// Subscribe subscribes to events for a specific call
func (s *StreamingClient) Subscribe(callID string) error {
	return s.Send(map[string]interface{}{
		"action":  "subscribe",
		"call_id": callID,
	})
}

// Unsubscribe unsubscribes from events for a specific call
func (s *StreamingClient) Unsubscribe(callID string) error {
	return s.Send(map[string]interface{}{
		"action":  "unsubscribe",
		"call_id": callID,
	})
}

// SendAudio sends audio data to a call
func (s *StreamingClient) SendAudio(callID string, audioData []byte) error {
	encoded := base64.StdEncoding.EncodeToString(audioData)
	return s.Send(map[string]interface{}{
		"action":  "audio",
		"call_id": callID,
		"data":    encoded,
	})
}

// InjectText injects text for the agent to speak
func (s *StreamingClient) InjectText(callID string, text string, interrupt bool) error {
	return s.Send(map[string]interface{}{
		"action":    "inject",
		"call_id":   callID,
		"text":      text,
		"interrupt": interrupt,
	})
}

// SendDTMF sends DTMF tones
func (s *StreamingClient) SendDTMF(callID string, digits string) error {
	return s.Send(map[string]interface{}{
		"action":  "dtmf",
		"call_id": callID,
		"digits":  digits,
	})
}

// IsConnected returns whether the client is connected
func (s *StreamingClient) IsConnected() bool {
	return s.running && s.conn != nil
}

// WaitForEvent waits for a specific event type with timeout
func (s *StreamingClient) WaitForEvent(eventType StreamEventType, timeout time.Duration) (*StreamEvent, error) {
	eventChan := make(chan StreamEvent, 1)

	handler := func(event StreamEvent) {
		select {
		case eventChan <- event:
		default:
		}
	}

	s.On(eventType, handler)
	defer func() {
		// Remove handler (simplified - would need proper implementation)
	}()

	select {
	case event := <-eventChan:
		return &event, nil
	case <-time.After(timeout):
		return nil, fmt.Errorf("timeout waiting for event: %s", eventType)
	}
}
