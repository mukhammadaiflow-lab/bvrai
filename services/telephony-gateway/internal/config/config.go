package config

import (
	"fmt"
	"time"

	"github.com/kelseyhightower/envconfig"
)

// Config holds all configuration for the telephony gateway
type Config struct {
	// Server settings
	Port            int           `envconfig:"PORT" default:"8080"`
	Host            string        `envconfig:"HOST" default:"0.0.0.0"`
	ShutdownTimeout time.Duration `envconfig:"SHUTDOWN_TIMEOUT" default:"30s"`

	// Twilio settings
	TwilioAccountSID   string `envconfig:"TWILIO_ACCOUNT_SID" required:"true"`
	TwilioAuthToken    string `envconfig:"TWILIO_AUTH_TOKEN" required:"true"`
	TwilioPhoneNumber  string `envconfig:"TWILIO_PHONE_NUMBER"`
	TwilioWebhookURL   string `envconfig:"TWILIO_WEBHOOK_URL"` // Public URL for webhooks

	// Media Pipeline connection
	MediaPipelineURL  string `envconfig:"MEDIA_PIPELINE_URL" default:"localhost:8081"`
	MediaPipelineGRPC string `envconfig:"MEDIA_PIPELINE_GRPC" default:"localhost:50051"`

	// WebSocket settings
	WSReadBufferSize  int           `envconfig:"WS_READ_BUFFER_SIZE" default:"1024"`
	WSWriteBufferSize int           `envconfig:"WS_WRITE_BUFFER_SIZE" default:"1024"`
	WSPingInterval    time.Duration `envconfig:"WS_PING_INTERVAL" default:"30s"`
	WSPongWait        time.Duration `envconfig:"WS_PONG_WAIT" default:"60s"`
	WSWriteWait       time.Duration `envconfig:"WS_WRITE_WAIT" default:"10s"`

	// Session settings
	MaxConcurrentCalls int           `envconfig:"MAX_CONCURRENT_CALLS" default:"100"`
	SessionTimeout     time.Duration `envconfig:"SESSION_TIMEOUT" default:"1h"`

	// Agent defaults
	DefaultAgentID     string `envconfig:"DEFAULT_AGENT_ID" default:"default"`
	DefaultGreeting    string `envconfig:"DEFAULT_GREETING" default:"Hello, how can I help you today?"`
	DefaultVoice       string `envconfig:"DEFAULT_VOICE" default:"Polly.Amy"`

	// Logging
	LogLevel  string `envconfig:"LOG_LEVEL" default:"info"`
	LogFormat string `envconfig:"LOG_FORMAT" default:"json"` // json or console

	// Health check
	HealthCheckPath string `envconfig:"HEALTH_CHECK_PATH" default:"/health"`

	// CORS
	AllowedOrigins []string `envconfig:"ALLOWED_ORIGINS" default:"*"`
}

// Load reads configuration from environment variables
func Load() (*Config, error) {
	var cfg Config
	if err := envconfig.Process("", &cfg); err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}
	return &cfg, nil
}

// Address returns the server address
func (c *Config) Address() string {
	return fmt.Sprintf("%s:%d", c.Host, c.Port)
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.TwilioAccountSID == "" {
		return fmt.Errorf("TWILIO_ACCOUNT_SID is required")
	}
	if c.TwilioAuthToken == "" {
		return fmt.Errorf("TWILIO_AUTH_TOKEN is required")
	}
	if c.MaxConcurrentCalls <= 0 {
		return fmt.Errorf("MAX_CONCURRENT_CALLS must be positive")
	}
	return nil
}
