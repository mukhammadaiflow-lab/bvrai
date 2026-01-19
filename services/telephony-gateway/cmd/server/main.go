package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/bvrai/telephony-gateway/internal/api"
	"github.com/bvrai/telephony-gateway/internal/config"
	"github.com/bvrai/telephony-gateway/internal/session"
	"github.com/bvrai/telephony-gateway/internal/twilio"
	"github.com/rs/zerolog"
)

func main() {
	// Initialize logger
	logger := initLogger()
	logger.Info().Msg("Starting Telephony Gateway")

	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		logger.Fatal().Err(err).Msg("Failed to load configuration")
	}

	if err := cfg.Validate(); err != nil {
		logger.Fatal().Err(err).Msg("Invalid configuration")
	}

	logger.Info().
		Str("host", cfg.Host).
		Int("port", cfg.Port).
		Int("max_concurrent", cfg.MaxConcurrentCalls).
		Msg("Configuration loaded")

	// Initialize components
	sessionManager := session.NewManager(
		cfg.MaxConcurrentCalls,
		cfg.SessionTimeout,
		logger,
	)

	webhookHandler := twilio.NewWebhookHandler(cfg, sessionManager, logger)
	mediaHandler := twilio.NewMediaHandler(cfg, sessionManager, logger)

	// Set up audio callback for media pipeline integration
	mediaHandler.SetAudioCallback(func(sessionID string, audio []byte, timestamp int64) {
		// Forward audio to media pipeline
		// This will be connected to gRPC client for media-pipeline service
		logger.Trace().
			Str("session_id", sessionID).
			Int("audio_len", len(audio)).
			Int64("timestamp", timestamp).
			Msg("Audio received")
	})

	// Set up session callbacks
	sessionManager.OnSessionCreated(func(s *session.Session) {
		logger.Info().
			Str("session_id", s.ID).
			Str("from", s.From).
			Str("to", s.To).
			Msg("New session created")
	})

	sessionManager.OnSessionEnded(func(s *session.Session) {
		logger.Info().
			Str("session_id", s.ID).
			Dur("duration", s.Duration()).
			Msg("Session ended")
	})

	// Initialize router
	router := api.NewRouter(cfg, sessionManager, webhookHandler, mediaHandler, logger)

	// Create HTTP server
	server := &http.Server{
		Addr:         cfg.Address(),
		Handler:      router.Handler(),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in goroutine
	go func() {
		logger.Info().Str("addr", cfg.Address()).Msg("HTTP server starting")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal().Err(err).Msg("HTTP server error")
		}
	}()

	// Wait for shutdown signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit

	logger.Info().Str("signal", sig.String()).Msg("Shutdown signal received")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()

	// Shutdown session manager first
	if err := sessionManager.Shutdown(ctx); err != nil {
		logger.Error().Err(err).Msg("Session manager shutdown error")
	}

	// Shutdown HTTP server
	if err := server.Shutdown(ctx); err != nil {
		logger.Error().Err(err).Msg("HTTP server shutdown error")
	}

	logger.Info().Msg("Telephony Gateway stopped")
}

func initLogger() zerolog.Logger {
	// Check log format from environment
	logFormat := os.Getenv("LOG_FORMAT")
	logLevel := os.Getenv("LOG_LEVEL")

	// Set log level
	level := zerolog.InfoLevel
	switch logLevel {
	case "debug":
		level = zerolog.DebugLevel
	case "trace":
		level = zerolog.TraceLevel
	case "warn":
		level = zerolog.WarnLevel
	case "error":
		level = zerolog.ErrorLevel
	}

	zerolog.SetGlobalLevel(level)

	// Console output for development
	if logFormat == "console" {
		output := zerolog.ConsoleWriter{Out: os.Stdout, TimeFormat: time.RFC3339}
		return zerolog.New(output).With().Timestamp().Caller().Logger()
	}

	// JSON output for production
	return zerolog.New(os.Stdout).With().Timestamp().Caller().Logger()
}

func init() {
	// Print banner
	banner := `
╔═══════════════════════════════════════════════════════════╗
║                  TELEPHONY GATEWAY                        ║
║              Builder Engine v1.0.0                        ║
╚═══════════════════════════════════════════════════════════╝
`
	fmt.Println(banner)
}
