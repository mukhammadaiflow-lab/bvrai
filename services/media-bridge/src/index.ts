/**
 * Media Bridge / Media Plane Service - Main Entry Point
 *
 * Self-hosted WebSocket-based media plane for voice AI agents.
 * Handles:
 * - Room management with participant routing
 * - WebSocket connections for real-time media/transcript streaming
 * - Integration with Dialog Manager for AI responses
 *
 * Architecture:
 * - HTTP server for REST API (room management, health checks)
 * - WebSocket server for real-time media connections
 * - Room Manager for in-memory room state
 * - Dialog Manager client for AI orchestration
 */
import 'dotenv/config';
import express, { Request, Response, NextFunction } from 'express';
import { createServer } from 'http';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import pinoHttp from 'pino-http';
import { config } from './config';
import { logger } from './logger';
import { router } from './routes';
import { initializeMediaServer } from './media-server';
import { roomManager } from './room-manager';

// Create Express application
const app = express();

// Create HTTP server for WebSocket support
const server = createServer(app);

// Initialize WebSocket-based media server
initializeMediaServer(server);

// Security middleware
app.use(helmet());

// CORS configuration
const corsOrigins = config.CORS_ORIGINS === '*'
  ? '*'
  : config.CORS_ORIGINS.split(',').map(o => o.trim());

app.use(cors({
  origin: corsOrigins,
  methods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
}));

// Request logging
app.use(pinoHttp({
  logger,
  autoLogging: {
    ignore: (req) => req.url === '/health' || req.url === '/ready',
  },
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(config.RATE_LIMIT_WINDOW_MS, 10),
  max: parseInt(config.RATE_LIMIT_MAX_REQUESTS, 10),
  message: { error: 'Too many requests', code: 'RATE_LIMIT_EXCEEDED' },
  skip: (req) => req.path === '/health' || req.path === '/ready',
});
app.use(limiter);

// Body parsing
app.use(express.json({ limit: '1mb' }));

// Mount routes
app.use('/', router);

// 404 handler
app.use((_req: Request, res: Response) => {
  res.status(404).json({ error: 'Not found', code: 'NOT_FOUND' });
});

// Global error handler
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  logger.error({
    msg: 'Unhandled error',
    error: err.message,
    stack: config.NODE_ENV === 'development' ? err.stack : undefined,
  });
  res.status(500).json({
    error: 'Internal server error',
    code: 'INTERNAL_ERROR',
    ...(config.NODE_ENV === 'development' && { details: err.message }),
  });
});

// Start server
const port = parseInt(config.PORT, 10);

server.listen(port, () => {
  logger.info({
    msg: 'Media Bridge service started',
    port,
    env: config.NODE_ENV,
    wsEndpoint: `/media/{room}?token=JWT`,
  });
});

// Graceful shutdown
const shutdown = async (signal: string) => {
  logger.info({ msg: `Received ${signal}, shutting down gracefully` });

  // Stop accepting new connections
  server.close(async () => {
    // Clean up all rooms
    roomManager.shutdown();
    logger.info({ msg: 'Server closed' });
    process.exit(0);
  });

  // Force shutdown after 30 seconds
  setTimeout(() => {
    logger.error({ msg: 'Forced shutdown after timeout' });
    process.exit(1);
  }, 30000);
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));

export { app, server };
