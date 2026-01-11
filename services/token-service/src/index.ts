/**
 * Token Service - Main Entry Point
 *
 * LiveKit access token generation service for the Builder Engine.
 * Provides JWT tokens for room participants and server-side agents.
 */
import 'dotenv/config';
import express, { Request, Response, NextFunction } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import pinoHttp from 'pino-http';
import { config } from './config';
import { logger } from './logger';
import { router } from './routes';

// Create Express application
const app = express();

// Security middleware
app.use(helmet());

// CORS configuration
const corsOrigins = config.CORS_ORIGINS === '*'
  ? '*'
  : config.CORS_ORIGINS.split(',').map(o => o.trim());

app.use(cors({
  origin: corsOrigins,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
  maxAge: 86400,
}));

// Request logging
app.use(pinoHttp({
  logger,
  autoLogging: {
    ignore: (req) => {
      // Don't log health check requests in production
      return req.url === '/health' || req.url === '/ready';
    },
  },
  customLogLevel: (_req, res) => {
    if (res.statusCode >= 500) return 'error';
    if (res.statusCode >= 400) return 'warn';
    return 'info';
  },
  serializers: {
    req: (req) => ({
      method: req.method,
      url: req.url,
      // Redact sensitive headers
      headers: {
        'user-agent': req.headers['user-agent'],
        'content-type': req.headers['content-type'],
      },
    }),
    res: (res) => ({
      statusCode: res.statusCode,
    }),
  },
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(config.RATE_LIMIT_WINDOW_MS, 10),
  max: parseInt(config.RATE_LIMIT_MAX_REQUESTS, 10),
  message: {
    error: 'Too many requests',
    code: 'RATE_LIMIT_EXCEEDED',
  },
  standardHeaders: true,
  legacyHeaders: false,
  skip: (req) => req.path === '/health' || req.path === '/ready',
});

app.use(limiter);

// Body parsing
app.use(express.json({ limit: '10kb' }));

// Mount routes
app.use('/', router);

// 404 handler
app.use((_req: Request, res: Response) => {
  res.status(404).json({
    error: 'Not found',
    code: 'NOT_FOUND',
  });
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

const server = app.listen(port, () => {
  logger.info({
    msg: 'Token service started',
    port,
    env: config.NODE_ENV,
  });
});

// Graceful shutdown
const shutdown = (signal: string) => {
  logger.info({ msg: `Received ${signal}, shutting down gracefully` });
  server.close(() => {
    logger.info({ msg: 'Server closed' });
    process.exit(0);
  });

  // Force shutdown after 10 seconds
  setTimeout(() => {
    logger.error({ msg: 'Forced shutdown after timeout' });
    process.exit(1);
  }, 10000);
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));

export { app };
