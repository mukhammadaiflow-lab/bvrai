/**
 * Structured JSON logging using Pino
 * Provides consistent log format across all services
 */
import pino from 'pino';
import { config } from './config';

// Redact sensitive fields from logs
const redactPaths = [
  'req.headers.authorization',
  'req.headers.cookie',
  'res.headers["set-cookie"]',
  'apiKey',
  'apiSecret',
  'token',
  'password',
  'secret',
];

export const logger = pino({
  name: 'token-service',
  level: config.NODE_ENV === 'production' ? 'info' : 'debug',
  redact: redactPaths,
  formatters: {
    level: (label) => ({ level: label }),
    bindings: (bindings) => ({
      pid: bindings.pid,
      host: bindings.hostname,
      service: 'token-service',
      version: process.env.npm_package_version || '1.0.0',
    }),
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  // In development, use pretty printing
  transport:
    config.NODE_ENV === 'development'
      ? {
          target: 'pino-pretty',
          options: {
            colorize: true,
            translateTime: 'SYS:standard',
          },
        }
      : undefined,
});

export type Logger = typeof logger;
