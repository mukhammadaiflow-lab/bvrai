/**
 * Structured JSON logging for Media Bridge
 */
import pino from 'pino';
import { config } from './config';

const redactPaths = [
  'req.headers.authorization',
  'token',
  'apiKey',
  'apiSecret',
];

export const logger = pino({
  name: 'media-bridge',
  level: config.NODE_ENV === 'production' ? 'info' : 'debug',
  redact: redactPaths,
  formatters: {
    level: (label) => ({ level: label }),
    bindings: (bindings) => ({
      pid: bindings.pid,
      host: bindings.hostname,
      service: 'media-bridge',
      version: process.env.npm_package_version || '1.0.0',
    }),
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  transport:
    config.NODE_ENV === 'development'
      ? {
          target: 'pino-pretty',
          options: { colorize: true, translateTime: 'SYS:standard' },
        }
      : undefined,
});

export type Logger = typeof logger;
