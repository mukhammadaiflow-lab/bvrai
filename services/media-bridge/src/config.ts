/**
 * Configuration module for Media Bridge / Media Plane Service
 *
 * This is our self-hosted WebSocket-based media plane.
 * It handles rooms, participants, and audio/transcript routing.
 */
import { z } from 'zod';

const configSchema = z.object({
  // Server configuration
  PORT: z.string().default('3002'),
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),

  // JWT Configuration (must match Token Service)
  // TODO: For production, consider using asymmetric keys or shared secret store
  JWT_SECRET: z.string().min(32, 'JWT_SECRET must be at least 32 characters'),

  // Dialog Manager service
  DIALOG_MANAGER_URL: z.string().default('http://localhost:3003'),

  // Room settings
  MAX_ROOMS: z.string().default('1000'),
  MAX_PARTICIPANTS_PER_ROOM: z.string().default('50'),
  ROOM_IDLE_TIMEOUT_MS: z.string().default('300000'), // 5 minutes

  // Rate limiting
  RATE_LIMIT_WINDOW_MS: z.string().default('60000'),
  RATE_LIMIT_MAX_REQUESTS: z.string().default('100'),

  // CORS
  CORS_ORIGINS: z.string().default('*'),
});

export type Config = z.infer<typeof configSchema>;

function loadConfig(): Config {
  const result = configSchema.safeParse(process.env);

  if (!result.success) {
    console.error('Configuration validation failed:');
    console.error(result.error.format());

    if (process.env.NODE_ENV === 'development' || process.env.NODE_ENV === 'test') {
      console.warn('Using development defaults');
      return {
        PORT: process.env.PORT || '3002',
        NODE_ENV: (process.env.NODE_ENV as Config['NODE_ENV']) || 'development',
        JWT_SECRET: process.env.JWT_SECRET || 'dev_secret_key_must_be_32_chars!',
        DIALOG_MANAGER_URL: process.env.DIALOG_MANAGER_URL || 'http://localhost:3003',
        MAX_ROOMS: '1000',
        MAX_PARTICIPANTS_PER_ROOM: '50',
        ROOM_IDLE_TIMEOUT_MS: '300000',
        RATE_LIMIT_WINDOW_MS: '60000',
        RATE_LIMIT_MAX_REQUESTS: '100',
        CORS_ORIGINS: '*',
      };
    }
    throw new Error('Invalid configuration');
  }

  return result.data;
}

export const config = loadConfig();
