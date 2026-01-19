/**
 * Configuration module for Token Service
 * All secrets are read from environment variables
 *
 * This service issues tokens for our self-hosted media plane.
 */
import { z } from 'zod';

const configSchema = z.object({
  // Server configuration
  PORT: z.string().default('3001'),
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),

  // Media Plane configuration (self-hosted)
  // TODO: For production, consider migrating to asymmetric keys (RS256) or KMS integration
  JWT_SECRET: z.string().min(32, 'JWT_SECRET must be at least 32 characters'),
  MEDIA_PLANE_WS_URL: z.string().default('ws://localhost:3002'),

  // Token defaults
  DEFAULT_TOKEN_TTL_SECONDS: z.string().default('3600'),
  MAX_TOKEN_TTL_SECONDS: z.string().default('86400'),

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

    // In development, provide helpful defaults
    if (process.env.NODE_ENV === 'development' || process.env.NODE_ENV === 'test') {
      console.warn('Using development defaults for missing config values');
      return {
        PORT: process.env.PORT || '3001',
        NODE_ENV: (process.env.NODE_ENV as Config['NODE_ENV']) || 'development',
        JWT_SECRET: process.env.JWT_SECRET || 'dev_secret_key_must_be_32_chars!',
        MEDIA_PLANE_WS_URL: process.env.MEDIA_PLANE_WS_URL || 'ws://localhost:3002',
        DEFAULT_TOKEN_TTL_SECONDS: '3600',
        MAX_TOKEN_TTL_SECONDS: '86400',
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
