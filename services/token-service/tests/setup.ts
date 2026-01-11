/**
 * Jest test setup
 * Sets up environment variables for testing
 */

// Set test environment variables before importing any modules
process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-jwt-secret-must-be-32-characters!';
process.env.MEDIA_PLANE_WS_URL = 'ws://localhost:3002';
process.env.PORT = '3099';

// Suppress console output during tests
if (process.env.SUPPRESS_LOGS !== 'false') {
  jest.spyOn(console, 'log').mockImplementation(() => {});
  jest.spyOn(console, 'info').mockImplementation(() => {});
  jest.spyOn(console, 'warn').mockImplementation(() => {});
  jest.spyOn(console, 'error').mockImplementation(() => {});
}
