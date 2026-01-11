/**
 * Jest test setup for Media Bridge
 */
process.env.NODE_ENV = 'test';
process.env.PORT = '3098';
process.env.JWT_SECRET = 'test-jwt-secret-must-be-32-characters!';
process.env.DIALOG_MANAGER_URL = 'http://localhost:3003';

// Suppress console output during tests
if (process.env.SUPPRESS_LOGS !== 'false') {
  jest.spyOn(console, 'log').mockImplementation(() => {});
  jest.spyOn(console, 'info').mockImplementation(() => {});
  jest.spyOn(console, 'warn').mockImplementation(() => {});
  jest.spyOn(console, 'error').mockImplementation(() => {});
}
