import type { BuilderEngine } from '../client';
import type { User } from '../models';

export class UsersResource {
  constructor(private readonly client: BuilderEngine) {}

  async getMe(): Promise<User> {
    return this.client.request({ method: 'GET', path: '/api/v1/users/me' });
  }

  async updateProfile(data: Partial<User>): Promise<User> {
    return this.client.request({ method: 'PATCH', path: '/api/v1/users/me/profile', body: data });
  }

  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    await this.client.request({ method: 'POST', path: '/api/v1/users/me/password', body: { current_password: currentPassword, new_password: newPassword } });
  }
}
