/**
 * Current User API Route
 *
 * Returns the currently authenticated user by proxying to the backend.
 */

import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8086/api/v1';

export async function GET(request: NextRequest) {
  try {
    const token = request.cookies.get('auth_token')?.value;

    if (!token) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }

    // Forward request to backend
    const backendResponse = await fetch(`${API_BASE_URL}/auth/me`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    if (!backendResponse.ok) {
      // If token is invalid, clear the cookie
      if (backendResponse.status === 401) {
        const response = NextResponse.json(
          { error: 'Session expired' },
          { status: 401 }
        );
        response.cookies.set('auth_token', '', {
          httpOnly: true,
          secure: process.env.NODE_ENV === 'production',
          sameSite: 'lax',
          path: '/',
          maxAge: 0,
        });
        return response;
      }

      const data = await backendResponse.json();
      return NextResponse.json(
        { error: data.detail || data.message || 'Failed to get user' },
        { status: backendResponse.status }
      );
    }

    const user = await backendResponse.json();
    return NextResponse.json(user);
  } catch (error) {
    console.error('Get current user error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
