/**
 * Login API Route
 *
 * Proxies login requests to the backend and sets httpOnly cookies
 * for secure token storage.
 */

import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8086/api/v1';

// Cookie settings for security
const COOKIE_OPTIONS = {
  httpOnly: true,
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'lax' as const,
  path: '/',
  maxAge: 60 * 60 * 24 * 7, // 7 days
};

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, password } = body;

    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email and password are required' },
        { status: 400 }
      );
    }

    // Forward login request to backend
    const backendResponse = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    const data = await backendResponse.json();

    if (!backendResponse.ok) {
      return NextResponse.json(
        { error: data.detail || data.message || 'Login failed' },
        { status: backendResponse.status }
      );
    }

    // Extract token (backend may return it as 'token' or 'access_token')
    const token = data.token || data.access_token;

    if (!token) {
      return NextResponse.json(
        { error: 'No token received from server' },
        { status: 500 }
      );
    }

    // Create response with user data (without token in body for security)
    const response = NextResponse.json({
      user: data.user,
      message: 'Login successful',
    });

    // Set httpOnly cookie with the token
    response.cookies.set('auth_token', token, COOKIE_OPTIONS);

    return response;
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
