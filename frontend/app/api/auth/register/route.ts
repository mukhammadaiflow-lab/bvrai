/**
 * Register API Route
 *
 * Proxies registration requests to the backend and sets httpOnly cookies.
 */

import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8086/api/v1';

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
    const { email, password, name, organization_name } = body;

    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email and password are required' },
        { status: 400 }
      );
    }

    // Forward registration request to backend
    const backendResponse = await fetch(`${API_BASE_URL}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password, name, organization_name }),
    });

    const data = await backendResponse.json();

    if (!backendResponse.ok) {
      return NextResponse.json(
        { error: data.detail || data.message || 'Registration failed' },
        { status: backendResponse.status }
      );
    }

    // Extract token
    const token = data.token || data.access_token;

    if (!token) {
      return NextResponse.json(
        { error: 'No token received from server' },
        { status: 500 }
      );
    }

    // Create response with user data
    const response = NextResponse.json({
      user: data.user,
      message: 'Registration successful',
    });

    // Set httpOnly cookie
    response.cookies.set('auth_token', token, COOKIE_OPTIONS);

    return response;
  } catch (error) {
    console.error('Registration error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
