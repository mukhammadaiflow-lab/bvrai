/**
 * API Proxy Route
 *
 * Proxies requests to the backend API, automatically including
 * the auth token from the httpOnly cookie.
 */

import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8086/api/v1';

async function proxyRequest(
  request: NextRequest,
  method: string,
  params: { path: string[] }
) {
  try {
    const token = request.cookies.get('auth_token')?.value;
    const path = params.path.join('/');
    const url = new URL(request.url);
    const queryString = url.search;

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    let body: string | undefined;
    if (method !== 'GET' && method !== 'HEAD') {
      try {
        body = await request.text();
      } catch {
        // No body
      }
    }

    const backendResponse = await fetch(`${API_BASE_URL}/${path}${queryString}`, {
      method,
      headers,
      body: body || undefined,
    });

    // Handle different response types
    const contentType = backendResponse.headers.get('content-type');

    if (contentType?.includes('application/json')) {
      const data = await backendResponse.json();
      return NextResponse.json(data, { status: backendResponse.status });
    }

    if (contentType?.includes('audio/') || contentType?.includes('application/octet-stream')) {
      const blob = await backendResponse.blob();
      return new NextResponse(blob, {
        status: backendResponse.status,
        headers: {
          'Content-Type': contentType,
        },
      });
    }

    const text = await backendResponse.text();
    return new NextResponse(text, {
      status: backendResponse.status,
      headers: {
        'Content-Type': contentType || 'text/plain',
      },
    });
  } catch (error) {
    console.error('Proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to connect to backend' },
      { status: 502 }
    );
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, 'GET', params);
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, 'POST', params);
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, 'PUT', params);
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, 'PATCH', params);
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, 'DELETE', params);
}
