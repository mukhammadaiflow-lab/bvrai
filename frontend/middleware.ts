/**
 * Next.js Middleware for Authentication
 *
 * Handles auth token management via httpOnly cookies for security.
 * Redirects unauthenticated users from protected routes.
 */

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Routes that require authentication
const PROTECTED_ROUTES = [
  '/dashboard',
  '/agents',
  '/calls',
  '/analytics',
  '/settings',
  '/campaigns',
  '/contacts',
  '/phone-numbers',
  '/recordings',
  '/workflows',
  '/team-management',
  '/billing',
  '/integrations',
  '/knowledge-base',
  '/scheduled-jobs',
  '/custom-reports',
  '/ab-testing',
];

// Routes that should redirect to dashboard if already authenticated
const AUTH_ROUTES = ['/auth/login', '/auth/register', '/login', '/register'];

// Public routes that don't require authentication
const PUBLIC_ROUTES = ['/', '/api/health', '/api/auth'];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Skip middleware for static files and API routes (except auth)
  if (
    pathname.startsWith('/_next') ||
    pathname.startsWith('/static') ||
    pathname.includes('.') ||
    pathname.startsWith('/api/health')
  ) {
    return NextResponse.next();
  }

  // Get auth token from cookie
  const token = request.cookies.get('auth_token')?.value;
  const isAuthenticated = !!token;

  // Check if current path is protected
  const isProtectedRoute = PROTECTED_ROUTES.some(
    (route) => pathname === route || pathname.startsWith(`${route}/`)
  );

  // Check if current path is an auth route
  const isAuthRoute = AUTH_ROUTES.some(
    (route) => pathname === route || pathname.startsWith(`${route}/`)
  );

  // Redirect unauthenticated users from protected routes to login
  if (isProtectedRoute && !isAuthenticated) {
    const loginUrl = new URL('/auth/login', request.url);
    loginUrl.searchParams.set('redirect', pathname);
    return NextResponse.redirect(loginUrl);
  }

  // Redirect authenticated users from auth routes to dashboard
  if (isAuthRoute && isAuthenticated) {
    return NextResponse.redirect(new URL('/dashboard', request.url));
  }

  // For API requests, forward the auth token in the Authorization header
  if (pathname.startsWith('/api/') && token) {
    const requestHeaders = new Headers(request.headers);
    requestHeaders.set('Authorization', `Bearer ${token}`);

    return NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    });
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|public/).*)',
  ],
};
