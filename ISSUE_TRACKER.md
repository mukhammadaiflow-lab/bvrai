# BVRAI Issue Tracker
## Complete List of All Identified Issues

**Total Issues:** 210+
**Last Updated:** January 16, 2026

---

## Issue Severity Legend

| Severity | Description | Response Time |
|----------|-------------|---------------|
| CRITICAL | Platform non-functional, security breach risk | Immediate |
| HIGH | Major feature broken, significant security risk | 24-48 hours |
| MEDIUM | Feature degraded, minor security concern | 1 week |
| LOW | Enhancement, code quality | 2-4 weeks |

---

## Phase 0: Emergency Fixes (BLOCKING)

### P0-001: Python Module Namespace Collision
- **Severity:** CRITICAL
- **Component:** Backend
- **File:** `/platform/` (entire directory)
- **Issue:** Folder name conflicts with Python's built-in `platform` module
- **Impact:** Entire backend non-functional
- **Fix:** Rename to `bvrai_core/` and update all imports
- **Status:** Pending
- **Assigned Phase:** 0

### P0-002: Missing Frontend Lib Utils
- **Severity:** CRITICAL
- **Component:** Frontend
- **File:** `frontend/lib/utils.ts`
- **Issue:** File does not exist but is imported by 29 files
- **Impact:** Frontend won't compile
- **Fix:** Create utils.ts with required functions
- **Status:** Pending
- **Assigned Phase:** 0

### P0-003: Missing Frontend API Client
- **Severity:** CRITICAL
- **Component:** Frontend
- **File:** `frontend/lib/api.ts`
- **Issue:** File does not exist but is imported by all hooks
- **Impact:** Frontend won't compile
- **Fix:** Create api.ts with axios client and API modules
- **Status:** Pending
- **Assigned Phase:** 0

### P0-004: Missing React Query Provider
- **Severity:** CRITICAL
- **Component:** Frontend
- **File:** `frontend/app/layout.tsx`
- **Issue:** No QueryClientProvider wrapping app
- **Impact:** All React Query hooks fail
- **Fix:** Create providers.tsx and wrap app
- **Status:** Pending
- **Assigned Phase:** 0

### P0-005: Next.js Security Vulnerability
- **Severity:** CRITICAL
- **Component:** Frontend
- **File:** `frontend/package.json`
- **Line:** 7
- **Issue:** Next.js 14.1.0 has known CVE
- **Impact:** Security vulnerability
- **Fix:** Upgrade to Next.js 14.2.28+
- **Status:** Pending
- **Assigned Phase:** 0

### P0-006: CORS Misconfiguration
- **Severity:** CRITICAL
- **Component:** Backend/Security
- **File:** `platform/api/app.py`
- **Line:** 75, 242-247
- **Issue:** `allow_origins=["*"]` with `allow_credentials=True`
- **Impact:** Any website can make authenticated requests
- **Fix:** Use explicit origin whitelist
- **Status:** Pending
- **Assigned Phase:** 0

---

## Phase 1: Backend Issues

### BE-001: Missing Await on Async Call
- **Severity:** HIGH
- **Component:** Backend
- **File:** `platform/voice_engine/pipeline.py`
- **Line:** 520
- **Issue:** `VADFactory.create()` called without await
- **Impact:** Coroutine stored instead of result
- **Fix:** Add `await`
- **Status:** Pending
- **Assigned Phase:** 1

### BE-002: Missing Await on Async Call
- **Severity:** HIGH
- **Component:** Backend
- **File:** `platform/voice_engine/pipeline.py`
- **Line:** 542
- **Issue:** `STTProviderFactory.create()` called without await
- **Impact:** Coroutine stored instead of result
- **Fix:** Add `await`
- **Status:** Pending
- **Assigned Phase:** 1

### BE-003: Wrong Type Passed to AudioBuffer
- **Severity:** HIGH
- **Component:** Backend
- **File:** `platform/voice_engine/interruption.py`
- **Line:** 166-169
- **Issue:** Passing `int` instead of `AudioFormat` object
- **Impact:** AudioBuffer initialization fails
- **Fix:** Create proper AudioFormat object
- **Status:** Pending
- **Assigned Phase:** 1

### BE-004: Dead Code Pattern
- **Severity:** LOW
- **Component:** Backend
- **File:** `platform/voice_engine/interruption.py`
- **Line:** 166
- **Issue:** `if True else None` is always True
- **Impact:** Code quality, confusing intent
- **Fix:** Remove dead branch
- **Status:** Pending
- **Assigned Phase:** 1

### BE-005: Race Condition in Audio Buffer
- **Severity:** HIGH
- **Component:** Backend
- **File:** `platform/voice_engine/audio.py`
- **Line:** 270-318
- **Issue:** Calculations outside lock can race
- **Impact:** Data corruption in multi-threaded use
- **Fix:** Expand lock scope
- **Status:** Pending
- **Assigned Phase:** 1

### BE-006: Silent Exception Swallowing
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/voice_engine/audio.py`
- **Line:** 180-181
- **Issue:** `except Exception: return 0.0`
- **Impact:** Bugs masked, debugging impossible
- **Fix:** Log exception before return
- **Status:** Pending
- **Assigned Phase:** 1

### BE-007: Silent Exception Swallowing
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/voice_engine/audio.py`
- **Line:** 190-191
- **Issue:** `except Exception: return 0`
- **Impact:** Bugs masked
- **Fix:** Log exception
- **Status:** Pending
- **Assigned Phase:** 1

### BE-008: Silent Exception Swallowing
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/voice_engine/audio.py`
- **Line:** 762-763
- **Issue:** `except Exception: return 0.0`
- **Impact:** Bugs masked
- **Fix:** Log exception
- **Status:** Pending
- **Assigned Phase:** 1

### BE-009: Silent Exception Swallowing
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/llm/context.py`
- **Line:** 71
- **Issue:** `except Exception: pass`
- **Impact:** Bugs masked
- **Fix:** Log exception
- **Status:** Pending
- **Assigned Phase:** 1

### BE-010: Division by Zero Risk
- **Severity:** HIGH
- **Component:** Backend
- **File:** `platform/core/engine.py`
- **Line:** 497
- **Issue:** `max_context_tokens` could be 0
- **Impact:** Runtime crash
- **Fix:** Add zero check
- **Status:** Pending
- **Assigned Phase:** 1

### BE-011: Blocking I/O in Async Context
- **Severity:** HIGH
- **Component:** Backend
- **File:** `platform/llm/providers.py`
- **Line:** 620-623
- **Issue:** `asyncio.to_thread` blocks thread pool
- **Impact:** Performance degradation
- **Fix:** Use native async API
- **Status:** Pending
- **Assigned Phase:** 1

### BE-012: Repeated Model Creation
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/llm/providers.py`
- **Line:** 676-726
- **Issue:** `GenerativeModel` created on every call
- **Impact:** Performance overhead
- **Fix:** Cache model instance
- **Status:** Pending
- **Assigned Phase:** 1

### BE-013: String Concatenation in Loop
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/voice_engine/pipeline.py`
- **Line:** 1070
- **Issue:** `response_text += chunk` is O(n²)
- **Impact:** Performance degradation
- **Fix:** Use list and join
- **Status:** Pending
- **Assigned Phase:** 1

### BE-014: No Connection Pooling
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/voice_engine/stt.py`
- **Issue:** New ClientSession per request
- **Impact:** Connection overhead
- **Fix:** Reuse session
- **Status:** Pending
- **Assigned Phase:** 1

### BE-015: No Connection Pooling
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/voice_engine/tts.py`
- **Issue:** New ClientSession per request
- **Impact:** Connection overhead
- **Fix:** Reuse session
- **Status:** Pending
- **Assigned Phase:** 1

### BE-016: Unbounded Cache Growth
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/voice_engine/tts.py`
- **Line:** 285-289
- **Issue:** Only 100 items removed when cache full
- **Impact:** Memory leak
- **Fix:** Proper LRU eviction
- **Status:** Pending
- **Assigned Phase:** 1

### BE-017: Magic Numbers
- **Severity:** LOW
- **Component:** Backend
- **File:** `platform/voice_engine/vad.py`
- **Line:** 620
- **Issue:** `0.7` threshold without constant
- **Impact:** Maintainability
- **Fix:** Define named constant
- **Status:** Pending
- **Assigned Phase:** 1

### BE-018: Duplicate Code
- **Severity:** LOW
- **Component:** Backend
- **File:** `platform/llm/providers.py`
- **Line:** 76-81, 270-316, 544-570
- **Issue:** `_convert_messages()` duplicated
- **Impact:** Maintainability
- **Fix:** Extract to base class
- **Status:** Pending
- **Assigned Phase:** 1

### BE-019: Missing Timeout Fallback
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/voice_engine/stt.py`
- **Line:** 374-375
- **Issue:** No fallback if timeout is 0/negative
- **Impact:** Potential hang
- **Fix:** Add validation
- **Status:** Pending
- **Assigned Phase:** 1

### BE-020: Incomplete SIP Implementation
- **Severity:** HIGH
- **Component:** Backend
- **File:** `platform/telephony/sip.py`
- **Line:** 495, 597
- **Issue:** TODO comments in production code
- **Impact:** SIP features incomplete
- **Fix:** Implement or remove
- **Status:** Pending
- **Assigned Phase:** 1

### BE-021: Misleading Implementation
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/llm/context.py`
- **Line:** 471-474
- **Issue:** Semantic strategy just calls priority
- **Impact:** Feature doesn't work as documented
- **Fix:** Implement or rename
- **Status:** Pending
- **Assigned Phase:** 1

### BE-022: WebSocket Leak Risk
- **Severity:** HIGH
- **Component:** Backend
- **File:** `platform/voice_engine/stt.py`
- **Line:** 352-357
- **Issue:** Exception between set and close leaks
- **Impact:** Resource leak
- **Fix:** Use try/finally
- **Status:** Pending
- **Assigned Phase:** 1

### BE-023: Task Cleanup Incomplete
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/voice_engine/pipeline.py`
- **Line:** 704-712
- **Issue:** Tasks may not clean up on timeout
- **Impact:** Resource leak
- **Fix:** Add timeout to gather
- **Status:** Pending
- **Assigned Phase:** 1

### BE-024: Unvalidated Input in Token Count
- **Severity:** MEDIUM
- **Component:** Backend
- **File:** `platform/llm/context.py`
- **Line:** 64
- **Issue:** No validation of model parameter
- **Impact:** Potential error
- **Fix:** Add validation
- **Status:** Pending
- **Assigned Phase:** 1

### BE-025: Inconsistent Null Handling
- **Severity:** LOW
- **Component:** Backend
- **File:** `platform/voice_engine/interruption.py`
- **Line:** 259
- **Issue:** Sometimes `if self.vad`, sometimes `if not self._vad`
- **Impact:** Code clarity
- **Fix:** Standardize
- **Status:** Pending
- **Assigned Phase:** 1

### BE-026 through BE-085: Additional Backend Issues
*(60 more issues with similar detail - abbreviated for space)*

- Missing logging context (structured logging)
- Inconsistent error handling patterns
- Missing type hints (10+ locations)
- Pass statements without @abstractmethod
- Circular dependency risks
- Missing input validation
- Unencrypted audio buffers
- API key exposure in logs
- Race conditions (3 more locations)
- Resource leaks (2 more locations)

---

## Phase 2: Security Issues

### SEC-001: JWT Secret Generated at Runtime
- **Severity:** CRITICAL
- **Component:** Security
- **File:** `platform/auth/session.py`
- **Line:** 47
- **Issue:** Random secret per instance
- **Impact:** Distributed systems fail
- **Fix:** Load from environment
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-002: Weak Password Hashing
- **Severity:** CRITICAL
- **Component:** Security
- **File:** `platform/auth/manager.py`
- **Line:** 1133-1164
- **Issue:** PBKDF2 with string salt
- **Impact:** Password security compromised
- **Fix:** Use bcrypt
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-003: Timing Attack Vulnerability
- **Severity:** CRITICAL
- **Component:** Security
- **File:** `platform/auth/session.py`
- **Line:** 180-182
- **Issue:** Direct string comparison
- **Impact:** Token can be extracted
- **Fix:** Use hmac.compare_digest
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-004: No MFA Enforcement
- **Severity:** CRITICAL
- **Component:** Security
- **File:** `platform/auth/manager.py`
- **Issue:** MFA fields exist but not enforced
- **Impact:** Sensitive operations unprotected
- **Fix:** Add MFA challenges
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-005: Custom JWT Implementation
- **Severity:** CRITICAL
- **Component:** Security
- **File:** `platform/auth/session.py`
- **Line:** 462-556
- **Issue:** Custom JWT without algorithm validation
- **Impact:** "none" algorithm attack possible
- **Fix:** Use PyJWT library
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-006: Credential Default Allow
- **Severity:** HIGH
- **Component:** Security
- **File:** `platform/security/credentials.py`
- **Line:** 720-747
- **Issue:** No policies = allow all
- **Impact:** Credentials accessible to anyone
- **Fix:** Default to deny
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-007: API Key Wildcard Scope
- **Severity:** HIGH
- **Component:** Security
- **File:** `platform/auth/session.py`
- **Line:** 355-409
- **Issue:** "*" scope grants all permissions
- **Impact:** Over-privileged keys
- **Fix:** Remove wildcard support
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-008: No Auth Rate Limiting
- **Severity:** HIGH
- **Component:** Security
- **File:** `platform/api/app.py`
- **Line:** 265-275
- **Issue:** Auth endpoints use global limits
- **Impact:** Brute force possible
- **Fix:** Add per-endpoint limits
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-009: SSO Attribute Injection
- **Severity:** HIGH
- **Component:** Security
- **File:** `platform/auth/manager.py`
- **Line:** 438-449
- **Issue:** No validation of extracted attributes
- **Impact:** Injection attacks
- **Fix:** Validate and sanitize
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-010: Path Traversal in Slug
- **Severity:** HIGH
- **Component:** Security
- **File:** `platform/auth/manager.py`
- **Line:** 647-650
- **Issue:** Slug only lowercased, not validated
- **Impact:** Path traversal attack
- **Fix:** Regex validation
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-011: Open Redirect
- **Severity:** MEDIUM
- **Component:** Security
- **File:** `platform/auth/manager.py`
- **Line:** 348, 370
- **Issue:** Redirect URI not whitelisted
- **Impact:** Phishing attacks
- **Fix:** Validate against whitelist
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-012: SSRF in Discovery
- **Severity:** MEDIUM
- **Component:** Security
- **File:** `platform/auth/oidc.py`, `saml.py`
- **Issue:** No internal IP filtering
- **Impact:** Internal network access
- **Fix:** Block private IP ranges
- **Status:** Pending
- **Assigned Phase:** 2

### SEC-013 through SEC-026: Additional Security Issues
*(14 more issues with similar detail)*

---

## Phase 1: Frontend Issues

### FE-001: Missing Utils Module
- **Severity:** CRITICAL
- **Component:** Frontend
- **File:** `frontend/lib/utils.ts`
- **Issue:** File missing, 29 imports fail
- **Status:** Pending
- **Assigned Phase:** 0

### FE-002: Missing API Module
- **Severity:** CRITICAL
- **Component:** Frontend
- **File:** `frontend/lib/api.ts`
- **Issue:** File missing, all hooks fail
- **Status:** Pending
- **Assigned Phase:** 0

### FE-003: Badge className Error
- **Severity:** HIGH
- **Component:** Frontend
- **File:** `frontend/app/agents/page.tsx`
- **Line:** 181
- **Issue:** className not in BadgeProps
- **Fix:** Add to interface
- **Status:** Pending
- **Assigned Phase:** 1

### FE-004: Unknown Type Error
- **Severity:** HIGH
- **Component:** Frontend
- **File:** `frontend/hooks/useBilling.ts`
- **Line:** 62-63
- **Issue:** data.checkout_url on unknown
- **Fix:** Add type guard
- **Status:** Pending
- **Assigned Phase:** 1

### FE-005: Blob Type Error
- **Severity:** HIGH
- **Component:** Frontend
- **File:** `frontend/hooks/useVoiceConfig.ts`
- **Line:** 108
- **Issue:** Unknown not assignable to Blob
- **Fix:** Type assertion
- **Status:** Pending
- **Assigned Phase:** 1

### FE-006: Missing Error Boundary
- **Severity:** HIGH
- **Component:** Frontend
- **File:** `frontend/app/layout.tsx`
- **Issue:** No error boundary
- **Impact:** App crashes on error
- **Fix:** Add ErrorBoundary component
- **Status:** Pending
- **Assigned Phase:** 1

### FE-007: Missing Loading States
- **Severity:** MEDIUM
- **Component:** Frontend
- **File:** Multiple pages
- **Issue:** No loading skeletons
- **Impact:** Poor UX
- **Fix:** Add Suspense/loading
- **Status:** Pending
- **Assigned Phase:** 1

### FE-008: Hardcoded Mock Data
- **Severity:** HIGH
- **Component:** Frontend
- **File:** `frontend/app/dashboard/page.tsx`
- **Line:** 20-85
- **Issue:** Mock data never replaced
- **Impact:** No real data
- **Fix:** Use React Query hooks
- **Status:** Pending
- **Assigned Phase:** 1

### FE-009: localStorage SSR Issue
- **Severity:** HIGH
- **Component:** Frontend
- **File:** `frontend/app/page.tsx`
- **Line:** 11
- **Issue:** localStorage access without SSR guard
- **Impact:** Hydration mismatch
- **Fix:** Check window exists
- **Status:** Pending
- **Assigned Phase:** 1

### FE-010: Token in WebSocket URL
- **Severity:** HIGH
- **Component:** Frontend/Security
- **File:** `frontend/hooks/useWebSocket.ts`
- **Line:** 50
- **Issue:** Auth token passed in URL
- **Impact:** Token in logs/history
- **Fix:** Use headers or first message
- **Status:** Pending
- **Assigned Phase:** 1

### FE-011: Missing Logout Handler
- **Severity:** MEDIUM
- **Component:** Frontend
- **File:** `frontend/components/layouts/sidebar.tsx`
- **Line:** 183-185
- **Issue:** Logout button has no onClick
- **Impact:** Cannot log out
- **Fix:** Add handler
- **Status:** Pending
- **Assigned Phase:** 1

### FE-012: No Form Validation
- **Severity:** MEDIUM
- **Component:** Frontend
- **File:** `frontend/app/auth/login/page.tsx`
- **Line:** 64-91
- **Issue:** No email/password validation
- **Impact:** Poor UX
- **Fix:** Add Zod validation
- **Status:** Pending
- **Assigned Phase:** 1

### FE-013 through FE-050: Additional Frontend Issues
*(37 more issues including accessibility, performance, i18n)*

---

## SDK Issues

### SDK-001: Duplicate Python SDKs
- **Severity:** HIGH
- **Component:** SDK
- **File:** `sdk/python/` and `sdks/python/`
- **Issue:** Two conflicting implementations
- **Fix:** Archive old, use new
- **Status:** Pending
- **Assigned Phase:** 1

### SDK-002: Duplicate TypeScript SDKs
- **Severity:** HIGH
- **Component:** SDK
- **File:** `sdk/javascript/` and `sdks/typescript/`
- **Issue:** Two conflicting implementations
- **Fix:** Archive old, use new
- **Status:** Pending
- **Assigned Phase:** 1

### SDK-003: Missing Rate Limit Auto-Retry
- **Severity:** MEDIUM
- **Component:** SDK
- **File:** All SDKs
- **Issue:** 429 throws instead of retrying
- **Fix:** Auto-retry with Retry-After
- **Status:** Pending
- **Assigned Phase:** 3

### SDK-004 through SDK-020: Additional SDK Issues
*(16 more issues)*

---

## Infrastructure Issues

### INFRA-001: Missing Pod Security Policy
- **Severity:** CRITICAL
- **Component:** Infrastructure
- **File:** `deploy/kubernetes/namespace.yaml`
- **Issue:** No PSP enforcement
- **Impact:** Containers can run as root
- **Fix:** Add restricted PSP
- **Status:** Pending
- **Assigned Phase:** 4

### INFRA-002: Public EKS Endpoint
- **Severity:** CRITICAL
- **Component:** Infrastructure
- **File:** `deploy/terraform/modules/eks/main.tf`
- **Line:** 160-161
- **Issue:** API exposed to internet
- **Fix:** Restrict to VPN IPs
- **Status:** Pending
- **Assigned Phase:** 4

### INFRA-003: Missing Container Scanning
- **Severity:** CRITICAL
- **Component:** Infrastructure
- **File:** `.github/workflows/ci.yml`
- **Issue:** No vulnerability scanning
- **Fix:** Add Trivy scanning
- **Status:** Pending
- **Assigned Phase:** 4

### INFRA-004 through INFRA-030: Additional Infrastructure Issues
*(26 more issues)*

---

## Issue Statistics

### By Component
| Component | Critical | High | Medium | Low | Total |
|-----------|----------|------|--------|-----|-------|
| Backend | 5 | 20 | 40 | 20 | 85 |
| Frontend | 4 | 15 | 20 | 11 | 50 |
| Security | 6 | 12 | 8 | 0 | 26 |
| SDK | 2 | 8 | 8 | 2 | 20 |
| Infrastructure | 3 | 10 | 12 | 5 | 30 |
| **Total** | **20** | **65** | **88** | **38** | **211** |

### By Phase
| Phase | Issues | Status |
|-------|--------|--------|
| Phase 0 | 6 | Pending |
| Phase 1 | 85 | Pending |
| Phase 2 | 26 | Pending |
| Phase 3 | 30 | Pending |
| Phase 4 | 40 | Pending |
| Phase 5 | 15 | Pending |
| Phase 6 | 9 | Pending |

---

## How to Use This Tracker

1. **Find Issue:** Search by ID (e.g., BE-001) or component
2. **Check Status:** Pending → In Progress → Fixed → Verified
3. **Update Status:** Edit this file when fixing issues
4. **Link Commits:** Reference issue ID in commit messages

```bash
git commit -m "fix(backend): resolve race condition in audio buffer

Fixes BE-005"
```

---

*Last Updated: January 16, 2026*
