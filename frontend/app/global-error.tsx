"use client";

import { useEffect } from "react";
import { AlertOctagon, RefreshCw } from "lucide-react";

interface GlobalErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

/**
 * Global Error Handler
 *
 * This catches errors in the root layout and provides a minimal fallback.
 * It must define its own <html> and <body> tags since it replaces the root layout.
 */
export default function GlobalError({ error, reset }: GlobalErrorProps) {
  useEffect(() => {
    // Log to error reporting service
    console.error("[Global Error]", {
      message: error.message,
      digest: error.digest,
      stack: error.stack,
      timestamp: new Date().toISOString(),
    });
  }, [error]);

  return (
    <html lang="en">
      <body className="font-sans antialiased">
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#0f172a",
            padding: "24px",
          }}
        >
          <div
            style={{
              maxWidth: "400px",
              width: "100%",
              backgroundColor: "#1e293b",
              borderRadius: "12px",
              padding: "32px",
              textAlign: "center",
              border: "1px solid #ef4444",
            }}
          >
            <div
              style={{
                width: "64px",
                height: "64px",
                borderRadius: "50%",
                backgroundColor: "rgba(239, 68, 68, 0.1)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 16px",
              }}
            >
              <AlertOctagon style={{ width: "32px", height: "32px", color: "#ef4444" }} />
            </div>

            <h1 style={{ color: "#f8fafc", fontSize: "20px", fontWeight: "600", marginBottom: "8px" }}>
              Critical Error
            </h1>

            <p style={{ color: "#94a3b8", fontSize: "14px", marginBottom: "24px" }}>
              A critical error occurred. Please reload the page to continue.
            </p>

            {error.digest && (
              <p style={{ color: "#64748b", fontSize: "12px", marginBottom: "24px" }}>
                Error ID: {error.digest}
              </p>
            )}

            <button
              onClick={reset}
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "8px",
                backgroundColor: "#3b82f6",
                color: "#ffffff",
                padding: "12px 24px",
                borderRadius: "8px",
                border: "none",
                cursor: "pointer",
                fontSize: "14px",
                fontWeight: "500",
                width: "100%",
              }}
            >
              <RefreshCw style={{ width: "16px", height: "16px" }} />
              Reload Application
            </button>
          </div>
        </div>
      </body>
    </html>
  );
}
