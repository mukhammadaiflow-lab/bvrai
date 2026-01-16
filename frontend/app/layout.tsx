import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers, ErrorBoundary } from "./providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Builder Voice AI - Dashboard",
  description: "Build and manage AI voice agents for your business",
  keywords: ["AI", "voice agents", "conversational AI", "phone automation"],
  authors: [{ name: "BVRAI Team" }],
  robots: "noindex, nofollow", // Dashboard should not be indexed
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ErrorBoundary>
          <Providers>{children}</Providers>
        </ErrorBoundary>
      </body>
    </html>
  );
}
