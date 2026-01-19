import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Builder Voice AI - Dashboard",
  description: "Build and manage AI voice agents for your business",
  keywords: ["AI", "voice agents", "conversational AI", "phone automation"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
