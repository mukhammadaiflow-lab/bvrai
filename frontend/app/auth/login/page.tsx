"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Bot, Eye, EyeOff, Loader2, Sparkles, ArrowRight, Mic, Phone, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      localStorage.setItem("auth_token", "demo_token");
      router.push("/dashboard");
    } catch (err) {
      setError("Invalid email or password");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex overflow-hidden">
      {/* Left Side - Form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-background relative">
        {/* Subtle background pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#8080800a_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:24px_24px]" />

        <div
          className={cn(
            "w-full max-w-[440px] relative z-10 transition-all duration-700",
            mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
          )}
        >
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3 mb-12 group">
            <div className="relative">
              <div className="absolute -inset-2 bg-gradient-to-r from-primary/20 to-accent/20 rounded-2xl blur-lg group-hover:blur-xl transition-all opacity-0 group-hover:opacity-100" />
              <div className="relative flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-accent shadow-lg shadow-primary/25">
                <Bot className="h-6 w-6 text-white" />
              </div>
            </div>
            <div>
              <span className="font-bold text-2xl bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                BVRAI
              </span>
              <span className="block text-xs text-muted-foreground -mt-0.5">Voice AI Platform</span>
            </div>
          </Link>

          {/* Heading */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold tracking-tight mb-2">
              Welcome back
            </h1>
            <p className="text-muted-foreground text-lg">
              Sign in to continue building voice AI experiences
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <div className="rounded-xl bg-destructive/10 border border-destructive/20 p-4 text-sm text-destructive flex items-center gap-3 animate-shake">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-destructive/20 flex items-center justify-center">
                  <span className="text-lg">!</span>
                </div>
                <span>{error}</span>
              </div>
            )}

            <div className="space-y-2">
              <label htmlFor="email" className="text-sm font-medium">
                Email address
              </label>
              <Input
                id="email"
                type="email"
                placeholder="you@company.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="h-12 rounded-xl border-2 border-border/50 bg-muted/30 px-4 text-base transition-all focus:border-primary focus:bg-background focus:ring-4 focus:ring-primary/10"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label htmlFor="password" className="text-sm font-medium">
                  Password
                </label>
                <Link
                  href="/auth/forgot-password"
                  className="text-sm text-primary hover:text-primary/80 transition-colors"
                >
                  Forgot password?
                </Link>
              </div>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="h-12 rounded-xl border-2 border-border/50 bg-muted/30 px-4 pr-12 text-base transition-all focus:border-primary focus:bg-background focus:ring-4 focus:ring-primary/10"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            <Button
              type="submit"
              disabled={loading}
              className="w-full h-12 rounded-xl text-base font-semibold bg-gradient-to-r from-primary to-primary/90 hover:from-primary/90 hover:to-primary shadow-lg shadow-primary/25 hover:shadow-xl hover:shadow-primary/30 transition-all"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Signing in...
                </>
              ) : (
                <>
                  Sign in
                  <ArrowRight className="ml-2 h-5 w-5" />
                </>
              )}
            </Button>
          </form>

          {/* Divider */}
          <div className="relative my-8">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t border-border/50" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-background px-4 text-muted-foreground font-medium">
                Or continue with
              </span>
            </div>
          </div>

          {/* Social Login */}
          <div className="grid grid-cols-2 gap-4">
            <Button
              variant="outline"
              className="h-12 rounded-xl border-2 hover:border-primary/30 hover:bg-primary/5 transition-all"
            >
              <svg className="mr-2 h-5 w-5" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
              Google
            </Button>
            <Button
              variant="outline"
              className="h-12 rounded-xl border-2 hover:border-primary/30 hover:bg-primary/5 transition-all"
            >
              <svg className="mr-2 h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              GitHub
            </Button>
          </div>

          {/* Sign Up Link */}
          <p className="mt-8 text-center text-muted-foreground">
            Don't have an account?{" "}
            <Link
              href="/auth/register"
              className="font-semibold text-primary hover:text-primary/80 transition-colors"
            >
              Create free account
            </Link>
          </p>
        </div>
      </div>

      {/* Right Side - Hero */}
      <div className="hidden lg:flex flex-1 relative overflow-hidden">
        {/* Animated Gradient Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary via-purple-600 to-accent" />

        {/* Animated Orbs */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-white/10 rounded-full blur-3xl animate-pulse-soft" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-accent/20 rounded-full blur-3xl animate-bounce-soft" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-white/5 rounded-full blur-3xl" />

        {/* Grid Pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:64px_64px]" />

        {/* Content */}
        <div className="relative z-10 flex flex-col justify-center px-16 text-white">
          <div className="mb-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 text-sm font-medium mb-6">
              <Sparkles className="h-4 w-4" />
              <span>AI-Powered Voice Platform</span>
            </div>

            <h2 className="text-5xl font-bold leading-tight mb-6">
              Build intelligent
              <br />
              <span className="text-white/90">voice experiences</span>
            </h2>

            <p className="text-xl text-white/70 max-w-md leading-relaxed">
              Create, deploy, and scale AI voice agents that understand and respond naturally to your customers.
            </p>
          </div>

          {/* Feature Pills */}
          <div className="flex flex-wrap gap-3 mb-12">
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/10">
              <Mic className="h-4 w-4" />
              <span className="text-sm">Natural Speech</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/10">
              <Phone className="h-4 w-4" />
              <span className="text-sm">Phone Integration</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/10">
              <Zap className="h-4 w-4" />
              <span className="text-sm">Real-time</span>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-8">
            <div>
              <div className="text-4xl font-bold mb-1">10M+</div>
              <div className="text-white/60 text-sm">Calls processed</div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-1">99.9%</div>
              <div className="text-white/60 text-sm">Uptime SLA</div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-1">&lt;200ms</div>
              <div className="text-white/60 text-sm">Response time</div>
            </div>
          </div>
        </div>

        {/* Decorative Elements */}
        <div className="absolute bottom-8 left-8 right-8">
          <div className="flex items-center gap-4 px-6 py-4 rounded-2xl bg-white/10 backdrop-blur-md border border-white/20">
            <div className="flex -space-x-2">
              {[1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="w-10 h-10 rounded-full bg-gradient-to-br from-white/20 to-white/5 border-2 border-white/20 flex items-center justify-center text-xs font-medium"
                >
                  {String.fromCharCode(64 + i)}
                </div>
              ))}
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium">Join 2,000+ companies</div>
              <div className="text-xs text-white/60">using BVRAI to power their voice AI</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
