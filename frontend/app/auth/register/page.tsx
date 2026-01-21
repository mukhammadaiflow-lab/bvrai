"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Bot,
  Eye,
  EyeOff,
  Loader2,
  Check,
  X,
  Sparkles,
  ArrowRight,
  Shield,
  Zap,
  Globe,
  Clock,
  Users,
  TrendingUp
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface PasswordStrength {
  score: number;
  requirements: {
    length: boolean;
    uppercase: boolean;
    lowercase: boolean;
    number: boolean;
    special: boolean;
  };
}

function checkPasswordStrength(password: string): PasswordStrength {
  const requirements = {
    length: password.length >= 8,
    uppercase: /[A-Z]/.test(password),
    lowercase: /[a-z]/.test(password),
    number: /[0-9]/.test(password),
    special: /[!@#$%^&*(),.?":{}|<>]/.test(password),
  };

  const score = Object.values(requirements).filter(Boolean).length;

  return { score, requirements };
}

export default function RegisterPage() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    company: "",
    password: "",
    confirmPassword: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [agreedToTerms, setAgreedToTerms] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const passwordStrength = checkPasswordStrength(formData.password);
  const passwordsMatch = formData.password === formData.confirmPassword;

  const handleChange = (field: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData((prev) => ({ ...prev, [field]: e.target.value }));
    setError("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!agreedToTerms) {
      setError("Please agree to the terms and conditions");
      return;
    }

    if (!passwordsMatch) {
      setError("Passwords do not match");
      return;
    }

    if (passwordStrength.score < 3) {
      setError("Please choose a stronger password");
      return;
    }

    setLoading(true);
    setError("");

    try {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      router.push("/auth/login?registered=true");
    } catch (err) {
      setError("Registration failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const strengthLabels = ["Very Weak", "Weak", "Fair", "Good", "Strong"];
  const strengthColors = [
    "from-red-500 to-red-600",
    "from-orange-500 to-orange-600",
    "from-yellow-500 to-yellow-600",
    "from-lime-500 to-lime-600",
    "from-green-500 to-emerald-600",
  ];

  return (
    <div className="min-h-screen flex overflow-hidden">
      {/* Left Side - Hero */}
      <div className="hidden lg:flex flex-1 relative overflow-hidden">
        {/* Animated Gradient Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-accent via-orange-500 to-primary" />

        {/* Animated Orbs */}
        <div className="absolute top-1/3 left-1/4 w-96 h-96 bg-white/10 rounded-full blur-3xl animate-pulse-soft" />
        <div className="absolute bottom-1/3 right-1/4 w-80 h-80 bg-primary/20 rounded-full blur-3xl animate-bounce-soft" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-white/5 rounded-full blur-3xl" />

        {/* Grid Pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:64px_64px]" />

        {/* Content */}
        <div className="relative z-10 flex flex-col justify-center px-16 text-white">
          <div className="mb-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 text-sm font-medium mb-6">
              <Shield className="h-4 w-4" />
              <span>Enterprise-Grade Security</span>
            </div>

            <h2 className="text-5xl font-bold leading-tight mb-6">
              Start building
              <br />
              <span className="text-white/90">voice AI today</span>
            </h2>

            <p className="text-xl text-white/70 max-w-md leading-relaxed">
              Join thousands of companies using BVRAI to create intelligent voice experiences that delight customers.
            </p>
          </div>

          {/* Benefits */}
          <div className="space-y-4 mb-12">
            {[
              { icon: <Zap className="h-5 w-5" />, title: "Setup in minutes", desc: "Get your first voice agent running quickly" },
              { icon: <Globe className="h-5 w-5" />, title: "Global infrastructure", desc: "Low latency worldwide with 99.9% uptime" },
              { icon: <Shield className="h-5 w-5" />, title: "SOC 2 compliant", desc: "Enterprise security & data protection" },
            ].map((benefit, idx) => (
              <div key={idx} className="flex items-start gap-4 p-4 rounded-2xl bg-white/10 backdrop-blur-sm border border-white/10">
                <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-white/20 flex items-center justify-center">
                  {benefit.icon}
                </div>
                <div>
                  <div className="font-semibold">{benefit.title}</div>
                  <div className="text-sm text-white/60">{benefit.desc}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Social Proof */}
          <div className="flex items-center gap-6">
            <div className="flex -space-x-3">
              {[1, 2, 3, 4, 5].map((i) => (
                <div
                  key={i}
                  className="w-10 h-10 rounded-full bg-gradient-to-br from-white/30 to-white/10 border-2 border-white/30 flex items-center justify-center text-xs font-bold"
                >
                  {String.fromCharCode(64 + i)}
                </div>
              ))}
            </div>
            <div>
              <div className="flex items-center gap-1">
                {[1, 2, 3, 4, 5].map((i) => (
                  <svg key={i} className="w-4 h-4 text-yellow-400 fill-current" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                  </svg>
                ))}
              </div>
              <div className="text-sm text-white/60">Loved by 2,000+ teams</div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-background relative">
        {/* Subtle background pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#8080800a_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:24px_24px]" />

        <div
          className={cn(
            "w-full max-w-[480px] relative z-10 transition-all duration-700",
            mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
          )}
        >
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3 mb-10 group">
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
          <div className="mb-6">
            <h1 className="text-3xl font-bold tracking-tight mb-2">
              Create your account
            </h1>
            <p className="text-muted-foreground text-lg">
              Start your 14-day free trial. No credit card required.
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="rounded-xl bg-destructive/10 border border-destructive/20 p-4 text-sm text-destructive flex items-center gap-3 animate-shake">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-destructive/20 flex items-center justify-center">
                  <span className="text-lg">!</span>
                </div>
                <span>{error}</span>
              </div>
            )}

            {/* Name & Company Row */}
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label htmlFor="name" className="text-sm font-medium">
                  Full name
                </label>
                <Input
                  id="name"
                  placeholder="John Doe"
                  value={formData.name}
                  onChange={handleChange("name")}
                  required
                  className="h-12 rounded-xl border-2 border-border/50 bg-muted/30 px-4 text-base transition-all focus:border-primary focus:bg-background focus:ring-4 focus:ring-primary/10"
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="company" className="text-sm font-medium">
                  Company <span className="text-muted-foreground">(optional)</span>
                </label>
                <Input
                  id="company"
                  placeholder="Acme Inc"
                  value={formData.company}
                  onChange={handleChange("company")}
                  className="h-12 rounded-xl border-2 border-border/50 bg-muted/30 px-4 text-base transition-all focus:border-primary focus:bg-background focus:ring-4 focus:ring-primary/10"
                />
              </div>
            </div>

            {/* Email */}
            <div className="space-y-2">
              <label htmlFor="email" className="text-sm font-medium">
                Work email
              </label>
              <Input
                id="email"
                type="email"
                placeholder="you@company.com"
                value={formData.email}
                onChange={handleChange("email")}
                required
                className="h-12 rounded-xl border-2 border-border/50 bg-muted/30 px-4 text-base transition-all focus:border-primary focus:bg-background focus:ring-4 focus:ring-primary/10"
              />
            </div>

            {/* Password */}
            <div className="space-y-2">
              <label htmlFor="password" className="text-sm font-medium">
                Password
              </label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Create a strong password"
                  value={formData.password}
                  onChange={handleChange("password")}
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

              {/* Password Strength Indicator */}
              {formData.password && (
                <div className="space-y-3 pt-2">
                  <div className="flex gap-1.5">
                    {[1, 2, 3, 4, 5].map((level) => (
                      <div
                        key={level}
                        className={cn(
                          "h-1.5 flex-1 rounded-full transition-all duration-300",
                          level <= passwordStrength.score
                            ? `bg-gradient-to-r ${strengthColors[passwordStrength.score - 1]}`
                            : "bg-muted"
                        )}
                      />
                    ))}
                  </div>
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-medium text-muted-foreground">
                      {strengthLabels[passwordStrength.score - 1] || "Very Weak"}
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { key: "length", label: "8+ characters" },
                      { key: "uppercase", label: "Uppercase" },
                      { key: "lowercase", label: "Lowercase" },
                      { key: "number", label: "Number" },
                    ].map((req) => (
                      <div
                        key={req.key}
                        className={cn(
                          "flex items-center gap-2 text-xs transition-colors",
                          passwordStrength.requirements[req.key as keyof typeof passwordStrength.requirements]
                            ? "text-green-600"
                            : "text-muted-foreground"
                        )}
                      >
                        <div className={cn(
                          "w-4 h-4 rounded-full flex items-center justify-center transition-all",
                          passwordStrength.requirements[req.key as keyof typeof passwordStrength.requirements]
                            ? "bg-green-500/20"
                            : "bg-muted"
                        )}>
                          {passwordStrength.requirements[req.key as keyof typeof passwordStrength.requirements] ? (
                            <Check className="h-2.5 w-2.5" />
                          ) : (
                            <X className="h-2.5 w-2.5" />
                          )}
                        </div>
                        {req.label}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Confirm Password */}
            <div className="space-y-2">
              <label htmlFor="confirmPassword" className="text-sm font-medium">
                Confirm password
              </label>
              <div className="relative">
                <Input
                  id="confirmPassword"
                  type={showConfirmPassword ? "text" : "password"}
                  placeholder="Confirm your password"
                  value={formData.confirmPassword}
                  onChange={handleChange("confirmPassword")}
                  required
                  className={cn(
                    "h-12 rounded-xl border-2 bg-muted/30 px-4 pr-12 text-base transition-all focus:bg-background focus:ring-4 focus:ring-primary/10",
                    formData.confirmPassword && !passwordsMatch
                      ? "border-destructive focus:border-destructive focus:ring-destructive/10"
                      : "border-border/50 focus:border-primary"
                  )}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showConfirmPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
              {formData.confirmPassword && !passwordsMatch && (
                <p className="text-xs text-destructive flex items-center gap-1">
                  <X className="h-3 w-3" />
                  Passwords do not match
                </p>
              )}
            </div>

            {/* Terms */}
            <div className="flex items-start gap-3 pt-2">
              <div className="relative flex items-center">
                <input
                  type="checkbox"
                  id="terms"
                  checked={agreedToTerms}
                  onChange={(e) => setAgreedToTerms(e.target.checked)}
                  className="peer sr-only"
                />
                <div
                  onClick={() => setAgreedToTerms(!agreedToTerms)}
                  className={cn(
                    "h-5 w-5 rounded-md border-2 cursor-pointer transition-all flex items-center justify-center",
                    agreedToTerms
                      ? "bg-primary border-primary"
                      : "border-border/50 hover:border-primary/50"
                  )}
                >
                  {agreedToTerms && <Check className="h-3 w-3 text-white" />}
                </div>
              </div>
              <label htmlFor="terms" className="text-sm text-muted-foreground cursor-pointer">
                I agree to the{" "}
                <Link href="/terms" className="text-primary hover:text-primary/80 transition-colors font-medium">
                  Terms of Service
                </Link>{" "}
                and{" "}
                <Link href="/privacy" className="text-primary hover:text-primary/80 transition-colors font-medium">
                  Privacy Policy
                </Link>
              </label>
            </div>

            <Button
              type="submit"
              disabled={loading || !agreedToTerms}
              className="w-full h-12 rounded-xl text-base font-semibold bg-gradient-to-r from-primary to-primary/90 hover:from-primary/90 hover:to-primary shadow-lg shadow-primary/25 hover:shadow-xl hover:shadow-primary/30 transition-all mt-2"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Creating account...
                </>
              ) : (
                <>
                  Create account
                  <ArrowRight className="ml-2 h-5 w-5" />
                </>
              )}
            </Button>
          </form>

          {/* Divider */}
          <div className="relative my-6">
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
              type="button"
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
              type="button"
            >
              <svg className="mr-2 h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              GitHub
            </Button>
          </div>

          {/* Sign In Link */}
          <p className="mt-6 text-center text-muted-foreground">
            Already have an account?{" "}
            <Link
              href="/auth/login"
              className="font-semibold text-primary hover:text-primary/80 transition-colors"
            >
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
