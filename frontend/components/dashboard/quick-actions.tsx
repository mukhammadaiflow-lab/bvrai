"use client";

import React from "react";
import Link from "next/link";
import {
  Bot,
  Phone,
  BarChart3,
  Settings,
  Key,
  Webhook,
  Users,
  Mic,
  Plus,
  ArrowRight,
  Sparkles,
  Zap,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface QuickAction {
  label: string;
  description?: string;
  href: string;
  icon: React.ElementType;
  variant?: "default" | "primary" | "featured";
}

const quickActions: QuickAction[] = [
  {
    label: "Create Agent",
    description: "Build a new AI voice agent",
    href: "/agents/new",
    icon: Bot,
    variant: "primary",
  },
  {
    label: "View Calls",
    description: "See call history and recordings",
    href: "/calls",
    icon: Phone,
  },
  {
    label: "Analytics",
    description: "View performance metrics",
    href: "/analytics",
    icon: BarChart3,
  },
  {
    label: "Voice Config",
    description: "Configure voice settings",
    href: "/voice-config",
    icon: Mic,
  },
];

const moreActions: QuickAction[] = [
  { label: "API Keys", href: "/api-keys", icon: Key },
  { label: "Webhooks", href: "/webhooks", icon: Webhook },
  { label: "Team", href: "/team", icon: Users },
  { label: "Settings", href: "/settings", icon: Settings },
];

interface QuickActionsProps {
  className?: string;
}

export function QuickActions({ className }: QuickActionsProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5" />
          Quick Actions
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {quickActions.map((action) => (
            <QuickActionButton key={action.href} action={action} />
          ))}
        </div>
        <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t">
          {moreActions.map((action) => (
            <Button
              key={action.href}
              variant="ghost"
              size="sm"
              className="text-muted-foreground"
              asChild
            >
              <Link href={action.href}>
                <action.icon className="mr-1.5 h-3.5 w-3.5" />
                {action.label}
              </Link>
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

interface QuickActionButtonProps {
  action: QuickAction;
}

function QuickActionButton({ action }: QuickActionButtonProps) {
  const Icon = action.icon;

  return (
    <Link
      href={action.href}
      className={cn(
        "group relative flex flex-col items-center justify-center gap-2 p-4 rounded-lg border text-center transition-all",
        action.variant === "primary"
          ? "bg-primary/5 border-primary/20 hover:bg-primary/10 hover:border-primary/30"
          : "hover:bg-muted hover:border-primary/20"
      )}
    >
      <div
        className={cn(
          "flex h-10 w-10 items-center justify-center rounded-full transition-colors",
          action.variant === "primary"
            ? "bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground"
            : "bg-muted text-muted-foreground group-hover:bg-primary/10 group-hover:text-primary"
        )}
      >
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <span className="text-sm font-medium">{action.label}</span>
        {action.description && (
          <p className="text-xs text-muted-foreground mt-0.5 hidden md:block">
            {action.description}
          </p>
        )}
      </div>
      {action.variant === "primary" && (
        <span className="absolute top-2 right-2">
          <Plus className="h-4 w-4 text-primary" />
        </span>
      )}
    </Link>
  );
}

interface QuickActionsCompactProps {
  className?: string;
}

export function QuickActionsCompact({ className }: QuickActionsCompactProps) {
  return (
    <div className={cn("flex flex-wrap gap-2", className)}>
      <Button asChild>
        <Link href="/agents/new">
          <Bot className="mr-2 h-4 w-4" />
          Create Agent
        </Link>
      </Button>
      <Button variant="outline" asChild>
        <Link href="/calls">
          <Phone className="mr-2 h-4 w-4" />
          View Calls
        </Link>
      </Button>
      <Button variant="outline" asChild>
        <Link href="/analytics">
          <BarChart3 className="mr-2 h-4 w-4" />
          Analytics
        </Link>
      </Button>
    </div>
  );
}

interface GettingStartedProps {
  completedSteps: number;
  totalSteps: number;
  className?: string;
}

export function GettingStarted({
  completedSteps,
  totalSteps,
  className,
}: GettingStartedProps) {
  const steps = [
    {
      title: "Create your first agent",
      description: "Set up an AI voice agent with custom prompts",
      href: "/agents/new",
      icon: Bot,
      completed: completedSteps >= 1,
    },
    {
      title: "Configure voice settings",
      description: "Choose a voice provider and customize speech",
      href: "/voice-config",
      icon: Mic,
      completed: completedSteps >= 2,
    },
    {
      title: "Connect your phone number",
      description: "Link a phone number to receive calls",
      href: "/settings",
      icon: Phone,
      completed: completedSteps >= 3,
    },
    {
      title: "Make a test call",
      description: "Try out your agent with a test call",
      href: "/calls",
      icon: Sparkles,
      completed: completedSteps >= 4,
    },
  ];

  if (completedSteps >= totalSteps) return null;

  return (
    <Card className={cn("overflow-hidden", className)}>
      <div className="bg-gradient-to-r from-primary/10 via-primary/5 to-transparent p-4 border-b">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              Getting Started
            </h3>
            <p className="text-sm text-muted-foreground">
              Complete these steps to set up your platform
            </p>
          </div>
          <div className="text-right">
            <span className="text-2xl font-bold text-primary">{completedSteps}</span>
            <span className="text-muted-foreground">/{totalSteps}</span>
          </div>
        </div>
      </div>
      <CardContent className="p-0">
        <div className="divide-y">
          {steps.map((step, index) => (
            <Link
              key={index}
              href={step.href}
              className={cn(
                "flex items-center gap-4 p-4 transition-colors",
                step.completed
                  ? "bg-muted/30 opacity-60"
                  : "hover:bg-muted/50"
              )}
            >
              <div
                className={cn(
                  "flex h-10 w-10 items-center justify-center rounded-full",
                  step.completed
                    ? "bg-green-100 text-green-600"
                    : "bg-primary/10 text-primary"
                )}
              >
                {step.completed ? (
                  <svg
                    className="h-5 w-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                ) : (
                  <step.icon className="h-5 w-5" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p
                  className={cn(
                    "font-medium",
                    step.completed && "line-through"
                  )}
                >
                  {step.title}
                </p>
                <p className="text-sm text-muted-foreground truncate">
                  {step.description}
                </p>
              </div>
              {!step.completed && (
                <ArrowRight className="h-5 w-5 text-muted-foreground" />
              )}
            </Link>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
