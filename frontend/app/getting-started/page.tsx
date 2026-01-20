"use client";

import React, { useState, useEffect } from "react";
import { DashboardLayout } from "@/components/layouts/dashboard-layout";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Button,
  Badge,
  Input,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  Label,
  Separator,
  Progress,
  Checkbox,
  Textarea,
} from "@/components/ui";
import {
  Check,
  CheckCircle,
  Circle,
  ChevronRight,
  ChevronLeft,
  Bot,
  Phone,
  Mic,
  Users,
  Webhook,
  Key,
  Sparkles,
  Play,
  Settings,
  ArrowRight,
  Zap,
  MessageSquare,
  Globe,
  Clock,
  Star,
  Rocket,
  Award,
  ExternalLink,
  Copy,
  Eye,
  EyeOff,
  HelpCircle,
  BookOpen,
  Video,
  FileText,
  Headphones,
  PartyPopper,
  Gift,
  Target,
  Lightbulb,
} from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";

// Types
interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  icon: React.ElementType;
  completed: boolean;
  href?: string;
  action?: string;
}

interface QuickStartTemplate {
  id: string;
  name: string;
  description: string;
  icon: React.ElementType;
  category: string;
  estimatedTime: string;
}

// Mock data
const initialSteps: OnboardingStep[] = [
  {
    id: "profile",
    title: "Complete your profile",
    description: "Add your organization details and preferences",
    icon: Users,
    completed: true,
    href: "/settings/profile",
  },
  {
    id: "voice",
    title: "Configure voice settings",
    description: "Choose your AI voice provider and settings",
    icon: Mic,
    completed: true,
    href: "/settings/voice",
  },
  {
    id: "agent",
    title: "Create your first agent",
    description: "Build a voice AI agent to handle calls",
    icon: Bot,
    completed: false,
    href: "/agents/new",
    action: "Create Agent",
  },
  {
    id: "test",
    title: "Make a test call",
    description: "Test your agent with a real phone call",
    icon: Phone,
    completed: false,
    action: "Start Test Call",
  },
  {
    id: "webhook",
    title: "Set up integrations",
    description: "Connect webhooks and external services",
    icon: Webhook,
    completed: false,
    href: "/webhooks",
  },
  {
    id: "api",
    title: "Get your API keys",
    description: "Access API keys for programmatic control",
    icon: Key,
    completed: false,
    href: "/settings/api-keys",
  },
];

const quickStartTemplates: QuickStartTemplate[] = [
  {
    id: "customer-support",
    name: "Customer Support",
    description: "Handle customer inquiries and support tickets",
    icon: Headphones,
    category: "Support",
    estimatedTime: "5 min",
  },
  {
    id: "appointment-scheduler",
    name: "Appointment Scheduler",
    description: "Book appointments and manage calendars",
    icon: Clock,
    category: "Scheduling",
    estimatedTime: "5 min",
  },
  {
    id: "lead-qualifier",
    name: "Lead Qualifier",
    description: "Qualify leads and collect information",
    icon: Target,
    category: "Sales",
    estimatedTime: "5 min",
  },
  {
    id: "order-status",
    name: "Order Status",
    description: "Provide order tracking and updates",
    icon: Globe,
    category: "E-commerce",
    estimatedTime: "5 min",
  },
];

const resources = [
  {
    title: "Documentation",
    description: "Comprehensive guides and API reference",
    icon: BookOpen,
    href: "/docs",
    external: true,
  },
  {
    title: "Video Tutorials",
    description: "Watch step-by-step tutorials",
    icon: Video,
    href: "/tutorials",
    external: true,
  },
  {
    title: "API Reference",
    description: "Full API documentation",
    icon: FileText,
    href: "/api-docs",
    external: true,
  },
  {
    title: "Community",
    description: "Join our Discord community",
    icon: MessageSquare,
    href: "https://discord.gg/example",
    external: true,
  },
];

// Components
function StepCard({
  step,
  index,
  onComplete,
}: {
  step: OnboardingStep;
  index: number;
  onComplete: () => void;
}) {
  const Icon = step.icon;

  return (
    <div
      className={cn(
        "relative flex items-start gap-4 p-4 rounded-lg border transition-all",
        step.completed
          ? "bg-green-50/50 border-green-200 dark:bg-green-950/20"
          : "hover:bg-muted/50 hover:border-primary/50"
      )}
    >
      <div
        className={cn(
          "flex h-10 w-10 shrink-0 items-center justify-center rounded-full",
          step.completed
            ? "bg-green-500 text-white"
            : "bg-muted text-muted-foreground"
        )}
      >
        {step.completed ? (
          <Check className="h-5 w-5" />
        ) : (
          <span className="text-sm font-medium">{index + 1}</span>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <h3
            className={cn(
              "font-medium",
              step.completed && "text-green-700 dark:text-green-400"
            )}
          >
            {step.title}
          </h3>
          {step.completed && (
            <Badge variant="secondary" className="text-xs bg-green-100 text-green-700">
              Completed
            </Badge>
          )}
        </div>
        <p className="text-sm text-muted-foreground mt-1">{step.description}</p>
        {!step.completed && (
          <div className="mt-3">
            {step.href ? (
              <Link href={step.href}>
                <Button size="sm">
                  {step.action || "Get Started"}
                  <ChevronRight className="ml-1 h-4 w-4" />
                </Button>
              </Link>
            ) : (
              <Button size="sm" onClick={onComplete}>
                {step.action || "Get Started"}
                <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            )}
          </div>
        )}
      </div>
      <Icon
        className={cn(
          "h-5 w-5 shrink-0",
          step.completed ? "text-green-500" : "text-muted-foreground"
        )}
      />
    </div>
  );
}

function TemplateCard({ template }: { template: QuickStartTemplate }) {
  const Icon = template.icon;

  return (
    <Link href={`/agents/new?template=${template.id}`}>
      <Card className="h-full hover:border-primary/50 hover:shadow-md transition-all cursor-pointer group">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
              <Icon className="h-5 w-5 text-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <h3 className="font-medium group-hover:text-primary transition-colors">
                  {template.name}
                </h3>
                <Badge variant="outline" className="text-xs">
                  {template.estimatedTime}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                {template.description}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

function ResourceCard({
  resource,
}: {
  resource: (typeof resources)[0];
}) {
  const Icon = resource.icon;

  return (
    <a
      href={resource.href}
      target={resource.external ? "_blank" : undefined}
      rel={resource.external ? "noopener noreferrer" : undefined}
      className="block"
    >
      <div className="flex items-center gap-3 p-3 rounded-lg border hover:bg-muted/50 hover:border-primary/50 transition-all cursor-pointer group">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted group-hover:bg-primary/10 transition-colors">
          <Icon className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium group-hover:text-primary transition-colors">
            {resource.title}
          </h4>
          <p className="text-xs text-muted-foreground">{resource.description}</p>
        </div>
        {resource.external && <ExternalLink className="h-4 w-4 text-muted-foreground" />}
      </div>
    </a>
  );
}

function WelcomeModal({
  open,
  onOpenChange,
  onGetStarted,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onGetStarted: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <div className="text-center py-6">
          <div className="flex justify-center mb-4">
            <div className="relative">
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-primary to-purple-600">
                <Rocket className="h-10 w-10 text-white" />
              </div>
              <div className="absolute -top-1 -right-1 flex h-6 w-6 items-center justify-center rounded-full bg-yellow-400">
                <Sparkles className="h-4 w-4 text-yellow-800" />
              </div>
            </div>
          </div>
          <DialogHeader className="text-center">
            <DialogTitle className="text-2xl">Welcome to Builder AI!</DialogTitle>
            <DialogDescription className="text-base mt-2">
              You're about to create powerful voice AI agents that can handle customer calls,
              schedule appointments, and much more.
            </DialogDescription>
          </DialogHeader>

          <div className="mt-6 space-y-3">
            <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 text-left">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-green-100 text-green-600">
                <Check className="h-4 w-4" />
              </div>
              <div>
                <p className="text-sm font-medium">No coding required</p>
                <p className="text-xs text-muted-foreground">Build agents with our visual editor</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 text-left">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                <Zap className="h-4 w-4" />
              </div>
              <div>
                <p className="text-sm font-medium">Ready in minutes</p>
                <p className="text-xs text-muted-foreground">Use templates to get started fast</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 text-left">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-100 text-purple-600">
                <Award className="h-4 w-4" />
              </div>
              <div>
                <p className="text-sm font-medium">Enterprise-grade quality</p>
                <p className="text-xs text-muted-foreground">Natural conversations that delight customers</p>
              </div>
            </div>
          </div>
        </div>
        <DialogFooter className="sm:justify-center">
          <Button size="lg" className="px-8" onClick={onGetStarted}>
            Let's Get Started
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function CompletionModal({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <div className="text-center py-6">
          <div className="flex justify-center mb-4">
            <div className="relative">
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-green-400 to-green-600">
                <PartyPopper className="h-10 w-10 text-white" />
              </div>
            </div>
          </div>
          <DialogHeader className="text-center">
            <DialogTitle className="text-2xl">Congratulations!</DialogTitle>
            <DialogDescription className="text-base mt-2">
              You've completed the onboarding! You're now ready to make the most of Builder AI.
            </DialogDescription>
          </DialogHeader>

          <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-950/30 dark:to-orange-950/30 border border-yellow-200 dark:border-yellow-800">
            <div className="flex items-center gap-3">
              <Gift className="h-8 w-8 text-yellow-600" />
              <div className="text-left">
                <p className="font-medium text-yellow-800 dark:text-yellow-200">Bonus Credits!</p>
                <p className="text-sm text-yellow-700 dark:text-yellow-300">
                  We've added 100 free call minutes to your account
                </p>
              </div>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-3">
            <Link href="/agents/new">
              <Button variant="outline" className="w-full">
                <Bot className="mr-2 h-4 w-4" />
                Create Agent
              </Button>
            </Link>
            <Link href="/dashboard">
              <Button className="w-full">
                Go to Dashboard
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default function GettingStartedPage() {
  const [steps, setSteps] = useState<OnboardingStep[]>(initialSteps);
  const [showWelcome, setShowWelcome] = useState(false);
  const [showCompletion, setShowCompletion] = useState(false);

  const completedSteps = steps.filter((s) => s.completed).length;
  const progress = (completedSteps / steps.length) * 100;
  const allCompleted = completedSteps === steps.length;

  // Check if first visit
  useEffect(() => {
    const hasSeenWelcome = localStorage.getItem("hasSeenWelcome");
    if (!hasSeenWelcome) {
      setShowWelcome(true);
    }
  }, []);

  const handleWelcomeClose = () => {
    localStorage.setItem("hasSeenWelcome", "true");
    setShowWelcome(false);
  };

  const handleStepComplete = (stepId: string) => {
    setSteps((prev) =>
      prev.map((s) => (s.id === stepId ? { ...s, completed: true } : s))
    );

    // Check if all steps are now completed
    const newCompletedCount = steps.filter((s) => s.completed || s.id === stepId).length;
    if (newCompletedCount === steps.length) {
      setTimeout(() => setShowCompletion(true), 500);
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-8">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Getting Started</h1>
            <p className="text-muted-foreground">
              Complete these steps to set up your voice AI platform
            </p>
          </div>
          {!allCompleted && (
            <div className="flex items-center gap-3">
              <div className="text-right">
                <p className="text-sm font-medium">
                  {completedSteps} of {steps.length} completed
                </p>
                <p className="text-xs text-muted-foreground">{Math.round(progress)}% done</p>
              </div>
              <div className="w-32">
                <Progress value={progress} className="h-2" />
              </div>
            </div>
          )}
        </div>

        {/* Completion Banner */}
        {allCompleted && (
          <Card className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/30 dark:to-emerald-950/30 border-green-200">
            <CardContent className="p-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-green-500">
                <CheckCircle className="h-7 w-7 text-white" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-green-800 dark:text-green-200">
                  All set! You've completed the setup.
                </h3>
                <p className="text-sm text-green-700 dark:text-green-300">
                  Your voice AI platform is ready to use. Start creating amazing agents!
                </p>
              </div>
              <Link href="/agents/new">
                <Button>
                  Create Your First Agent
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        )}

        <div className="grid gap-8 lg:grid-cols-3">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Setup Steps */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Rocket className="h-5 w-5 text-primary" />
                  <CardTitle>Setup Checklist</CardTitle>
                </div>
                <CardDescription>
                  Follow these steps to get your voice AI platform up and running
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {steps.map((step, index) => (
                  <StepCard
                    key={step.id}
                    step={step}
                    index={index}
                    onComplete={() => handleStepComplete(step.id)}
                  />
                ))}
              </CardContent>
            </Card>

            {/* Quick Start Templates */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-primary" />
                    <CardTitle>Quick Start Templates</CardTitle>
                  </div>
                  <Link href="/templates">
                    <Button variant="ghost" size="sm">
                      View All
                      <ChevronRight className="ml-1 h-4 w-4" />
                    </Button>
                  </Link>
                </div>
                <CardDescription>
                  Get started quickly with pre-built agent templates
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3 sm:grid-cols-2">
                  {quickStartTemplates.map((template) => (
                    <TemplateCard key={template.id} template={template} />
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Pro Tips */}
            <Card className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border-blue-200">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Lightbulb className="h-5 w-5 text-blue-600" />
                  <CardTitle className="text-base">Pro Tips</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-start gap-2">
                  <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-200 text-blue-700 text-xs font-medium">
                    1
                  </div>
                  <p className="text-sm text-blue-800 dark:text-blue-200">
                    Start with a template and customize it to fit your needs
                  </p>
                </div>
                <div className="flex items-start gap-2">
                  <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-200 text-blue-700 text-xs font-medium">
                    2
                  </div>
                  <p className="text-sm text-blue-800 dark:text-blue-200">
                    Test your agent with a real call before going live
                  </p>
                </div>
                <div className="flex items-start gap-2">
                  <div className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-200 text-blue-700 text-xs font-medium">
                    3
                  </div>
                  <p className="text-sm text-blue-800 dark:text-blue-200">
                    Use webhooks to integrate with your existing tools
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Resources */}
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <BookOpen className="h-5 w-5 text-primary" />
                  <CardTitle className="text-base">Resources</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                {resources.map((resource) => (
                  <ResourceCard key={resource.title} resource={resource} />
                ))}
              </CardContent>
            </Card>

            {/* Help */}
            <Card>
              <CardContent className="p-4 flex items-center gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900/30">
                  <HelpCircle className="h-5 w-5 text-purple-600" />
                </div>
                <div className="flex-1">
                  <p className="font-medium text-sm">Need help?</p>
                  <p className="text-xs text-muted-foreground">Our team is here to assist you</p>
                </div>
                <Button variant="outline" size="sm">
                  Contact Support
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Modals */}
      <WelcomeModal
        open={showWelcome}
        onOpenChange={setShowWelcome}
        onGetStarted={handleWelcomeClose}
      />
      <CompletionModal open={showCompletion} onOpenChange={setShowCompletion} />
    </DashboardLayout>
  );
}
