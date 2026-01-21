"use client";

import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { DashboardLayout } from "@/components/layouts/dashboard-layout";
import { billing as billingApi, analytics } from "@/lib/api";
import { toast } from "sonner";
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
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  Label,
  Separator,
  Progress,
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui";
import {
  CreditCard,
  Download,
  Check,
  Zap,
  Phone,
  Clock,
  Bot,
  Users,
  TrendingUp,
  TrendingDown,
  Calendar,
  FileText,
  ExternalLink,
  AlertCircle,
  AlertTriangle,
  Plus,
  Trash2,
  Edit,
  Star,
  Building,
  Receipt,
  DollarSign,
  Activity,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  ChevronRight,
  Shield,
  Info,
  Wallet,
  X,
  CheckCircle,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Types
interface Plan {
  id: string;
  name: string;
  description: string;
  price: number | null;
  billingCycle: "monthly" | "yearly";
  features: string[];
  limits: {
    calls: number;
    minutes: number;
    agents: number;
    team: number;
  };
  popular?: boolean;
  current?: boolean;
}

interface UsageMetric {
  name: string;
  icon: React.ElementType;
  used: number;
  limit: number;
  unit?: string;
  trend?: number;
}

interface Invoice {
  id: string;
  date: string;
  amount: number;
  status: "paid" | "pending" | "failed" | "refunded";
  description: string;
  downloadUrl?: string;
}

interface PaymentMethod {
  id: string;
  type: "card" | "bank";
  brand?: string;
  last4: string;
  expiryMonth?: number;
  expiryYear?: number;
  isDefault: boolean;
}

// Static plans configuration
const planDefinitions: Plan[] = [
  {
    id: "free",
    name: "Free",
    description: "Get started with basics",
    price: 0,
    billingCycle: "monthly",
    features: [
      "100 calls/month",
      "500 minutes",
      "1 agent",
      "1 team member",
      "Community support",
    ],
    limits: { calls: 100, minutes: 500, agents: 1, team: 1 },
  },
  {
    id: "starter",
    name: "Starter",
    description: "For small teams",
    price: 29,
    billingCycle: "monthly",
    features: [
      "1,000 calls/month",
      "2,500 minutes",
      "3 agents",
      "3 team members",
      "Email support",
      "Basic analytics",
    ],
    limits: { calls: 1000, minutes: 2500, agents: 3, team: 3 },
  },
  {
    id: "professional",
    name: "Professional",
    description: "Perfect for growing teams",
    price: 99,
    billingCycle: "monthly",
    features: [
      "5,000 calls/month",
      "10,000 minutes",
      "10 agents",
      "10 team members",
      "Priority support",
      "Advanced analytics",
      "Webhooks",
      "API access",
    ],
    limits: { calls: 5000, minutes: 10000, agents: 10, team: 10 },
    popular: true,
  },
  {
    id: "enterprise",
    name: "Enterprise",
    description: "For large organizations",
    price: null,
    billingCycle: "monthly",
    features: [
      "Unlimited calls",
      "Unlimited minutes",
      "Unlimited agents",
      "Unlimited team members",
      "24/7 dedicated support",
      "Custom SLA",
      "SSO & SAML",
      "Custom integrations",
      "On-premise deployment",
    ],
    limits: { calls: -1, minutes: -1, agents: -1, team: -1 },
  },
];

// Utility functions
const formatNumber = (num: number): string => {
  return new Intl.NumberFormat().format(num);
};

const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(amount);
};

const formatDate = (date: string): string => {
  return new Date(date).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
};

const getUsageColor = (percentage: number): string => {
  if (percentage >= 90) return "text-red-500";
  if (percentage >= 70) return "text-yellow-500";
  return "text-green-500";
};

const getUsageProgressColor = (percentage: number): string => {
  if (percentage >= 90) return "bg-red-500";
  if (percentage >= 70) return "bg-yellow-500";
  return "bg-primary";
};

// Components
function UsageCard({ metric }: { metric: UsageMetric }) {
  const percentage = metric.limit > 0 ? (metric.used / metric.limit) * 100 : 0;
  const Icon = metric.icon;

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
              <Icon className="h-4 w-4 text-primary" />
            </div>
            <span className="font-medium text-sm">{metric.name}</span>
          </div>
          {metric.trend !== undefined && (
            <div className={cn("flex items-center gap-1 text-xs", metric.trend >= 0 ? "text-green-500" : "text-red-500")}>
              {metric.trend >= 0 ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
              {Math.abs(metric.trend)}%
            </div>
          )}
        </div>
        <div className="space-y-2">
          <div className="flex items-baseline justify-between">
            <span className="text-2xl font-bold">{formatNumber(metric.used)}</span>
            <span className="text-sm text-muted-foreground">
              / {metric.limit === -1 ? "∞" : formatNumber(metric.limit)} {metric.unit || ""}
            </span>
          </div>
          <div className="h-2 rounded-full bg-secondary overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all", getUsageProgressColor(percentage))}
              style={{ width: `${Math.min(percentage, 100)}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className={getUsageColor(percentage)}>{percentage.toFixed(0)}% used</span>
            {metric.limit > 0 && (
              <span className="text-muted-foreground">{formatNumber(metric.limit - metric.used)} remaining</span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function UsageChart() {
  const maxCalls = Math.max(...dailyUsage.map((d) => d.calls));

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Daily Usage</CardTitle>
            <CardDescription>Calls over the past 7 days</CardDescription>
          </div>
          <Select defaultValue="7d">
            <SelectTrigger className="w-24 h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">7 days</SelectItem>
              <SelectItem value="30d">30 days</SelectItem>
              <SelectItem value="90d">90 days</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-end gap-2 h-40">
          {dailyUsage.map((day, index) => (
            <div key={day.day} className="flex-1 flex flex-col items-center gap-1">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div
                      className="w-full bg-primary/80 rounded-t hover:bg-primary transition-colors cursor-pointer"
                      style={{ height: `${(day.calls / maxCalls) * 100}%`, minHeight: "8px" }}
                    />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{day.calls} calls</p>
                    <p className="text-xs text-muted-foreground">{day.minutes} minutes</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <span className="text-xs text-muted-foreground">{day.day}</span>
            </div>
          ))}
        </div>
        <div className="flex items-center justify-between mt-4 pt-4 border-t">
          <div className="text-sm">
            <span className="font-medium">385</span>
            <span className="text-muted-foreground"> total calls this week</span>
          </div>
          <div className="flex items-center gap-1 text-sm text-green-500">
            <ArrowUpRight className="h-4 w-4" />
            <span>+12% from last week</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function PlanCard({
  plan,
  onSelect,
  billingCycle,
}: {
  plan: Plan;
  onSelect: () => void;
  billingCycle: "monthly" | "yearly";
}) {
  const yearlyPrice = plan.price ? plan.price * 10 : null; // 2 months free
  const displayPrice = billingCycle === "yearly" && yearlyPrice ? yearlyPrice : plan.price;
  const monthlyEquivalent = billingCycle === "yearly" && yearlyPrice ? yearlyPrice / 12 : plan.price;

  return (
    <div
      className={cn(
        "relative rounded-xl border p-6 transition-all hover:shadow-md",
        plan.current && "border-primary ring-2 ring-primary/20",
        plan.popular && !plan.current && "border-purple-300"
      )}
    >
      {plan.popular && (
        <Badge className="absolute -top-3 left-1/2 -translate-x-1/2 bg-purple-500">
          Most Popular
        </Badge>
      )}
      {plan.current && (
        <Badge className="absolute -top-3 right-4 bg-primary">
          Current Plan
        </Badge>
      )}

      <div className="mb-4">
        <h3 className="text-xl font-bold">{plan.name}</h3>
        <p className="text-sm text-muted-foreground">{plan.description}</p>
      </div>

      <div className="mb-6">
        {displayPrice !== null ? (
          <div className="flex items-baseline gap-1">
            <span className="text-4xl font-bold">{formatCurrency(displayPrice)}</span>
            <span className="text-muted-foreground">/{billingCycle === "yearly" ? "year" : "month"}</span>
          </div>
        ) : (
          <span className="text-3xl font-bold">Custom</span>
        )}
        {billingCycle === "yearly" && plan.price && (
          <p className="text-sm text-muted-foreground mt-1">
            {formatCurrency(monthlyEquivalent!)}/month billed annually
          </p>
        )}
      </div>

      <ul className="space-y-3 mb-6">
        {plan.features.map((feature, i) => (
          <li key={i} className="flex items-start gap-2 text-sm">
            <Check className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>{feature}</span>
          </li>
        ))}
      </ul>

      <Button
        className="w-full"
        variant={plan.current ? "outline" : plan.popular ? "default" : "outline"}
        disabled={plan.current}
        onClick={onSelect}
      >
        {plan.current ? (
          "Current Plan"
        ) : plan.price === null ? (
          "Contact Sales"
        ) : (
          <>
            {plan.id === "free" ? "Downgrade" : "Upgrade"}
            <ChevronRight className="ml-1 h-4 w-4" />
          </>
        )}
      </Button>
    </div>
  );
}

function PaymentMethodCard({
  method,
  onEdit,
  onRemove,
  onSetDefault,
}: {
  method: PaymentMethod;
  onEdit: () => void;
  onRemove: () => void;
  onSetDefault: () => void;
}) {
  const brandIcons: Record<string, string> = {
    visa: "💳",
    mastercard: "💳",
    amex: "💳",
  };

  return (
    <div
      className={cn(
        "flex items-center justify-between p-4 rounded-lg border",
        method.isDefault && "border-primary bg-primary/5"
      )}
    >
      <div className="flex items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-muted text-2xl">
          {brandIcons[method.brand || ""] || "💳"}
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-medium capitalize">{method.brand || "Card"}</span>
            <span className="text-muted-foreground">•••• {method.last4}</span>
            {method.isDefault && (
              <Badge variant="secondary" className="text-xs">
                Default
              </Badge>
            )}
          </div>
          {method.expiryMonth && method.expiryYear && (
            <p className="text-sm text-muted-foreground">
              Expires {method.expiryMonth}/{method.expiryYear}
            </p>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2">
        {!method.isDefault && (
          <Button variant="ghost" size="sm" onClick={onSetDefault}>
            Set Default
          </Button>
        )}
        <Button variant="ghost" size="icon-sm" onClick={onEdit}>
          <Edit className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onRemove}
          className="text-red-500 hover:text-red-600"
          disabled={method.isDefault}
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

function InvoiceStatusBadge({ status }: { status: Invoice["status"] }) {
  const config = {
    paid: { color: "bg-green-100 text-green-700", icon: CheckCircle, label: "Paid" },
    pending: { color: "bg-yellow-100 text-yellow-700", icon: Clock, label: "Pending" },
    failed: { color: "bg-red-100 text-red-700", icon: AlertCircle, label: "Failed" },
    refunded: { color: "bg-gray-100 text-gray-700", icon: RefreshCw, label: "Refunded" },
  };

  const { color, icon: Icon, label } = config[status];

  return (
    <Badge className={cn("gap-1", color)}>
      <Icon className="h-3 w-3" />
      {label}
    </Badge>
  );
}

export default function BillingPage() {
  const [activeTab, setActiveTab] = useState("overview");
  const [billingCycle, setBillingCycle] = useState<"monthly" | "yearly">("monthly");
  const [showUpgradeDialog, setShowUpgradeDialog] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null);

  // Fetch current plan and usage from API
  const { data: planData, isLoading: planLoading } = useQuery({
    queryKey: ["billing", "plan"],
    queryFn: () => billingApi.getCurrentPlan(),
  });

  // Fetch invoices from API
  const { data: invoicesData, isLoading: invoicesLoading } = useQuery({
    queryKey: ["billing", "invoices"],
    queryFn: () => billingApi.getInvoices(),
  });

  // Fetch dashboard stats for additional usage data
  const { data: dashboardData } = useQuery({
    queryKey: ["analytics", "dashboard"],
    queryFn: () => analytics.getDashboard(),
  });

  // Upgrade mutation
  const upgradeMutation = useMutation({
    mutationFn: (planId: string) => billingApi.createCheckoutSession(planId),
    onSuccess: (data) => {
      if (data.checkout_url) {
        window.location.href = data.checkout_url;
      } else {
        toast.success("Plan upgraded successfully");
        setShowUpgradeDialog(false);
      }
    },
    onError: (error: Error) => {
      toast.error(`Failed to upgrade: ${error.message}`);
    },
  });

  // Derive current plan from API or use default
  const currentPlanId = planData?.plan || "professional";
  const currentPlan: Plan = {
    ...(planDefinitions.find((p) => p.id === currentPlanId) || planDefinitions[2]),
    current: true,
  };

  // Build usage metrics from API data
  const apiUsage = planData?.usage || dashboardData?.usage || {};
  const usageMetrics: UsageMetric[] = [
    {
      name: "Calls",
      icon: Phone,
      used: apiUsage.calls_used || dashboardData?.today?.total_calls || 0,
      limit: apiUsage.calls_limit || currentPlan.limits.calls,
      trend: 12,
    },
    {
      name: "Minutes",
      icon: Clock,
      used: apiUsage.minutes_used || dashboardData?.today?.total_minutes || 0,
      limit: apiUsage.minutes_limit || currentPlan.limits.minutes,
      unit: "min",
      trend: 8,
    },
    {
      name: "Agents",
      icon: Bot,
      used: dashboardData?.agents?.active || 0,
      limit: currentPlan.limits.agents,
    },
    {
      name: "Team Members",
      icon: Users,
      used: 4, // Would need team API
      limit: currentPlan.limits.team,
    },
  ];

  // Daily usage placeholder (would need time-series API)
  const dailyUsage = [
    { day: "Mon", calls: 45, minutes: 180 },
    { day: "Tue", calls: 62, minutes: 248 },
    { day: "Wed", calls: 58, minutes: 232 },
    { day: "Thu", calls: 71, minutes: 284 },
    { day: "Fri", calls: 89, minutes: 356 },
    { day: "Sat", calls: 32, minutes: 128 },
    { day: "Sun", calls: 28, minutes: 112 },
  ];

  // Transform invoices from API
  const invoices: Invoice[] = (invoicesData || []).map((inv: any) => ({
    id: inv.id,
    date: inv.created_at,
    amount: (inv.amount_cents || 0) / 100,
    status: inv.status as Invoice["status"],
    description: inv.description || `${currentPlan.name} Plan`,
    downloadUrl: inv.pdf_url,
  }));

  // Payment methods placeholder (would need Stripe integration)
  const paymentMethods: PaymentMethod[] = [];

  // Mark plans with current
  const plans = planDefinitions.map((p) => ({
    ...p,
    current: p.id === currentPlanId,
  }));

  const totalSpentThisYear = invoices
    .filter((inv) => inv.date?.startsWith(new Date().getFullYear().toString()))
    .reduce((sum, inv) => sum + inv.amount, 0);

  const nextBillingDate = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
  const nextBillingAmount = currentPlan.price || 0;

  const handleUpgrade = () => {
    if (selectedPlan && selectedPlan.price !== null) {
      upgradeMutation.mutate(selectedPlan.id);
    } else {
      // Contact sales for enterprise
      window.location.href = "mailto:sales@bvrai.com?subject=Enterprise Plan Inquiry";
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Billing & Usage</h1>
            <p className="text-muted-foreground">
              Manage your subscription, usage, and payment methods
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline">
              <Receipt className="mr-2 h-4 w-4" />
              Billing Portal
            </Button>
            <Button variant="outline">
              <CreditCard className="mr-2 h-4 w-4" />
              Update Payment
            </Button>
          </div>
        </div>

        {/* Current Plan Summary */}
        <Card className="border-primary bg-gradient-to-r from-primary/5 to-transparent">
          <CardContent className="p-6">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex items-start gap-4">
                <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary text-primary-foreground">
                  <Zap className="h-7 w-7" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Badge className="bg-primary text-primary-foreground">{currentPlan.name}</Badge>
                    <span className="text-sm text-muted-foreground">Current Plan</span>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">
                    {formatCurrency(currentPlan.price || 0)}/month • Next billing: {nextBillingDate}
                  </p>
                  <div className="flex items-center gap-3">
                    <Button size="sm" onClick={() => setShowUpgradeDialog(true)}>
                      <ArrowUpRight className="mr-1 h-3 w-3" />
                      Upgrade
                    </Button>
                    <Button variant="ghost" size="sm" className="text-muted-foreground">
                      View Plan Details
                    </Button>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-6 lg:grid-cols-3">
                <div className="text-center lg:text-right">
                  <p className="text-sm text-muted-foreground">Next Invoice</p>
                  <p className="text-2xl font-bold">{formatCurrency(nextBillingAmount)}</p>
                </div>
                <div className="text-center lg:text-right">
                  <p className="text-sm text-muted-foreground">This Year</p>
                  <p className="text-2xl font-bold">{formatCurrency(totalSpentThisYear)}</p>
                </div>
                <div className="col-span-2 text-center lg:col-span-1 lg:text-right">
                  <p className="text-sm text-muted-foreground">Status</p>
                  <div className="flex items-center justify-center lg:justify-end gap-1 mt-1">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="font-medium text-green-600">Active</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="overview" className="gap-2">
              <Activity className="h-4 w-4" />
              Usage
            </TabsTrigger>
            <TabsTrigger value="plans" className="gap-2">
              <Zap className="h-4 w-4" />
              Plans
            </TabsTrigger>
            <TabsTrigger value="invoices" className="gap-2">
              <FileText className="h-4 w-4" />
              Invoices
            </TabsTrigger>
            <TabsTrigger value="payment" className="gap-2">
              <CreditCard className="h-4 w-4" />
              Payment Methods
            </TabsTrigger>
          </TabsList>

          {/* Usage Tab */}
          <TabsContent value="overview" className="mt-4 space-y-4">
            {/* Warning if near limit */}
            {usageMetrics.some((m) => m.limit > 0 && (m.used / m.limit) >= 0.9) && (
              <Card className="border-yellow-200 bg-yellow-50 dark:bg-yellow-950/20">
                <CardContent className="p-4 flex items-center gap-3">
                  <AlertTriangle className="h-5 w-5 text-yellow-600" />
                  <div className="flex-1">
                    <p className="font-medium text-yellow-800 dark:text-yellow-200">Approaching usage limits</p>
                    <p className="text-sm text-yellow-700 dark:text-yellow-300">
                      You're using over 90% of some resources. Consider upgrading your plan.
                    </p>
                  </div>
                  <Button size="sm" variant="outline" onClick={() => setShowUpgradeDialog(true)}>
                    Upgrade Now
                  </Button>
                </CardContent>
              </Card>
            )}

            {/* Usage Cards */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {usageMetrics.map((metric) => (
                <UsageCard key={metric.name} metric={metric} />
              ))}
            </div>

            {/* Usage Chart */}
            <div className="grid gap-4 lg:grid-cols-2">
              <UsageChart />
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Cost Breakdown</CardTitle>
                  <CardDescription>This billing period</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between py-2">
                    <div className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full bg-primary" />
                      <span className="text-sm">Base Plan</span>
                    </div>
                    <span className="font-medium">{formatCurrency(currentPlan.price || 0)}</span>
                  </div>
                  <div className="flex items-center justify-between py-2">
                    <div className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full bg-green-500" />
                      <span className="text-sm">Additional Usage</span>
                    </div>
                    <span className="font-medium">{formatCurrency(0)}</span>
                  </div>
                  <div className="flex items-center justify-between py-2">
                    <div className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full bg-purple-500" />
                      <span className="text-sm">Add-ons</span>
                    </div>
                    <span className="font-medium">{formatCurrency(0)}</span>
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between py-2">
                    <span className="font-medium">Estimated Total</span>
                    <span className="text-xl font-bold">{formatCurrency(currentPlan.price || 0)}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Next invoice on {nextBillingDate}
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Plans Tab */}
          <TabsContent value="plans" className="mt-4 space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium">Choose Your Plan</h3>
                <p className="text-sm text-muted-foreground">
                  Select the plan that best fits your needs
                </p>
              </div>
              <div className="flex items-center gap-2 p-1 bg-muted rounded-lg">
                <Button
                  variant={billingCycle === "monthly" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setBillingCycle("monthly")}
                >
                  Monthly
                </Button>
                <Button
                  variant={billingCycle === "yearly" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setBillingCycle("yearly")}
                  className="gap-1"
                >
                  Yearly
                  <Badge variant="secondary" className="text-xs bg-green-100 text-green-700">
                    Save 17%
                  </Badge>
                </Button>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {plans.map((plan) => (
                <PlanCard
                  key={plan.id}
                  plan={plan}
                  billingCycle={billingCycle}
                  onSelect={() => {
                    setSelectedPlan(plan);
                    setShowUpgradeDialog(true);
                  }}
                />
              ))}
            </div>

            <Card className="bg-muted/50">
              <CardContent className="p-6 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <Building className="h-8 w-8 text-muted-foreground" />
                  <div>
                    <h4 className="font-medium">Need a custom solution?</h4>
                    <p className="text-sm text-muted-foreground">
                      Contact our sales team for enterprise pricing and custom features
                    </p>
                  </div>
                </div>
                <Button>Contact Sales</Button>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Invoices Tab */}
          <TabsContent value="invoices" className="mt-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Invoice History</CardTitle>
                    <CardDescription>View and download your past invoices</CardDescription>
                  </div>
                  <Button variant="outline" size="sm">
                    <Download className="mr-2 h-4 w-4" />
                    Export All
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b text-left text-sm text-muted-foreground bg-muted/30">
                        <th className="py-3 px-4 font-medium">Invoice</th>
                        <th className="py-3 font-medium">Date</th>
                        <th className="py-3 font-medium">Description</th>
                        <th className="py-3 font-medium">Amount</th>
                        <th className="py-3 font-medium">Status</th>
                        <th className="py-3 px-4 font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {invoices.map((invoice) => (
                        <tr key={invoice.id} className="border-b last:border-0 hover:bg-muted/30">
                          <td className="py-4 px-4">
                            <div className="flex items-center gap-2">
                              <FileText className="h-4 w-4 text-muted-foreground" />
                              <span className="font-medium font-mono text-sm">{invoice.id}</span>
                            </div>
                          </td>
                          <td className="py-4 text-sm">{formatDate(invoice.date)}</td>
                          <td className="py-4 text-sm text-muted-foreground">{invoice.description}</td>
                          <td className="py-4 font-medium">{formatCurrency(invoice.amount)}</td>
                          <td className="py-4">
                            <InvoiceStatusBadge status={invoice.status} />
                          </td>
                          <td className="py-4 px-4">
                            <div className="flex items-center gap-2">
                              <Button variant="ghost" size="sm">
                                <Download className="mr-1 h-4 w-4" />
                                PDF
                              </Button>
                              <Button variant="ghost" size="sm">
                                <ExternalLink className="mr-1 h-4 w-4" />
                                View
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Payment Methods Tab */}
          <TabsContent value="payment" className="mt-4 space-y-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Payment Methods</CardTitle>
                    <CardDescription>Manage your payment methods</CardDescription>
                  </div>
                  <Button size="sm">
                    <Plus className="mr-2 h-4 w-4" />
                    Add Payment Method
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {paymentMethods.map((method) => (
                  <PaymentMethodCard
                    key={method.id}
                    method={method}
                    onEdit={() => {}}
                    onRemove={() => {}}
                    onSetDefault={() => {}}
                  />
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Billing Address</CardTitle>
                <CardDescription>Used for invoices and receipts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <p className="font-medium">Acme Inc.</p>
                    <p className="text-sm text-muted-foreground">123 Main Street</p>
                    <p className="text-sm text-muted-foreground">San Francisco, CA 94102</p>
                    <p className="text-sm text-muted-foreground">United States</p>
                  </div>
                  <Button variant="outline" size="sm">
                    <Edit className="mr-2 h-4 w-4" />
                    Edit
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card className="border-yellow-200 bg-yellow-50/50 dark:bg-yellow-950/20">
              <CardContent className="p-4 flex items-start gap-3">
                <Shield className="h-5 w-5 text-yellow-600 mt-0.5" />
                <div>
                  <p className="font-medium text-yellow-800 dark:text-yellow-200">Secure Payments</p>
                  <p className="text-sm text-yellow-700 dark:text-yellow-300">
                    Your payment information is encrypted and securely stored. We use Stripe for payment processing
                    and never store your card details on our servers.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Upgrade Dialog */}
      <Dialog open={showUpgradeDialog} onOpenChange={setShowUpgradeDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Upgrade Your Plan</DialogTitle>
            <DialogDescription>
              You're about to upgrade to the {selectedPlan?.name || "new"} plan
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <div className="p-4 bg-muted rounded-lg mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">New Plan</span>
                <Badge>{selectedPlan?.name || "Selected Plan"}</Badge>
              </div>
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>Monthly cost</span>
                <span>{selectedPlan?.price ? formatCurrency(selectedPlan.price) : "Custom"}</span>
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              Your new plan will be effective immediately. You'll be charged a prorated amount for the
              remainder of this billing period.
            </p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowUpgradeDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleUpgrade} disabled={upgradeMutation.isPending}>
              {upgradeMutation.isPending ? "Processing..." : "Confirm Upgrade"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </DashboardLayout>
  );
}
