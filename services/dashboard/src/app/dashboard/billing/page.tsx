"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  CreditCard,
  Download,
  Plus,
  CheckCircle,
  AlertCircle,
  Clock,
  TrendingUp,
  Phone,
  Bot,
  Zap,
  ArrowUpRight,
  FileText,
  DollarSign,
  Calendar,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { billingApi } from "@/lib/api";
import { formatCurrency, formatRelativeTime, cn } from "@/lib/utils";

// Types
interface Subscription {
  id: string;
  plan: "starter" | "professional" | "enterprise";
  status: "active" | "past_due" | "canceled" | "trialing";
  current_period_start: string;
  current_period_end: string;
  cancel_at_period_end: boolean;
}

interface UsageData {
  minutes_used: number;
  minutes_limit: number;
  calls_made: number;
  calls_limit: number;
  agents_active: number;
  agents_limit: number;
  api_calls: number;
  api_calls_limit: number;
}

interface Invoice {
  id: string;
  number: string;
  amount: number;
  status: "paid" | "pending" | "failed";
  created_at: string;
  paid_at?: string;
  pdf_url?: string;
}

interface PaymentMethod {
  id: string;
  type: "card";
  brand: string;
  last4: string;
  exp_month: number;
  exp_year: number;
  is_default: boolean;
}

// Plan details
const plans = {
  starter: {
    name: "Starter",
    price: 49,
    features: [
      "500 minutes/month",
      "2 AI agents",
      "Basic analytics",
      "Email support",
      "1 phone number",
    ],
  },
  professional: {
    name: "Professional",
    price: 199,
    features: [
      "2,500 minutes/month",
      "10 AI agents",
      "Advanced analytics",
      "Priority support",
      "5 phone numbers",
      "Custom voices",
      "API access",
    ],
  },
  enterprise: {
    name: "Enterprise",
    price: null,
    features: [
      "Unlimited minutes",
      "Unlimited agents",
      "Enterprise analytics",
      "24/7 support",
      "Unlimited phone numbers",
      "Custom voices",
      "Full API access",
      "SLA guarantee",
      "Dedicated account manager",
    ],
  },
};

// Usage Card
function UsageCard({
  title,
  used,
  limit,
  icon: Icon,
}: {
  title: string;
  used: number;
  limit: number;
  icon: React.ComponentType<{ className?: string }>;
}) {
  const percentage = limit > 0 ? (used / limit) * 100 : 0;
  const isNearLimit = percentage >= 80;
  const isOverLimit = percentage >= 100;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline justify-between mb-2">
          <span className="text-2xl font-bold">{used.toLocaleString()}</span>
          <span className="text-sm text-muted-foreground">/ {limit.toLocaleString()}</span>
        </div>
        <Progress
          value={Math.min(percentage, 100)}
          className={cn(
            "h-2",
            isOverLimit && "[&>div]:bg-destructive",
            isNearLimit && !isOverLimit && "[&>div]:bg-warning"
          )}
        />
        {isNearLimit && (
          <p className="text-xs text-muted-foreground mt-2">
            {isOverLimit ? (
              <span className="text-destructive">Over limit - upgrade your plan</span>
            ) : (
              <span className="text-warning">Approaching limit</span>
            )}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// Current Plan Card
function CurrentPlanCard({
  subscription,
  onUpgrade,
  onCancel,
}: {
  subscription: Subscription;
  onUpgrade: () => void;
  onCancel: () => void;
}) {
  const plan = plans[subscription.plan];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Current Plan</CardTitle>
            <CardDescription>Your subscription details</CardDescription>
          </div>
          <Badge
            variant={
              subscription.status === "active"
                ? "default"
                : subscription.status === "trialing"
                ? "secondary"
                : "destructive"
            }
            className={cn(
              subscription.status === "active" && "bg-success text-success-foreground"
            )}
          >
            {subscription.status === "active" && <CheckCircle className="mr-1 h-3 w-3" />}
            {subscription.status === "past_due" && <AlertCircle className="mr-1 h-3 w-3" />}
            {subscription.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-2xl font-bold">{plan.name}</h3>
            <p className="text-muted-foreground">
              {plan.price ? `$${plan.price}/month` : "Custom pricing"}
            </p>
          </div>
          {subscription.plan !== "enterprise" && (
            <Button onClick={onUpgrade}>
              <ArrowUpRight className="mr-2 h-4 w-4" />
              Upgrade
            </Button>
          )}
        </div>

        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">Plan includes:</p>
          <ul className="grid grid-cols-2 gap-2">
            {plan.features.map((feature, i) => (
              <li key={i} className="flex items-center gap-2 text-sm">
                <CheckCircle className="h-4 w-4 text-success" />
                {feature}
              </li>
            ))}
          </ul>
        </div>

        <div className="flex items-center justify-between pt-4 border-t">
          <div className="text-sm text-muted-foreground">
            <Calendar className="inline h-4 w-4 mr-1" />
            {subscription.cancel_at_period_end
              ? `Cancels on ${new Date(subscription.current_period_end).toLocaleDateString()}`
              : `Renews on ${new Date(subscription.current_period_end).toLocaleDateString()}`}
          </div>
          {!subscription.cancel_at_period_end && (
            <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={onCancel}>
              Cancel subscription
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// Payment Methods Card
function PaymentMethodsCard({
  paymentMethods,
  onAdd,
  onSetDefault,
  onRemove,
}: {
  paymentMethods: PaymentMethod[];
  onAdd: () => void;
  onSetDefault: (id: string) => void;
  onRemove: (id: string) => void;
}) {
  const cardBrandIcons: Record<string, string> = {
    visa: "💳",
    mastercard: "💳",
    amex: "💳",
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Payment Methods</CardTitle>
            <CardDescription>Manage your payment methods</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={onAdd}>
            <Plus className="mr-2 h-4 w-4" />
            Add Card
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {paymentMethods.length === 0 ? (
          <div className="text-center py-8">
            <CreditCard className="mx-auto h-12 w-12 text-muted-foreground/50" />
            <p className="mt-4 text-sm text-muted-foreground">No payment methods added</p>
          </div>
        ) : (
          <div className="space-y-3">
            {paymentMethods.map((method) => (
              <div
                key={method.id}
                className="flex items-center justify-between rounded-lg border p-4"
              >
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded bg-muted text-xl">
                    {cardBrandIcons[method.brand.toLowerCase()] || "💳"}
                  </div>
                  <div>
                    <p className="font-medium">
                      {method.brand} •••• {method.last4}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Expires {method.exp_month}/{method.exp_year}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {method.is_default ? (
                    <Badge variant="outline">Default</Badge>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onSetDefault(method.id)}
                    >
                      Set default
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-destructive hover:text-destructive"
                    onClick={() => onRemove(method.id)}
                    disabled={method.is_default}
                  >
                    ×
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Invoices Table
function InvoicesTable({ invoices }: { invoices: Invoice[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Billing History</CardTitle>
        <CardDescription>View and download past invoices</CardDescription>
      </CardHeader>
      <CardContent>
        {invoices.length === 0 ? (
          <div className="text-center py-8">
            <FileText className="mx-auto h-12 w-12 text-muted-foreground/50" />
            <p className="mt-4 text-sm text-muted-foreground">No invoices yet</p>
          </div>
        ) : (
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Invoice</TableHead>
                  <TableHead>Amount</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead className="w-[50px]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {invoices.map((invoice) => (
                  <TableRow key={invoice.id}>
                    <TableCell className="font-medium">{invoice.number}</TableCell>
                    <TableCell>{formatCurrency(invoice.amount)}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          invoice.status === "paid"
                            ? "default"
                            : invoice.status === "pending"
                            ? "secondary"
                            : "destructive"
                        }
                        className={cn(
                          invoice.status === "paid" && "bg-success text-success-foreground"
                        )}
                      >
                        {invoice.status === "paid" && <CheckCircle className="mr-1 h-3 w-3" />}
                        {invoice.status === "failed" && <AlertCircle className="mr-1 h-3 w-3" />}
                        {invoice.status === "pending" && <Clock className="mr-1 h-3 w-3" />}
                        {invoice.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {new Date(invoice.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      {invoice.pdf_url && (
                        <Button
                          variant="ghost"
                          size="icon"
                          asChild
                        >
                          <a href={invoice.pdf_url} target="_blank" rel="noopener noreferrer">
                            <Download className="h-4 w-4" />
                          </a>
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Upgrade Dialog
function UpgradeDialog({
  open,
  onOpenChange,
  currentPlan,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  currentPlan: string;
}) {
  const [selectedPlan, setSelectedPlan] = useState<"professional" | "enterprise">("professional");
  const queryClient = useQueryClient();

  const upgradeMutation = useMutation({
    mutationFn: (plan: string) => billingApi.upgradePlan(plan),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["subscription"] });
      onOpenChange(false);
    },
  });

  const planOptions = [
    {
      id: "professional",
      name: "Professional",
      price: "$199/mo",
      description: "For growing businesses",
    },
    {
      id: "enterprise",
      name: "Enterprise",
      price: "Custom",
      description: "For large organizations",
    },
  ].filter((p) => p.id !== currentPlan);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Upgrade Your Plan</DialogTitle>
          <DialogDescription>Choose a plan that fits your needs</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {planOptions.map((plan) => (
            <div
              key={plan.id}
              className={cn(
                "flex items-center justify-between rounded-lg border p-4 cursor-pointer transition-colors",
                selectedPlan === plan.id
                  ? "border-primary bg-primary/5"
                  : "hover:border-muted-foreground/50"
              )}
              onClick={() => setSelectedPlan(plan.id as typeof selectedPlan)}
            >
              <div>
                <p className="font-medium">{plan.name}</p>
                <p className="text-sm text-muted-foreground">{plan.description}</p>
              </div>
              <div className="text-right">
                <p className="font-bold">{plan.price}</p>
                {selectedPlan === plan.id && (
                  <CheckCircle className="h-5 w-5 text-primary ml-auto mt-1" />
                )}
              </div>
            </div>
          ))}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => upgradeMutation.mutate(selectedPlan)}
            disabled={upgradeMutation.isPending}
          >
            {upgradeMutation.isPending
              ? "Processing..."
              : selectedPlan === "enterprise"
              ? "Contact Sales"
              : "Upgrade Now"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function BillingPage() {
  const [upgradeDialogOpen, setUpgradeDialogOpen] = useState(false);
  const queryClient = useQueryClient();

  const { data: subscriptionData } = useQuery({
    queryKey: ["subscription"],
    queryFn: () => billingApi.getSubscription(),
  });

  const { data: usageData } = useQuery({
    queryKey: ["usage"],
    queryFn: () => billingApi.getUsage(),
  });

  const { data: invoicesData } = useQuery({
    queryKey: ["invoices"],
    queryFn: () => billingApi.getInvoices(),
  });

  const { data: paymentMethodsData } = useQuery({
    queryKey: ["payment-methods"],
    queryFn: () => billingApi.getPaymentMethods(),
  });

  const cancelMutation = useMutation({
    mutationFn: () => billingApi.cancelSubscription(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["subscription"] });
    },
  });

  // Mock data
  const mockSubscription: Subscription = {
    id: "sub_1",
    plan: "professional",
    status: "active",
    current_period_start: "2024-01-01T00:00:00Z",
    current_period_end: "2024-02-01T00:00:00Z",
    cancel_at_period_end: false,
  };

  const mockUsage: UsageData = {
    minutes_used: 1847,
    minutes_limit: 2500,
    calls_made: 423,
    calls_limit: 1000,
    agents_active: 6,
    agents_limit: 10,
    api_calls: 15420,
    api_calls_limit: 50000,
  };

  const mockInvoices: Invoice[] = [
    {
      id: "inv_1",
      number: "INV-2024-001",
      amount: 199,
      status: "paid",
      created_at: "2024-01-01T00:00:00Z",
      paid_at: "2024-01-01T00:05:00Z",
      pdf_url: "#",
    },
    {
      id: "inv_2",
      number: "INV-2023-012",
      amount: 199,
      status: "paid",
      created_at: "2023-12-01T00:00:00Z",
      paid_at: "2023-12-01T00:03:00Z",
      pdf_url: "#",
    },
    {
      id: "inv_3",
      number: "INV-2023-011",
      amount: 199,
      status: "paid",
      created_at: "2023-11-01T00:00:00Z",
      paid_at: "2023-11-01T00:02:00Z",
      pdf_url: "#",
    },
  ];

  const mockPaymentMethods: PaymentMethod[] = [
    {
      id: "pm_1",
      type: "card",
      brand: "Visa",
      last4: "4242",
      exp_month: 12,
      exp_year: 2025,
      is_default: true,
    },
    {
      id: "pm_2",
      type: "card",
      brand: "Mastercard",
      last4: "8888",
      exp_month: 6,
      exp_year: 2024,
      is_default: false,
    },
  ];

  const subscription = subscriptionData || mockSubscription;
  const usage = usageData || mockUsage;
  const invoices = invoicesData?.invoices || mockInvoices;
  const paymentMethods = paymentMethodsData?.payment_methods || mockPaymentMethods;

  const handleCancel = () => {
    if (
      confirm(
        "Are you sure you want to cancel your subscription? You'll retain access until the end of your billing period."
      )
    ) {
      cancelMutation.mutate();
    }
  };

  const handleAddPaymentMethod = () => {
    // Would open Stripe Elements
    console.log("Add payment method");
  };

  const handleSetDefaultPaymentMethod = (id: string) => {
    console.log("Set default:", id);
  };

  const handleRemovePaymentMethod = (id: string) => {
    console.log("Remove:", id);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold">Billing</h1>
        <p className="text-muted-foreground">Manage your subscription and payment methods</p>
      </div>

      {/* Usage Overview */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Current Usage</h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <UsageCard
            title="Minutes Used"
            used={usage.minutes_used}
            limit={usage.minutes_limit}
            icon={Clock}
          />
          <UsageCard
            title="Calls Made"
            used={usage.calls_made}
            limit={usage.calls_limit}
            icon={Phone}
          />
          <UsageCard
            title="Active Agents"
            used={usage.agents_active}
            limit={usage.agents_limit}
            icon={Bot}
          />
          <UsageCard
            title="API Calls"
            used={usage.api_calls}
            limit={usage.api_calls_limit}
            icon={Zap}
          />
        </div>
      </div>

      {/* Subscription & Payment */}
      <div className="grid gap-6 lg:grid-cols-2">
        <CurrentPlanCard
          subscription={subscription}
          onUpgrade={() => setUpgradeDialogOpen(true)}
          onCancel={handleCancel}
        />
        <PaymentMethodsCard
          paymentMethods={paymentMethods}
          onAdd={handleAddPaymentMethod}
          onSetDefault={handleSetDefaultPaymentMethod}
          onRemove={handleRemovePaymentMethod}
        />
      </div>

      {/* Invoices */}
      <InvoicesTable invoices={invoices} />

      {/* Upgrade Dialog */}
      <UpgradeDialog
        open={upgradeDialogOpen}
        onOpenChange={setUpgradeDialogOpen}
        currentPlan={subscription.plan}
      />
    </div>
  );
}
