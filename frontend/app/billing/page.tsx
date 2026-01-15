"use client";

import React, { useState } from "react";
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
  Calendar,
  FileText,
  ExternalLink,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn, formatNumber } from "@/lib/utils";

// Mock billing data
const currentPlan = {
  name: "Professional",
  price: 99,
  billingCycle: "monthly",
  nextBillingDate: "2024-02-01",
};

const usage = {
  calls: { used: 1247, limit: 5000 },
  minutes: { used: 3842, limit: 10000 },
  agents: { used: 5, limit: 10 },
  teamMembers: { used: 4, limit: 10 },
};

const plans = [
  {
    id: "free",
    name: "Free",
    price: 0,
    features: ["100 calls/month", "500 minutes", "1 agent", "Community support"],
    limits: { calls: 100, minutes: 500, agents: 1, team: 1 },
    popular: false,
  },
  {
    id: "starter",
    name: "Starter",
    price: 29,
    features: ["1,000 calls/month", "2,500 minutes", "3 agents", "Email support"],
    limits: { calls: 1000, minutes: 2500, agents: 3, team: 3 },
    popular: false,
  },
  {
    id: "professional",
    name: "Professional",
    price: 99,
    features: ["5,000 calls/month", "10,000 minutes", "10 agents", "Priority support", "Analytics", "Webhooks"],
    limits: { calls: 5000, minutes: 10000, agents: 10, team: 10 },
    popular: true,
    current: true,
  },
  {
    id: "enterprise",
    name: "Enterprise",
    price: null,
    features: ["Unlimited calls", "Unlimited minutes", "Unlimited agents", "24/7 support", "SLA", "Custom integrations"],
    limits: { calls: -1, minutes: -1, agents: -1, team: -1 },
    popular: false,
  },
];

const invoices = [
  { id: "INV-001", date: "2024-01-01", amount: 99, status: "paid" },
  { id: "INV-002", date: "2023-12-01", amount: 99, status: "paid" },
  { id: "INV-003", date: "2023-11-01", amount: 99, status: "paid" },
  { id: "INV-004", date: "2023-10-01", amount: 29, status: "paid" },
];

export default function BillingPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Billing</h1>
          <p className="text-muted-foreground">
            Manage your subscription and view usage
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <CreditCard className="mr-2 h-4 w-4" />
            Update Payment
          </Button>
        </div>
      </div>

      {/* Current Plan */}
      <Card className="border-primary">
        <CardContent className="p-6">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Badge className="bg-primary text-primary-foreground">Current Plan</Badge>
                <h2 className="text-2xl font-bold">{currentPlan.name}</h2>
              </div>
              <p className="text-muted-foreground mb-4">
                ${currentPlan.price}/month &bull; Next billing: {currentPlan.nextBillingDate}
              </p>
              <div className="flex gap-2">
                <Button variant="outline" size="sm">Change Plan</Button>
                <Button variant="ghost" size="sm" className="text-red-600 hover:text-red-700">
                  Cancel Subscription
                </Button>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground">This month's estimated cost</p>
              <p className="text-3xl font-bold">${currentPlan.price}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Usage Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3 mb-3">
              <Phone className="h-5 w-5 text-muted-foreground" />
              <span className="font-medium">Calls</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>{formatNumber(usage.calls.used)}</span>
                <span className="text-muted-foreground">/ {formatNumber(usage.calls.limit)}</span>
              </div>
              <div className="h-2 rounded-full bg-secondary">
                <div
                  className={cn(
                    "h-2 rounded-full transition-all",
                    (usage.calls.used / usage.calls.limit) > 0.9 ? "bg-red-500" :
                    (usage.calls.used / usage.calls.limit) > 0.7 ? "bg-yellow-500" : "bg-primary"
                  )}
                  style={{ width: `${(usage.calls.used / usage.calls.limit) * 100}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                {Math.round((usage.calls.used / usage.calls.limit) * 100)}% used
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3 mb-3">
              <Clock className="h-5 w-5 text-muted-foreground" />
              <span className="font-medium">Minutes</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>{formatNumber(usage.minutes.used)}</span>
                <span className="text-muted-foreground">/ {formatNumber(usage.minutes.limit)}</span>
              </div>
              <div className="h-2 rounded-full bg-secondary">
                <div
                  className="h-2 rounded-full bg-primary transition-all"
                  style={{ width: `${(usage.minutes.used / usage.minutes.limit) * 100}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                {Math.round((usage.minutes.used / usage.minutes.limit) * 100)}% used
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3 mb-3">
              <Bot className="h-5 w-5 text-muted-foreground" />
              <span className="font-medium">Agents</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>{usage.agents.used}</span>
                <span className="text-muted-foreground">/ {usage.agents.limit}</span>
              </div>
              <div className="h-2 rounded-full bg-secondary">
                <div
                  className="h-2 rounded-full bg-primary transition-all"
                  style={{ width: `${(usage.agents.used / usage.agents.limit) * 100}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                {usage.agents.limit - usage.agents.used} remaining
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3 mb-3">
              <Users className="h-5 w-5 text-muted-foreground" />
              <span className="font-medium">Team Members</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>{usage.teamMembers.used}</span>
                <span className="text-muted-foreground">/ {usage.teamMembers.limit}</span>
              </div>
              <div className="h-2 rounded-full bg-secondary">
                <div
                  className="h-2 rounded-full bg-primary transition-all"
                  style={{ width: `${(usage.teamMembers.used / usage.teamMembers.limit) * 100}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                {usage.teamMembers.limit - usage.teamMembers.used} remaining
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Plans Comparison */}
      <Card>
        <CardHeader>
          <CardTitle>Available Plans</CardTitle>
          <CardDescription>
            Choose the plan that best fits your needs
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {plans.map((plan) => (
              <div
                key={plan.id}
                className={cn(
                  "rounded-lg border p-4 relative",
                  plan.current && "border-primary ring-1 ring-primary",
                  plan.popular && "border-purple-500"
                )}
              >
                {plan.popular && (
                  <Badge className="absolute -top-2 left-4 bg-purple-500">
                    Most Popular
                  </Badge>
                )}
                {plan.current && (
                  <Badge className="absolute -top-2 right-4 bg-primary">
                    Current
                  </Badge>
                )}
                <h3 className="font-bold text-lg mt-2">{plan.name}</h3>
                <p className="text-2xl font-bold mt-2">
                  {plan.price !== null ? `$${plan.price}` : "Custom"}
                  {plan.price !== null && <span className="text-sm font-normal text-muted-foreground">/mo</span>}
                </p>
                <ul className="mt-4 space-y-2">
                  {plan.features.map((feature, i) => (
                    <li key={i} className="flex items-center gap-2 text-sm">
                      <Check className="h-4 w-4 text-green-600" />
                      {feature}
                    </li>
                  ))}
                </ul>
                <Button
                  className="w-full mt-4"
                  variant={plan.current ? "outline" : "default"}
                  disabled={plan.current}
                >
                  {plan.current ? "Current Plan" : plan.price === null ? "Contact Sales" : "Upgrade"}
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Invoice History */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Invoice History</CardTitle>
              <CardDescription>Your recent invoices and payments</CardDescription>
            </div>
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Download All
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b text-left text-sm text-muted-foreground">
                  <th className="pb-3 font-medium">Invoice</th>
                  <th className="pb-3 font-medium">Date</th>
                  <th className="pb-3 font-medium">Amount</th>
                  <th className="pb-3 font-medium">Status</th>
                  <th className="pb-3 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {invoices.map((invoice) => (
                  <tr key={invoice.id} className="border-b last:border-0">
                    <td className="py-3">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <span className="font-medium">{invoice.id}</span>
                      </div>
                    </td>
                    <td className="py-3">{invoice.date}</td>
                    <td className="py-3">${invoice.amount}.00</td>
                    <td className="py-3">
                      <Badge className="bg-green-100 text-green-800">
                        <Check className="mr-1 h-3 w-3" />
                        {invoice.status}
                      </Badge>
                    </td>
                    <td className="py-3">
                      <Button variant="ghost" size="sm">
                        <Download className="mr-1 h-4 w-4" />
                        PDF
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
