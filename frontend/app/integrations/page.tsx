"use client";

import React, { useState } from "react";
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
  Switch,
  Separator,
  ScrollArea,
} from "@/components/ui";
import {
  Search,
  Plus,
  Check,
  CheckCircle,
  ExternalLink,
  Settings,
  Trash2,
  RefreshCw,
  Zap,
  Database,
  Calendar,
  Mail,
  MessageSquare,
  Phone,
  CreditCard,
  FileText,
  Users,
  Globe,
  Lock,
  Unlock,
  Star,
  ArrowRight,
  ChevronRight,
  Puzzle,
  Code,
  Webhook,
  CloudCog,
  BarChart3,
  ShoppingCart,
  Headphones,
  Building,
  Briefcase,
  HelpCircle,
  AlertCircle,
  Bot,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";

// Types
interface Integration {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  status: "connected" | "available" | "coming_soon";
  isPremium: boolean;
  isPopular: boolean;
  features: string[];
  setupTime: string;
  docsUrl?: string;
  connectedAt?: string;
  config?: Record<string, any>;
}

interface IntegrationCategory {
  id: string;
  name: string;
  icon: React.ElementType;
  count: number;
}

// Integration icons mapping
const integrationIcons: Record<string, React.ReactNode> = {
  salesforce: <Building className="h-8 w-8" style={{ color: "#00A1E0" }} />,
  hubspot: <Building className="h-8 w-8" style={{ color: "#FF7A59" }} />,
  slack: <MessageSquare className="h-8 w-8" style={{ color: "#4A154B" }} />,
  zapier: <Zap className="h-8 w-8" style={{ color: "#FF4A00" }} />,
  google_calendar: <Calendar className="h-8 w-8" style={{ color: "#4285F4" }} />,
  outlook: <Mail className="h-8 w-8" style={{ color: "#0078D4" }} />,
  twilio: <Phone className="h-8 w-8" style={{ color: "#F22F46" }} />,
  stripe: <CreditCard className="h-8 w-8" style={{ color: "#635BFF" }} />,
  zendesk: <Headphones className="h-8 w-8" style={{ color: "#03363D" }} />,
  intercom: <MessageSquare className="h-8 w-8" style={{ color: "#1F8DED" }} />,
  notion: <FileText className="h-8 w-8" style={{ color: "#000000" }} />,
  airtable: <Database className="h-8 w-8" style={{ color: "#18BFFF" }} />,
  shopify: <ShoppingCart className="h-8 w-8" style={{ color: "#7AB55C" }} />,
  pipedrive: <Briefcase className="h-8 w-8" style={{ color: "#017737" }} />,
  freshdesk: <Headphones className="h-8 w-8" style={{ color: "#25C16F" }} />,
  calendly: <Calendar className="h-8 w-8" style={{ color: "#006BFF" }} />,
  mailchimp: <Mail className="h-8 w-8" style={{ color: "#FFE01B" }} />,
  sendgrid: <Mail className="h-8 w-8" style={{ color: "#1A82E2" }} />,
  segment: <BarChart3 className="h-8 w-8" style={{ color: "#52BD95" }} />,
  mixpanel: <BarChart3 className="h-8 w-8" style={{ color: "#7856FF" }} />,
  google_sheets: <FileText className="h-8 w-8" style={{ color: "#0F9D58" }} />,
  webhook: <Webhook className="h-8 w-8" style={{ color: "#6366F1" }} />,
  api: <Code className="h-8 w-8" style={{ color: "#1E293B" }} />,
  openai: <Sparkles className="h-8 w-8" style={{ color: "#000000" }} />,
};

// Mock integrations data
const mockIntegrations: Integration[] = [
  {
    id: "salesforce",
    name: "Salesforce",
    description: "Sync contacts, leads, and call data with Salesforce CRM",
    icon: "salesforce",
    category: "crm",
    status: "connected",
    isPremium: false,
    isPopular: true,
    features: ["Contact sync", "Lead creation", "Call logging", "Custom fields"],
    setupTime: "5 min",
    connectedAt: "2024-01-10T10:00:00Z",
    config: { syncContacts: true, createLeads: true },
  },
  {
    id: "hubspot",
    name: "HubSpot",
    description: "Connect with HubSpot CRM for contact and deal management",
    icon: "hubspot",
    category: "crm",
    status: "available",
    isPremium: false,
    isPopular: true,
    features: ["Contact sync", "Deal tracking", "Activity logging", "Workflows"],
    setupTime: "5 min",
  },
  {
    id: "slack",
    name: "Slack",
    description: "Get real-time notifications and alerts in Slack channels",
    icon: "slack",
    category: "communication",
    status: "connected",
    isPremium: false,
    isPopular: true,
    features: ["Call notifications", "Summary alerts", "Channel routing", "Commands"],
    setupTime: "2 min",
    connectedAt: "2024-01-08T14:30:00Z",
    config: { channel: "#voice-ai-alerts" },
  },
  {
    id: "zapier",
    name: "Zapier",
    description: "Connect to 5000+ apps through Zapier automations",
    icon: "zapier",
    category: "automation",
    status: "available",
    isPremium: false,
    isPopular: true,
    features: ["Triggers", "Actions", "Multi-step Zaps", "Filters"],
    setupTime: "3 min",
  },
  {
    id: "google_calendar",
    name: "Google Calendar",
    description: "Schedule appointments and sync with Google Calendar",
    icon: "google_calendar",
    category: "scheduling",
    status: "connected",
    isPremium: false,
    isPopular: true,
    features: ["Booking sync", "Availability check", "Event creation", "Reminders"],
    setupTime: "2 min",
    connectedAt: "2024-01-05T09:00:00Z",
  },
  {
    id: "outlook",
    name: "Microsoft Outlook",
    description: "Sync with Outlook calendar and email",
    icon: "outlook",
    category: "scheduling",
    status: "available",
    isPremium: false,
    isPopular: false,
    features: ["Calendar sync", "Email integration", "Contact sync", "Meeting links"],
    setupTime: "3 min",
  },
  {
    id: "twilio",
    name: "Twilio",
    description: "Use your own Twilio account for phone numbers",
    icon: "twilio",
    category: "telephony",
    status: "available",
    isPremium: true,
    isPopular: true,
    features: ["BYOC", "Custom numbers", "SMS", "SIP trunking"],
    setupTime: "10 min",
  },
  {
    id: "stripe",
    name: "Stripe",
    description: "Process payments and manage subscriptions",
    icon: "stripe",
    category: "payments",
    status: "available",
    isPremium: false,
    isPopular: true,
    features: ["Payment links", "Invoicing", "Subscription management", "Refunds"],
    setupTime: "5 min",
  },
  {
    id: "zendesk",
    name: "Zendesk",
    description: "Create and manage support tickets in Zendesk",
    icon: "zendesk",
    category: "support",
    status: "available",
    isPremium: false,
    isPopular: true,
    features: ["Ticket creation", "Status updates", "Agent routing", "Macros"],
    setupTime: "5 min",
  },
  {
    id: "intercom",
    name: "Intercom",
    description: "Sync conversations and user data with Intercom",
    icon: "intercom",
    category: "support",
    status: "available",
    isPremium: false,
    isPopular: false,
    features: ["Conversation sync", "User profiles", "Tags", "Custom attributes"],
    setupTime: "5 min",
  },
  {
    id: "notion",
    name: "Notion",
    description: "Log call data and notes to Notion databases",
    icon: "notion",
    category: "productivity",
    status: "available",
    isPremium: false,
    isPopular: true,
    features: ["Database entries", "Page creation", "Templates", "Properties"],
    setupTime: "3 min",
  },
  {
    id: "airtable",
    name: "Airtable",
    description: "Sync data with Airtable bases and tables",
    icon: "airtable",
    category: "productivity",
    status: "available",
    isPremium: false,
    isPopular: true,
    features: ["Record sync", "Views", "Automations", "Custom fields"],
    setupTime: "3 min",
  },
  {
    id: "shopify",
    name: "Shopify",
    description: "Access order info and customer data from Shopify",
    icon: "shopify",
    category: "ecommerce",
    status: "available",
    isPremium: true,
    isPopular: true,
    features: ["Order lookup", "Customer data", "Inventory check", "Returns"],
    setupTime: "5 min",
  },
  {
    id: "pipedrive",
    name: "Pipedrive",
    description: "Sync deals and contacts with Pipedrive CRM",
    icon: "pipedrive",
    category: "crm",
    status: "available",
    isPremium: false,
    isPopular: false,
    features: ["Deal sync", "Contact management", "Activities", "Notes"],
    setupTime: "5 min",
  },
  {
    id: "freshdesk",
    name: "Freshdesk",
    description: "Create and manage tickets in Freshdesk",
    icon: "freshdesk",
    category: "support",
    status: "coming_soon",
    isPremium: false,
    isPopular: false,
    features: ["Ticket creation", "Agent routing", "Canned responses", "SLA"],
    setupTime: "5 min",
  },
  {
    id: "calendly",
    name: "Calendly",
    description: "Book appointments through Calendly scheduling",
    icon: "calendly",
    category: "scheduling",
    status: "available",
    isPremium: false,
    isPopular: true,
    features: ["Event types", "Availability", "Booking links", "Confirmations"],
    setupTime: "2 min",
  },
  {
    id: "mailchimp",
    name: "Mailchimp",
    description: "Add contacts to Mailchimp lists and campaigns",
    icon: "mailchimp",
    category: "marketing",
    status: "available",
    isPremium: false,
    isPopular: false,
    features: ["List management", "Tags", "Campaigns", "Automations"],
    setupTime: "3 min",
  },
  {
    id: "sendgrid",
    name: "SendGrid",
    description: "Send transactional emails via SendGrid",
    icon: "sendgrid",
    category: "marketing",
    status: "coming_soon",
    isPremium: false,
    isPopular: false,
    features: ["Email sending", "Templates", "Tracking", "Analytics"],
    setupTime: "5 min",
  },
  {
    id: "segment",
    name: "Segment",
    description: "Send events and data to Segment for analytics",
    icon: "segment",
    category: "analytics",
    status: "available",
    isPremium: true,
    isPopular: false,
    features: ["Event tracking", "Identify", "Track", "Page"],
    setupTime: "5 min",
  },
  {
    id: "mixpanel",
    name: "Mixpanel",
    description: "Track user events and analytics in Mixpanel",
    icon: "mixpanel",
    category: "analytics",
    status: "coming_soon",
    isPremium: true,
    isPopular: false,
    features: ["Event tracking", "User profiles", "Funnels", "Retention"],
    setupTime: "5 min",
  },
  {
    id: "google_sheets",
    name: "Google Sheets",
    description: "Log call data to Google Sheets spreadsheets",
    icon: "google_sheets",
    category: "productivity",
    status: "available",
    isPremium: false,
    isPopular: true,
    features: ["Row creation", "Cell updates", "Multiple sheets", "Formulas"],
    setupTime: "2 min",
  },
  {
    id: "webhook",
    name: "Custom Webhook",
    description: "Send data to any HTTP endpoint",
    icon: "webhook",
    category: "developer",
    status: "connected",
    isPremium: false,
    isPopular: false,
    features: ["HTTP POST", "Custom headers", "Retry logic", "Authentication"],
    setupTime: "1 min",
    connectedAt: "2024-01-12T16:00:00Z",
  },
  {
    id: "api",
    name: "REST API",
    description: "Full API access for custom integrations",
    icon: "api",
    category: "developer",
    status: "available",
    isPremium: false,
    isPopular: false,
    features: ["Full CRUD", "Webhooks", "Authentication", "Rate limiting"],
    setupTime: "Varies",
  },
  {
    id: "openai",
    name: "OpenAI (Custom)",
    description: "Use your own OpenAI API key for AI features",
    icon: "openai",
    category: "ai",
    status: "available",
    isPremium: true,
    isPopular: true,
    features: ["GPT-4", "Custom prompts", "Fine-tuning", "Usage tracking"],
    setupTime: "2 min",
  },
];

const categories: IntegrationCategory[] = [
  { id: "all", name: "All Integrations", icon: Puzzle, count: mockIntegrations.length },
  { id: "crm", name: "CRM", icon: Building, count: mockIntegrations.filter(i => i.category === "crm").length },
  { id: "communication", name: "Communication", icon: MessageSquare, count: mockIntegrations.filter(i => i.category === "communication").length },
  { id: "scheduling", name: "Scheduling", icon: Calendar, count: mockIntegrations.filter(i => i.category === "scheduling").length },
  { id: "support", name: "Support", icon: Headphones, count: mockIntegrations.filter(i => i.category === "support").length },
  { id: "productivity", name: "Productivity", icon: FileText, count: mockIntegrations.filter(i => i.category === "productivity").length },
  { id: "automation", name: "Automation", icon: Zap, count: mockIntegrations.filter(i => i.category === "automation").length },
  { id: "marketing", name: "Marketing", icon: Mail, count: mockIntegrations.filter(i => i.category === "marketing").length },
  { id: "analytics", name: "Analytics", icon: BarChart3, count: mockIntegrations.filter(i => i.category === "analytics").length },
  { id: "ecommerce", name: "E-commerce", icon: ShoppingCart, count: mockIntegrations.filter(i => i.category === "ecommerce").length },
  { id: "telephony", name: "Telephony", icon: Phone, count: mockIntegrations.filter(i => i.category === "telephony").length },
  { id: "payments", name: "Payments", icon: CreditCard, count: mockIntegrations.filter(i => i.category === "payments").length },
  { id: "developer", name: "Developer", icon: Code, count: mockIntegrations.filter(i => i.category === "developer").length },
  { id: "ai", name: "AI & ML", icon: Sparkles, count: mockIntegrations.filter(i => i.category === "ai").length },
];

// Utility functions
const formatDate = (dateString: string): string => {
  return new Date(dateString).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
};

// Components
function IntegrationCard({
  integration,
  onConnect,
  onManage,
  onDisconnect,
}: {
  integration: Integration;
  onConnect: () => void;
  onManage: () => void;
  onDisconnect: () => void;
}) {
  const isConnected = integration.status === "connected";
  const isComingSoon = integration.status === "coming_soon";

  return (
    <Card
      className={cn(
        "relative hover:shadow-md transition-all group",
        isConnected && "border-green-200 bg-green-50/30 dark:bg-green-950/10",
        isComingSoon && "opacity-60"
      )}
    >
      {integration.isPremium && (
        <Badge className="absolute -top-2 -right-2 bg-gradient-to-r from-yellow-500 to-orange-500 text-white border-0">
          <Star className="h-3 w-3 mr-1" />
          Premium
        </Badge>
      )}
      {integration.isPopular && !integration.isPremium && (
        <Badge className="absolute -top-2 -right-2 bg-primary text-primary-foreground">
          Popular
        </Badge>
      )}
      <CardContent className="p-5">
        <div className="flex items-start gap-4">
          <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-muted">
            {integrationIcons[integration.icon] || <Puzzle className="h-8 w-8" />}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-semibold">{integration.name}</h3>
              {isConnected && (
                <Badge variant="outline" className="text-xs bg-green-100 text-green-700 border-green-200">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Connected
                </Badge>
              )}
              {isComingSoon && (
                <Badge variant="outline" className="text-xs">
                  Coming Soon
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
              {integration.description}
            </p>
            <div className="flex flex-wrap gap-1 mb-3">
              {integration.features.slice(0, 3).map((feature) => (
                <Badge key={feature} variant="secondary" className="text-xs">
                  {feature}
                </Badge>
              ))}
              {integration.features.length > 3 && (
                <Badge variant="secondary" className="text-xs">
                  +{integration.features.length - 3}
                </Badge>
              )}
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">
                Setup: {integration.setupTime}
              </span>
              {isConnected ? (
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" onClick={onManage}>
                    <Settings className="h-3.5 w-3.5 mr-1" />
                    Manage
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onDisconnect}
                    className="text-red-600 hover:text-red-700 hover:bg-red-50"
                  >
                    Disconnect
                  </Button>
                </div>
              ) : isComingSoon ? (
                <Button variant="outline" size="sm" disabled>
                  Notify Me
                </Button>
              ) : (
                <Button size="sm" onClick={onConnect}>
                  <Plus className="h-3.5 w-3.5 mr-1" />
                  Connect
                </Button>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function ConnectedIntegrationRow({
  integration,
  onManage,
  onRefresh,
  onDisconnect,
}: {
  integration: Integration;
  onManage: () => void;
  onRefresh: () => void;
  onDisconnect: () => void;
}) {
  return (
    <div className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-muted/30 transition-colors">
      <div className="flex items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted">
          {integrationIcons[integration.icon] || <Puzzle className="h-6 w-6" />}
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h4 className="font-medium">{integration.name}</h4>
            <Badge variant="outline" className="text-xs bg-green-100 text-green-700 border-green-200">
              <CheckCircle className="h-3 w-3 mr-1" />
              Active
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground">
            Connected {integration.connectedAt ? formatDate(integration.connectedAt) : "recently"}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon-sm" onClick={onRefresh}>
          <RefreshCw className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="sm" onClick={onManage}>
          <Settings className="h-4 w-4 mr-1" />
          Settings
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onDisconnect}
          className="text-red-500 hover:text-red-600 hover:bg-red-50"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

function IntegrationSetupDialog({
  open,
  onOpenChange,
  integration,
  onConnect,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  integration: Integration | null;
  onConnect: () => void;
}) {
  const [step, setStep] = useState(1);

  if (!integration) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted">
              {integrationIcons[integration.icon] || <Puzzle className="h-6 w-6" />}
            </div>
            <div>
              <DialogTitle>Connect {integration.name}</DialogTitle>
              <DialogDescription>
                Set up the integration in just a few steps
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="py-4">
          {/* Progress */}
          <div className="flex items-center justify-between mb-6">
            {[1, 2, 3].map((s) => (
              <React.Fragment key={s}>
                <div
                  className={cn(
                    "flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium",
                    step >= s
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground"
                  )}
                >
                  {step > s ? <Check className="h-4 w-4" /> : s}
                </div>
                {s < 3 && (
                  <div
                    className={cn(
                      "flex-1 h-1 mx-2 rounded",
                      step > s ? "bg-primary" : "bg-muted"
                    )}
                  />
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Step content */}
          {step === 1 && (
            <div className="space-y-4">
              <h4 className="font-medium">Features you'll get</h4>
              <div className="space-y-2">
                {integration.features.map((feature) => (
                  <div key={feature} className="flex items-center gap-2 text-sm">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span>{feature}</span>
                  </div>
                ))}
              </div>
              <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200">
                <div className="flex items-start gap-2">
                  <AlertCircle className="h-4 w-4 text-blue-600 mt-0.5" />
                  <div className="text-sm">
                    <p className="font-medium text-blue-700 dark:text-blue-300">
                      Estimated setup time: {integration.setupTime}
                    </p>
                    <p className="text-blue-600 dark:text-blue-400">
                      You can disconnect this integration at any time.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-4">
              <h4 className="font-medium">Connect your account</h4>
              <p className="text-sm text-muted-foreground">
                Click the button below to authorize Builder AI to connect with {integration.name}.
              </p>
              <Button className="w-full" size="lg">
                <ExternalLink className="mr-2 h-4 w-4" />
                Authorize with {integration.name}
              </Button>
              <Separator />
              <div className="text-center">
                <span className="text-sm text-muted-foreground">or connect manually</span>
              </div>
              <div className="space-y-3">
                <div className="space-y-2">
                  <Label>API Key</Label>
                  <Input type="password" placeholder="Enter your API key" />
                </div>
                {integration.id === "salesforce" && (
                  <div className="space-y-2">
                    <Label>Instance URL</Label>
                    <Input placeholder="https://your-instance.salesforce.com" />
                  </div>
                )}
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="space-y-4">
              <h4 className="font-medium">Configure settings</h4>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">Sync contacts</p>
                    <p className="text-xs text-muted-foreground">
                      Automatically sync contact information
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">Log call data</p>
                    <p className="text-xs text-muted-foreground">
                      Create activity records for each call
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">Create new records</p>
                    <p className="text-xs text-muted-foreground">
                      Automatically create new contacts/leads
                    </p>
                  </div>
                  <Switch />
                </div>
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          {step > 1 && (
            <Button variant="outline" onClick={() => setStep(step - 1)}>
              Back
            </Button>
          )}
          {step < 3 ? (
            <Button onClick={() => setStep(step + 1)}>
              Continue
              <ChevronRight className="ml-1 h-4 w-4" />
            </Button>
          ) : (
            <Button onClick={onConnect}>
              <Check className="mr-2 h-4 w-4" />
              Complete Setup
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function IntegrationsPage() {
  const [integrations, setIntegrations] = useState<Integration[]>(mockIntegrations);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [activeTab, setActiveTab] = useState("browse");
  const [showSetupDialog, setShowSetupDialog] = useState(false);
  const [selectedIntegration, setSelectedIntegration] = useState<Integration | null>(null);

  // Filter integrations
  const filteredIntegrations = integrations.filter((integration) => {
    const matchesSearch =
      integration.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      integration.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory =
      selectedCategory === "all" || integration.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const connectedIntegrations = integrations.filter((i) => i.status === "connected");

  // Handlers
  const handleConnect = (integration: Integration) => {
    setSelectedIntegration(integration);
    setShowSetupDialog(true);
  };

  const handleCompleteSetup = () => {
    if (selectedIntegration) {
      setIntegrations((prev) =>
        prev.map((i) =>
          i.id === selectedIntegration.id
            ? { ...i, status: "connected" as const, connectedAt: new Date().toISOString() }
            : i
        )
      );
    }
    setShowSetupDialog(false);
    setSelectedIntegration(null);
  };

  const handleDisconnect = (integrationId: string) => {
    setIntegrations((prev) =>
      prev.map((i) =>
        i.id === integrationId
          ? { ...i, status: "available" as const, connectedAt: undefined }
          : i
      )
    );
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Integrations</h1>
            <p className="text-muted-foreground">
              Connect your favorite tools and services
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" asChild>
              <Link href="/webhooks">
                <Webhook className="mr-2 h-4 w-4" />
                Webhooks
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/settings/api-keys">
                <Code className="mr-2 h-4 w-4" />
                API Keys
              </Link>
            </Button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardContent className="p-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30">
                <CheckCircle className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{connectedIntegrations.length}</p>
                <p className="text-sm text-muted-foreground">Connected</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/30">
                <Puzzle className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {integrations.filter((i) => i.status === "available").length}
                </p>
                <p className="text-sm text-muted-foreground">Available</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900/30">
                <Zap className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {integrations.filter((i) => i.isPopular).length}
                </p>
                <p className="text-sm text-muted-foreground">Popular</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100 dark:bg-yellow-900/30">
                <Star className="h-5 w-5 text-yellow-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {integrations.filter((i) => i.isPremium).length}
                </p>
                <p className="text-sm text-muted-foreground">Premium</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <div className="flex items-center justify-between flex-wrap gap-4">
            <TabsList>
              <TabsTrigger value="browse" className="gap-2">
                <Puzzle className="h-4 w-4" />
                Browse
              </TabsTrigger>
              <TabsTrigger value="connected" className="gap-2">
                <CheckCircle className="h-4 w-4" />
                Connected
                {connectedIntegrations.length > 0 && (
                  <Badge variant="secondary" className="ml-1">
                    {connectedIntegrations.length}
                  </Badge>
                )}
              </TabsTrigger>
            </TabsList>

            {activeTab === "browse" && (
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search integrations..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 w-64"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Browse Tab */}
          <TabsContent value="browse" className="mt-6">
            <div className="grid gap-6 lg:grid-cols-[240px_1fr]">
              {/* Categories Sidebar */}
              <Card className="h-fit">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Categories</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <ScrollArea className="h-[500px]">
                    <div className="space-y-1 p-2">
                      {categories.map((category) => {
                        const Icon = category.icon;
                        return (
                          <button
                            key={category.id}
                            className={cn(
                              "flex items-center justify-between w-full px-3 py-2 rounded-lg text-sm transition-colors",
                              selectedCategory === category.id
                                ? "bg-primary text-primary-foreground"
                                : "hover:bg-muted"
                            )}
                            onClick={() => setSelectedCategory(category.id)}
                          >
                            <div className="flex items-center gap-2">
                              <Icon className="h-4 w-4" />
                              <span>{category.name}</span>
                            </div>
                            <Badge
                              variant="secondary"
                              className={cn(
                                "text-xs",
                                selectedCategory === category.id &&
                                  "bg-primary-foreground/20 text-primary-foreground"
                              )}
                            >
                              {category.count}
                            </Badge>
                          </button>
                        );
                      })}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>

              {/* Integrations Grid */}
              <div className="space-y-4">
                {filteredIntegrations.length > 0 ? (
                  <div className="grid gap-4 md:grid-cols-2">
                    {filteredIntegrations.map((integration) => (
                      <IntegrationCard
                        key={integration.id}
                        integration={integration}
                        onConnect={() => handleConnect(integration)}
                        onManage={() => {}}
                        onDisconnect={() => handleDisconnect(integration.id)}
                      />
                    ))}
                  </div>
                ) : (
                  <Card>
                    <CardContent className="p-12 text-center">
                      <Puzzle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                      <h3 className="text-lg font-medium mb-1">No integrations found</h3>
                      <p className="text-sm text-muted-foreground">
                        Try adjusting your search or category filter
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>

          {/* Connected Tab */}
          <TabsContent value="connected" className="mt-6">
            {connectedIntegrations.length > 0 ? (
              <div className="space-y-3">
                {connectedIntegrations.map((integration) => (
                  <ConnectedIntegrationRow
                    key={integration.id}
                    integration={integration}
                    onManage={() => {}}
                    onRefresh={() => {}}
                    onDisconnect={() => handleDisconnect(integration.id)}
                  />
                ))}
              </div>
            ) : (
              <Card>
                <CardContent className="p-12 text-center">
                  <CheckCircle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium mb-1">No connected integrations</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Connect your first integration to get started
                  </p>
                  <Button onClick={() => setActiveTab("browse")}>
                    <Plus className="mr-2 h-4 w-4" />
                    Browse Integrations
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>

        {/* Request Integration */}
        <Card className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-950/30 dark:to-blue-950/30 border-purple-200">
          <CardContent className="p-6 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-purple-100 dark:bg-purple-900/50">
                <HelpCircle className="h-6 w-6 text-purple-600" />
              </div>
              <div>
                <h3 className="font-semibold">Don't see what you need?</h3>
                <p className="text-sm text-muted-foreground">
                  Request a new integration and we'll consider adding it
                </p>
              </div>
            </div>
            <Button variant="outline">
              Request Integration
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Setup Dialog */}
      <IntegrationSetupDialog
        open={showSetupDialog}
        onOpenChange={setShowSetupDialog}
        integration={selectedIntegration}
        onConnect={handleCompleteSetup}
      />
    </DashboardLayout>
  );
}
