"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Search,
  Plus,
  Settings,
  CheckCircle,
  ExternalLink,
  Unplug,
  RefreshCw,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { integrationsApi } from "@/lib/api";
import { formatRelativeTime, cn } from "@/lib/utils";

// Integration types
interface Integration {
  id: string;
  name: string;
  slug: string;
  description: string;
  category: "crm" | "telephony" | "calendar" | "communication" | "analytics" | "ai" | "storage";
  logo_url: string;
  status: "connected" | "disconnected" | "error";
  connected_at?: string;
  config?: Record<string, unknown>;
}

// Integration logos (using placeholder colors)
const integrationLogos: Record<string, { bg: string; text: string }> = {
  salesforce: { bg: "bg-blue-500", text: "SF" },
  hubspot: { bg: "bg-orange-500", text: "HS" },
  pipedrive: { bg: "bg-green-500", text: "PD" },
  twilio: { bg: "bg-red-500", text: "TW" },
  vonage: { bg: "bg-purple-500", text: "VG" },
  "google-calendar": { bg: "bg-blue-600", text: "GC" },
  calendly: { bg: "bg-blue-400", text: "CL" },
  slack: { bg: "bg-purple-600", text: "SL" },
  "microsoft-teams": { bg: "bg-blue-700", text: "MT" },
  zapier: { bg: "bg-orange-600", text: "ZP" },
  segment: { bg: "bg-green-600", text: "SG" },
  openai: { bg: "bg-gray-900", text: "AI" },
  anthropic: { bg: "bg-amber-700", text: "AN" },
  elevenlabs: { bg: "bg-pink-500", text: "11" },
  deepgram: { bg: "bg-teal-500", text: "DG" },
  aws: { bg: "bg-yellow-600", text: "AWS" },
  gcs: { bg: "bg-blue-500", text: "GCS" },
};

// Integration Card
function IntegrationCard({
  integration,
  onConnect,
  onDisconnect,
  onConfigure,
}: {
  integration: Integration;
  onConnect: (id: string) => void;
  onDisconnect: (id: string) => void;
  onConfigure: (integration: Integration) => void;
}) {
  const logo = integrationLogos[integration.slug] || { bg: "bg-gray-500", text: integration.name.substring(0, 2).toUpperCase() };

  return (
    <Card className="transition-all hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "flex h-12 w-12 items-center justify-center rounded-lg text-white font-bold",
                logo.bg
              )}
            >
              {logo.text}
            </div>
            <div>
              <CardTitle className="text-base">{integration.name}</CardTitle>
              <Badge
                variant={
                  integration.status === "connected"
                    ? "default"
                    : integration.status === "error"
                    ? "destructive"
                    : "secondary"
                }
                className={cn(
                  "mt-1",
                  integration.status === "connected" && "bg-success text-success-foreground"
                )}
              >
                {integration.status === "connected" && <CheckCircle className="mr-1 h-3 w-3" />}
                {integration.status === "error" && <AlertCircle className="mr-1 h-3 w-3" />}
                {integration.status}
              </Badge>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">{integration.description}</p>
        <div className="flex items-center justify-between">
          {integration.status === "connected" ? (
            <>
              <span className="text-xs text-muted-foreground">
                Connected {formatRelativeTime(integration.connected_at!)}
              </span>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" onClick={() => onConfigure(integration)}>
                  <Settings className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:text-destructive"
                  onClick={() => onDisconnect(integration.id)}
                >
                  <Unplug className="h-4 w-4" />
                </Button>
              </div>
            </>
          ) : (
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={() => onConnect(integration.id)}
            >
              <Plus className="mr-2 h-4 w-4" />
              Connect
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// Connect Dialog
function ConnectDialog({
  open,
  onOpenChange,
  integration,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  integration: Integration | null;
}) {
  const queryClient = useQueryClient();
  const [apiKey, setApiKey] = useState("");
  const [selectedAccount, setSelectedAccount] = useState("");

  const connectMutation = useMutation({
    mutationFn: (data: { integrationId: string; config: Record<string, string> }) =>
      integrationsApi.connect(data.integrationId, data.config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["integrations"] });
      onOpenChange(false);
      setApiKey("");
    },
  });

  if (!integration) return null;

  const logo = integrationLogos[integration.slug] || { bg: "bg-gray-500", text: integration.name.substring(0, 2).toUpperCase() };

  const handleConnect = () => {
    // Different integrations have different auth flows
    if (integration.slug === "openai" || integration.slug === "anthropic" || integration.slug === "elevenlabs") {
      connectMutation.mutate({
        integrationId: integration.id,
        config: { api_key: apiKey },
      });
    } else {
      // OAuth flow would redirect
      window.location.href = `/api/integrations/${integration.slug}/oauth`;
    }
  };

  const isApiKeyIntegration = ["openai", "anthropic", "elevenlabs", "deepgram", "twilio", "vonage"].includes(integration.slug);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "flex h-10 w-10 items-center justify-center rounded-lg text-white font-bold text-sm",
                logo.bg
              )}
            >
              {logo.text}
            </div>
            <div>
              <DialogTitle>Connect {integration.name}</DialogTitle>
              <DialogDescription>
                {isApiKeyIntegration
                  ? "Enter your API key to connect"
                  : "Sign in to connect your account"}
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        {isApiKeyIntegration ? (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="api-key">API Key</Label>
              <Input
                id="api-key"
                type="password"
                placeholder="Enter your API key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Find your API key in your {integration.name} dashboard
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              You&apos;ll be redirected to {integration.name} to authorize the connection.
              We only request the minimum permissions necessary.
            </p>
            <div className="rounded-lg bg-muted p-4">
              <p className="text-sm font-medium mb-2">Permissions requested:</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Read and write contacts</li>
                <li>• Create and update deals</li>
                <li>• Access activity history</li>
              </ul>
            </div>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleConnect}
            disabled={isApiKeyIntegration && !apiKey.trim()}
          >
            {isApiKeyIntegration ? (
              connectMutation.isPending ? "Connecting..." : "Connect"
            ) : (
              <>
                <ExternalLink className="mr-2 h-4 w-4" />
                Continue to {integration.name}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Configure Dialog
function ConfigureDialog({
  open,
  onOpenChange,
  integration,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  integration: Integration | null;
}) {
  const queryClient = useQueryClient();
  const [syncEnabled, setSyncEnabled] = useState(true);
  const [syncInterval, setSyncInterval] = useState("15");

  const updateMutation = useMutation({
    mutationFn: (config: Record<string, unknown>) =>
      integrationsApi.update(integration!.id, config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["integrations"] });
      onOpenChange(false);
    },
  });

  const testMutation = useMutation({
    mutationFn: () => integrationsApi.test(integration!.id),
  });

  if (!integration) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Configure {integration.name}</DialogTitle>
          <DialogDescription>Manage your integration settings</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="flex items-center justify-between rounded-lg border p-4">
            <div>
              <p className="font-medium">Connection Status</p>
              <p className="text-sm text-muted-foreground">
                Connected {formatRelativeTime(integration.connected_at!)}
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => testMutation.mutate()}
              disabled={testMutation.isPending}
            >
              {testMutation.isPending ? (
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-2 h-4 w-4" />
              )}
              Test Connection
            </Button>
          </div>

          <div className="space-y-2">
            <Label>Sync Interval</Label>
            <Select value={syncInterval} onValueChange={setSyncInterval}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="5">Every 5 minutes</SelectItem>
                <SelectItem value="15">Every 15 minutes</SelectItem>
                <SelectItem value="30">Every 30 minutes</SelectItem>
                <SelectItem value="60">Every hour</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between rounded-lg border p-4">
            <div>
              <p className="font-medium">Auto-sync</p>
              <p className="text-sm text-muted-foreground">
                Automatically sync data between systems
              </p>
            </div>
            <button
              className={cn(
                "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                syncEnabled ? "bg-primary" : "bg-muted"
              )}
              onClick={() => setSyncEnabled(!syncEnabled)}
            >
              <span
                className={cn(
                  "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                  syncEnabled ? "translate-x-6" : "translate-x-1"
                )}
              />
            </button>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => updateMutation.mutate({ sync_enabled: syncEnabled, sync_interval: parseInt(syncInterval) })}
            disabled={updateMutation.isPending}
          >
            {updateMutation.isPending ? "Saving..." : "Save Changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function IntegrationsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("all");
  const [connectDialogOpen, setConnectDialogOpen] = useState(false);
  const [configureDialogOpen, setConfigureDialogOpen] = useState(false);
  const [selectedIntegration, setSelectedIntegration] = useState<Integration | null>(null);
  const queryClient = useQueryClient();

  const { data: integrationsData, isLoading } = useQuery({
    queryKey: ["integrations"],
    queryFn: () => integrationsApi.list(),
  });

  const disconnectMutation = useMutation({
    mutationFn: (id: string) => integrationsApi.disconnect(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["integrations"] });
    },
  });

  // Mock integrations data
  const mockIntegrations: Integration[] = [
    // CRM
    {
      id: "1",
      name: "Salesforce",
      slug: "salesforce",
      description: "Sync contacts, leads, and opportunities with Salesforce CRM",
      category: "crm",
      logo_url: "",
      status: "connected",
      connected_at: "2024-01-10T10:00:00Z",
    },
    {
      id: "2",
      name: "HubSpot",
      slug: "hubspot",
      description: "Connect your HubSpot CRM for seamless contact management",
      category: "crm",
      logo_url: "",
      status: "disconnected",
    },
    {
      id: "3",
      name: "Pipedrive",
      slug: "pipedrive",
      description: "Integrate with Pipedrive for deal tracking and automation",
      category: "crm",
      logo_url: "",
      status: "disconnected",
    },
    // Telephony
    {
      id: "4",
      name: "Twilio",
      slug: "twilio",
      description: "Use Twilio for voice and SMS capabilities",
      category: "telephony",
      logo_url: "",
      status: "connected",
      connected_at: "2024-01-05T09:00:00Z",
    },
    {
      id: "5",
      name: "Vonage",
      slug: "vonage",
      description: "Alternative telephony provider with global coverage",
      category: "telephony",
      logo_url: "",
      status: "disconnected",
    },
    // Calendar
    {
      id: "6",
      name: "Google Calendar",
      slug: "google-calendar",
      description: "Schedule and manage appointments with Google Calendar",
      category: "calendar",
      logo_url: "",
      status: "connected",
      connected_at: "2024-01-08T11:00:00Z",
    },
    {
      id: "7",
      name: "Calendly",
      slug: "calendly",
      description: "Automate appointment scheduling with Calendly",
      category: "calendar",
      logo_url: "",
      status: "disconnected",
    },
    // Communication
    {
      id: "8",
      name: "Slack",
      slug: "slack",
      description: "Get notifications and updates in your Slack workspace",
      category: "communication",
      logo_url: "",
      status: "connected",
      connected_at: "2024-01-12T14:00:00Z",
    },
    {
      id: "9",
      name: "Microsoft Teams",
      slug: "microsoft-teams",
      description: "Integrate with Microsoft Teams for enterprise communication",
      category: "communication",
      logo_url: "",
      status: "disconnected",
    },
    // Analytics
    {
      id: "10",
      name: "Zapier",
      slug: "zapier",
      description: "Connect to thousands of apps through Zapier automations",
      category: "analytics",
      logo_url: "",
      status: "disconnected",
    },
    {
      id: "11",
      name: "Segment",
      slug: "segment",
      description: "Send call data to your analytics stack via Segment",
      category: "analytics",
      logo_url: "",
      status: "disconnected",
    },
    // AI
    {
      id: "12",
      name: "OpenAI",
      slug: "openai",
      description: "Use OpenAI GPT models for conversation intelligence",
      category: "ai",
      logo_url: "",
      status: "connected",
      connected_at: "2024-01-01T08:00:00Z",
    },
    {
      id: "13",
      name: "Anthropic",
      slug: "anthropic",
      description: "Use Claude models for advanced reasoning",
      category: "ai",
      logo_url: "",
      status: "disconnected",
    },
    {
      id: "14",
      name: "ElevenLabs",
      slug: "elevenlabs",
      description: "Premium voice synthesis with ElevenLabs",
      category: "ai",
      logo_url: "",
      status: "connected",
      connected_at: "2024-01-03T10:00:00Z",
    },
    {
      id: "15",
      name: "Deepgram",
      slug: "deepgram",
      description: "Real-time speech recognition with Deepgram",
      category: "ai",
      logo_url: "",
      status: "connected",
      connected_at: "2024-01-02T09:00:00Z",
    },
    // Storage
    {
      id: "16",
      name: "AWS S3",
      slug: "aws",
      description: "Store recordings and transcripts in AWS S3",
      category: "storage",
      logo_url: "",
      status: "disconnected",
    },
    {
      id: "17",
      name: "Google Cloud Storage",
      slug: "gcs",
      description: "Use Google Cloud Storage for file storage",
      category: "storage",
      logo_url: "",
      status: "disconnected",
    },
  ];

  const integrations = integrationsData?.integrations || mockIntegrations;

  // Categories
  const categories = [
    { value: "all", label: "All" },
    { value: "crm", label: "CRM" },
    { value: "telephony", label: "Telephony" },
    { value: "calendar", label: "Calendar" },
    { value: "communication", label: "Communication" },
    { value: "analytics", label: "Analytics" },
    { value: "ai", label: "AI & ML" },
    { value: "storage", label: "Storage" },
  ];

  // Filter integrations
  const filteredIntegrations = integrations.filter((integration: Integration) => {
    const matchesSearch =
      integration.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      integration.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = activeCategory === "all" || integration.category === activeCategory;
    return matchesSearch && matchesCategory;
  });

  // Connected count
  const connectedCount = integrations.filter((i: Integration) => i.status === "connected").length;

  const handleConnect = (id: string) => {
    const integration = integrations.find((i: Integration) => i.id === id);
    setSelectedIntegration(integration || null);
    setConnectDialogOpen(true);
  };

  const handleDisconnect = (id: string) => {
    if (confirm("Are you sure you want to disconnect this integration?")) {
      disconnectMutation.mutate(id);
    }
  };

  const handleConfigure = (integration: Integration) => {
    setSelectedIntegration(integration);
    setConfigureDialogOpen(true);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Integrations</h1>
          <p className="text-muted-foreground">
            Connect your favorite tools and services
          </p>
        </div>
        <Badge variant="outline" className="text-sm w-fit">
          {connectedCount} connected
        </Badge>
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search integrations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
        <Tabs value={activeCategory} onValueChange={setActiveCategory}>
          <TabsList className="flex-wrap h-auto">
            {categories.map((cat) => (
              <TabsTrigger key={cat.value} value={cat.value} className="text-xs sm:text-sm">
                {cat.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      {/* Integrations Grid */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      ) : filteredIntegrations.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Search className="h-12 w-12 text-muted-foreground/50" />
            <h3 className="mt-4 text-lg font-medium">No integrations found</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              Try adjusting your search or filter
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredIntegrations.map((integration: Integration) => (
            <IntegrationCard
              key={integration.id}
              integration={integration}
              onConnect={handleConnect}
              onDisconnect={handleDisconnect}
              onConfigure={handleConfigure}
            />
          ))}
        </div>
      )}

      {/* Dialogs */}
      <ConnectDialog
        open={connectDialogOpen}
        onOpenChange={setConnectDialogOpen}
        integration={selectedIntegration}
      />
      <ConfigureDialog
        open={configureDialogOpen}
        onOpenChange={setConfigureDialogOpen}
        integration={selectedIntegration}
      />
    </div>
  );
}
