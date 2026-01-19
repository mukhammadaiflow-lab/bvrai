"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  User,
  Building2,
  Key,
  Bell,
  Shield,
  CreditCard,
  Webhook,
  Globe,
  Moon,
  Sun,
  Laptop,
  Save,
  Copy,
  Eye,
  EyeOff,
  Plus,
  Trash2,
  RefreshCw,
  CheckCircle,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useAuth } from "@/hooks/use-auth";
import { settingsApi, apiKeysApi, webhooksApi } from "@/lib/api";
import { formatRelativeTime, cn } from "@/lib/utils";

interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at?: string;
  expires_at?: string;
  scopes: string[];
}

interface WebhookEndpoint {
  id: string;
  url: string;
  events: string[];
  active: boolean;
  secret: string;
  created_at: string;
  last_triggered_at?: string;
}

// Profile Settings
function ProfileSettings() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [name, setName] = useState(user?.name || "");
  const [email, setEmail] = useState(user?.email || "");
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await settingsApi.updateProfile({ name, email });
      queryClient.invalidateQueries({ queryKey: ["user"] });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Profile Settings</CardTitle>
        <CardDescription>Manage your personal information</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-center gap-6">
          <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary text-2xl font-bold text-primary-foreground">
            {user?.name?.charAt(0) || "U"}
          </div>
          <div>
            <Button variant="outline" size="sm">
              Change Avatar
            </Button>
            <p className="mt-1 text-xs text-muted-foreground">JPG, PNG or GIF. Max 2MB.</p>
          </div>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="name">Full Name</Label>
            <Input id="name" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="email">Email Address</Label>
            <Input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
        </div>
        <div className="space-y-2">
          <Label>Account ID</Label>
          <div className="flex items-center gap-2">
            <code className="flex-1 rounded bg-muted px-3 py-2 text-sm">{user?.id || "usr_xxx"}</code>
            <Button variant="ghost" size="icon">
              <Copy className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <Button onClick={handleSave} disabled={isSaving}>
          {isSaving ? "Saving..." : "Save Changes"}
        </Button>
      </CardContent>
    </Card>
  );
}

// Organization Settings
function OrganizationSettings() {
  const [orgName, setOrgName] = useState("Acme Inc");
  const [orgSlug, setOrgSlug] = useState("acme");
  const [timezone, setTimezone] = useState("America/Los_Angeles");
  const [isSaving, setIsSaving] = useState(false);

  const timezones = [
    { value: "America/New_York", label: "Eastern Time (ET)" },
    { value: "America/Chicago", label: "Central Time (CT)" },
    { value: "America/Denver", label: "Mountain Time (MT)" },
    { value: "America/Los_Angeles", label: "Pacific Time (PT)" },
    { value: "Europe/London", label: "Greenwich Mean Time (GMT)" },
    { value: "Europe/Paris", label: "Central European Time (CET)" },
    { value: "Asia/Tokyo", label: "Japan Standard Time (JST)" },
  ];

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await settingsApi.updateOrganization({ name: orgName, slug: orgSlug, timezone });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Organization Settings</CardTitle>
        <CardDescription>Manage your organization details</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="org-name">Organization Name</Label>
            <Input id="org-name" value={orgName} onChange={(e) => setOrgName(e.target.value)} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="org-slug">URL Slug</Label>
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">app.builderengine.io/</span>
              <Input
                id="org-slug"
                value={orgSlug}
                onChange={(e) => setOrgSlug(e.target.value)}
                className="max-w-[150px]"
              />
            </div>
          </div>
        </div>
        <div className="space-y-2">
          <Label>Default Timezone</Label>
          <Select value={timezone} onValueChange={setTimezone}>
            <SelectTrigger className="max-w-md">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {timezones.map((tz) => (
                <SelectItem key={tz.value} value={tz.value}>
                  {tz.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button onClick={handleSave} disabled={isSaving}>
          {isSaving ? "Saving..." : "Save Changes"}
        </Button>
      </CardContent>
    </Card>
  );
}

// API Keys Management
function ApiKeysSettings() {
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyScopes, setNewKeyScopes] = useState<string[]>(["read"]);
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const queryClient = useQueryClient();

  const { data: keysData } = useQuery({
    queryKey: ["api-keys"],
    queryFn: () => apiKeysApi.list(),
  });

  const createMutation = useMutation({
    mutationFn: (data: { name: string; scopes: string[] }) => apiKeysApi.create(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["api-keys"] });
      setCreatedKey(data.key);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiKeysApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["api-keys"] });
    },
  });

  const mockKeys: ApiKey[] = [
    {
      id: "1",
      name: "Production API",
      prefix: "sk_live_xxxx",
      created_at: "2024-01-10T10:00:00Z",
      last_used_at: "2024-01-20T14:30:00Z",
      scopes: ["read", "write", "calls"],
    },
    {
      id: "2",
      name: "Development",
      prefix: "sk_test_xxxx",
      created_at: "2024-01-05T09:00:00Z",
      scopes: ["read", "write"],
    },
    {
      id: "3",
      name: "Analytics Only",
      prefix: "sk_live_yyyy",
      created_at: "2024-01-15T11:00:00Z",
      last_used_at: "2024-01-19T08:15:00Z",
      expires_at: "2024-06-15T11:00:00Z",
      scopes: ["read", "analytics"],
    },
  ];

  const apiKeys = keysData?.keys || mockKeys;

  const availableScopes = [
    { value: "read", label: "Read", description: "Read access to resources" },
    { value: "write", label: "Write", description: "Create and update resources" },
    { value: "calls", label: "Calls", description: "Make and manage calls" },
    { value: "analytics", label: "Analytics", description: "Access analytics data" },
    { value: "billing", label: "Billing", description: "Manage billing" },
  ];

  const handleCreate = () => {
    createMutation.mutate({ name: newKeyName, scopes: newKeyScopes });
  };

  const copyKey = (key: string) => {
    navigator.clipboard.writeText(key);
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>API Keys</CardTitle>
            <CardDescription>Manage API keys for programmatic access</CardDescription>
          </div>
          <Button onClick={() => setShowCreateDialog(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create Key
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Key</TableHead>
                <TableHead>Scopes</TableHead>
                <TableHead>Last Used</TableHead>
                <TableHead className="w-[50px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {apiKeys.map((key: ApiKey) => (
                <TableRow key={key.id}>
                  <TableCell>
                    <div>
                      <p className="font-medium">{key.name}</p>
                      <p className="text-xs text-muted-foreground">
                        Created {formatRelativeTime(key.created_at)}
                      </p>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <code className="rounded bg-muted px-2 py-1 text-xs">
                        {showKeys[key.id] ? `${key.prefix}...masked` : key.prefix}
                      </code>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={() => setShowKeys({ ...showKeys, [key.id]: !showKeys[key.id] })}
                      >
                        {showKeys[key.id] ? (
                          <EyeOff className="h-3 w-3" />
                        ) : (
                          <Eye className="h-3 w-3" />
                        )}
                      </Button>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {key.scopes.map((scope) => (
                        <Badge key={scope} variant="outline" className="text-xs">
                          {scope}
                        </Badge>
                      ))}
                    </div>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {key.last_used_at ? formatRelativeTime(key.last_used_at) : "Never"}
                  </TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-destructive hover:text-destructive"
                      onClick={() => {
                        if (confirm("Delete this API key?")) {
                          deleteMutation.mutate(key.id);
                        }
                      }}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>

      {/* Create Key Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{createdKey ? "API Key Created" : "Create API Key"}</DialogTitle>
            <DialogDescription>
              {createdKey
                ? "Copy your API key now. You won't be able to see it again."
                : "Create a new API key for your application"}
            </DialogDescription>
          </DialogHeader>
          {createdKey ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2 rounded-lg border bg-muted p-4">
                <code className="flex-1 text-sm break-all">{createdKey}</code>
                <Button variant="ghost" size="icon" onClick={() => copyKey(createdKey)}>
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex items-center gap-2 rounded-lg bg-warning/10 p-3 text-sm text-warning">
                <Shield className="h-4 w-4" />
                Store this key securely. It won&apos;t be shown again.
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="key-name">Key Name</Label>
                <Input
                  id="key-name"
                  placeholder="e.g., Production API"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>Permissions</Label>
                <div className="space-y-2">
                  {availableScopes.map((scope) => (
                    <div
                      key={scope.value}
                      className={cn(
                        "flex items-center justify-between rounded-lg border p-3 cursor-pointer transition-colors",
                        newKeyScopes.includes(scope.value)
                          ? "border-primary bg-primary/5"
                          : "hover:border-muted-foreground/50"
                      )}
                      onClick={() => {
                        setNewKeyScopes(
                          newKeyScopes.includes(scope.value)
                            ? newKeyScopes.filter((s) => s !== scope.value)
                            : [...newKeyScopes, scope.value]
                        );
                      }}
                    >
                      <div>
                        <p className="font-medium">{scope.label}</p>
                        <p className="text-xs text-muted-foreground">{scope.description}</p>
                      </div>
                      {newKeyScopes.includes(scope.value) && (
                        <CheckCircle className="h-5 w-5 text-primary" />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
          <DialogFooter>
            {createdKey ? (
              <Button
                onClick={() => {
                  setShowCreateDialog(false);
                  setCreatedKey(null);
                  setNewKeyName("");
                  setNewKeyScopes(["read"]);
                }}
              >
                Done
              </Button>
            ) : (
              <>
                <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                  Cancel
                </Button>
                <Button
                  onClick={handleCreate}
                  disabled={!newKeyName.trim() || newKeyScopes.length === 0 || createMutation.isPending}
                >
                  {createMutation.isPending ? "Creating..." : "Create Key"}
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

// Webhooks Settings
function WebhooksSettings() {
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newWebhookUrl, setNewWebhookUrl] = useState("");
  const [newWebhookEvents, setNewWebhookEvents] = useState<string[]>(["call.completed"]);
  const queryClient = useQueryClient();

  const { data: webhooksData } = useQuery({
    queryKey: ["webhooks"],
    queryFn: () => webhooksApi.list(),
  });

  const createMutation = useMutation({
    mutationFn: (data: { url: string; events: string[] }) => webhooksApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["webhooks"] });
      setShowCreateDialog(false);
      setNewWebhookUrl("");
      setNewWebhookEvents(["call.completed"]);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => webhooksApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["webhooks"] });
    },
  });

  const mockWebhooks: WebhookEndpoint[] = [
    {
      id: "1",
      url: "https://api.myapp.com/webhooks/builderengine",
      events: ["call.started", "call.completed", "call.failed"],
      active: true,
      secret: "whsec_xxxx",
      created_at: "2024-01-10T10:00:00Z",
      last_triggered_at: "2024-01-20T14:30:00Z",
    },
    {
      id: "2",
      url: "https://hooks.slack.com/services/xxx",
      events: ["call.completed"],
      active: true,
      secret: "whsec_yyyy",
      created_at: "2024-01-15T09:00:00Z",
    },
  ];

  const webhooks = webhooksData?.webhooks || mockWebhooks;

  const availableEvents = [
    { value: "call.started", label: "Call Started" },
    { value: "call.completed", label: "Call Completed" },
    { value: "call.failed", label: "Call Failed" },
    { value: "agent.created", label: "Agent Created" },
    { value: "agent.updated", label: "Agent Updated" },
    { value: "recording.ready", label: "Recording Ready" },
    { value: "transcript.ready", label: "Transcript Ready" },
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Webhooks</CardTitle>
            <CardDescription>Configure webhook endpoints for real-time events</CardDescription>
          </div>
          <Button onClick={() => setShowCreateDialog(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Add Endpoint
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {webhooks.length === 0 ? (
          <div className="text-center py-8">
            <Webhook className="mx-auto h-12 w-12 text-muted-foreground/50" />
            <h3 className="mt-4 text-lg font-medium">No webhooks configured</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              Add a webhook endpoint to receive real-time events
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {webhooks.map((webhook: WebhookEndpoint) => (
              <div key={webhook.id} className="flex items-start justify-between rounded-lg border p-4">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <code className="text-sm">{webhook.url}</code>
                    <Badge variant={webhook.active ? "default" : "secondary"}>
                      {webhook.active ? "Active" : "Inactive"}
                    </Badge>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {webhook.events.map((event) => (
                      <Badge key={event} variant="outline" className="text-xs">
                        {event}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Last triggered:{" "}
                    {webhook.last_triggered_at
                      ? formatRelativeTime(webhook.last_triggered_at)
                      : "Never"}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="icon">
                    <RefreshCw className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-destructive hover:text-destructive"
                    onClick={() => {
                      if (confirm("Delete this webhook?")) {
                        deleteMutation.mutate(webhook.id);
                      }
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>

      {/* Create Webhook Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Webhook Endpoint</DialogTitle>
            <DialogDescription>Configure a new webhook to receive events</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="webhook-url">Endpoint URL</Label>
              <Input
                id="webhook-url"
                type="url"
                placeholder="https://api.yourapp.com/webhooks"
                value={newWebhookUrl}
                onChange={(e) => setNewWebhookUrl(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Events to subscribe</Label>
              <div className="grid grid-cols-2 gap-2">
                {availableEvents.map((event) => (
                  <div
                    key={event.value}
                    className={cn(
                      "flex items-center gap-2 rounded-lg border p-2 cursor-pointer transition-colors text-sm",
                      newWebhookEvents.includes(event.value)
                        ? "border-primary bg-primary/5"
                        : "hover:border-muted-foreground/50"
                    )}
                    onClick={() => {
                      setNewWebhookEvents(
                        newWebhookEvents.includes(event.value)
                          ? newWebhookEvents.filter((e) => e !== event.value)
                          : [...newWebhookEvents, event.value]
                      );
                    }}
                  >
                    <div
                      className={cn(
                        "h-4 w-4 rounded border flex items-center justify-center",
                        newWebhookEvents.includes(event.value)
                          ? "bg-primary border-primary"
                          : "border-muted-foreground"
                      )}
                    >
                      {newWebhookEvents.includes(event.value) && (
                        <CheckCircle className="h-3 w-3 text-primary-foreground" />
                      )}
                    </div>
                    {event.label}
                  </div>
                ))}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createMutation.mutate({ url: newWebhookUrl, events: newWebhookEvents })}
              disabled={!newWebhookUrl.trim() || newWebhookEvents.length === 0 || createMutation.isPending}
            >
              {createMutation.isPending ? "Creating..." : "Create Webhook"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

// Appearance Settings
function AppearanceSettings() {
  const [theme, setTheme] = useState<"light" | "dark" | "system">("system");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Appearance</CardTitle>
        <CardDescription>Customize how the dashboard looks</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <Label>Theme</Label>
          <div className="grid grid-cols-3 gap-4">
            <div
              className={cn(
                "flex flex-col items-center gap-2 rounded-lg border p-4 cursor-pointer transition-colors",
                theme === "light" ? "border-primary bg-primary/5" : "hover:border-muted-foreground/50"
              )}
              onClick={() => setTheme("light")}
            >
              <Sun className="h-6 w-6" />
              <span className="text-sm font-medium">Light</span>
            </div>
            <div
              className={cn(
                "flex flex-col items-center gap-2 rounded-lg border p-4 cursor-pointer transition-colors",
                theme === "dark" ? "border-primary bg-primary/5" : "hover:border-muted-foreground/50"
              )}
              onClick={() => setTheme("dark")}
            >
              <Moon className="h-6 w-6" />
              <span className="text-sm font-medium">Dark</span>
            </div>
            <div
              className={cn(
                "flex flex-col items-center gap-2 rounded-lg border p-4 cursor-pointer transition-colors",
                theme === "system" ? "border-primary bg-primary/5" : "hover:border-muted-foreground/50"
              )}
              onClick={() => setTheme("system")}
            >
              <Laptop className="h-6 w-6" />
              <span className="text-sm font-medium">System</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Notifications Settings
function NotificationsSettings() {
  const [emailNotifications, setEmailNotifications] = useState({
    callCompleted: true,
    dailySummary: true,
    weeklyReport: false,
    billingAlerts: true,
    securityAlerts: true,
    productUpdates: false,
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Notifications</CardTitle>
        <CardDescription>Configure how you want to be notified</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {[
            { key: "callCompleted", label: "Call Completed", desc: "Get notified when calls finish" },
            { key: "dailySummary", label: "Daily Summary", desc: "Receive daily activity digest" },
            { key: "weeklyReport", label: "Weekly Report", desc: "Detailed weekly analytics report" },
            { key: "billingAlerts", label: "Billing Alerts", desc: "Usage limits and payment reminders" },
            { key: "securityAlerts", label: "Security Alerts", desc: "Login attempts and security events" },
            { key: "productUpdates", label: "Product Updates", desc: "New features and improvements" },
          ].map((item) => (
            <div key={item.key} className="flex items-center justify-between rounded-lg border p-4">
              <div>
                <p className="font-medium">{item.label}</p>
                <p className="text-sm text-muted-foreground">{item.desc}</p>
              </div>
              <button
                className={cn(
                  "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                  emailNotifications[item.key as keyof typeof emailNotifications]
                    ? "bg-primary"
                    : "bg-muted"
                )}
                onClick={() =>
                  setEmailNotifications({
                    ...emailNotifications,
                    [item.key]: !emailNotifications[item.key as keyof typeof emailNotifications],
                  })
                }
              >
                <span
                  className={cn(
                    "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                    emailNotifications[item.key as keyof typeof emailNotifications]
                      ? "translate-x-6"
                      : "translate-x-1"
                  )}
                />
              </button>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// Security Settings
function SecuritySettings() {
  const [showPasswordDialog, setShowPasswordDialog] = useState(false);
  const [show2FADialog, setShow2FADialog] = useState(false);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Security</CardTitle>
        <CardDescription>Manage your account security settings</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between rounded-lg border p-4">
          <div className="flex items-center gap-3">
            <Key className="h-5 w-5 text-muted-foreground" />
            <div>
              <p className="font-medium">Password</p>
              <p className="text-sm text-muted-foreground">Last changed 30 days ago</p>
            </div>
          </div>
          <Button variant="outline" onClick={() => setShowPasswordDialog(true)}>
            Change Password
          </Button>
        </div>
        <div className="flex items-center justify-between rounded-lg border p-4">
          <div className="flex items-center gap-3">
            <Shield className="h-5 w-5 text-muted-foreground" />
            <div>
              <p className="font-medium">Two-Factor Authentication</p>
              <p className="text-sm text-muted-foreground">Add an extra layer of security</p>
            </div>
          </div>
          <Button variant="outline" onClick={() => setShow2FADialog(true)}>
            Enable 2FA
          </Button>
        </div>
        <div className="flex items-center justify-between rounded-lg border p-4">
          <div className="flex items-center gap-3">
            <Globe className="h-5 w-5 text-muted-foreground" />
            <div>
              <p className="font-medium">Active Sessions</p>
              <p className="text-sm text-muted-foreground">2 active sessions</p>
            </div>
          </div>
          <Button variant="outline">Manage Sessions</Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState("profile");

  const tabs = [
    { value: "profile", label: "Profile", icon: User },
    { value: "organization", label: "Organization", icon: Building2 },
    { value: "api-keys", label: "API Keys", icon: Key },
    { value: "webhooks", label: "Webhooks", icon: Webhook },
    { value: "notifications", label: "Notifications", icon: Bell },
    { value: "security", label: "Security", icon: Shield },
    { value: "appearance", label: "Appearance", icon: Moon },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground">Manage your account and preferences</p>
      </div>

      {/* Settings Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-7 h-auto">
          {tabs.map((tab) => (
            <TabsTrigger key={tab.value} value={tab.value} className="flex items-center gap-2 py-3">
              <tab.icon className="h-4 w-4" />
              <span className="hidden sm:inline">{tab.label}</span>
            </TabsTrigger>
          ))}
        </TabsList>

        <div className="mt-6">
          <TabsContent value="profile">
            <ProfileSettings />
          </TabsContent>
          <TabsContent value="organization">
            <OrganizationSettings />
          </TabsContent>
          <TabsContent value="api-keys">
            <ApiKeysSettings />
          </TabsContent>
          <TabsContent value="webhooks">
            <WebhooksSettings />
          </TabsContent>
          <TabsContent value="notifications">
            <NotificationsSettings />
          </TabsContent>
          <TabsContent value="security">
            <SecuritySettings />
          </TabsContent>
          <TabsContent value="appearance">
            <AppearanceSettings />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
