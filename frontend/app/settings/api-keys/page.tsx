"use client";

import React, { useState } from "react";
import {
  Key,
  Plus,
  Copy,
  Eye,
  EyeOff,
  Trash2,
  Check,
  AlertCircle,
  Clock,
  Shield,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
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
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { cn, formatRelativeTime } from "@/lib/utils";
import { toast } from "sonner";

interface APIKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  scopes: string[];
  status: "active" | "revoked";
}

// Demo data
const demoKeys: APIKey[] = [
  {
    id: "key_1",
    name: "Production API Key",
    prefix: "bvr_sk_live_XXXX",
    created_at: "2024-01-15T10:00:00Z",
    last_used_at: "2024-01-20T14:30:00Z",
    scopes: ["agents:read", "agents:write", "calls:read", "calls:write"],
    status: "active",
  },
  {
    id: "key_2",
    name: "Development Key",
    prefix: "bvr_sk_test_XXXX",
    created_at: "2024-01-10T08:00:00Z",
    last_used_at: "2024-01-19T16:45:00Z",
    scopes: ["agents:read", "calls:read"],
    status: "active",
  },
  {
    id: "key_3",
    name: "Old Integration",
    prefix: "bvr_sk_live_YYYY",
    created_at: "2023-12-01T12:00:00Z",
    last_used_at: null,
    scopes: ["calls:read"],
    status: "revoked",
  },
];

const availableScopes = [
  { id: "agents:read", label: "Read Agents", description: "View agent configurations" },
  { id: "agents:write", label: "Write Agents", description: "Create and modify agents" },
  { id: "calls:read", label: "Read Calls", description: "View call logs and transcripts" },
  { id: "calls:write", label: "Make Calls", description: "Initiate outbound calls" },
  { id: "analytics:read", label: "Read Analytics", description: "Access analytics data" },
  { id: "webhooks:manage", label: "Manage Webhooks", description: "Configure webhooks" },
];

export default function APIKeysPage() {
  const [keys, setKeys] = useState<APIKey[]>(demoKeys);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState<string | null>(null);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyScopes, setNewKeyScopes] = useState<string[]>(["agents:read", "calls:read"]);
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [creating, setCreating] = useState(false);

  const handleCreateKey = async () => {
    setCreating(true);

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));

    const newKey: APIKey = {
      id: `key_${Date.now()}`,
      name: newKeyName,
      prefix: "bvr_sk_live_ZZZZ",
      created_at: new Date().toISOString(),
      last_used_at: null,
      scopes: newKeyScopes,
      status: "active",
    };

    setKeys((prev) => [newKey, ...prev]);
    setCreatedKey("bvr_sk_live_" + Math.random().toString(36).substring(2, 34));
    setCreating(false);
  };

  const handleCopyKey = async () => {
    if (createdKey) {
      await navigator.clipboard.writeText(createdKey);
      setCopied(true);
      toast.success("API key copied to clipboard");
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleRevokeKey = (keyId: string) => {
    setKeys((prev) =>
      prev.map((key) =>
        key.id === keyId ? { ...key, status: "revoked" as const } : key
      )
    );
    setShowDeleteDialog(null);
    toast.success("API key revoked");
  };

  const handleCloseCreateDialog = () => {
    setShowCreateDialog(false);
    setCreatedKey(null);
    setNewKeyName("");
    setNewKeyScopes(["agents:read", "calls:read"]);
    setCopied(false);
  };

  const toggleScope = (scopeId: string) => {
    setNewKeyScopes((prev) =>
      prev.includes(scopeId)
        ? prev.filter((s) => s !== scopeId)
        : [...prev, scopeId]
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">API Keys</h1>
          <p className="text-muted-foreground">
            Manage API keys for programmatic access to Builder Voice AI
          </p>
        </div>
        <Button onClick={() => setShowCreateDialog(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Create API Key
        </Button>
      </div>

      {/* Security Notice */}
      <Alert>
        <Shield className="h-4 w-4" />
        <AlertTitle>Keep your API keys secure</AlertTitle>
        <AlertDescription>
          API keys provide full access to your account. Never share them in public repositories or client-side code.
        </AlertDescription>
      </Alert>

      {/* API Keys List */}
      <Card>
        <CardHeader>
          <CardTitle>Your API Keys</CardTitle>
          <CardDescription>
            {keys.filter((k) => k.status === "active").length} active keys
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {keys.map((key) => (
              <div
                key={key.id}
                className={cn(
                  "flex items-center justify-between p-4 rounded-lg border",
                  key.status === "revoked" && "opacity-60"
                )}
              >
                <div className="flex items-center gap-4">
                  <div
                    className={cn(
                      "flex h-10 w-10 items-center justify-center rounded-lg",
                      key.status === "active"
                        ? "bg-primary/10 text-primary"
                        : "bg-muted text-muted-foreground"
                    )}
                  >
                    <Key className="h-5 w-5" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{key.name}</span>
                      <Badge
                        variant={key.status === "active" ? "default" : "secondary"}
                      >
                        {key.status}
                      </Badge>
                    </div>
                    <p className="text-sm font-mono text-muted-foreground">
                      {key.prefix}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-6">
                  <div className="text-right text-sm">
                    <p className="text-muted-foreground">Last used</p>
                    <p>
                      {key.last_used_at
                        ? formatRelativeTime(key.last_used_at)
                        : "Never"}
                    </p>
                  </div>
                  <div className="text-right text-sm">
                    <p className="text-muted-foreground">Created</p>
                    <p>{formatRelativeTime(key.created_at)}</p>
                  </div>
                  {key.status === "active" && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-destructive hover:text-destructive"
                      onClick={() => setShowDeleteDialog(key.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Usage Examples */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Start</CardTitle>
          <CardDescription>
            Use your API key to authenticate requests
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="text-sm font-medium mb-2">cURL</p>
            <div className="rounded-lg bg-muted p-4 font-mono text-sm overflow-x-auto">
              <pre>{`curl -X GET "https://api.buildervoice.ai/v1/agents" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json"`}</pre>
            </div>
          </div>
          <div>
            <p className="text-sm font-medium mb-2">Python</p>
            <div className="rounded-lg bg-muted p-4 font-mono text-sm overflow-x-auto">
              <pre>{`import requests

headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

response = requests.get(
    "https://api.buildervoice.ai/v1/agents",
    headers=headers
)`}</pre>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Create Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={handleCloseCreateDialog}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>
              {createdKey ? "API Key Created" : "Create API Key"}
            </DialogTitle>
            <DialogDescription>
              {createdKey
                ? "Your new API key has been created. Copy it now - you won't be able to see it again."
                : "Create a new API key with specific permissions."}
            </DialogDescription>
          </DialogHeader>

          {createdKey ? (
            <div className="space-y-4">
              <Alert variant="warning">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Save your key</AlertTitle>
                <AlertDescription>
                  This key will only be shown once. Store it securely.
                </AlertDescription>
              </Alert>

              <div className="rounded-lg bg-muted p-4">
                <div className="flex items-center justify-between">
                  <code className="text-sm font-mono break-all">{createdKey}</code>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleCopyKey}
                  >
                    {copied ? (
                      <Check className="h-4 w-4 text-green-600" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              <DialogFooter>
                <Button onClick={handleCloseCreateDialog}>Done</Button>
              </DialogFooter>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Key Name</label>
                <Input
                  placeholder="e.g., Production API Key"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Permissions</label>
                <div className="grid grid-cols-2 gap-2">
                  {availableScopes.map((scope) => (
                    <button
                      key={scope.id}
                      type="button"
                      onClick={() => toggleScope(scope.id)}
                      className={cn(
                        "flex flex-col items-start p-3 rounded-lg border text-left transition-all",
                        newKeyScopes.includes(scope.id)
                          ? "border-primary bg-primary/5"
                          : "hover:border-primary/50"
                      )}
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className={cn(
                            "h-4 w-4 rounded border flex items-center justify-center",
                            newKeyScopes.includes(scope.id)
                              ? "bg-primary border-primary"
                              : "border-muted-foreground"
                          )}
                        >
                          {newKeyScopes.includes(scope.id) && (
                            <Check className="h-3 w-3 text-primary-foreground" />
                          )}
                        </div>
                        <span className="text-sm font-medium">{scope.label}</span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 ml-6">
                        {scope.description}
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              <DialogFooter>
                <Button variant="outline" onClick={handleCloseCreateDialog}>
                  Cancel
                </Button>
                <Button
                  onClick={handleCreateKey}
                  disabled={!newKeyName || newKeyScopes.length === 0 || creating}
                >
                  {creating ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    "Create Key"
                  )}
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Revoke Dialog */}
      <Dialog open={!!showDeleteDialog} onOpenChange={() => setShowDeleteDialog(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Revoke API Key</DialogTitle>
            <DialogDescription>
              Are you sure you want to revoke this API key? This action cannot be undone.
              Any applications using this key will immediately lose access.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteDialog(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => showDeleteDialog && handleRevokeKey(showDeleteDialog)}
            >
              Revoke Key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
