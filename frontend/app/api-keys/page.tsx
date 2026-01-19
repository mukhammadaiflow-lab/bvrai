"use client";

import React, { useState } from "react";
import {
  Key,
  Plus,
  Copy,
  Trash2,
  Eye,
  EyeOff,
  Clock,
  Shield,
  AlertTriangle,
  Check,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn, formatRelativeTime } from "@/lib/utils";

// Mock API keys data
const mockApiKeys = [
  {
    id: "1",
    name: "Production API Key",
    keyPrefix: "sk_live_xxxxxxxxxxxxx",
    scopes: ["agents:read", "agents:write", "calls:read", "calls:write", "analytics:read"],
    createdAt: "2024-01-01T00:00:00Z",
    lastUsed: "2024-01-14T10:30:00Z",
    expiresAt: null,
    isActive: true,
  },
  {
    id: "2",
    name: "Development Key",
    keyPrefix: "sk_test_xxxxxxxxxxxxx",
    scopes: ["agents:read", "agents:write", "calls:read"],
    createdAt: "2024-01-10T00:00:00Z",
    lastUsed: "2024-01-14T09:15:00Z",
    expiresAt: null,
    isActive: true,
  },
  {
    id: "3",
    name: "Analytics Service",
    keyPrefix: "sk_live_yyyyyyyyyyyy",
    scopes: ["analytics:read"],
    createdAt: "2024-01-05T00:00:00Z",
    lastUsed: "2024-01-14T08:00:00Z",
    expiresAt: "2024-06-01T00:00:00Z",
    isActive: true,
  },
  {
    id: "4",
    name: "Old Integration (Deprecated)",
    keyPrefix: "sk_live_zzzzzzzzzzzz",
    scopes: ["agents:read"],
    createdAt: "2023-12-01T00:00:00Z",
    lastUsed: "2024-01-01T00:00:00Z",
    expiresAt: null,
    isActive: false,
  },
];

const availableScopes = [
  { id: "agents:read", label: "Read Agents", description: "View agent configurations" },
  { id: "agents:write", label: "Write Agents", description: "Create and modify agents" },
  { id: "calls:read", label: "Read Calls", description: "View call history and details" },
  { id: "calls:write", label: "Write Calls", description: "Initiate and manage calls" },
  { id: "analytics:read", label: "Read Analytics", description: "Access analytics data" },
  { id: "webhooks:read", label: "Read Webhooks", description: "View webhook configurations" },
  { id: "webhooks:write", label: "Write Webhooks", description: "Manage webhooks" },
  { id: "billing:read", label: "Read Billing", description: "View billing information" },
];

export default function ApiKeysPage() {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const handleCopyKey = (keyPrefix: string) => {
    navigator.clipboard.writeText(keyPrefix);
    setCopiedKey(keyPrefix);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">API Keys</h1>
          <p className="text-muted-foreground">
            Manage API keys for programmatic access to your account
          </p>
        </div>
        <Button onClick={() => setShowCreateModal(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Create API Key
        </Button>
      </div>

      {/* Security Notice */}
      <Card className="border-yellow-200 bg-yellow-50">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-yellow-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-yellow-800">Keep your API keys secure</p>
              <p className="text-sm text-yellow-700">
                Never expose API keys in client-side code or public repositories. Use environment variables and secret management tools.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* API Keys List */}
      <div className="space-y-4">
        {mockApiKeys.map((apiKey) => (
          <Card key={apiKey.id} className={cn(!apiKey.isActive && "opacity-60")}>
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4">
                  <div className={cn(
                    "flex h-12 w-12 items-center justify-center rounded-lg",
                    apiKey.isActive ? "bg-primary/10" : "bg-muted"
                  )}>
                    <Key className={cn(
                      "h-6 w-6",
                      apiKey.isActive ? "text-primary" : "text-muted-foreground"
                    )} />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold">{apiKey.name}</h3>
                      {!apiKey.isActive && (
                        <Badge variant="secondary">Revoked</Badge>
                      )}
                      {apiKey.expiresAt && (
                        <Badge variant="outline" className="text-yellow-600 border-yellow-600">
                          <Clock className="mr-1 h-3 w-3" />
                          Expires {formatRelativeTime(apiKey.expiresAt)}
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mb-3">
                      <code className="bg-muted px-2 py-1 rounded text-sm font-mono">
                        {apiKey.keyPrefix}
                      </code>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => handleCopyKey(apiKey.keyPrefix)}
                      >
                        {copiedKey === apiKey.keyPrefix ? (
                          <Check className="h-3 w-3 text-green-600" />
                        ) : (
                          <Copy className="h-3 w-3" />
                        )}
                      </Button>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {apiKey.scopes.map((scope) => (
                        <Badge key={scope} variant="outline" className="text-xs">
                          <Shield className="mr-1 h-3 w-3" />
                          {scope}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {apiKey.isActive && (
                    <Button variant="destructive" size="sm">
                      <Trash2 className="mr-1 h-4 w-4" />
                      Revoke
                    </Button>
                  )}
                </div>
              </div>
              <div className="mt-4 pt-4 border-t">
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Created</p>
                    <p className="font-medium">{formatRelativeTime(apiKey.createdAt)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Last Used</p>
                    <p className="font-medium">{formatRelativeTime(apiKey.lastUsed)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Scopes</p>
                    <p className="font-medium">{apiKey.scopes.length} permissions</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Available Scopes Reference */}
      <Card>
        <CardHeader>
          <CardTitle>Available Scopes</CardTitle>
          <CardDescription>
            Permissions you can grant to your API keys
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            {availableScopes.map((scope) => (
              <div key={scope.id} className="rounded-lg border p-3">
                <div className="flex items-center gap-2 mb-1">
                  <Shield className="h-4 w-4 text-muted-foreground" />
                  <code className="text-sm font-medium">{scope.id}</code>
                </div>
                <p className="text-xs text-muted-foreground">{scope.description}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
