"use client";

import React, { useState } from "react";
import {
  Settings,
  Building2,
  Bell,
  Shield,
  Globe,
  Palette,
  Save,
  RefreshCw,
  Check,
  AlertCircle,
  Mic,
  Volume2,
  Bot,
  Database,
  Zap,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type SettingsTab = "general" | "defaults" | "notifications" | "security" | "advanced";

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<SettingsTab>("general");
  const [hasChanges, setHasChanges] = useState(false);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setSaving(false);
    setHasChanges(false);
  };

  const tabs = [
    { id: "general", label: "General", icon: Building2 },
    { id: "defaults", label: "Defaults", icon: Settings },
    { id: "notifications", label: "Notifications", icon: Bell },
    { id: "security", label: "Security", icon: Shield },
    { id: "advanced", label: "Advanced", icon: Zap },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground">
            Manage your organization settings and preferences
          </p>
        </div>
        {hasChanges && (
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="text-yellow-600 border-yellow-600">
              <AlertCircle className="mr-1 h-3 w-3" />
              Unsaved changes
            </Badge>
            <Button variant="outline" onClick={() => setHasChanges(false)}>
              Discard
            </Button>
            <Button onClick={handleSave} disabled={saving}>
              {saving ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="mr-2 h-4 w-4" />
                  Save Changes
                </>
              )}
            </Button>
          </div>
        )}
      </div>

      <div className="flex gap-6">
        {/* Sidebar Tabs */}
        <div className="w-48 shrink-0">
          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as SettingsTab)}
                className={cn(
                  "flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors",
                  activeTab === tab.id
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
              >
                <tab.icon className="h-4 w-4" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1 space-y-6">
          {activeTab === "general" && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Organization Information</CardTitle>
                  <CardDescription>
                    Basic information about your organization
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Organization Name</label>
                      <Input
                        defaultValue="Acme Corporation"
                        onChange={() => setHasChanges(true)}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Slug</label>
                      <Input
                        defaultValue="acme-corp"
                        disabled
                        className="bg-muted"
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Website</label>
                    <Input
                      type="url"
                      defaultValue="https://acme.example.com"
                      onChange={() => setHasChanges(true)}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Industry</label>
                    <Input
                      defaultValue="Technology"
                      onChange={() => setHasChanges(true)}
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Timezone & Locale</CardTitle>
                  <CardDescription>
                    Set your preferred timezone and language
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Timezone</label>
                      <Input
                        defaultValue="America/New_York (UTC-05:00)"
                        onChange={() => setHasChanges(true)}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Date Format</label>
                      <Input
                        defaultValue="MM/DD/YYYY"
                        onChange={() => setHasChanges(true)}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )}

          {activeTab === "defaults" && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Default STT Provider</CardTitle>
                  <CardDescription>
                    Speech-to-text provider used for new agents
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-3 md:grid-cols-3">
                    {["Deepgram", "OpenAI Whisper", "Google Cloud"].map((provider) => (
                      <div
                        key={provider}
                        className={cn(
                          "rounded-lg border p-4 cursor-pointer transition-all",
                          provider === "Deepgram" ? "border-primary bg-primary/5" : "hover:border-primary/50"
                        )}
                        onClick={() => setHasChanges(true)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Mic className="h-4 w-4" />
                            <span className="font-medium">{provider}</span>
                          </div>
                          {provider === "Deepgram" && (
                            <Check className="h-4 w-4 text-primary" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Default TTS Provider</CardTitle>
                  <CardDescription>
                    Text-to-speech provider used for new agents
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-3 md:grid-cols-3">
                    {["ElevenLabs", "OpenAI TTS", "PlayHT"].map((provider) => (
                      <div
                        key={provider}
                        className={cn(
                          "rounded-lg border p-4 cursor-pointer transition-all",
                          provider === "ElevenLabs" ? "border-primary bg-primary/5" : "hover:border-primary/50"
                        )}
                        onClick={() => setHasChanges(true)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Volume2 className="h-4 w-4" />
                            <span className="font-medium">{provider}</span>
                          </div>
                          {provider === "ElevenLabs" && (
                            <Check className="h-4 w-4 text-primary" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Default LLM Provider</CardTitle>
                  <CardDescription>
                    Large language model used for new agents
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-3 md:grid-cols-3">
                    {["OpenAI GPT-4o", "Anthropic Claude", "Google Gemini"].map((provider) => (
                      <div
                        key={provider}
                        className={cn(
                          "rounded-lg border p-4 cursor-pointer transition-all",
                          provider === "OpenAI GPT-4o" ? "border-primary bg-primary/5" : "hover:border-primary/50"
                        )}
                        onClick={() => setHasChanges(true)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Bot className="h-4 w-4" />
                            <span className="font-medium">{provider}</span>
                          </div>
                          {provider === "OpenAI GPT-4o" && (
                            <Check className="h-4 w-4 text-primary" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </>
          )}

          {activeTab === "notifications" && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Email Notifications</CardTitle>
                  <CardDescription>
                    Configure which events trigger email notifications
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {[
                    { label: "Call completed", description: "Get notified when calls end", enabled: true },
                    { label: "Agent errors", description: "Receive alerts for agent failures", enabled: true },
                    { label: "Usage alerts", description: "Warnings when approaching limits", enabled: true },
                    { label: "Weekly reports", description: "Summary of weekly performance", enabled: false },
                    { label: "Billing updates", description: "Invoice and payment notifications", enabled: true },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center justify-between rounded-lg border p-4">
                      <div>
                        <p className="font-medium">{item.label}</p>
                        <p className="text-sm text-muted-foreground">{item.description}</p>
                      </div>
                      <button
                        className={cn(
                          "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                          item.enabled ? "bg-primary" : "bg-muted"
                        )}
                        onClick={() => setHasChanges(true)}
                      >
                        <span
                          className={cn(
                            "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                            item.enabled ? "translate-x-6" : "translate-x-1"
                          )}
                        />
                      </button>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Webhook Events</CardTitle>
                  <CardDescription>
                    Events that trigger webhook notifications
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {[
                    { label: "call.started", enabled: true },
                    { label: "call.ended", enabled: true },
                    { label: "call.failed", enabled: true },
                    { label: "conversation.message", enabled: false },
                    { label: "transfer.initiated", enabled: true },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center justify-between rounded-lg border p-4">
                      <code className="text-sm bg-muted px-2 py-1 rounded">{item.label}</code>
                      <button
                        className={cn(
                          "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                          item.enabled ? "bg-primary" : "bg-muted"
                        )}
                        onClick={() => setHasChanges(true)}
                      >
                        <span
                          className={cn(
                            "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                            item.enabled ? "translate-x-6" : "translate-x-1"
                          )}
                        />
                      </button>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </>
          )}

          {activeTab === "security" && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Call Recording</CardTitle>
                  <CardDescription>
                    Configure call recording settings
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between rounded-lg border p-4">
                    <div>
                      <p className="font-medium">Enable call recording</p>
                      <p className="text-sm text-muted-foreground">Record all calls for quality assurance</p>
                    </div>
                    <button
                      className="relative inline-flex h-6 w-11 items-center rounded-full bg-primary transition-colors"
                      onClick={() => setHasChanges(true)}
                    >
                      <span className="inline-block h-4 w-4 transform rounded-full bg-white translate-x-6 transition-transform" />
                    </button>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Recording retention (days)</label>
                    <Input
                      type="number"
                      defaultValue="90"
                      onChange={() => setHasChanges(true)}
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Data Privacy</CardTitle>
                  <CardDescription>
                    Configure data handling and privacy settings
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between rounded-lg border p-4">
                    <div>
                      <p className="font-medium">PII Redaction</p>
                      <p className="text-sm text-muted-foreground">Automatically redact personal information</p>
                    </div>
                    <button
                      className="relative inline-flex h-6 w-11 items-center rounded-full bg-primary transition-colors"
                      onClick={() => setHasChanges(true)}
                    >
                      <span className="inline-block h-4 w-4 transform rounded-full bg-white translate-x-6 transition-transform" />
                    </button>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border p-4">
                    <div>
                      <p className="font-medium">HIPAA Compliance Mode</p>
                      <p className="text-sm text-muted-foreground">Enable additional safeguards for healthcare data</p>
                    </div>
                    <button
                      className="relative inline-flex h-6 w-11 items-center rounded-full bg-muted transition-colors"
                      onClick={() => setHasChanges(true)}
                    >
                      <span className="inline-block h-4 w-4 transform rounded-full bg-white translate-x-1 transition-transform" />
                    </button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>IP Restrictions</CardTitle>
                  <CardDescription>
                    Limit API access to specific IP addresses
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between rounded-lg border p-4">
                    <div>
                      <p className="font-medium">Enable IP allowlist</p>
                      <p className="text-sm text-muted-foreground">Only allow API calls from specified IPs</p>
                    </div>
                    <button
                      className="relative inline-flex h-6 w-11 items-center rounded-full bg-muted transition-colors"
                      onClick={() => setHasChanges(true)}
                    >
                      <span className="inline-block h-4 w-4 transform rounded-full bg-white translate-x-1 transition-transform" />
                    </button>
                  </div>
                </CardContent>
              </Card>
            </>
          )}

          {activeTab === "advanced" && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Concurrency Limits</CardTitle>
                  <CardDescription>
                    Configure concurrent call handling
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Max concurrent calls</label>
                    <Input
                      type="number"
                      defaultValue="50"
                      onChange={() => setHasChanges(true)}
                    />
                    <p className="text-xs text-muted-foreground">
                      Maximum number of simultaneous calls (plan limit: 100)
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Webhook Settings</CardTitle>
                  <CardDescription>
                    Configure webhook retry behavior
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Retry attempts</label>
                      <Input
                        type="number"
                        defaultValue="3"
                        onChange={() => setHasChanges(true)}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Timeout (seconds)</label>
                      <Input
                        type="number"
                        defaultValue="30"
                        onChange={() => setHasChanges(true)}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Analytics</CardTitle>
                  <CardDescription>
                    Configure analytics and data collection
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between rounded-lg border p-4">
                    <div>
                      <p className="font-medium">Enable analytics</p>
                      <p className="text-sm text-muted-foreground">Collect call and conversation analytics</p>
                    </div>
                    <button
                      className="relative inline-flex h-6 w-11 items-center rounded-full bg-primary transition-colors"
                      onClick={() => setHasChanges(true)}
                    >
                      <span className="inline-block h-4 w-4 transform rounded-full bg-white translate-x-6 transition-transform" />
                    </button>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border p-4">
                    <div>
                      <p className="font-medium">Sentiment analysis</p>
                      <p className="text-sm text-muted-foreground">Analyze customer sentiment in real-time</p>
                    </div>
                    <button
                      className="relative inline-flex h-6 w-11 items-center rounded-full bg-primary transition-colors"
                      onClick={() => setHasChanges(true)}
                    >
                      <span className="inline-block h-4 w-4 transform rounded-full bg-white translate-x-6 transition-transform" />
                    </button>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-red-200">
                <CardHeader>
                  <CardTitle className="text-red-600">Danger Zone</CardTitle>
                  <CardDescription>
                    Irreversible actions for your organization
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between rounded-lg border border-red-200 p-4">
                    <div>
                      <p className="font-medium">Delete all data</p>
                      <p className="text-sm text-muted-foreground">Permanently delete all calls and conversations</p>
                    </div>
                    <Button variant="destructive" size="sm">Delete Data</Button>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border border-red-200 p-4">
                    <div>
                      <p className="font-medium">Delete organization</p>
                      <p className="text-sm text-muted-foreground">Permanently delete your entire organization</p>
                    </div>
                    <Button variant="destructive" size="sm">Delete Organization</Button>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
