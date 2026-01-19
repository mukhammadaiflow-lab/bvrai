"use client";

import React, { useState } from "react";
import {
  User,
  Mail,
  Phone,
  Building2,
  Globe,
  Camera,
  Save,
  Loader2,
  Check,
  Shield,
  Bell,
  Key,
  Smartphone,
  LogOut,
  Trash2,
  AlertTriangle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
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
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { cn, formatRelativeTime } from "@/lib/utils";
import { toast } from "sonner";

interface UserProfile {
  name: string;
  email: string;
  phone: string;
  company: string;
  role: string;
  timezone: string;
  language: string;
  avatar_url: string | null;
  email_verified: boolean;
  two_factor_enabled: boolean;
  created_at: string;
}

interface Session {
  id: string;
  device: string;
  browser: string;
  location: string;
  ip: string;
  last_active: string;
  is_current: boolean;
}

// Demo data
const demoProfile: UserProfile = {
  name: "John Doe",
  email: "john@company.com",
  phone: "+1 (555) 123-4567",
  company: "Acme Inc",
  role: "Admin",
  timezone: "America/New_York",
  language: "en",
  avatar_url: null,
  email_verified: true,
  two_factor_enabled: false,
  created_at: "2024-01-01T00:00:00Z",
};

const demoSessions: Session[] = [
  {
    id: "1",
    device: "MacBook Pro",
    browser: "Chrome 120",
    location: "New York, US",
    ip: "192.168.1.***",
    last_active: new Date().toISOString(),
    is_current: true,
  },
  {
    id: "2",
    device: "iPhone 15",
    browser: "Safari Mobile",
    location: "New York, US",
    ip: "192.168.1.***",
    last_active: new Date(Date.now() - 3600000).toISOString(),
    is_current: false,
  },
];

const timezones = [
  { value: "America/New_York", label: "Eastern Time (ET)" },
  { value: "America/Chicago", label: "Central Time (CT)" },
  { value: "America/Denver", label: "Mountain Time (MT)" },
  { value: "America/Los_Angeles", label: "Pacific Time (PT)" },
  { value: "Europe/London", label: "London (GMT)" },
  { value: "Europe/Paris", label: "Paris (CET)" },
  { value: "Asia/Tokyo", label: "Tokyo (JST)" },
  { value: "Asia/Shanghai", label: "Shanghai (CST)" },
];

const languages = [
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "ja", label: "Japanese" },
  { value: "zh", label: "Chinese" },
];

export default function ProfilePage() {
  const [profile, setProfile] = useState<UserProfile>(demoProfile);
  const [sessions] = useState<Session[]>(demoSessions);
  const [saving, setSaving] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showLogoutAllDialog, setShowLogoutAllDialog] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");

  // Notification preferences
  const [notifications, setNotifications] = useState({
    email_calls: true,
    email_weekly_report: true,
    email_product_updates: false,
    push_calls: true,
    push_agent_status: true,
  });

  const handleProfileChange = (field: keyof UserProfile, value: string) => {
    setProfile((prev) => ({ ...prev, [field]: value }));
  };

  const handleSaveProfile = async () => {
    setSaving(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setSaving(false);
    toast.success("Profile updated successfully");
  };

  const handleEnable2FA = async () => {
    toast.info("2FA setup would open here");
  };

  const handleLogoutSession = async (sessionId: string) => {
    toast.success("Session logged out");
  };

  const handleLogoutAll = async () => {
    setShowLogoutAllDialog(false);
    toast.success("All other sessions logged out");
  };

  const handleDeleteAccount = async () => {
    if (deleteConfirmText !== "DELETE") return;
    toast.error("Account deletion would be processed");
    setShowDeleteDialog(false);
  };

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase();
  };

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Profile Settings</h1>
        <p className="text-muted-foreground">
          Manage your account settings and preferences
        </p>
      </div>

      {/* Profile Card */}
      <Card>
        <CardHeader>
          <CardTitle>Personal Information</CardTitle>
          <CardDescription>Update your personal details</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Avatar */}
          <div className="flex items-center gap-6">
            <div className="relative">
              <Avatar className="h-20 w-20">
                <AvatarImage src={profile.avatar_url || undefined} />
                <AvatarFallback className="text-xl">
                  {getInitials(profile.name)}
                </AvatarFallback>
              </Avatar>
              <button
                className="absolute bottom-0 right-0 flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-lg hover:bg-primary/90"
                onClick={() => toast.info("Avatar upload would open")}
              >
                <Camera className="h-4 w-4" />
              </button>
            </div>
            <div>
              <p className="font-medium">{profile.name}</p>
              <p className="text-sm text-muted-foreground">{profile.email}</p>
              <div className="flex gap-2 mt-2">
                <Badge variant="secondary">{profile.role}</Badge>
                {profile.email_verified && (
                  <Badge variant="success" className="bg-green-100 text-green-800">
                    <Check className="mr-1 h-3 w-3" />
                    Verified
                  </Badge>
                )}
              </div>
            </div>
          </div>

          <Separator />

          {/* Form Fields */}
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">Full Name</label>
              <Input
                value={profile.name}
                onChange={(e) => handleProfileChange("name", e.target.value)}
                leftIcon={<User className="h-4 w-4" />}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Email</label>
              <Input
                type="email"
                value={profile.email}
                onChange={(e) => handleProfileChange("email", e.target.value)}
                leftIcon={<Mail className="h-4 w-4" />}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Phone</label>
              <Input
                type="tel"
                value={profile.phone}
                onChange={(e) => handleProfileChange("phone", e.target.value)}
                leftIcon={<Phone className="h-4 w-4" />}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Company</label>
              <Input
                value={profile.company}
                onChange={(e) => handleProfileChange("company", e.target.value)}
                leftIcon={<Building2 className="h-4 w-4" />}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Timezone</label>
              <Select
                value={profile.timezone}
                onValueChange={(value) => handleProfileChange("timezone", value)}
              >
                <SelectTrigger>
                  <Globe className="mr-2 h-4 w-4 text-muted-foreground" />
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
            <div className="space-y-2">
              <label className="text-sm font-medium">Language</label>
              <Select
                value={profile.language}
                onValueChange={(value) => handleProfileChange("language", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {languages.map((lang) => (
                    <SelectItem key={lang.value} value={lang.value}>
                      {lang.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex justify-end">
            <Button onClick={handleSaveProfile} disabled={saving}>
              {saving ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
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
        </CardContent>
      </Card>

      {/* Security Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Security
          </CardTitle>
          <CardDescription>Manage your account security settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Password */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                <Key className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <p className="font-medium">Password</p>
                <p className="text-sm text-muted-foreground">
                  Last changed 30 days ago
                </p>
              </div>
            </div>
            <Button variant="outline" onClick={() => toast.info("Password change dialog would open")}>
              Change Password
            </Button>
          </div>

          <Separator />

          {/* Two-Factor Auth */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                <Smartphone className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <p className="font-medium">Two-Factor Authentication</p>
                <p className="text-sm text-muted-foreground">
                  {profile.two_factor_enabled
                    ? "Enabled - Your account is more secure"
                    : "Add an extra layer of security to your account"}
                </p>
              </div>
            </div>
            {profile.two_factor_enabled ? (
              <Badge variant="success" className="bg-green-100 text-green-800">
                <Check className="mr-1 h-3 w-3" />
                Enabled
              </Badge>
            ) : (
              <Button onClick={handleEnable2FA}>Enable 2FA</Button>
            )}
          </div>

          <Separator />

          {/* Active Sessions */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="font-medium">Active Sessions</p>
                <p className="text-sm text-muted-foreground">
                  Manage your active login sessions
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowLogoutAllDialog(true)}
              >
                <LogOut className="mr-2 h-4 w-4" />
                Log Out All
              </Button>
            </div>

            <div className="space-y-3">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  className="flex items-center justify-between p-3 rounded-lg border"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                      {session.device.includes("iPhone") ? (
                        <Smartphone className="h-5 w-5 text-muted-foreground" />
                      ) : (
                        <Globe className="h-5 w-5 text-muted-foreground" />
                      )}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="font-medium">{session.device}</p>
                        {session.is_current && (
                          <Badge variant="secondary" className="text-xs">
                            Current
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {session.browser} · {session.location} · {session.ip}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-muted-foreground">
                      {session.is_current ? "Active now" : formatRelativeTime(session.last_active)}
                    </p>
                    {!session.is_current && (
                      <button
                        className="text-sm text-destructive hover:underline"
                        onClick={() => handleLogoutSession(session.id)}
                      >
                        Log out
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Notifications Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notifications
          </CardTitle>
          <CardDescription>Configure how you receive notifications</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <p className="font-medium mb-4">Email Notifications</p>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Call Notifications</p>
                  <p className="text-sm text-muted-foreground">
                    Receive emails about missed calls and call summaries
                  </p>
                </div>
                <Switch
                  checked={notifications.email_calls}
                  onCheckedChange={(checked) =>
                    setNotifications((prev) => ({ ...prev, email_calls: checked }))
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Weekly Reports</p>
                  <p className="text-sm text-muted-foreground">
                    Receive weekly performance summaries
                  </p>
                </div>
                <Switch
                  checked={notifications.email_weekly_report}
                  onCheckedChange={(checked) =>
                    setNotifications((prev) => ({ ...prev, email_weekly_report: checked }))
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Product Updates</p>
                  <p className="text-sm text-muted-foreground">
                    News about new features and improvements
                  </p>
                </div>
                <Switch
                  checked={notifications.email_product_updates}
                  onCheckedChange={(checked) =>
                    setNotifications((prev) => ({ ...prev, email_product_updates: checked }))
                  }
                />
              </div>
            </div>
          </div>

          <Separator />

          <div>
            <p className="font-medium mb-4">Push Notifications</p>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Incoming Calls</p>
                  <p className="text-sm text-muted-foreground">
                    Get notified about incoming calls in real-time
                  </p>
                </div>
                <Switch
                  checked={notifications.push_calls}
                  onCheckedChange={(checked) =>
                    setNotifications((prev) => ({ ...prev, push_calls: checked }))
                  }
                />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Agent Status Changes</p>
                  <p className="text-sm text-muted-foreground">
                    Notifications when agents go online/offline
                  </p>
                </div>
                <Switch
                  checked={notifications.push_agent_status}
                  onCheckedChange={(checked) =>
                    setNotifications((prev) => ({ ...prev, push_agent_status: checked }))
                  }
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Danger Zone */}
      <Card className="border-destructive/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-5 w-5" />
            Danger Zone
          </CardTitle>
          <CardDescription>
            Irreversible and destructive actions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Delete Account</p>
              <p className="text-sm text-muted-foreground">
                Permanently delete your account and all associated data
              </p>
            </div>
            <Button
              variant="destructive"
              onClick={() => setShowDeleteDialog(true)}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete Account
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Logout All Dialog */}
      <Dialog open={showLogoutAllDialog} onOpenChange={setShowLogoutAllDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Log Out All Sessions</DialogTitle>
            <DialogDescription>
              This will log you out of all devices except your current session.
              You'll need to sign in again on those devices.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowLogoutAllDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleLogoutAll}>Log Out All</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Account Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Account</DialogTitle>
            <DialogDescription>
              This action cannot be undone. This will permanently delete your
              account and remove all associated data from our servers.
            </DialogDescription>
          </DialogHeader>

          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Warning</AlertTitle>
            <AlertDescription>
              All your agents, call history, and settings will be permanently deleted.
            </AlertDescription>
          </Alert>

          <div className="space-y-2">
            <label className="text-sm font-medium">
              Type <span className="font-mono font-bold">DELETE</span> to confirm
            </label>
            <Input
              value={deleteConfirmText}
              onChange={(e) => setDeleteConfirmText(e.target.value)}
              placeholder="DELETE"
            />
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteDialog(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteAccount}
              disabled={deleteConfirmText !== "DELETE"}
            >
              Delete Account
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
