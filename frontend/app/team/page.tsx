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
  Checkbox,
  Separator,
  ScrollArea,
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
  Skeleton,
  Textarea,
} from "@/components/ui";
import {
  Users,
  Plus,
  Search,
  Mail,
  Shield,
  ShieldCheck,
  ShieldAlert,
  MoreHorizontal,
  Trash2,
  Edit,
  Crown,
  Clock,
  Check,
  X,
  UserPlus,
  UserMinus,
  UserCog,
  Copy,
  ExternalLink,
  AlertTriangle,
  RefreshCw,
  Send,
  Key,
  Settings,
  Eye,
  Bot,
  Phone,
  Webhook,
  CreditCard,
  BarChart3,
  Activity,
  Lock,
  Unlock,
  ChevronDown,
  ChevronUp,
  LogOut,
  AlertCircle,
  CheckCircle,
  History,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Types
interface Permission {
  id: string;
  name: string;
  description: string;
  category: string;
}

interface Role {
  id: string;
  name: string;
  description: string;
  color: string;
  permissions: string[];
  isCustom?: boolean;
}

interface TeamMember {
  id: string;
  name: string;
  email: string;
  role: string;
  status: "active" | "invited" | "suspended";
  lastActive: string;
  joinedAt: string;
  avatar?: string;
  twoFactorEnabled: boolean;
}

interface Invitation {
  id: string;
  email: string;
  role: string;
  sentAt: string;
  expiresAt: string;
  status: "pending" | "expired";
}

interface ActivityLog {
  id: string;
  user: string;
  action: string;
  target: string;
  timestamp: string;
}

// Permission definitions
const allPermissions: Permission[] = [
  // Agents
  { id: "agents.view", name: "View Agents", description: "View agent configurations", category: "Agents" },
  { id: "agents.create", name: "Create Agents", description: "Create new voice agents", category: "Agents" },
  { id: "agents.edit", name: "Edit Agents", description: "Modify existing agents", category: "Agents" },
  { id: "agents.delete", name: "Delete Agents", description: "Remove agents permanently", category: "Agents" },
  { id: "agents.deploy", name: "Deploy Agents", description: "Deploy agents to production", category: "Agents" },
  // Calls
  { id: "calls.view", name: "View Calls", description: "View call history and recordings", category: "Calls" },
  { id: "calls.monitor", name: "Monitor Calls", description: "Listen to live calls", category: "Calls" },
  { id: "calls.manage", name: "Manage Calls", description: "End or transfer calls", category: "Calls" },
  // Analytics
  { id: "analytics.view", name: "View Analytics", description: "Access analytics dashboard", category: "Analytics" },
  { id: "analytics.export", name: "Export Data", description: "Export analytics data", category: "Analytics" },
  // Team
  { id: "team.view", name: "View Team", description: "See team member list", category: "Team" },
  { id: "team.invite", name: "Invite Members", description: "Invite new team members", category: "Team" },
  { id: "team.manage", name: "Manage Members", description: "Change roles and remove members", category: "Team" },
  { id: "team.roles", name: "Manage Roles", description: "Create and edit custom roles", category: "Team" },
  // Settings
  { id: "settings.view", name: "View Settings", description: "View organization settings", category: "Settings" },
  { id: "settings.edit", name: "Edit Settings", description: "Modify organization settings", category: "Settings" },
  { id: "settings.billing", name: "Manage Billing", description: "Access billing and subscription", category: "Settings" },
  { id: "settings.api", name: "Manage API Keys", description: "Create and revoke API keys", category: "Settings" },
  // Webhooks
  { id: "webhooks.view", name: "View Webhooks", description: "View webhook configurations", category: "Webhooks" },
  { id: "webhooks.manage", name: "Manage Webhooks", description: "Create and edit webhooks", category: "Webhooks" },
];

// Default roles
const defaultRoles: Role[] = [
  {
    id: "owner",
    name: "Owner",
    description: "Full access to all features. Cannot be removed.",
    color: "bg-purple-100 text-purple-700 border-purple-200",
    permissions: allPermissions.map((p) => p.id),
  },
  {
    id: "admin",
    name: "Admin",
    description: "Manage team, agents, and most settings",
    color: "bg-blue-100 text-blue-700 border-blue-200",
    permissions: [
      "agents.view", "agents.create", "agents.edit", "agents.delete", "agents.deploy",
      "calls.view", "calls.monitor", "calls.manage",
      "analytics.view", "analytics.export",
      "team.view", "team.invite", "team.manage",
      "settings.view", "settings.edit",
      "webhooks.view", "webhooks.manage",
    ],
  },
  {
    id: "member",
    name: "Member",
    description: "Create and manage agents, view analytics",
    color: "bg-green-100 text-green-700 border-green-200",
    permissions: [
      "agents.view", "agents.create", "agents.edit",
      "calls.view", "calls.monitor",
      "analytics.view",
      "team.view",
      "settings.view",
      "webhooks.view",
    ],
  },
  {
    id: "viewer",
    name: "Viewer",
    description: "Read-only access to most features",
    color: "bg-gray-100 text-gray-700 border-gray-200",
    permissions: [
      "agents.view",
      "calls.view",
      "analytics.view",
      "team.view",
      "settings.view",
      "webhooks.view",
    ],
  },
];

// Mock data
const mockTeamMembers: TeamMember[] = [
  {
    id: "1",
    name: "John Smith",
    email: "john@example.com",
    role: "owner",
    status: "active",
    lastActive: "2024-01-14T10:30:00Z",
    joinedAt: "2023-01-15T00:00:00Z",
    twoFactorEnabled: true,
  },
  {
    id: "2",
    name: "Sarah Johnson",
    email: "sarah@example.com",
    role: "admin",
    status: "active",
    lastActive: "2024-01-14T09:15:00Z",
    joinedAt: "2023-06-20T00:00:00Z",
    twoFactorEnabled: true,
  },
  {
    id: "3",
    name: "Mike Brown",
    email: "mike@example.com",
    role: "member",
    status: "active",
    lastActive: "2024-01-13T16:45:00Z",
    joinedAt: "2023-09-10T00:00:00Z",
    twoFactorEnabled: false,
  },
  {
    id: "4",
    name: "Emily Davis",
    email: "emily@example.com",
    role: "viewer",
    status: "active",
    lastActive: "2024-01-12T11:00:00Z",
    joinedAt: "2024-01-01T00:00:00Z",
    twoFactorEnabled: false,
  },
  {
    id: "5",
    name: "Alex Wilson",
    email: "alex@example.com",
    role: "member",
    status: "suspended",
    lastActive: "2024-01-10T08:30:00Z",
    joinedAt: "2023-11-15T00:00:00Z",
    twoFactorEnabled: false,
  },
];

const mockInvitations: Invitation[] = [
  {
    id: "inv-1",
    email: "chris@example.com",
    role: "member",
    sentAt: "2024-01-13T10:00:00Z",
    expiresAt: "2024-01-20T10:00:00Z",
    status: "pending",
  },
  {
    id: "inv-2",
    email: "jessica@example.com",
    role: "viewer",
    sentAt: "2024-01-12T14:00:00Z",
    expiresAt: "2024-01-19T14:00:00Z",
    status: "pending",
  },
  {
    id: "inv-3",
    email: "old@example.com",
    role: "member",
    sentAt: "2024-01-01T10:00:00Z",
    expiresAt: "2024-01-08T10:00:00Z",
    status: "expired",
  },
];

const mockActivityLogs: ActivityLog[] = [
  { id: "1", user: "John Smith", action: "invited", target: "chris@example.com as Member", timestamp: "2024-01-13T10:00:00Z" },
  { id: "2", user: "Sarah Johnson", action: "changed role", target: "Mike Brown from Viewer to Member", timestamp: "2024-01-12T15:30:00Z" },
  { id: "3", user: "John Smith", action: "removed", target: "Tom Anderson", timestamp: "2024-01-11T09:00:00Z" },
  { id: "4", user: "Sarah Johnson", action: "suspended", target: "Alex Wilson", timestamp: "2024-01-10T08:30:00Z" },
  { id: "5", user: "John Smith", action: "invited", target: "jessica@example.com as Viewer", timestamp: "2024-01-12T14:00:00Z" },
];

// Utility functions
const formatRelativeTime = (date: string): string => {
  const now = new Date();
  const past = new Date(date);
  const diffMs = now.getTime() - past.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return past.toLocaleDateString();
};

const getInitials = (name: string): string => {
  return name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase();
};

// Components
function RoleBadge({ roleId, roles }: { roleId: string; roles: Role[] }) {
  const role = roles.find((r) => r.id === roleId);
  if (!role) return null;

  return (
    <Badge className={cn("gap-1", role.color)}>
      {roleId === "owner" && <Crown className="h-3 w-3" />}
      {role.name}
    </Badge>
  );
}

function InviteMemberDialog({
  open,
  onOpenChange,
  roles,
  onInvite,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  roles: Role[];
  onInvite: (email: string, role: string, message?: string) => void;
}) {
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("member");
  const [message, setMessage] = useState("");
  const [bulkMode, setBulkMode] = useState(false);
  const [bulkEmails, setBulkEmails] = useState("");

  const handleInvite = () => {
    if (bulkMode) {
      const emails = bulkEmails.split(/[,\n]/).map((e) => e.trim()).filter(Boolean);
      emails.forEach((e) => onInvite(e, role, message));
    } else {
      onInvite(email, role, message);
    }
    setEmail("");
    setBulkEmails("");
    setMessage("");
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Invite Team Member</DialogTitle>
          <DialogDescription>
            Send an invitation to join your organization
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Toggle bulk mode */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="bulkMode"
              checked={bulkMode}
              onCheckedChange={(checked) => setBulkMode(checked === true)}
            />
            <Label htmlFor="bulkMode" className="text-sm cursor-pointer">
              Invite multiple people at once
            </Label>
          </div>

          {/* Email input */}
          {bulkMode ? (
            <div className="space-y-2">
              <Label>Email Addresses</Label>
              <Textarea
                placeholder="Enter email addresses (separated by commas or new lines)&#10;john@example.com&#10;jane@example.com"
                value={bulkEmails}
                onChange={(e) => setBulkEmails(e.target.value)}
                rows={4}
              />
              <p className="text-xs text-muted-foreground">
                {bulkEmails.split(/[,\n]/).filter((e) => e.trim()).length} email(s) entered
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <Label htmlFor="email">Email Address</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  placeholder="colleague@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>
          )}

          {/* Role selection */}
          <div className="space-y-2">
            <Label>Role</Label>
            <Select value={role} onValueChange={setRole}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {roles.filter((r) => r.id !== "owner").map((r) => (
                  <SelectItem key={r.id} value={r.id}>
                    <div className="flex items-center gap-2">
                      <Badge className={cn("text-xs", r.color)}>{r.name}</Badge>
                      <span className="text-xs text-muted-foreground">{r.description}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Personal message */}
          <div className="space-y-2">
            <Label htmlFor="message">Personal Message (optional)</Label>
            <Textarea
              id="message"
              placeholder="Add a personal note to the invitation..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows={3}
            />
          </div>

          {/* Permission preview */}
          <div className="p-3 bg-muted/50 rounded-lg">
            <p className="text-sm font-medium mb-2">This role can:</p>
            <div className="flex flex-wrap gap-1">
              {roles.find((r) => r.id === role)?.permissions.slice(0, 6).map((p) => (
                <Badge key={p} variant="outline" className="text-xs">
                  {allPermissions.find((perm) => perm.id === p)?.name}
                </Badge>
              ))}
              {(roles.find((r) => r.id === role)?.permissions.length || 0) > 6 && (
                <Badge variant="outline" className="text-xs">
                  +{(roles.find((r) => r.id === role)?.permissions.length || 0) - 6} more
                </Badge>
              )}
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleInvite}
            disabled={bulkMode ? !bulkEmails.trim() : !email.trim()}
          >
            <Send className="mr-2 h-4 w-4" />
            Send Invitation
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function EditMemberDialog({
  open,
  onOpenChange,
  member,
  roles,
  onSave,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  member: TeamMember | null;
  roles: Role[];
  onSave: (memberId: string, role: string) => void;
}) {
  const [selectedRole, setSelectedRole] = useState(member?.role || "member");

  if (!member) return null;

  const handleSave = () => {
    onSave(member.id, selectedRole);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Edit Team Member</DialogTitle>
          <DialogDescription>
            Change role and permissions for {member.name}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Member info */}
          <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary font-medium">
              {getInitials(member.name)}
            </div>
            <div>
              <p className="font-medium">{member.name}</p>
              <p className="text-sm text-muted-foreground">{member.email}</p>
            </div>
          </div>

          {/* Role selection */}
          <div className="space-y-3">
            <Label>Role</Label>
            <div className="space-y-2">
              {roles.filter((r) => r.id !== "owner").map((r) => (
                <div
                  key={r.id}
                  className={cn(
                    "flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors",
                    selectedRole === r.id
                      ? "border-primary bg-primary/5"
                      : "hover:bg-muted/50"
                  )}
                  onClick={() => setSelectedRole(r.id)}
                >
                  <div className="pt-0.5">
                    <div
                      className={cn(
                        "h-4 w-4 rounded-full border-2",
                        selectedRole === r.id
                          ? "border-primary bg-primary"
                          : "border-muted-foreground"
                      )}
                    >
                      {selectedRole === r.id && (
                        <Check className="h-3 w-3 text-primary-foreground" />
                      )}
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <Badge className={cn("text-xs", r.color)}>{r.name}</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">{r.description}</p>
                    <div className="flex flex-wrap gap-1 mt-2">
                      {r.permissions.slice(0, 4).map((p) => (
                        <span key={p} className="text-xs text-muted-foreground">
                          {allPermissions.find((perm) => perm.id === p)?.name}
                          {r.permissions.indexOf(p) < Math.min(3, r.permissions.length - 1) && ","}
                        </span>
                      ))}
                      {r.permissions.length > 4 && (
                        <span className="text-xs text-muted-foreground">
                          +{r.permissions.length - 4} more
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save Changes</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function TeamMemberRow({
  member,
  roles,
  onEdit,
  onRemove,
  onSuspend,
  onReactivate,
}: {
  member: TeamMember;
  roles: Role[];
  onEdit: () => void;
  onRemove: () => void;
  onSuspend: () => void;
  onReactivate: () => void;
}) {
  const isOwner = member.role === "owner";

  return (
    <tr className="border-b last:border-0 hover:bg-muted/30 transition-colors">
      <td className="py-4 pl-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 font-medium text-primary">
              {getInitials(member.name)}
            </div>
            {member.status === "active" && (
              <div className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full bg-green-500 border-2 border-white" />
            )}
            {member.status === "suspended" && (
              <div className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full bg-red-500 border-2 border-white" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <p className="font-medium">{member.name}</p>
              {isOwner && <Crown className="h-4 w-4 text-yellow-500" />}
              {member.twoFactorEnabled && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <ShieldCheck className="h-4 w-4 text-green-500" />
                    </TooltipTrigger>
                    <TooltipContent>2FA Enabled</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
            <p className="text-sm text-muted-foreground">{member.email}</p>
          </div>
        </div>
      </td>
      <td className="py-4">
        <RoleBadge roleId={member.role} roles={roles} />
      </td>
      <td className="py-4">
        <Badge
          variant="outline"
          className={cn(
            member.status === "active" && "bg-green-50 text-green-700 border-green-200",
            member.status === "suspended" && "bg-red-50 text-red-700 border-red-200",
            member.status === "invited" && "bg-yellow-50 text-yellow-700 border-yellow-200"
          )}
        >
          {member.status}
        </Badge>
      </td>
      <td className="py-4">
        <span className="text-sm text-muted-foreground">
          {formatRelativeTime(member.lastActive)}
        </span>
      </td>
      <td className="py-4 pr-4">
        <div className="flex items-center gap-1">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={onEdit}
                  disabled={isOwner}
                >
                  <Edit className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Edit role</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          {member.status === "active" && !isOwner && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={onSuspend}
                    className="text-yellow-600 hover:text-yellow-700"
                  >
                    <Lock className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Suspend access</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}

          {member.status === "suspended" && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={onReactivate}
                    className="text-green-600 hover:text-green-700"
                  >
                    <Unlock className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Reactivate</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={onRemove}
                  disabled={isOwner}
                  className="text-red-600 hover:text-red-700 disabled:text-muted-foreground"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Remove from team</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </td>
    </tr>
  );
}

function InvitationRow({
  invitation,
  onResend,
  onCancel,
}: {
  invitation: Invitation;
  onResend: () => void;
  onCancel: () => void;
}) {
  const isExpired = invitation.status === "expired";

  return (
    <div
      className={cn(
        "flex items-center justify-between rounded-lg border p-4",
        isExpired && "bg-muted/50 opacity-75"
      )}
    >
      <div className="flex items-center gap-3">
        <div
          className={cn(
            "flex h-10 w-10 items-center justify-center rounded-full",
            isExpired ? "bg-gray-100" : "bg-yellow-100"
          )}
        >
          <Mail className={cn("h-5 w-5", isExpired ? "text-gray-500" : "text-yellow-600")} />
        </div>
        <div>
          <p className="font-medium">{invitation.email}</p>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Badge variant="outline" className="text-xs">
              {invitation.role}
            </Badge>
            <span>•</span>
            <Clock className="h-3 w-3" />
            <span>Sent {formatRelativeTime(invitation.sentAt)}</span>
            {isExpired && (
              <>
                <span>•</span>
                <Badge variant="outline" className="text-xs bg-red-50 text-red-700 border-red-200">
                  Expired
                </Badge>
              </>
            )}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={onResend}>
          <RefreshCw className="mr-1 h-3 w-3" />
          {isExpired ? "Resend" : "Resend"}
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          className="text-red-600 hover:text-red-700"
          onClick={onCancel}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

function RolePermissionsCard({ role, allPerms }: { role: Role; allPerms: Permission[] }) {
  const [expanded, setExpanded] = useState(false);
  const categories = Array.from(new Set(allPerms.map((p) => p.category)));

  return (
    <Card className={cn(role.id === "owner" && "border-purple-200")}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div>
            <div className="flex items-center gap-2">
              <Badge className={role.color}>{role.name}</Badge>
              {role.id === "owner" && <Crown className="h-4 w-4 text-yellow-500" />}
              {role.isCustom && (
                <Badge variant="outline" className="text-xs">
                  Custom
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground mt-1">{role.description}</p>
          </div>
          {role.id !== "owner" && (
            <Button variant="ghost" size="icon-sm">
              <Edit className="h-4 w-4" />
            </Button>
          )}
        </div>

        <div className="space-y-2">
          <div className="text-xs text-muted-foreground">
            {role.permissions.length} / {allPerms.length} permissions
          </div>

          {expanded ? (
            <div className="space-y-3">
              {categories.map((category) => {
                const categoryPerms = allPerms.filter((p) => p.category === category);
                const enabledPerms = categoryPerms.filter((p) =>
                  role.permissions.includes(p.id)
                );

                return (
                  <div key={category}>
                    <p className="text-xs font-medium text-muted-foreground mb-1">
                      {category} ({enabledPerms.length}/{categoryPerms.length})
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {categoryPerms.map((perm) => (
                        <Badge
                          key={perm.id}
                          variant="outline"
                          className={cn(
                            "text-xs",
                            role.permissions.includes(perm.id)
                              ? "bg-green-50 text-green-700 border-green-200"
                              : "bg-gray-50 text-gray-400 border-gray-200"
                          )}
                        >
                          {role.permissions.includes(perm.id) ? (
                            <Check className="h-2.5 w-2.5 mr-1" />
                          ) : (
                            <X className="h-2.5 w-2.5 mr-1" />
                          )}
                          {perm.name}
                        </Badge>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="flex flex-wrap gap-1">
              {role.permissions.slice(0, 5).map((p) => (
                <Badge key={p} variant="outline" className="text-xs">
                  {allPerms.find((perm) => perm.id === p)?.name}
                </Badge>
              ))}
              {role.permissions.length > 5 && (
                <Badge variant="outline" className="text-xs">
                  +{role.permissions.length - 5}
                </Badge>
              )}
            </div>
          )}

          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded(!expanded)}
            className="w-full text-xs"
          >
            {expanded ? (
              <>
                <ChevronUp className="mr-1 h-3 w-3" />
                Show Less
              </>
            ) : (
              <>
                <ChevronDown className="mr-1 h-3 w-3" />
                Show All Permissions
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function TeamPage() {
  const [members, setMembers] = useState<TeamMember[]>(mockTeamMembers);
  const [invitations, setInvitations] = useState<Invitation[]>(mockInvitations);
  const [roles] = useState<Role[]>(defaultRoles);
  const [activityLogs] = useState<ActivityLog[]>(mockActivityLogs);

  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [roleFilter, setRoleFilter] = useState<string>("all");
  const [activeTab, setActiveTab] = useState("members");

  const [showInviteDialog, setShowInviteDialog] = useState(false);
  const [editingMember, setEditingMember] = useState<TeamMember | null>(null);

  // Filter members
  const filteredMembers = members.filter((member) => {
    const matchesSearch =
      member.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      member.email.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === "all" || member.status === statusFilter;
    const matchesRole = roleFilter === "all" || member.role === roleFilter;
    return matchesSearch && matchesStatus && matchesRole;
  });

  // Stats
  const stats = {
    total: members.length,
    active: members.filter((m) => m.status === "active").length,
    pending: invitations.filter((i) => i.status === "pending").length,
    seatsUsed: members.length,
    seatsTotal: 10,
  };

  // Handlers
  const handleInvite = (email: string, role: string) => {
    const newInvitation: Invitation = {
      id: `inv-${Date.now()}`,
      email,
      role,
      sentAt: new Date().toISOString(),
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      status: "pending",
    };
    setInvitations((prev) => [newInvitation, ...prev]);
  };

  const handleSaveMember = (memberId: string, role: string) => {
    setMembers((prev) =>
      prev.map((m) => (m.id === memberId ? { ...m, role } : m))
    );
  };

  const handleRemoveMember = (memberId: string) => {
    setMembers((prev) => prev.filter((m) => m.id !== memberId));
  };

  const handleSuspendMember = (memberId: string) => {
    setMembers((prev) =>
      prev.map((m) => (m.id === memberId ? { ...m, status: "suspended" } : m))
    );
  };

  const handleReactivateMember = (memberId: string) => {
    setMembers((prev) =>
      prev.map((m) => (m.id === memberId ? { ...m, status: "active" } : m))
    );
  };

  const handleResendInvitation = (invitationId: string) => {
    setInvitations((prev) =>
      prev.map((inv) =>
        inv.id === invitationId
          ? {
              ...inv,
              sentAt: new Date().toISOString(),
              expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
              status: "pending" as const,
            }
          : inv
      )
    );
  };

  const handleCancelInvitation = (invitationId: string) => {
    setInvitations((prev) => prev.filter((inv) => inv.id !== invitationId));
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Team Management</h1>
            <p className="text-muted-foreground">
              Manage your team members, roles, and permissions
            </p>
          </div>
          <Button onClick={() => setShowInviteDialog(true)}>
            <UserPlus className="mr-2 h-4 w-4" />
            Invite Member
          </Button>
        </div>

        {/* Stats */}
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/30">
                  <Users className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.total}</p>
                  <p className="text-sm text-muted-foreground">Team Members</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.active}</p>
                  <p className="text-sm text-muted-foreground">Active</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100 dark:bg-yellow-900/30">
                  <Mail className="h-5 w-5 text-yellow-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stats.pending}</p>
                  <p className="text-sm text-muted-foreground">Pending Invites</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900/30">
                  <Shield className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">
                    {stats.seatsUsed}/{stats.seatsTotal}
                  </p>
                  <p className="text-sm text-muted-foreground">Seats Used</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <div className="flex items-center justify-between">
            <TabsList>
              <TabsTrigger value="members" className="gap-2">
                <Users className="h-4 w-4" />
                Members
              </TabsTrigger>
              <TabsTrigger value="invitations" className="gap-2">
                <Mail className="h-4 w-4" />
                Invitations
                {stats.pending > 0 && (
                  <Badge variant="secondary" className="ml-1 text-xs">
                    {stats.pending}
                  </Badge>
                )}
              </TabsTrigger>
              <TabsTrigger value="roles" className="gap-2">
                <Shield className="h-4 w-4" />
                Roles & Permissions
              </TabsTrigger>
              <TabsTrigger value="activity" className="gap-2">
                <History className="h-4 w-4" />
                Activity
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Members Tab */}
          <TabsContent value="members" className="mt-4">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Team Members</CardTitle>
                    <CardDescription>
                      {filteredMembers.length} of {members.length} members
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search members..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-9 w-64"
                      />
                    </div>
                    <Select value={statusFilter} onValueChange={setStatusFilter}>
                      <SelectTrigger className="w-32">
                        <SelectValue placeholder="Status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Status</SelectItem>
                        <SelectItem value="active">Active</SelectItem>
                        <SelectItem value="suspended">Suspended</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={roleFilter} onValueChange={setRoleFilter}>
                      <SelectTrigger className="w-32">
                        <SelectValue placeholder="Role" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Roles</SelectItem>
                        {roles.map((r) => (
                          <SelectItem key={r.id} value={r.id}>
                            {r.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b text-left text-sm text-muted-foreground bg-muted/30">
                        <th className="py-3 pl-4 font-medium">Member</th>
                        <th className="py-3 font-medium">Role</th>
                        <th className="py-3 font-medium">Status</th>
                        <th className="py-3 font-medium">Last Active</th>
                        <th className="py-3 pr-4 font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredMembers.map((member) => (
                        <TeamMemberRow
                          key={member.id}
                          member={member}
                          roles={roles}
                          onEdit={() => setEditingMember(member)}
                          onRemove={() => handleRemoveMember(member.id)}
                          onSuspend={() => handleSuspendMember(member.id)}
                          onReactivate={() => handleReactivateMember(member.id)}
                        />
                      ))}
                    </tbody>
                  </table>
                </div>
                {filteredMembers.length === 0 && (
                  <div className="py-12 text-center">
                    <Users className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-medium">No members found</h3>
                    <p className="text-sm text-muted-foreground">
                      Try adjusting your search or filters
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Invitations Tab */}
          <TabsContent value="invitations" className="mt-4">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Pending Invitations</CardTitle>
                    <CardDescription>
                      {invitations.filter((i) => i.status === "pending").length} pending,{" "}
                      {invitations.filter((i) => i.status === "expired").length} expired
                    </CardDescription>
                  </div>
                  <Button variant="outline" onClick={() => setShowInviteDialog(true)}>
                    <Plus className="mr-2 h-4 w-4" />
                    New Invitation
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {invitations.length > 0 ? (
                  invitations.map((invitation) => (
                    <InvitationRow
                      key={invitation.id}
                      invitation={invitation}
                      onResend={() => handleResendInvitation(invitation.id)}
                      onCancel={() => handleCancelInvitation(invitation.id)}
                    />
                  ))
                ) : (
                  <div className="py-12 text-center">
                    <Mail className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-medium">No invitations</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Invite team members to collaborate
                    </p>
                    <Button onClick={() => setShowInviteDialog(true)}>
                      <UserPlus className="mr-2 h-4 w-4" />
                      Invite Member
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Roles Tab */}
          <TabsContent value="roles" className="mt-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-medium">Roles & Permissions</h3>
                  <p className="text-sm text-muted-foreground">
                    Manage access levels and what each role can do
                  </p>
                </div>
                <Button variant="outline">
                  <Plus className="mr-2 h-4 w-4" />
                  Create Custom Role
                </Button>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                {roles.map((role) => (
                  <RolePermissionsCard key={role.id} role={role} allPerms={allPermissions} />
                ))}
              </div>
            </div>
          </TabsContent>

          {/* Activity Tab */}
          <TabsContent value="activity" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Activity Log</CardTitle>
                <CardDescription>
                  Recent team management activity
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <div className="divide-y">
                  {activityLogs.map((log) => (
                    <div key={log.id} className="flex items-center gap-4 p-4 hover:bg-muted/30">
                      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-muted">
                        {log.action === "invited" && <UserPlus className="h-4 w-4 text-green-600" />}
                        {log.action === "removed" && <UserMinus className="h-4 w-4 text-red-600" />}
                        {log.action === "changed role" && <UserCog className="h-4 w-4 text-blue-600" />}
                        {log.action === "suspended" && <Lock className="h-4 w-4 text-yellow-600" />}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm">
                          <span className="font-medium">{log.user}</span>{" "}
                          <span className="text-muted-foreground">{log.action}</span>{" "}
                          <span className="font-medium">{log.target}</span>
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {formatRelativeTime(log.timestamp)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Invite Dialog */}
      <InviteMemberDialog
        open={showInviteDialog}
        onOpenChange={setShowInviteDialog}
        roles={roles}
        onInvite={handleInvite}
      />

      {/* Edit Member Dialog */}
      <EditMemberDialog
        open={!!editingMember}
        onOpenChange={(open) => !open && setEditingMember(null)}
        member={editingMember}
        roles={roles}
        onSave={handleSaveMember}
      />
    </DashboardLayout>
  );
}
