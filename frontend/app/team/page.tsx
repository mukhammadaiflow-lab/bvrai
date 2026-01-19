"use client";

import React, { useState } from "react";
import {
  Users,
  Plus,
  Search,
  Mail,
  Shield,
  MoreHorizontal,
  Trash2,
  Edit,
  Crown,
  Clock,
  Check,
  X,
  UserPlus,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn, formatRelativeTime } from "@/lib/utils";

// Mock team members data
const teamMembers = [
  {
    id: "1",
    name: "John Smith",
    email: "john@example.com",
    role: "owner",
    status: "active",
    lastActive: "2024-01-14T10:30:00Z",
    joinedAt: "2023-01-15T00:00:00Z",
    avatar: null,
  },
  {
    id: "2",
    name: "Sarah Johnson",
    email: "sarah@example.com",
    role: "admin",
    status: "active",
    lastActive: "2024-01-14T09:15:00Z",
    joinedAt: "2023-06-20T00:00:00Z",
    avatar: null,
  },
  {
    id: "3",
    name: "Mike Brown",
    email: "mike@example.com",
    role: "member",
    status: "active",
    lastActive: "2024-01-13T16:45:00Z",
    joinedAt: "2023-09-10T00:00:00Z",
    avatar: null,
  },
  {
    id: "4",
    name: "Emily Davis",
    email: "emily@example.com",
    role: "viewer",
    status: "active",
    lastActive: "2024-01-12T11:00:00Z",
    joinedAt: "2024-01-01T00:00:00Z",
    avatar: null,
  },
];

const pendingInvitations = [
  {
    id: "inv-1",
    email: "alex@example.com",
    role: "member",
    sentAt: "2024-01-13T10:00:00Z",
    expiresAt: "2024-01-20T10:00:00Z",
  },
  {
    id: "inv-2",
    email: "chris@example.com",
    role: "viewer",
    sentAt: "2024-01-12T14:00:00Z",
    expiresAt: "2024-01-19T14:00:00Z",
  },
];

const roles = [
  {
    id: "owner",
    name: "Owner",
    description: "Full access to all features and settings",
    permissions: ["All permissions"],
  },
  {
    id: "admin",
    name: "Admin",
    description: "Manage team, agents, and settings",
    permissions: ["Manage team", "Manage agents", "View analytics", "Configure webhooks"],
  },
  {
    id: "member",
    name: "Member",
    description: "Create and manage agents",
    permissions: ["Create agents", "Edit agents", "View calls", "View analytics"],
  },
  {
    id: "viewer",
    name: "Viewer",
    description: "View-only access",
    permissions: ["View agents", "View calls", "View analytics"],
  },
];

export default function TeamPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [showInviteModal, setShowInviteModal] = useState(false);

  const filteredMembers = teamMembers.filter(
    (member) =>
      member.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      member.email.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getRoleBadgeColor = (role: string) => {
    switch (role) {
      case "owner":
        return "bg-purple-100 text-purple-800";
      case "admin":
        return "bg-blue-100 text-blue-800";
      case "member":
        return "bg-green-100 text-green-800";
      case "viewer":
        return "bg-gray-100 text-gray-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Team</h1>
          <p className="text-muted-foreground">
            Manage your team members and their permissions
          </p>
        </div>
        <Button onClick={() => setShowInviteModal(true)}>
          <UserPlus className="mr-2 h-4 w-4" />
          Invite Member
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                <Users className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{teamMembers.length}</p>
                <p className="text-sm text-muted-foreground">Team Members</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100">
                <Mail className="h-5 w-5 text-yellow-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{pendingInvitations.length}</p>
                <p className="text-sm text-muted-foreground">Pending Invitations</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
                <Shield className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">6</p>
                <p className="text-sm text-muted-foreground">Seats Available</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="flex gap-4">
        <Input
          placeholder="Search members..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          leftIcon={<Search className="h-4 w-4" />}
          className="max-w-md"
        />
      </div>

      {/* Team Members */}
      <Card>
        <CardHeader>
          <CardTitle>Team Members</CardTitle>
          <CardDescription>
            People with access to your organization
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b text-left text-sm text-muted-foreground">
                  <th className="pb-3 font-medium">Member</th>
                  <th className="pb-3 font-medium">Role</th>
                  <th className="pb-3 font-medium">Status</th>
                  <th className="pb-3 font-medium">Last Active</th>
                  <th className="pb-3 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredMembers.map((member) => (
                  <tr key={member.id} className="border-b last:border-0">
                    <td className="py-4">
                      <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 font-medium text-primary">
                          {getInitials(member.name)}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <p className="font-medium">{member.name}</p>
                            {member.role === "owner" && (
                              <Crown className="h-4 w-4 text-yellow-500" />
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground">{member.email}</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-4">
                      <Badge className={getRoleBadgeColor(member.role)}>
                        {member.role}
                      </Badge>
                    </td>
                    <td className="py-4">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full bg-green-500" />
                        <span className="text-sm capitalize">{member.status}</span>
                      </div>
                    </td>
                    <td className="py-4">
                      <span className="text-sm">{formatRelativeTime(member.lastActive)}</span>
                    </td>
                    <td className="py-4">
                      <div className="flex items-center gap-2">
                        <Button variant="ghost" size="icon-sm" disabled={member.role === "owner"}>
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          className="text-red-600"
                          disabled={member.role === "owner"}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Pending Invitations */}
      {pendingInvitations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Pending Invitations</CardTitle>
            <CardDescription>
              Invitations waiting for acceptance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {pendingInvitations.map((invitation) => (
                <div
                  key={invitation.id}
                  className="flex items-center justify-between rounded-lg border p-4"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100">
                      <Mail className="h-5 w-5 text-yellow-600" />
                    </div>
                    <div>
                      <p className="font-medium">{invitation.email}</p>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Badge variant="outline" className="text-xs">
                          {invitation.role}
                        </Badge>
                        <span>&bull;</span>
                        <Clock className="h-3 w-3" />
                        <span>Sent {formatRelativeTime(invitation.sentAt)}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm">
                      Resend
                    </Button>
                    <Button variant="ghost" size="icon-sm" className="text-red-600">
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Roles Reference */}
      <Card>
        <CardHeader>
          <CardTitle>Role Permissions</CardTitle>
          <CardDescription>
            Understanding what each role can do
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {roles.map((role) => (
              <div key={role.id} className="rounded-lg border p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Badge className={getRoleBadgeColor(role.id)}>{role.name}</Badge>
                  {role.id === "owner" && <Crown className="h-4 w-4 text-yellow-500" />}
                </div>
                <p className="text-sm text-muted-foreground mb-3">{role.description}</p>
                <ul className="space-y-1">
                  {role.permissions.map((permission, i) => (
                    <li key={i} className="flex items-center gap-2 text-xs">
                      <Check className="h-3 w-3 text-green-600" />
                      {permission}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
