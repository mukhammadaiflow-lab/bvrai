"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Bot,
  Plus,
  Search,
  MoreHorizontal,
  Play,
  Pause,
  Trash2,
  Edit,
  Copy,
  Phone,
  BarChart3,
  Settings2,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { agentsApi, Agent } from "@/lib/api";
import { formatNumber, formatRelativeTime } from "@/lib/utils";
import { toast } from "sonner";

// Agent Card Component
function AgentCard({ agent, onEdit, onDelete }: { agent: Agent; onEdit: () => void; onDelete: () => void }) {
  const queryClient = useQueryClient();

  const activateMutation = useMutation({
    mutationFn: () => agentsApi.activate(agent.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agents"] });
      toast.success("Agent activated");
    },
  });

  const pauseMutation = useMutation({
    mutationFn: () => agentsApi.pause(agent.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agents"] });
      toast.success("Agent paused");
    },
  });

  return (
    <Card className="card-hover">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Bot className="h-6 w-6 text-primary" />
            </div>
            <div>
              <CardTitle className="text-lg">{agent.name}</CardTitle>
              <CardDescription className="line-clamp-1">
                {agent.description || "No description"}
              </CardDescription>
            </div>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onEdit}>
                <Edit className="mr-2 h-4 w-4" />
                Edit
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Copy className="mr-2 h-4 w-4" />
                Duplicate
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Phone className="mr-2 h-4 w-4" />
                Test Call
              </DropdownMenuItem>
              <DropdownMenuItem>
                <BarChart3 className="mr-2 h-4 w-4" />
                Analytics
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings2 className="mr-2 h-4 w-4" />
                Configure
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              {agent.status === "active" ? (
                <DropdownMenuItem onClick={() => pauseMutation.mutate()}>
                  <Pause className="mr-2 h-4 w-4" />
                  Pause
                </DropdownMenuItem>
              ) : (
                <DropdownMenuItem onClick={() => activateMutation.mutate()}>
                  <Play className="mr-2 h-4 w-4" />
                  Activate
                </DropdownMenuItem>
              )}
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onDelete} className="text-destructive">
                <Trash2 className="mr-2 h-4 w-4" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Phone className="h-4 w-4" />
              <span>{formatNumber(0)} calls</span>
            </div>
            <span>•</span>
            <span>{formatRelativeTime(agent.created_at)}</span>
          </div>
          <Badge
            variant={agent.status === "active" ? "default" : "secondary"}
            className={agent.status === "active" ? "bg-success hover:bg-success/80" : ""}
          >
            <span className={`mr-1.5 h-1.5 w-1.5 rounded-full ${agent.status === "active" ? "bg-success-foreground" : "bg-muted-foreground"}`} />
            {agent.status}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

// Create/Edit Agent Dialog
function AgentDialog({
  open,
  onOpenChange,
  agent,
  onSuccess,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  agent?: Agent | null;
  onSuccess: () => void;
}) {
  const [name, setName] = useState(agent?.name || "");
  const [description, setDescription] = useState(agent?.description || "");
  const [systemPrompt, setSystemPrompt] = useState(agent?.system_prompt || "");
  const [voiceId, setVoiceId] = useState(agent?.voice_id || "");

  const createMutation = useMutation({
    mutationFn: (data: { name: string; description?: string; system_prompt?: string; voice_id?: string }) =>
      agent ? agentsApi.update(agent.id, data) : agentsApi.create(data),
    onSuccess: () => {
      toast.success(agent ? "Agent updated" : "Agent created");
      onSuccess();
      onOpenChange(false);
    },
    onError: () => {
      toast.error("Failed to save agent");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createMutation.mutate({
      name,
      description: description || undefined,
      system_prompt: systemPrompt || undefined,
      voice_id: voiceId || undefined,
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>{agent ? "Edit Agent" : "Create New Agent"}</DialogTitle>
            <DialogDescription>
              {agent
                ? "Update your agent's configuration"
                : "Configure your new AI voice agent"}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Customer Support Agent"
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="description">Description</Label>
              <Input
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Handles customer inquiries and support tickets"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="voice">Voice</Label>
              <Select value={voiceId} onValueChange={setVoiceId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a voice" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="alloy">Alloy (Neutral)</SelectItem>
                  <SelectItem value="echo">Echo (Male)</SelectItem>
                  <SelectItem value="fable">Fable (British)</SelectItem>
                  <SelectItem value="onyx">Onyx (Deep Male)</SelectItem>
                  <SelectItem value="nova">Nova (Female)</SelectItem>
                  <SelectItem value="shimmer">Shimmer (Soft Female)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="prompt">System Prompt</Label>
              <Textarea
                id="prompt"
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder="You are a helpful customer support agent..."
                rows={6}
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" loading={createMutation.isPending}>
              {agent ? "Save Changes" : "Create Agent"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export default function AgentsPage() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deletingAgent, setDeletingAgent] = useState<Agent | null>(null);

  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["agents", statusFilter],
    queryFn: () => agentsApi.list({ status: statusFilter === "all" ? undefined : statusFilter }),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => agentsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agents"] });
      toast.success("Agent deleted");
      setDeleteDialogOpen(false);
    },
  });

  const agents = data?.agents || [];
  const filteredAgents = agents.filter((agent: Agent) =>
    agent.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Agents</h1>
          <p className="text-muted-foreground">
            Create and manage your AI voice agents
          </p>
        </div>
        <Button onClick={() => { setEditingAgent(null); setDialogOpen(true); }}>
          <Plus className="mr-2 h-4 w-4" />
          Create Agent
        </Button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search agents..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-10"
          />
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-[150px]">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="inactive">Inactive</SelectItem>
            <SelectItem value="paused">Paused</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Agents Grid */}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <Card key={i} className="h-[180px] animate-pulse bg-muted" />
          ))}
        </div>
      ) : filteredAgents.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Bot className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No agents found</h3>
            <p className="text-muted-foreground text-center mb-4">
              {search
                ? "No agents match your search criteria"
                : "Get started by creating your first AI voice agent"}
            </p>
            {!search && (
              <Button onClick={() => setDialogOpen(true)}>
                <Plus className="mr-2 h-4 w-4" />
                Create Agent
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredAgents.map((agent: Agent) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              onEdit={() => { setEditingAgent(agent); setDialogOpen(true); }}
              onDelete={() => { setDeletingAgent(agent); setDeleteDialogOpen(true); }}
            />
          ))}
        </div>
      )}

      {/* Create/Edit Dialog */}
      <AgentDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        agent={editingAgent}
        onSuccess={() => queryClient.invalidateQueries({ queryKey: ["agents"] })}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Agent</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete &quot;{deletingAgent?.name}&quot;? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => deletingAgent && deleteMutation.mutate(deletingAgent.id)}
              loading={deleteMutation.isPending}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
