"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Search,
  Plus,
  MoreVertical,
  Play,
  Pause,
  Edit,
  Copy,
  Trash2,
  GitBranch,
  Clock,
  CheckCircle,
  XCircle,
  ArrowRight,
  Zap,
  MessageSquare,
  Phone,
  Database,
  Send,
  Filter,
  Settings,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { workflowsApi } from "@/lib/api";
import { formatRelativeTime, cn } from "@/lib/utils";

// Types
interface WorkflowNode {
  id: string;
  type: "trigger" | "action" | "condition" | "delay";
  name: string;
  config: Record<string, unknown>;
}

interface Workflow {
  id: string;
  name: string;
  description?: string;
  status: "active" | "inactive" | "draft";
  trigger_type: "call_started" | "call_ended" | "scheduled" | "webhook" | "manual";
  nodes: WorkflowNode[];
  executions_count: number;
  success_rate: number;
  last_executed_at?: string;
  created_at: string;
  updated_at: string;
}

// Node type icons
const nodeIcons: Record<string, React.ComponentType<{ className?: string }>> = {
  trigger: Zap,
  action: ArrowRight,
  condition: GitBranch,
  delay: Clock,
  message: MessageSquare,
  call: Phone,
  crm: Database,
  webhook: Send,
};

// Workflow Card
function WorkflowCard({
  workflow,
  onEdit,
  onDuplicate,
  onDelete,
  onToggle,
}: {
  workflow: Workflow;
  onEdit: (id: string) => void;
  onDuplicate: (id: string) => void;
  onDelete: (id: string) => void;
  onToggle: (id: string, status: "active" | "inactive") => void;
}) {
  const triggerLabels: Record<string, string> = {
    call_started: "When call starts",
    call_ended: "When call ends",
    scheduled: "On schedule",
    webhook: "Webhook received",
    manual: "Manual trigger",
  };

  return (
    <Card className="transition-all hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "flex h-10 w-10 items-center justify-center rounded-lg",
                workflow.status === "active"
                  ? "bg-success/10 text-success"
                  : workflow.status === "draft"
                  ? "bg-warning/10 text-warning"
                  : "bg-muted text-muted-foreground"
              )}
            >
              <GitBranch className="h-5 w-5" />
            </div>
            <div>
              <CardTitle className="text-base">{workflow.name}</CardTitle>
              <Badge
                variant={
                  workflow.status === "active"
                    ? "default"
                    : workflow.status === "draft"
                    ? "secondary"
                    : "outline"
                }
                className={cn(
                  "mt-1",
                  workflow.status === "active" && "bg-success text-success-foreground"
                )}
              >
                {workflow.status}
              </Badge>
            </div>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => onEdit(workflow.id)}>
                <Edit className="mr-2 h-4 w-4" />
                Edit
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onDuplicate(workflow.id)}>
                <Copy className="mr-2 h-4 w-4" />
                Duplicate
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => onToggle(workflow.id, workflow.status === "active" ? "inactive" : "active")}
              >
                {workflow.status === "active" ? (
                  <>
                    <Pause className="mr-2 h-4 w-4" />
                    Deactivate
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Activate
                  </>
                )}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => onDelete(workflow.id)}
                className="text-destructive focus:text-destructive"
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>
      <CardContent>
        {workflow.description && (
          <p className="text-sm text-muted-foreground mb-4 line-clamp-2">{workflow.description}</p>
        )}

        {/* Trigger info */}
        <div className="flex items-center gap-2 mb-4">
          <Zap className="h-4 w-4 text-primary" />
          <span className="text-sm">{triggerLabels[workflow.trigger_type]}</span>
        </div>

        {/* Nodes preview */}
        <div className="flex items-center gap-1 mb-4 overflow-x-auto pb-2">
          {workflow.nodes.slice(0, 5).map((node, index) => {
            const Icon = nodeIcons[node.type] || ArrowRight;
            return (
              <div key={node.id} className="flex items-center">
                <div className="flex h-8 w-8 items-center justify-center rounded bg-muted">
                  <Icon className="h-4 w-4" />
                </div>
                {index < workflow.nodes.length - 1 && index < 4 && (
                  <ArrowRight className="h-4 w-4 mx-1 text-muted-foreground" />
                )}
              </div>
            );
          })}
          {workflow.nodes.length > 5 && (
            <span className="text-xs text-muted-foreground ml-2">
              +{workflow.nodes.length - 5} more
            </span>
          )}
        </div>

        {/* Stats */}
        <div className="flex items-center justify-between text-xs text-muted-foreground border-t pt-4">
          <div className="flex items-center gap-4">
            <span>{workflow.executions_count} executions</span>
            <span className="flex items-center gap-1">
              {workflow.success_rate >= 90 ? (
                <CheckCircle className="h-3 w-3 text-success" />
              ) : workflow.success_rate >= 70 ? (
                <CheckCircle className="h-3 w-3 text-warning" />
              ) : (
                <XCircle className="h-3 w-3 text-destructive" />
              )}
              {workflow.success_rate}% success
            </span>
          </div>
          <span>
            {workflow.last_executed_at
              ? `Last run ${formatRelativeTime(workflow.last_executed_at)}`
              : "Never run"}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

// Create Workflow Dialog
function CreateWorkflowDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [triggerType, setTriggerType] = useState<Workflow["trigger_type"]>("call_ended");
  const queryClient = useQueryClient();

  const createMutation = useMutation({
    mutationFn: (data: { name: string; description?: string; trigger_type: string }) =>
      workflowsApi.create(data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["workflows"] });
      onOpenChange(false);
      setName("");
      setDescription("");
      // Would redirect to workflow editor
      window.location.href = `/dashboard/workflows/${data.id}/edit`;
    },
  });

  const triggerOptions = [
    { value: "call_started", label: "When call starts", description: "Trigger when a new call begins" },
    { value: "call_ended", label: "When call ends", description: "Trigger after a call completes" },
    { value: "scheduled", label: "On schedule", description: "Run on a recurring schedule" },
    { value: "webhook", label: "Webhook received", description: "Trigger via external webhook" },
    { value: "manual", label: "Manual trigger", description: "Run manually when needed" },
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Create Workflow</DialogTitle>
          <DialogDescription>
            Create a new automation workflow for your voice agents
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Workflow Name</Label>
            <Input
              id="name"
              placeholder="e.g., Post-call CRM Update"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description (optional)</Label>
            <Textarea
              id="description"
              placeholder="Describe what this workflow does..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
            />
          </div>

          <div className="space-y-2">
            <Label>Trigger</Label>
            <div className="grid gap-2">
              {triggerOptions.map((option) => (
                <div
                  key={option.value}
                  className={cn(
                    "flex items-center gap-3 rounded-lg border p-3 cursor-pointer transition-colors",
                    triggerType === option.value
                      ? "border-primary bg-primary/5"
                      : "hover:border-muted-foreground/50"
                  )}
                  onClick={() => setTriggerType(option.value as Workflow["trigger_type"])}
                >
                  <div
                    className={cn(
                      "flex h-8 w-8 items-center justify-center rounded-full",
                      triggerType === option.value ? "bg-primary text-primary-foreground" : "bg-muted"
                    )}
                  >
                    <Zap className="h-4 w-4" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-sm">{option.label}</p>
                    <p className="text-xs text-muted-foreground">{option.description}</p>
                  </div>
                  {triggerType === option.value && (
                    <CheckCircle className="h-5 w-5 text-primary" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => createMutation.mutate({ name, description, trigger_type: triggerType })}
            disabled={!name.trim() || createMutation.isPending}
          >
            {createMutation.isPending ? "Creating..." : "Create Workflow"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Workflow Templates
function WorkflowTemplates({ onUseTemplate }: { onUseTemplate: (templateId: string) => void }) {
  const templates = [
    {
      id: "post-call-crm",
      name: "Post-Call CRM Update",
      description: "Automatically update your CRM after each call with notes and outcomes",
      trigger: "call_ended",
      nodes: 4,
    },
    {
      id: "appointment-confirmation",
      name: "Appointment Confirmation",
      description: "Send SMS/email confirmation after scheduling an appointment",
      trigger: "call_ended",
      nodes: 3,
    },
    {
      id: "lead-qualification",
      name: "Lead Qualification",
      description: "Route qualified leads to your sales team automatically",
      trigger: "call_ended",
      nodes: 5,
    },
    {
      id: "follow-up-sequence",
      name: "Follow-up Sequence",
      description: "Schedule automated follow-up calls for missed connections",
      trigger: "call_ended",
      nodes: 4,
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Templates</CardTitle>
        <CardDescription>Start with a pre-built workflow</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-2">
          {templates.map((template) => (
            <div
              key={template.id}
              className="flex items-start gap-3 rounded-lg border p-3 cursor-pointer hover:border-primary/50 transition-colors"
              onClick={() => onUseTemplate(template.id)}
            >
              <div className="flex h-8 w-8 items-center justify-center rounded bg-primary/10">
                <GitBranch className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium text-sm">{template.name}</p>
                <p className="text-xs text-muted-foreground line-clamp-2">{template.description}</p>
                <p className="text-xs text-muted-foreground mt-1">{template.nodes} steps</p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default function WorkflowsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "active" | "inactive" | "draft">("all");
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const queryClient = useQueryClient();

  const { data: workflowsData, isLoading } = useQuery({
    queryKey: ["workflows"],
    queryFn: () => workflowsApi.list(),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => workflowsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workflows"] });
    },
  });

  const toggleMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: "active" | "inactive" }) =>
      workflowsApi.update(id, { status }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workflows"] });
    },
  });

  // Mock workflows
  const mockWorkflows: Workflow[] = [
    {
      id: "1",
      name: "Post-Call CRM Sync",
      description: "Automatically creates or updates contact records in Salesforce after each call",
      status: "active",
      trigger_type: "call_ended",
      nodes: [
        { id: "1", type: "trigger", name: "Call Ended", config: {} },
        { id: "2", type: "condition", name: "Check Outcome", config: {} },
        { id: "3", type: "action", name: "Update CRM", config: {} },
        { id: "4", type: "action", name: "Send Slack", config: {} },
      ],
      executions_count: 1234,
      success_rate: 98.5,
      last_executed_at: "2024-01-20T14:30:00Z",
      created_at: "2024-01-01T10:00:00Z",
      updated_at: "2024-01-15T09:00:00Z",
    },
    {
      id: "2",
      name: "Appointment Confirmation",
      description: "Sends confirmation SMS and email when an appointment is scheduled",
      status: "active",
      trigger_type: "call_ended",
      nodes: [
        { id: "1", type: "trigger", name: "Call Ended", config: {} },
        { id: "2", type: "condition", name: "Appointment Booked?", config: {} },
        { id: "3", type: "action", name: "Send SMS", config: {} },
        { id: "4", type: "action", name: "Send Email", config: {} },
        { id: "5", type: "action", name: "Create Calendar", config: {} },
      ],
      executions_count: 567,
      success_rate: 99.1,
      last_executed_at: "2024-01-20T16:45:00Z",
      created_at: "2024-01-05T11:00:00Z",
      updated_at: "2024-01-18T14:00:00Z",
    },
    {
      id: "3",
      name: "Lead Routing",
      description: "Routes qualified leads to appropriate sales reps based on criteria",
      status: "inactive",
      trigger_type: "call_ended",
      nodes: [
        { id: "1", type: "trigger", name: "Call Ended", config: {} },
        { id: "2", type: "condition", name: "Lead Score", config: {} },
        { id: "3", type: "action", name: "Assign Rep", config: {} },
      ],
      executions_count: 234,
      success_rate: 95.2,
      last_executed_at: "2024-01-15T10:00:00Z",
      created_at: "2024-01-08T09:00:00Z",
      updated_at: "2024-01-10T11:00:00Z",
    },
    {
      id: "4",
      name: "Daily Summary Report",
      description: "Generates and sends daily summary of all calls to team",
      status: "active",
      trigger_type: "scheduled",
      nodes: [
        { id: "1", type: "trigger", name: "Daily at 6 PM", config: {} },
        { id: "2", type: "action", name: "Generate Report", config: {} },
        { id: "3", type: "action", name: "Send Email", config: {} },
      ],
      executions_count: 30,
      success_rate: 100,
      last_executed_at: "2024-01-19T18:00:00Z",
      created_at: "2024-01-01T08:00:00Z",
      updated_at: "2024-01-01T08:00:00Z",
    },
    {
      id: "5",
      name: "Follow-up Reminder (Draft)",
      description: "Work in progress - schedules follow-up calls",
      status: "draft",
      trigger_type: "call_ended",
      nodes: [
        { id: "1", type: "trigger", name: "Call Ended", config: {} },
        { id: "2", type: "delay", name: "Wait 24h", config: {} },
      ],
      executions_count: 0,
      success_rate: 0,
      created_at: "2024-01-18T15:00:00Z",
      updated_at: "2024-01-19T10:00:00Z",
    },
  ];

  const workflows = workflowsData?.workflows || mockWorkflows;

  // Filter workflows
  const filteredWorkflows = workflows.filter((workflow: Workflow) => {
    const matchesSearch =
      workflow.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      workflow.description?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === "all" || workflow.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  // Stats
  const activeCount = workflows.filter((w: Workflow) => w.status === "active").length;
  const totalExecutions = workflows.reduce((sum: number, w: Workflow) => sum + w.executions_count, 0);

  const handleEdit = (id: string) => {
    window.location.href = `/dashboard/workflows/${id}/edit`;
  };

  const handleDuplicate = (id: string) => {
    console.log("Duplicate workflow:", id);
  };

  const handleDelete = (id: string) => {
    if (confirm("Are you sure you want to delete this workflow?")) {
      deleteMutation.mutate(id);
    }
  };

  const handleToggle = (id: string, status: "active" | "inactive") => {
    toggleMutation.mutate({ id, status });
  };

  const handleUseTemplate = (templateId: string) => {
    console.log("Use template:", templateId);
    setCreateDialogOpen(true);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Workflows</h1>
          <p className="text-muted-foreground">
            Automate actions based on call events and triggers
          </p>
        </div>
        <Button onClick={() => setCreateDialogOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Create Workflow
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Workflows</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{workflows.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Active</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-success">{activeCount}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Executions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalExecutions.toLocaleString()}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Avg Success Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(
                workflows.reduce((sum: number, w: Workflow) => sum + w.success_rate, 0) /
                workflows.length
              ).toFixed(1)}
              %
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search workflows..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
        <Select value={statusFilter} onValueChange={(v) => setStatusFilter(v as typeof statusFilter)}>
          <SelectTrigger className="w-[150px]">
            <Filter className="mr-2 h-4 w-4" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="inactive">Inactive</SelectItem>
            <SelectItem value="draft">Draft</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Workflows List */}
        <div className="lg:col-span-2 space-y-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
            </div>
          ) : filteredWorkflows.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <GitBranch className="h-12 w-12 text-muted-foreground/50" />
                <h3 className="mt-4 text-lg font-medium">No workflows found</h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  {searchQuery || statusFilter !== "all"
                    ? "Try adjusting your search or filter"
                    : "Create your first workflow to automate tasks"}
                </p>
                {!searchQuery && statusFilter === "all" && (
                  <Button className="mt-4" onClick={() => setCreateDialogOpen(true)}>
                    <Plus className="mr-2 h-4 w-4" />
                    Create Workflow
                  </Button>
                )}
              </CardContent>
            </Card>
          ) : (
            filteredWorkflows.map((workflow: Workflow) => (
              <WorkflowCard
                key={workflow.id}
                workflow={workflow}
                onEdit={handleEdit}
                onDuplicate={handleDuplicate}
                onDelete={handleDelete}
                onToggle={handleToggle}
              />
            ))
          )}
        </div>

        {/* Templates Sidebar */}
        <div>
          <WorkflowTemplates onUseTemplate={handleUseTemplate} />
        </div>
      </div>

      {/* Create Dialog */}
      <CreateWorkflowDialog open={createDialogOpen} onOpenChange={setCreateDialogOpen} />
    </div>
  );
}
