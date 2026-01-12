"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  FileText,
  Plus,
  Search,
  MoreVertical,
  Upload,
  Link as LinkIcon,
  Trash2,
  Edit,
  Eye,
  Database,
  Brain,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader2,
  File,
  FileCode,
  FileJson,
  Globe,
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { knowledgeBaseApi } from "@/lib/api";
import { formatRelativeTime, formatBytes, cn } from "@/lib/utils";

interface KnowledgeBase {
  id: string;
  name: string;
  description?: string;
  document_count: number;
  total_chunks: number;
  total_size_bytes: number;
  status: "active" | "processing" | "error";
  created_at: string;
  updated_at: string;
}

interface Document {
  id: string;
  knowledge_base_id: string;
  name: string;
  type: "pdf" | "txt" | "md" | "html" | "url" | "json";
  source_url?: string;
  size_bytes: number;
  chunk_count: number;
  status: "pending" | "processing" | "completed" | "failed";
  error_message?: string;
  created_at: string;
  processed_at?: string;
}

// Document Icon
function DocumentIcon({ type }: { type: string }) {
  const iconClass = "h-5 w-5";
  switch (type) {
    case "pdf":
      return <FileText className={`${iconClass} text-red-500`} />;
    case "json":
      return <FileJson className={`${iconClass} text-yellow-500`} />;
    case "md":
    case "txt":
      return <FileCode className={`${iconClass} text-blue-500`} />;
    case "url":
    case "html":
      return <Globe className={`${iconClass} text-green-500`} />;
    default:
      return <File className={`${iconClass} text-muted-foreground`} />;
  }
}

// Status Badge
function StatusBadge({ status }: { status: string }) {
  const variants: Record<string, { variant: "default" | "secondary" | "destructive" | "outline"; icon: React.ReactNode }> = {
    active: { variant: "default", icon: <CheckCircle className="h-3 w-3" /> },
    completed: { variant: "default", icon: <CheckCircle className="h-3 w-3" /> },
    processing: { variant: "secondary", icon: <Loader2 className="h-3 w-3 animate-spin" /> },
    pending: { variant: "outline", icon: <Clock className="h-3 w-3" /> },
    error: { variant: "destructive", icon: <AlertCircle className="h-3 w-3" /> },
    failed: { variant: "destructive", icon: <AlertCircle className="h-3 w-3" /> },
  };

  const { variant, icon } = variants[status] || variants.pending;

  return (
    <Badge variant={variant} className="gap-1">
      {icon}
      {status}
    </Badge>
  );
}

// Knowledge Base Card
function KnowledgeBaseCard({
  kb,
  onSelect,
  onEdit,
  onDelete,
}: {
  kb: KnowledgeBase;
  onSelect: (id: string) => void;
  onEdit: (kb: KnowledgeBase) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <Card
      className="cursor-pointer transition-all hover:shadow-md hover:border-primary/50"
      onClick={() => onSelect(kb.id)}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
              <Database className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-base">{kb.name}</CardTitle>
              {kb.description && (
                <CardDescription className="line-clamp-1">{kb.description}</CardDescription>
              )}
            </div>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
              <Button variant="ghost" size="icon">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onEdit(kb); }}>
                <Edit className="mr-2 h-4 w-4" />
                Edit
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={(e) => { e.stopPropagation(); onDelete(kb.id); }}
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
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1 text-muted-foreground">
              <FileText className="h-4 w-4" />
              <span>{kb.document_count} docs</span>
            </div>
            <div className="flex items-center gap-1 text-muted-foreground">
              <Brain className="h-4 w-4" />
              <span>{kb.total_chunks} chunks</span>
            </div>
          </div>
          <StatusBadge status={kb.status} />
        </div>
        <div className="mt-2 text-xs text-muted-foreground">
          Updated {formatRelativeTime(kb.updated_at)}
        </div>
      </CardContent>
    </Card>
  );
}

// Document Row
function DocumentRow({
  doc,
  onView,
  onDelete,
}: {
  doc: Document;
  onView: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border p-4">
      <div className="flex items-center gap-3">
        <DocumentIcon type={doc.type} />
        <div>
          <p className="font-medium">{doc.name}</p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>{doc.type.toUpperCase()}</span>
            <span>•</span>
            <span>{formatBytes(doc.size_bytes)}</span>
            <span>•</span>
            <span>{doc.chunk_count} chunks</span>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <StatusBadge status={doc.status} />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon">
              <MoreVertical className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => onView(doc.id)}>
              <Eye className="mr-2 h-4 w-4" />
              View Content
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => onDelete(doc.id)}
              className="text-destructive focus:text-destructive"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}

// Create Knowledge Base Dialog
function CreateKnowledgeBaseDialog({
  open,
  onOpenChange,
  editingKb,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editingKb?: KnowledgeBase | null;
}) {
  const [name, setName] = useState(editingKb?.name || "");
  const [description, setDescription] = useState(editingKb?.description || "");
  const queryClient = useQueryClient();

  const createMutation = useMutation({
    mutationFn: (data: { name: string; description?: string }) =>
      editingKb
        ? knowledgeBaseApi.update(editingKb.id, data)
        : knowledgeBaseApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] });
      onOpenChange(false);
      setName("");
      setDescription("");
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{editingKb ? "Edit Knowledge Base" : "Create Knowledge Base"}</DialogTitle>
          <DialogDescription>
            {editingKb
              ? "Update your knowledge base details"
              : "Create a new knowledge base to store and query your documents"}
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              placeholder="e.g., Product Documentation"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="description">Description (optional)</Label>
            <Textarea
              id="description"
              placeholder="Brief description of what this knowledge base contains..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => createMutation.mutate({ name, description })}
            disabled={!name.trim() || createMutation.isPending}
          >
            {createMutation.isPending
              ? editingKb
                ? "Saving..."
                : "Creating..."
              : editingKb
              ? "Save Changes"
              : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Add Document Dialog
function AddDocumentDialog({
  open,
  onOpenChange,
  knowledgeBaseId,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  knowledgeBaseId: string;
}) {
  const [activeTab, setActiveTab] = useState<"upload" | "url" | "text">("upload");
  const [url, setUrl] = useState("");
  const [textContent, setTextContent] = useState("");
  const [textName, setTextName] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: async (data: { type: string; content: File | string; name?: string; url?: string }) => {
      if (data.type === "file" && data.content instanceof File) {
        return knowledgeBaseApi.uploadDocument(knowledgeBaseId, data.content);
      } else if (data.type === "url") {
        return knowledgeBaseApi.addUrl(knowledgeBaseId, data.url!);
      } else {
        return knowledgeBaseApi.addText(knowledgeBaseId, {
          name: data.name!,
          content: data.content as string,
        });
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["knowledge-base-documents", knowledgeBaseId] });
      queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] });
      onOpenChange(false);
      setUrl("");
      setTextContent("");
      setTextName("");
      setFiles(null);
    },
  });

  const handleSubmit = () => {
    if (activeTab === "upload" && files && files.length > 0) {
      uploadMutation.mutate({ type: "file", content: files[0] });
    } else if (activeTab === "url" && url) {
      uploadMutation.mutate({ type: "url", content: "", url });
    } else if (activeTab === "text" && textContent && textName) {
      uploadMutation.mutate({ type: "text", content: textContent, name: textName });
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Add Document</DialogTitle>
          <DialogDescription>Add a new document to your knowledge base</DialogDescription>
        </DialogHeader>
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="upload">
              <Upload className="mr-2 h-4 w-4" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="url">
              <LinkIcon className="mr-2 h-4 w-4" />
              URL
            </TabsTrigger>
            <TabsTrigger value="text">
              <FileText className="mr-2 h-4 w-4" />
              Text
            </TabsTrigger>
          </TabsList>
          <TabsContent value="upload" className="mt-4">
            <div className="space-y-4">
              <div
                className={cn(
                  "flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors",
                  files ? "border-primary bg-primary/5" : "border-muted-foreground/25"
                )}
              >
                <Upload className="h-8 w-8 text-muted-foreground" />
                <p className="mt-2 text-sm text-muted-foreground">
                  Drag & drop or click to upload
                </p>
                <p className="text-xs text-muted-foreground">
                  Supports PDF, TXT, MD, HTML, JSON
                </p>
                <input
                  type="file"
                  className="absolute inset-0 cursor-pointer opacity-0"
                  accept=".pdf,.txt,.md,.html,.json"
                  onChange={(e) => setFiles(e.target.files)}
                />
              </div>
              {files && files.length > 0 && (
                <div className="flex items-center gap-2 rounded-lg bg-muted p-3">
                  <FileText className="h-5 w-5" />
                  <span className="flex-1 text-sm">{files[0].name}</span>
                  <Button variant="ghost" size="sm" onClick={() => setFiles(null)}>
                    Remove
                  </Button>
                </div>
              )}
            </div>
          </TabsContent>
          <TabsContent value="url" className="mt-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="url">Website URL</Label>
                <Input
                  id="url"
                  type="url"
                  placeholder="https://example.com/docs"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                We&apos;ll crawl and extract content from this URL
              </p>
            </div>
          </TabsContent>
          <TabsContent value="text" className="mt-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="text-name">Document Name</Label>
                <Input
                  id="text-name"
                  placeholder="e.g., FAQ"
                  value={textName}
                  onChange={(e) => setTextName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="text-content">Content</Label>
                <Textarea
                  id="text-content"
                  placeholder="Paste your text content here..."
                  value={textContent}
                  onChange={(e) => setTextContent(e.target.value)}
                  rows={8}
                />
              </div>
            </div>
          </TabsContent>
        </Tabs>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={
              uploadMutation.isPending ||
              (activeTab === "upload" && !files) ||
              (activeTab === "url" && !url) ||
              (activeTab === "text" && (!textContent || !textName))
            }
          >
            {uploadMutation.isPending ? "Adding..." : "Add Document"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Knowledge Base Detail View
function KnowledgeBaseDetail({
  kb,
  onBack,
}: {
  kb: KnowledgeBase;
  onBack: () => void;
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const [addDocOpen, setAddDocOpen] = useState(false);
  const queryClient = useQueryClient();

  const { data: documentsData, isLoading } = useQuery({
    queryKey: ["knowledge-base-documents", kb.id],
    queryFn: () => knowledgeBaseApi.getDocuments(kb.id),
  });

  const deleteMutation = useMutation({
    mutationFn: (docId: string) => knowledgeBaseApi.deleteDocument(kb.id, docId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["knowledge-base-documents", kb.id] });
      queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] });
    },
  });

  // Mock documents
  const mockDocuments: Document[] = [
    {
      id: "1",
      knowledge_base_id: kb.id,
      name: "Product Manual v2.0.pdf",
      type: "pdf",
      size_bytes: 2450000,
      chunk_count: 145,
      status: "completed",
      created_at: "2024-01-15T10:00:00Z",
      processed_at: "2024-01-15T10:05:00Z",
    },
    {
      id: "2",
      knowledge_base_id: kb.id,
      name: "FAQ.md",
      type: "md",
      size_bytes: 45000,
      chunk_count: 32,
      status: "completed",
      created_at: "2024-01-14T09:00:00Z",
      processed_at: "2024-01-14T09:01:00Z",
    },
    {
      id: "3",
      knowledge_base_id: kb.id,
      name: "API Documentation",
      type: "url",
      source_url: "https://docs.example.com/api",
      size_bytes: 128000,
      chunk_count: 78,
      status: "processing",
      created_at: "2024-01-16T14:00:00Z",
    },
    {
      id: "4",
      knowledge_base_id: kb.id,
      name: "pricing.json",
      type: "json",
      size_bytes: 12000,
      chunk_count: 15,
      status: "completed",
      created_at: "2024-01-13T11:00:00Z",
      processed_at: "2024-01-13T11:01:00Z",
    },
  ];

  const documents = documentsData?.documents || mockDocuments;

  const filteredDocuments = documents.filter((doc: Document) =>
    doc.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleView = (id: string) => {
    console.log("View document:", id);
  };

  const handleDelete = (id: string) => {
    if (confirm("Are you sure you want to delete this document?")) {
      deleteMutation.mutate(id);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" onClick={onBack}>
          ←
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold">{kb.name}</h1>
          {kb.description && <p className="text-muted-foreground">{kb.description}</p>}
        </div>
        <Button onClick={() => setAddDocOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Add Document
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{kb.document_count}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Chunks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{kb.total_chunks}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Size</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatBytes(kb.total_size_bytes)}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Status</CardTitle>
          </CardHeader>
          <CardContent>
            <StatusBadge status={kb.status} />
          </CardContent>
        </Card>
      </div>

      {/* Documents List */}
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle>Documents</CardTitle>
              <CardDescription>All documents in this knowledge base</CardDescription>
            </div>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 w-[300px]"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
            </div>
          ) : filteredDocuments.length === 0 ? (
            <div className="text-center py-8">
              <FileText className="mx-auto h-12 w-12 text-muted-foreground/50" />
              <h3 className="mt-4 text-lg font-medium">No documents found</h3>
              <p className="mt-2 text-sm text-muted-foreground">
                {searchQuery
                  ? "Try adjusting your search"
                  : "Add your first document to this knowledge base"}
              </p>
              {!searchQuery && (
                <Button className="mt-4" onClick={() => setAddDocOpen(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Document
                </Button>
              )}
            </div>
          ) : (
            <div className="space-y-2">
              {filteredDocuments.map((doc: Document) => (
                <DocumentRow
                  key={doc.id}
                  doc={doc}
                  onView={handleView}
                  onDelete={handleDelete}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <AddDocumentDialog
        open={addDocOpen}
        onOpenChange={setAddDocOpen}
        knowledgeBaseId={kb.id}
      />
    </div>
  );
}

export default function KnowledgeBasePage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editingKb, setEditingKb] = useState<KnowledgeBase | null>(null);
  const [selectedKbId, setSelectedKbId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Fetch knowledge bases
  const { data: kbData, isLoading } = useQuery({
    queryKey: ["knowledge-bases"],
    queryFn: () => knowledgeBaseApi.list(),
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => knowledgeBaseApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] });
    },
  });

  // Mock data
  const mockKnowledgeBases: KnowledgeBase[] = [
    {
      id: "1",
      name: "Product Documentation",
      description: "All product manuals, guides, and specifications",
      document_count: 24,
      total_chunks: 1856,
      total_size_bytes: 45000000,
      status: "active",
      created_at: "2024-01-10T10:00:00Z",
      updated_at: "2024-01-20T14:00:00Z",
    },
    {
      id: "2",
      name: "Customer Support FAQ",
      description: "Common questions and troubleshooting guides",
      document_count: 12,
      total_chunks: 524,
      total_size_bytes: 8500000,
      status: "active",
      created_at: "2024-01-05T09:00:00Z",
      updated_at: "2024-01-18T11:00:00Z",
    },
    {
      id: "3",
      name: "Sales Playbook",
      description: "Sales scripts, objection handling, and product comparisons",
      document_count: 8,
      total_chunks: 312,
      total_size_bytes: 5200000,
      status: "processing",
      created_at: "2024-01-15T14:00:00Z",
      updated_at: "2024-01-15T14:30:00Z",
    },
    {
      id: "4",
      name: "Legal & Compliance",
      description: "Terms of service, privacy policy, and compliance documents",
      document_count: 6,
      total_chunks: 189,
      total_size_bytes: 3100000,
      status: "active",
      created_at: "2024-01-01T08:00:00Z",
      updated_at: "2024-01-12T16:00:00Z",
    },
  ];

  const knowledgeBases = kbData?.knowledge_bases || mockKnowledgeBases;

  // Filter knowledge bases
  const filteredKnowledgeBases = knowledgeBases.filter((kb: KnowledgeBase) =>
    kb.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    kb.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Find selected KB
  const selectedKb = selectedKbId
    ? knowledgeBases.find((kb: KnowledgeBase) => kb.id === selectedKbId)
    : null;

  const handleEdit = (kb: KnowledgeBase) => {
    setEditingKb(kb);
    setCreateDialogOpen(true);
  };

  const handleDelete = (id: string) => {
    if (confirm("Are you sure you want to delete this knowledge base? All documents will be permanently removed.")) {
      deleteMutation.mutate(id);
    }
  };

  // Show detail view if KB is selected
  if (selectedKb) {
    return (
      <KnowledgeBaseDetail
        kb={selectedKb}
        onBack={() => setSelectedKbId(null)}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Knowledge Base</h1>
          <p className="text-muted-foreground">
            Manage your documents and training data for AI agents
          </p>
        </div>
        <Button onClick={() => setCreateDialogOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Create Knowledge Base
        </Button>
      </div>

      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search knowledge bases..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10"
        />
      </div>

      {/* Knowledge Bases Grid */}
      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      ) : filteredKnowledgeBases.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Database className="h-12 w-12 text-muted-foreground/50" />
            <h3 className="mt-4 text-lg font-medium">No knowledge bases found</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              {searchQuery
                ? "Try adjusting your search"
                : "Create your first knowledge base to get started"}
            </p>
            {!searchQuery && (
              <Button className="mt-4" onClick={() => setCreateDialogOpen(true)}>
                <Plus className="mr-2 h-4 w-4" />
                Create Knowledge Base
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredKnowledgeBases.map((kb: KnowledgeBase) => (
            <KnowledgeBaseCard
              key={kb.id}
              kb={kb}
              onSelect={setSelectedKbId}
              onEdit={handleEdit}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}

      {/* Create/Edit Dialog */}
      <CreateKnowledgeBaseDialog
        open={createDialogOpen}
        onOpenChange={(open) => {
          setCreateDialogOpen(open);
          if (!open) setEditingKb(null);
        }}
        editingKb={editingKb}
      />
    </div>
  );
}
