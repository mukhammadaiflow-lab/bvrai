"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Bot,
  X,
  Play,
  Pause,
  Volume2,
  Copy,
  Check,
  Zap,
  Clock,
  Users,
  Wrench,
  MessageSquare,
  ChevronRight,
  Loader2,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { AgentTemplate } from "./template-card";

interface TemplatePreviewProps {
  template: AgentTemplate | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUse: (template: AgentTemplate) => void;
}

const categoryColors: Record<string, string> = {
  "Customer Support": "bg-blue-100 text-blue-700",
  "Sales": "bg-green-100 text-green-700",
  "Scheduling": "bg-purple-100 text-purple-700",
  "Healthcare": "bg-red-100 text-red-700",
  "Real Estate": "bg-amber-100 text-amber-700",
  "E-commerce": "bg-pink-100 text-pink-700",
  "Travel": "bg-cyan-100 text-cyan-700",
  "Education": "bg-indigo-100 text-indigo-700",
  "Financial": "bg-emerald-100 text-emerald-700",
};

export function TemplatePreview({
  template,
  open,
  onOpenChange,
  onUse,
}: TemplatePreviewProps) {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState("overview");
  const [isPlaying, setIsPlaying] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isCreating, setIsCreating] = useState(false);

  if (!template) return null;

  const handlePlayDemo = () => {
    setIsPlaying(!isPlaying);
    if (!isPlaying) {
      // Simulate playing demo
      setTimeout(() => setIsPlaying(false), 5000);
    }
  };

  const handleCopyPrompt = async () => {
    await navigator.clipboard.writeText(template.sample_prompt);
    setCopied(true);
    toast.success("Prompt copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleUseTemplate = async () => {
    setIsCreating(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsCreating(false);
    onUse(template);
    onOpenChange(false);
    router.push(`/agents/new?template=${template.id}`);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <div className="flex items-start gap-4">
            <div
              className={cn(
                "flex h-14 w-14 shrink-0 items-center justify-center rounded-xl",
                categoryColors[template.category] || "bg-gray-100 text-gray-700"
              )}
            >
              <Bot className="h-7 w-7" />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <DialogTitle className="text-xl">{template.name}</DialogTitle>
                {template.is_featured && (
                  <Badge className="bg-amber-100 text-amber-700">Popular</Badge>
                )}
              </div>
              <DialogDescription className="mt-1">
                {template.description}
              </DialogDescription>
              <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  {template.estimated_setup_time} setup
                </span>
                <span className="flex items-center gap-1">
                  <Users className="h-4 w-4" />
                  {template.popularity}+ uses
                </span>
                <Badge variant="outline" className={categoryColors[template.category]}>
                  {template.category}
                </Badge>
              </div>
            </div>
          </div>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 overflow-hidden flex flex-col">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="prompt">Prompt</TabsTrigger>
            <TabsTrigger value="tools">Tools</TabsTrigger>
            <TabsTrigger value="demo">Demo</TabsTrigger>
          </TabsList>

          <div className="flex-1 overflow-y-auto mt-4">
            <TabsContent value="overview" className="m-0 space-y-4">
              {/* Use Cases */}
              <div>
                <h4 className="font-medium mb-2">Use Cases</h4>
                <div className="grid grid-cols-2 gap-2">
                  {template.use_cases.map((useCase) => (
                    <div
                      key={useCase}
                      className="flex items-center gap-2 p-2 rounded-lg bg-muted/50"
                    >
                      <ChevronRight className="h-4 w-4 text-primary" />
                      <span className="text-sm">{useCase}</span>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              {/* Features */}
              <div>
                <h4 className="font-medium mb-2">Key Features</h4>
                <div className="flex flex-wrap gap-2">
                  {template.features.map((feature) => (
                    <Badge key={feature} variant="secondary">
                      {feature}
                    </Badge>
                  ))}
                </div>
              </div>

              <Separator />

              {/* Sample Greeting */}
              <div>
                <h4 className="font-medium mb-2">Sample Greeting</h4>
                <div className="rounded-lg bg-muted p-4">
                  <div className="flex items-start gap-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                      <Bot className="h-4 w-4 text-primary" />
                    </div>
                    <p className="text-sm leading-relaxed">{template.sample_greeting}</p>
                  </div>
                </div>
              </div>

              {/* Complexity & Industry */}
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg border p-4">
                  <p className="text-sm text-muted-foreground">Complexity</p>
                  <p className="font-medium capitalize mt-1">{template.complexity}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {template.complexity === "basic"
                      ? "Quick setup, minimal configuration"
                      : template.complexity === "intermediate"
                      ? "Some customization required"
                      : "Advanced configuration options"}
                  </p>
                </div>
                <div className="rounded-lg border p-4">
                  <p className="text-sm text-muted-foreground">Industry</p>
                  <p className="font-medium mt-1">{template.industry}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Optimized for {template.industry.toLowerCase()} use cases
                  </p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="prompt" className="m-0 space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="font-medium">System Prompt</h4>
                <Button variant="outline" size="sm" onClick={handleCopyPrompt}>
                  {copied ? (
                    <>
                      <Check className="mr-2 h-4 w-4" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="mr-2 h-4 w-4" />
                      Copy Prompt
                    </>
                  )}
                </Button>
              </div>
              <div className="rounded-lg bg-muted p-4 font-mono text-sm whitespace-pre-wrap">
                {template.sample_prompt}
              </div>
              <p className="text-sm text-muted-foreground">
                This prompt will be customized with your business details when you create the agent.
              </p>
            </TabsContent>

            <TabsContent value="tools" className="m-0 space-y-4">
              <h4 className="font-medium">Included Tools</h4>
              <div className="space-y-2">
                {template.tools.map((tool) => (
                  <div
                    key={tool}
                    className="flex items-center gap-3 p-3 rounded-lg border"
                  >
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                      <Wrench className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium capitalize">
                        {tool.replace(/_/g, " ")}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {getToolDescription(tool)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-sm text-muted-foreground">
                Tools can be added or removed after creating the agent.
              </p>
            </TabsContent>

            <TabsContent value="demo" className="m-0 space-y-4">
              <div className="rounded-lg border p-6 text-center">
                <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary/10 mx-auto mb-4">
                  <Volume2 className="h-10 w-10 text-primary" />
                </div>
                <h4 className="font-medium">Voice Demo</h4>
                <p className="text-sm text-muted-foreground mt-1 mb-4">
                  Listen to how this agent sounds in a sample conversation
                </p>
                <Button
                  size="lg"
                  onClick={handlePlayDemo}
                  disabled={isPlaying}
                >
                  {isPlaying ? (
                    <>
                      <Pause className="mr-2 h-5 w-5" />
                      Playing Demo...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-5 w-5" />
                      Play Demo Call
                    </>
                  )}
                </Button>
              </div>

              {/* Sample Conversation */}
              <div>
                <h4 className="font-medium mb-3">Sample Conversation</h4>
                <div className="space-y-3">
                  <div className="flex gap-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                      <MessageSquare className="h-4 w-4" />
                    </div>
                    <div className="rounded-lg bg-muted p-3 text-sm">
                      Hi, I'd like to {template.use_cases[0]?.toLowerCase() || "get help"}.
                    </div>
                  </div>
                  <div className="flex gap-3 justify-end">
                    <div className="rounded-lg bg-primary/10 p-3 text-sm max-w-[80%]">
                      {template.sample_greeting}
                    </div>
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                      <Bot className="h-4 w-4 text-primary" />
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>
          </div>
        </Tabs>

        {/* Footer */}
        <div className="flex items-center justify-between pt-4 border-t mt-4">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleUseTemplate} disabled={isCreating}>
            {isCreating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" />
                Use This Template
              </>
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function getToolDescription(tool: string): string {
  const descriptions: Record<string, string> = {
    knowledge_base: "Access company knowledge base and FAQs",
    create_ticket: "Create support tickets in your helpdesk",
    transfer_call: "Transfer calls to human agents",
    send_email: "Send follow-up emails to customers",
    check_availability: "Check calendar availability",
    book_appointment: "Book and confirm appointments",
    send_reminder: "Send appointment reminders",
    reschedule: "Reschedule existing appointments",
    check_inventory: "Check product availability",
    calculate_quote: "Generate price quotes",
    schedule_demo: "Schedule product demonstrations",
    update_crm: "Update customer records in CRM",
    track_order: "Track order status and shipping",
    process_return: "Process return and refund requests",
    apply_discount: "Apply discount codes",
    search_properties: "Search property listings",
    schedule_viewing: "Schedule property viewings",
    send_listing: "Send property listings via email",
    qualify_lead: "Qualify and score leads",
    verify_insurance: "Verify insurance coverage",
    check_schedule: "Check appointment schedule",
    run_diagnostic: "Run system diagnostics",
    screen_share_request: "Request screen sharing",
    start_claim: "Start new insurance claims",
    check_claim_status: "Check existing claim status",
    request_document: "Request required documents",
    verify_coverage: "Verify policy coverage",
    verify_identity: "Verify customer identity",
    check_balance: "Check account balance",
    report_card: "Report lost/stolen cards",
    search_flights: "Search available flights",
    book_hotel: "Book hotel reservations",
    create_itinerary: "Create travel itineraries",
    check_visa: "Check visa requirements",
    search_inventory: "Search vehicle inventory",
    schedule_test_drive: "Schedule test drives",
    calculate_payment: "Calculate monthly payments",
    estimate_trade: "Estimate trade-in value",
    search_programs: "Search academic programs",
    check_application: "Check application status",
    schedule_tour: "Schedule campus tours",
    calculate_aid: "Calculate financial aid",
    search_policies: "Search company policies",
    check_benefits: "Check employee benefits",
    submit_request: "Submit HR requests",
    lookup_employee: "Look up employee directory",
    check_prescription: "Check prescription status",
    request_refill: "Request prescription refills",
    schedule_pickup: "Schedule prescription pickup",
    book_table: "Book restaurant tables",
    add_waitlist: "Add to waitlist",
    send_confirmation: "Send booking confirmations",
  };

  return descriptions[tool] || "Custom tool functionality";
}
