"use client";

import React, { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Bot,
  Mic,
  Brain,
  Wrench,
  CheckCircle2,
  ChevronRight,
  ChevronLeft,
  Save,
  Play,
  AlertCircle,
  Sparkles,
  Volume2,
  Settings,
  Zap,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch, SwitchWithLabel } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { agents, voiceConfig } from "@/lib/api";
import { toast } from "sonner";

interface WizardStep {
  id: string;
  title: string;
  description: string;
  icon: React.ElementType;
}

const steps: WizardStep[] = [
  {
    id: "basics",
    title: "Basic Info",
    description: "Name and purpose",
    icon: Bot,
  },
  {
    id: "personality",
    title: "Personality",
    description: "Voice and behavior",
    icon: Mic,
  },
  {
    id: "llm",
    title: "AI Model",
    description: "Intelligence settings",
    icon: Brain,
  },
  {
    id: "tools",
    title: "Tools",
    description: "Capabilities",
    icon: Wrench,
  },
  {
    id: "review",
    title: "Review",
    description: "Finalize agent",
    icon: CheckCircle2,
  },
];

interface AgentFormData {
  name: string;
  description: string;
  system_prompt: string;
  greeting_message: string;
  voice_provider: string;
  voice_id: string;
  voice_name: string;
  language: string;
  speaking_rate: number;
  pitch: number;
  llm_provider: string;
  llm_model: string;
  temperature: number;
  max_tokens: number;
  tools: string[];
  is_active: boolean;
  allow_interruptions: boolean;
  end_call_on_silence: boolean;
  silence_timeout: number;
}

const defaultFormData: AgentFormData = {
  name: "",
  description: "",
  system_prompt: "",
  greeting_message: "Hello! How can I help you today?",
  voice_provider: "elevenlabs",
  voice_id: "",
  voice_name: "",
  language: "en-US",
  speaking_rate: 1.0,
  pitch: 1.0,
  llm_provider: "openai",
  llm_model: "gpt-4o",
  temperature: 0.7,
  max_tokens: 1024,
  tools: [],
  is_active: false,
  allow_interruptions: true,
  end_call_on_silence: true,
  silence_timeout: 5,
};

const voiceProviders = [
  { id: "elevenlabs", name: "ElevenLabs", description: "High-quality neural voices" },
  { id: "openai", name: "OpenAI TTS", description: "Fast, natural sounding" },
  { id: "deepgram", name: "Deepgram Aura", description: "Ultra-low latency" },
  { id: "azure", name: "Azure Speech", description: "Enterprise grade" },
];

const llmModels = [
  { id: "gpt-4o", name: "GPT-4o", description: "Most capable, recommended" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", description: "Fast and cost-effective" },
  { id: "gpt-4-turbo", name: "GPT-4 Turbo", description: "Previous generation" },
  { id: "claude-3-5-sonnet", name: "Claude 3.5 Sonnet", description: "Great reasoning" },
];

const availableTools = [
  { id: "calendar", name: "Calendar", description: "Schedule appointments" },
  { id: "transfer", name: "Call Transfer", description: "Transfer to human" },
  { id: "sms", name: "Send SMS", description: "Send text messages" },
  { id: "email", name: "Send Email", description: "Send emails" },
  { id: "lookup", name: "Database Lookup", description: "Query customer data" },
  { id: "webhook", name: "Webhook", description: "Call external APIs" },
];

const promptTemplates = [
  {
    id: "customer-service",
    name: "Customer Service",
    prompt: `You are a friendly and professional customer service agent. Your goal is to help customers with their inquiries in a warm and efficient manner.

Key behaviors:
- Greet customers warmly and ask how you can help
- Listen carefully to understand their needs
- Provide clear, accurate information
- Offer to transfer to a human agent when needed
- Always thank the customer for their time`,
  },
  {
    id: "appointment-booking",
    name: "Appointment Booking",
    prompt: `You are a scheduling assistant. Your role is to help customers book, modify, or cancel appointments.

Key behaviors:
- Ask for the type of appointment they need
- Check availability in the calendar
- Confirm all details (date, time, purpose)
- Send confirmation via SMS or email
- Handle rescheduling requests professionally`,
  },
  {
    id: "lead-qualification",
    name: "Lead Qualification",
    prompt: `You are a sales qualification specialist. Your goal is to understand potential customers' needs and determine if they're a good fit for our services.

Key behaviors:
- Introduce yourself and the company briefly
- Ask qualifying questions about their needs
- Gauge their budget and timeline
- Take notes on key information
- Schedule follow-up calls with sales team when appropriate`,
  },
];

interface AgentWizardProps {
  mode?: "create" | "edit";
  initialData?: Partial<AgentFormData>;
  agentId?: string;
}

export function AgentWizard({ mode = "create", initialData, agentId }: AgentWizardProps) {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState<AgentFormData>({
    ...defaultFormData,
    ...initialData,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Fetch available voices
  const { data: voicesData } = useQuery({
    queryKey: ["voices", formData.voice_provider],
    queryFn: () => voiceConfig.listVoices(formData.voice_provider),
    enabled: !!formData.voice_provider,
  });

  const voices = voicesData || [];

  // Create/update mutation
  const saveMutation = useMutation({
    mutationFn: async (data: AgentFormData) => {
      const payload = {
        name: data.name,
        description: data.description,
        system_prompt: data.system_prompt,
        greeting_message: data.greeting_message,
        is_active: data.is_active,
        llm_config: {
          provider: data.llm_provider,
          model: data.llm_model,
          temperature: data.temperature,
          max_tokens: data.max_tokens,
        },
        voice_config: {
          provider: data.voice_provider,
          voice_id: data.voice_id,
          voice_name: data.voice_name,
          language: data.language,
          speaking_rate: data.speaking_rate,
          pitch: data.pitch,
        },
        tools: data.tools.map((id) => ({ type: id, enabled: true })),
        settings: {
          allow_interruptions: data.allow_interruptions,
          end_call_on_silence: data.end_call_on_silence,
          silence_timeout: data.silence_timeout,
        },
      };

      if (mode === "edit" && agentId) {
        return agents.update(agentId, payload);
      }
      return agents.create(payload);
    },
    onSuccess: (data) => {
      toast.success(mode === "create" ? "Agent created successfully!" : "Agent updated successfully!");
      router.push(`/agents/${data.id || agentId}`);
    },
    onError: (error: Error) => {
      toast.error(`Failed to save agent: ${error.message}`);
    },
  });

  const updateField = useCallback(<K extends keyof AgentFormData>(
    field: K,
    value: AgentFormData[K]
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    setErrors((prev) => ({ ...prev, [field]: "" }));
  }, []);

  const validateStep = (step: number): boolean => {
    const newErrors: Record<string, string> = {};

    switch (step) {
      case 0: // Basics
        if (!formData.name.trim()) {
          newErrors.name = "Agent name is required";
        }
        if (!formData.system_prompt.trim()) {
          newErrors.system_prompt = "System prompt is required";
        }
        break;
      case 1: // Voice
        if (!formData.voice_provider) {
          newErrors.voice_provider = "Voice provider is required";
        }
        break;
      case 2: // LLM
        if (!formData.llm_model) {
          newErrors.llm_model = "Model selection is required";
        }
        break;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep((prev) => Math.min(prev + 1, steps.length - 1));
    }
  };

  const handlePrevious = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 0));
  };

  const handleSave = () => {
    if (validateStep(currentStep)) {
      saveMutation.mutate(formData);
    }
  };

  const applyTemplate = (template: typeof promptTemplates[0]) => {
    updateField("system_prompt", template.prompt);
  };

  const toggleTool = (toolId: string) => {
    setFormData((prev) => ({
      ...prev,
      tools: prev.tools.includes(toolId)
        ? prev.tools.filter((t) => t !== toolId)
        : [...prev.tools, toolId],
    }));
  };

  const progress = ((currentStep + 1) / steps.length) * 100;

  return (
    <div className="max-w-4xl mx-auto">
      {/* Progress Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold">
            {mode === "create" ? "Create New Agent" : "Edit Agent"}
          </h1>
          <Badge variant="secondary">
            Step {currentStep + 1} of {steps.length}
          </Badge>
        </div>
        <Progress value={progress} className="h-2" />
        <div className="flex justify-between mt-4">
          {steps.map((step, index) => (
            <button
              key={step.id}
              onClick={() => index <= currentStep && setCurrentStep(index)}
              disabled={index > currentStep}
              className={cn(
                "flex flex-col items-center gap-1 text-xs transition-colors",
                index <= currentStep
                  ? "text-foreground cursor-pointer"
                  : "text-muted-foreground cursor-not-allowed"
              )}
            >
              <div
                className={cn(
                  "flex h-10 w-10 items-center justify-center rounded-full transition-colors",
                  index < currentStep
                    ? "bg-primary text-primary-foreground"
                    : index === currentStep
                    ? "bg-primary/20 text-primary border-2 border-primary"
                    : "bg-muted text-muted-foreground"
                )}
              >
                {index < currentStep ? (
                  <CheckCircle2 className="h-5 w-5" />
                ) : (
                  <step.icon className="h-5 w-5" />
                )}
              </div>
              <span className="hidden sm:block">{step.title}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Step Content */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {React.createElement(steps[currentStep].icon, { className: "h-5 w-5" })}
            {steps[currentStep].title}
          </CardTitle>
          <CardDescription>{steps[currentStep].description}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Step 1: Basic Info */}
          {currentStep === 0 && (
            <>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name" required>Agent Name</Label>
                  <Input
                    id="name"
                    placeholder="e.g., Customer Support Agent"
                    value={formData.name}
                    onChange={(e) => updateField("name", e.target.value)}
                    error={errors.name}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description" optional>Description</Label>
                  <Textarea
                    id="description"
                    placeholder="Brief description of what this agent does..."
                    value={formData.description}
                    onChange={(e) => updateField("description", e.target.value)}
                    rows={2}
                  />
                </div>

                <Separator />

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="system_prompt" required>System Prompt</Label>
                    <div className="flex gap-2">
                      {promptTemplates.map((template) => (
                        <Button
                          key={template.id}
                          variant="outline"
                          size="sm"
                          onClick={() => applyTemplate(template)}
                        >
                          <Sparkles className="mr-1 h-3 w-3" />
                          {template.name}
                        </Button>
                      ))}
                    </div>
                  </div>
                  <Textarea
                    id="system_prompt"
                    placeholder="Define your agent's personality, role, and instructions..."
                    value={formData.system_prompt}
                    onChange={(e) => updateField("system_prompt", e.target.value)}
                    rows={10}
                    error={errors.system_prompt}
                    showCount
                    maxLength={4000}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="greeting">Greeting Message</Label>
                  <Textarea
                    id="greeting"
                    placeholder="First message the agent says when answering..."
                    value={formData.greeting_message}
                    onChange={(e) => updateField("greeting_message", e.target.value)}
                    rows={2}
                  />
                </div>
              </div>
            </>
          )}

          {/* Step 2: Voice & Personality */}
          {currentStep === 1 && (
            <>
              <div className="space-y-6">
                <div className="space-y-2">
                  <Label required>Voice Provider</Label>
                  <div className="grid grid-cols-2 gap-3">
                    {voiceProviders.map((provider) => (
                      <button
                        key={provider.id}
                        type="button"
                        onClick={() => updateField("voice_provider", provider.id)}
                        className={cn(
                          "flex flex-col items-start p-4 rounded-lg border transition-all text-left",
                          formData.voice_provider === provider.id
                            ? "border-primary bg-primary/5"
                            : "hover:border-primary/50"
                        )}
                      >
                        <span className="font-medium">{provider.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {provider.description}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {voices.length > 0 && (
                  <div className="space-y-2">
                    <Label required>Voice</Label>
                    <Select
                      value={formData.voice_id}
                      onValueChange={(value) => {
                        const voice = voices.find((v: any) => v.id === value);
                        updateField("voice_id", value);
                        updateField("voice_name", voice?.name || "");
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a voice" />
                      </SelectTrigger>
                      <SelectContent>
                        {voices.map((voice: any) => (
                          <SelectItem key={voice.id} value={voice.id}>
                            <div className="flex items-center gap-2">
                              <span>{voice.name}</span>
                              {voice.gender && (
                                <Badge variant="secondary" className="text-xs">
                                  {voice.gender}
                                </Badge>
                              )}
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div className="space-y-2">
                  <Label>Language</Label>
                  <Select
                    value={formData.language}
                    onValueChange={(value) => updateField("language", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en-US">English (US)</SelectItem>
                      <SelectItem value="en-GB">English (UK)</SelectItem>
                      <SelectItem value="es-ES">Spanish</SelectItem>
                      <SelectItem value="fr-FR">French</SelectItem>
                      <SelectItem value="de-DE">German</SelectItem>
                      <SelectItem value="it-IT">Italian</SelectItem>
                      <SelectItem value="pt-BR">Portuguese (Brazil)</SelectItem>
                      <SelectItem value="ja-JP">Japanese</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label>Speaking Rate</Label>
                    <Slider
                      value={[formData.speaking_rate]}
                      onValueChange={([value]) => updateField("speaking_rate", value)}
                      min={0.5}
                      max={2.0}
                      step={0.1}
                      showValue
                      valueFormat={(v) => `${v.toFixed(1)}x`}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Pitch</Label>
                    <Slider
                      value={[formData.pitch]}
                      onValueChange={([value]) => updateField("pitch", value)}
                      min={0.5}
                      max={2.0}
                      step={0.1}
                      showValue
                      valueFormat={(v) => `${v.toFixed(1)}x`}
                    />
                  </div>
                </div>

                {formData.voice_id && (
                  <div className="flex items-center justify-center p-4 bg-muted rounded-lg">
                    <Button variant="outline">
                      <Volume2 className="mr-2 h-4 w-4" />
                      Preview Voice
                    </Button>
                  </div>
                )}
              </div>
            </>
          )}

          {/* Step 3: AI Model */}
          {currentStep === 2 && (
            <>
              <div className="space-y-6">
                <div className="space-y-2">
                  <Label required>AI Model</Label>
                  <div className="grid grid-cols-2 gap-3">
                    {llmModels.map((model) => (
                      <button
                        key={model.id}
                        type="button"
                        onClick={() => updateField("llm_model", model.id)}
                        className={cn(
                          "flex flex-col items-start p-4 rounded-lg border transition-all text-left",
                          formData.llm_model === model.id
                            ? "border-primary bg-primary/5"
                            : "hover:border-primary/50"
                        )}
                      >
                        <span className="font-medium">{model.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {model.description}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Temperature</Label>
                    <span className="text-sm text-muted-foreground">
                      {formData.temperature.toFixed(1)} - {formData.temperature < 0.3 ? "Focused" : formData.temperature > 0.7 ? "Creative" : "Balanced"}
                    </span>
                  </div>
                  <Slider
                    value={[formData.temperature]}
                    onValueChange={([value]) => updateField("temperature", value)}
                    min={0}
                    max={1}
                    step={0.1}
                  />
                  <p className="text-xs text-muted-foreground">
                    Lower values make responses more focused and deterministic. Higher values increase creativity and variability.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label>Max Response Length</Label>
                  <Select
                    value={String(formData.max_tokens)}
                    onValueChange={(value) => updateField("max_tokens", Number(value))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="512">Short (512 tokens)</SelectItem>
                      <SelectItem value="1024">Medium (1024 tokens)</SelectItem>
                      <SelectItem value="2048">Long (2048 tokens)</SelectItem>
                      <SelectItem value="4096">Very Long (4096 tokens)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Separator />

                <div className="space-y-4">
                  <Label>Conversation Settings</Label>
                  <SwitchWithLabel
                    label="Allow Interruptions"
                    description="Let callers interrupt while the agent is speaking"
                    checked={formData.allow_interruptions}
                    onCheckedChange={(checked) => updateField("allow_interruptions", checked)}
                  />
                  <SwitchWithLabel
                    label="End Call on Silence"
                    description="Automatically end the call after prolonged silence"
                    checked={formData.end_call_on_silence}
                    onCheckedChange={(checked) => updateField("end_call_on_silence", checked)}
                  />
                  {formData.end_call_on_silence && (
                    <div className="pl-4 border-l-2 border-muted space-y-2">
                      <Label>Silence Timeout (seconds)</Label>
                      <Slider
                        value={[formData.silence_timeout]}
                        onValueChange={([value]) => updateField("silence_timeout", value)}
                        min={3}
                        max={30}
                        step={1}
                        showValue
                        valueFormat={(v) => `${v}s`}
                      />
                    </div>
                  )}
                </div>
              </div>
            </>
          )}

          {/* Step 4: Tools */}
          {currentStep === 3 && (
            <>
              <div className="space-y-4">
                <Alert>
                  <Zap className="h-4 w-4" />
                  <AlertTitle>Extend Your Agent's Capabilities</AlertTitle>
                  <AlertDescription>
                    Enable tools to let your agent perform actions like scheduling appointments, sending messages, or looking up data.
                  </AlertDescription>
                </Alert>

                <div className="grid grid-cols-2 gap-3">
                  {availableTools.map((tool) => (
                    <button
                      key={tool.id}
                      type="button"
                      onClick={() => toggleTool(tool.id)}
                      className={cn(
                        "flex items-start gap-3 p-4 rounded-lg border transition-all text-left",
                        formData.tools.includes(tool.id)
                          ? "border-primary bg-primary/5"
                          : "hover:border-primary/50"
                      )}
                    >
                      <div
                        className={cn(
                          "flex h-8 w-8 items-center justify-center rounded-full",
                          formData.tools.includes(tool.id)
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted"
                        )}
                      >
                        <Wrench className="h-4 w-4" />
                      </div>
                      <div>
                        <span className="font-medium">{tool.name}</span>
                        <p className="text-xs text-muted-foreground">
                          {tool.description}
                        </p>
                      </div>
                    </button>
                  ))}
                </div>

                {formData.tools.length === 0 && (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    No tools selected. Your agent will respond conversationally without taking actions.
                  </p>
                )}
              </div>
            </>
          )}

          {/* Step 5: Review */}
          {currentStep === 4 && (
            <>
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <Label className="text-muted-foreground">Agent Name</Label>
                      <p className="font-medium">{formData.name || "Not set"}</p>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Description</Label>
                      <p className="text-sm">{formData.description || "No description"}</p>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Voice</Label>
                      <p className="font-medium">
                        {formData.voice_name || "Default"} ({formData.voice_provider})
                      </p>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <Label className="text-muted-foreground">AI Model</Label>
                      <p className="font-medium">{formData.llm_model}</p>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Language</Label>
                      <p className="font-medium">{formData.language}</p>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Tools</Label>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {formData.tools.length > 0 ? (
                          formData.tools.map((toolId) => {
                            const tool = availableTools.find((t) => t.id === toolId);
                            return (
                              <Badge key={toolId} variant="secondary">
                                {tool?.name}
                              </Badge>
                            );
                          })
                        ) : (
                          <span className="text-sm text-muted-foreground">None</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                <Separator />

                <div>
                  <Label className="text-muted-foreground">System Prompt</Label>
                  <div className="mt-2 p-4 rounded-lg bg-muted/50 max-h-48 overflow-y-auto">
                    <p className="text-sm whitespace-pre-wrap">{formData.system_prompt}</p>
                  </div>
                </div>

                <div>
                  <Label className="text-muted-foreground">Greeting Message</Label>
                  <p className="mt-1 text-sm">{formData.greeting_message}</p>
                </div>

                <SwitchWithLabel
                  label="Activate Agent Immediately"
                  description="Make this agent available to receive calls right away"
                  checked={formData.is_active}
                  onCheckedChange={(checked) => updateField("is_active", checked)}
                />
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Navigation Buttons */}
      <div className="flex items-center justify-between mt-6">
        <Button
          variant="outline"
          onClick={handlePrevious}
          disabled={currentStep === 0}
        >
          <ChevronLeft className="mr-2 h-4 w-4" />
          Previous
        </Button>

        <div className="flex gap-2">
          {currentStep < steps.length - 1 ? (
            <Button onClick={handleNext}>
              Next
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
          ) : (
            <Button
              onClick={handleSave}
              disabled={saveMutation.isPending}
            >
              {saveMutation.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Save className="mr-2 h-4 w-4" />
              )}
              {mode === "create" ? "Create Agent" : "Save Changes"}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
