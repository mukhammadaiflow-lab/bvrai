"use client";

import React, { useState, useRef } from "react";
import {
  Volume2,
  Play,
  Pause,
  Square,
  Mic,
  Upload,
  Check,
  Loader2,
  Settings,
  Zap,
  Globe,
  Star,
  Clock,
  RefreshCw,
  Headphones,
  Activity,
  ChevronRight,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface VoiceProvider {
  id: string;
  name: string;
  description: string;
  logo: string;
  features: string[];
  pricing: string;
  latency: string;
  quality: number;
  supported_languages: number;
  is_connected: boolean;
}

interface Voice {
  id: string;
  name: string;
  provider: string;
  gender: "male" | "female" | "neutral";
  language: string;
  accent: string;
  style: string;
  preview_url: string;
  is_premium: boolean;
  is_favorite: boolean;
}

// Demo data
const voiceProviders: VoiceProvider[] = [
  {
    id: "elevenlabs",
    name: "ElevenLabs",
    description: "Ultra-realistic AI voices with emotional range",
    logo: "🎙️",
    features: ["Voice Cloning", "Multilingual", "Emotion Control", "Low Latency"],
    pricing: "From $5/month",
    latency: "~150ms",
    quality: 98,
    supported_languages: 29,
    is_connected: true,
  },
  {
    id: "openai",
    name: "OpenAI TTS",
    description: "High-quality text-to-speech from OpenAI",
    logo: "🤖",
    features: ["HD Quality", "6 Voices", "Fast Generation", "Simple API"],
    pricing: "Pay per use",
    latency: "~200ms",
    quality: 92,
    supported_languages: 15,
    is_connected: true,
  },
  {
    id: "deepgram",
    name: "Deepgram Aura",
    description: "Real-time conversational AI voices",
    logo: "🌊",
    features: ["Ultra-Low Latency", "Streaming", "Custom Voices", "Enterprise Ready"],
    pricing: "From $4/month",
    latency: "~80ms",
    quality: 94,
    supported_languages: 12,
    is_connected: false,
  },
  {
    id: "azure",
    name: "Azure Neural TTS",
    description: "Microsoft's neural text-to-speech service",
    logo: "☁️",
    features: ["500+ Voices", "SSML Support", "Custom Neural", "Enterprise"],
    pricing: "Pay per character",
    latency: "~180ms",
    quality: 95,
    supported_languages: 140,
    is_connected: false,
  },
  {
    id: "playht",
    name: "Play.ht",
    description: "AI voice generation platform",
    logo: "▶️",
    features: ["Voice Cloning", "API Access", "Audio Downloads", "Podcast Ready"],
    pricing: "From $29/month",
    latency: "~250ms",
    quality: 90,
    supported_languages: 20,
    is_connected: false,
  },
];

const demoVoices: Voice[] = [
  {
    id: "rachel",
    name: "Rachel",
    provider: "elevenlabs",
    gender: "female",
    language: "English",
    accent: "American",
    style: "Conversational",
    preview_url: "/voices/rachel-preview.mp3",
    is_premium: false,
    is_favorite: true,
  },
  {
    id: "adam",
    name: "Adam",
    provider: "elevenlabs",
    gender: "male",
    language: "English",
    accent: "American",
    style: "Professional",
    preview_url: "/voices/adam-preview.mp3",
    is_premium: false,
    is_favorite: false,
  },
  {
    id: "bella",
    name: "Bella",
    provider: "elevenlabs",
    gender: "female",
    language: "English",
    accent: "British",
    style: "Warm",
    preview_url: "/voices/bella-preview.mp3",
    is_premium: true,
    is_favorite: true,
  },
  {
    id: "alloy",
    name: "Alloy",
    provider: "openai",
    gender: "neutral",
    language: "English",
    accent: "Neutral",
    style: "Clear",
    preview_url: "/voices/alloy-preview.mp3",
    is_premium: false,
    is_favorite: false,
  },
  {
    id: "echo",
    name: "Echo",
    provider: "openai",
    gender: "male",
    language: "English",
    accent: "American",
    style: "Deep",
    preview_url: "/voices/echo-preview.mp3",
    is_premium: false,
    is_favorite: false,
  },
  {
    id: "nova",
    name: "Nova",
    provider: "openai",
    gender: "female",
    language: "English",
    accent: "American",
    style: "Friendly",
    preview_url: "/voices/nova-preview.mp3",
    is_premium: false,
    is_favorite: true,
  },
];

export default function VoiceConfigPage() {
  const [activeTab, setActiveTab] = useState("providers");
  const [selectedProvider, setSelectedProvider] = useState("elevenlabs");
  const [selectedVoice, setSelectedVoice] = useState<Voice | null>(null);
  const [voices, setVoices] = useState<Voice[]>(demoVoices);
  const [playingVoice, setPlayingVoice] = useState<string | null>(null);
  const [testText, setTestText] = useState("Hello! Welcome to Builder Voice AI. How can I assist you today?");
  const [isGenerating, setIsGenerating] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Voice settings
  const [voiceSettings, setVoiceSettings] = useState({
    speed: 1.0,
    pitch: 1.0,
    volume: 1.0,
    stability: 0.5,
    clarity: 0.75,
  });

  // Filter states
  const [genderFilter, setGenderFilter] = useState<string>("all");
  const [languageFilter, setLanguageFilter] = useState<string>("all");

  const handlePlayVoice = (voiceId: string) => {
    if (playingVoice === voiceId) {
      setPlayingVoice(null);
      audioRef.current?.pause();
    } else {
      setPlayingVoice(voiceId);
      // Simulate playing audio
      setTimeout(() => setPlayingVoice(null), 3000);
    }
  };

  const handleTestVoice = async () => {
    if (!selectedVoice) {
      toast.error("Please select a voice first");
      return;
    }
    setIsGenerating(true);
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setIsGenerating(false);
    toast.success("Audio generated successfully");
  };

  const handleConnectProvider = async (providerId: string) => {
    toast.info(`${providerId} integration setup would open here`);
  };

  const toggleFavorite = (voiceId: string) => {
    setVoices((prev) =>
      prev.map((v) =>
        v.id === voiceId ? { ...v, is_favorite: !v.is_favorite } : v
      )
    );
  };

  const filteredVoices = voices.filter((voice) => {
    const providerMatch = voice.provider === selectedProvider;
    const genderMatch = genderFilter === "all" || voice.gender === genderFilter;
    const languageMatch = languageFilter === "all" || voice.language === languageFilter;
    return providerMatch && genderMatch && languageMatch;
  });

  const connectedProvider = voiceProviders.find((p) => p.id === selectedProvider);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Voice Configuration</h1>
          <p className="text-muted-foreground">
            Configure voice providers and customize voice settings
          </p>
        </div>
        <Button>
          <Mic className="mr-2 h-4 w-4" />
          Clone Voice
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="providers">Providers</TabsTrigger>
          <TabsTrigger value="voices">Voice Library</TabsTrigger>
          <TabsTrigger value="testing">Voice Testing</TabsTrigger>
          <TabsTrigger value="custom">Custom Voices</TabsTrigger>
        </TabsList>

        {/* Providers Tab */}
        <TabsContent value="providers" className="space-y-6 mt-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {voiceProviders.map((provider) => (
              <Card
                key={provider.id}
                className={cn(
                  "cursor-pointer transition-all hover:shadow-md",
                  selectedProvider === provider.id && "ring-2 ring-primary"
                )}
                onClick={() => setSelectedProvider(provider.id)}
              >
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-3xl">{provider.logo}</span>
                      <div>
                        <CardTitle className="text-lg">{provider.name}</CardTitle>
                        <CardDescription className="text-xs">
                          {provider.pricing}
                        </CardDescription>
                      </div>
                    </div>
                    {provider.is_connected ? (
                      <Badge variant="success" className="bg-green-100 text-green-800">
                        <Check className="mr-1 h-3 w-3" />
                        Connected
                      </Badge>
                    ) : (
                      <Badge variant="secondary">Not Connected</Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    {provider.description}
                  </p>

                  <div className="flex flex-wrap gap-1">
                    {provider.features.slice(0, 3).map((feature) => (
                      <Badge key={feature} variant="outline" className="text-xs">
                        {feature}
                      </Badge>
                    ))}
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="p-2 rounded bg-muted">
                      <p className="text-muted-foreground">Quality</p>
                      <p className="font-bold">{provider.quality}%</p>
                    </div>
                    <div className="p-2 rounded bg-muted">
                      <p className="text-muted-foreground">Latency</p>
                      <p className="font-bold">{provider.latency}</p>
                    </div>
                    <div className="p-2 rounded bg-muted">
                      <p className="text-muted-foreground">Languages</p>
                      <p className="font-bold">{provider.supported_languages}</p>
                    </div>
                  </div>

                  {!provider.is_connected && (
                    <Button
                      className="w-full"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleConnectProvider(provider.id);
                      }}
                    >
                      Connect {provider.name}
                    </Button>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>

          {connectedProvider?.is_connected && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  {connectedProvider.name} Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">API Key</label>
                    <Input
                      type="password"
                      value="sk-xxxxxxxxxxxxxxxxxxxx"
                      placeholder="Enter your API key"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Default Model</label>
                    <Select defaultValue="eleven_multilingual_v2">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="eleven_multilingual_v2">Multilingual v2</SelectItem>
                        <SelectItem value="eleven_turbo_v2">Turbo v2</SelectItem>
                        <SelectItem value="eleven_monolingual_v1">Monolingual v1</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="flex justify-end">
                  <Button>Save Settings</Button>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Voices Tab */}
        <TabsContent value="voices" className="space-y-6 mt-6">
          {/* Filters */}
          <div className="flex flex-wrap gap-4">
            <Select value={selectedProvider} onValueChange={setSelectedProvider}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Provider" />
              </SelectTrigger>
              <SelectContent>
                {voiceProviders.filter((p) => p.is_connected).map((provider) => (
                  <SelectItem key={provider.id} value={provider.id}>
                    {provider.logo} {provider.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={genderFilter} onValueChange={setGenderFilter}>
              <SelectTrigger className="w-36">
                <SelectValue placeholder="Gender" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Genders</SelectItem>
                <SelectItem value="male">Male</SelectItem>
                <SelectItem value="female">Female</SelectItem>
                <SelectItem value="neutral">Neutral</SelectItem>
              </SelectContent>
            </Select>

            <Select value={languageFilter} onValueChange={setLanguageFilter}>
              <SelectTrigger className="w-36">
                <SelectValue placeholder="Language" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Languages</SelectItem>
                <SelectItem value="English">English</SelectItem>
                <SelectItem value="Spanish">Spanish</SelectItem>
                <SelectItem value="French">French</SelectItem>
              </SelectContent>
            </Select>

            <Button variant="outline" size="icon">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>

          {/* Voice Grid */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filteredVoices.map((voice) => (
              <Card
                key={voice.id}
                className={cn(
                  "cursor-pointer transition-all hover:shadow-md",
                  selectedVoice?.id === voice.id && "ring-2 ring-primary"
                )}
                onClick={() => setSelectedVoice(voice)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className={cn(
                          "flex h-12 w-12 items-center justify-center rounded-full",
                          voice.gender === "female"
                            ? "bg-pink-100 text-pink-600"
                            : voice.gender === "male"
                            ? "bg-blue-100 text-blue-600"
                            : "bg-purple-100 text-purple-600"
                        )}
                      >
                        <Volume2 className="h-6 w-6" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-medium">{voice.name}</h3>
                          {voice.is_premium && (
                            <Badge variant="secondary" className="text-xs">
                              <Zap className="mr-1 h-3 w-3" />
                              Premium
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {voice.accent} · {voice.style}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleFavorite(voice.id);
                      }}
                    >
                      <Star
                        className={cn(
                          "h-5 w-5",
                          voice.is_favorite
                            ? "fill-yellow-400 text-yellow-400"
                            : "text-muted-foreground"
                        )}
                      />
                    </button>
                  </div>

                  <div className="flex items-center gap-2 mt-4">
                    <Badge variant="outline" className="text-xs">
                      <Globe className="mr-1 h-3 w-3" />
                      {voice.language}
                    </Badge>
                    <Badge variant="outline" className="text-xs capitalize">
                      {voice.gender}
                    </Badge>
                  </div>

                  <div className="flex items-center justify-between mt-4 pt-4 border-t">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePlayVoice(voice.id);
                      }}
                    >
                      {playingVoice === voice.id ? (
                        <>
                          <Pause className="mr-2 h-4 w-4" />
                          Playing...
                        </>
                      ) : (
                        <>
                          <Play className="mr-2 h-4 w-4" />
                          Preview
                        </>
                      )}
                    </Button>
                    <Button
                      size="sm"
                      variant={selectedVoice?.id === voice.id ? "default" : "outline"}
                    >
                      {selectedVoice?.id === voice.id ? (
                        <>
                          <Check className="mr-2 h-4 w-4" />
                          Selected
                        </>
                      ) : (
                        "Select"
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Testing Tab */}
        <TabsContent value="testing" className="space-y-6 mt-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Voice Selection & Settings */}
            <Card>
              <CardHeader>
                <CardTitle>Voice Settings</CardTitle>
                <CardDescription>
                  Adjust voice parameters for testing
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Selected Voice</label>
                  <Select
                    value={selectedVoice?.id || ""}
                    onValueChange={(id) => {
                      const voice = voices.find((v) => v.id === id);
                      setSelectedVoice(voice || null);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Choose a voice" />
                    </SelectTrigger>
                    <SelectContent>
                      {voices.map((voice) => (
                        <SelectItem key={voice.id} value={voice.id}>
                          {voice.name} ({voice.provider})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <Separator />

                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <label className="text-sm font-medium">Speed</label>
                      <span className="text-sm text-muted-foreground">
                        {voiceSettings.speed.toFixed(1)}x
                      </span>
                    </div>
                    <Slider
                      value={[voiceSettings.speed]}
                      min={0.5}
                      max={2}
                      step={0.1}
                      onValueChange={([value]) =>
                        setVoiceSettings((prev) => ({ ...prev, speed: value }))
                      }
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <label className="text-sm font-medium">Pitch</label>
                      <span className="text-sm text-muted-foreground">
                        {voiceSettings.pitch.toFixed(1)}
                      </span>
                    </div>
                    <Slider
                      value={[voiceSettings.pitch]}
                      min={0.5}
                      max={1.5}
                      step={0.1}
                      onValueChange={([value]) =>
                        setVoiceSettings((prev) => ({ ...prev, pitch: value }))
                      }
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <label className="text-sm font-medium">Stability</label>
                      <span className="text-sm text-muted-foreground">
                        {(voiceSettings.stability * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Slider
                      value={[voiceSettings.stability]}
                      min={0}
                      max={1}
                      step={0.05}
                      onValueChange={([value]) =>
                        setVoiceSettings((prev) => ({ ...prev, stability: value }))
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      Higher stability makes the voice more consistent
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <label className="text-sm font-medium">Clarity + Similarity</label>
                      <span className="text-sm text-muted-foreground">
                        {(voiceSettings.clarity * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Slider
                      value={[voiceSettings.clarity]}
                      min={0}
                      max={1}
                      step={0.05}
                      onValueChange={([value]) =>
                        setVoiceSettings((prev) => ({ ...prev, clarity: value }))
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      Higher values enhance clarity and voice similarity
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Test Area */}
            <Card>
              <CardHeader>
                <CardTitle>Test Your Voice</CardTitle>
                <CardDescription>
                  Enter text and generate audio to test
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Test Text</label>
                  <textarea
                    className="w-full min-h-[150px] rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    placeholder="Enter text to synthesize..."
                    value={testText}
                    onChange={(e) => setTestText(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    {testText.length} characters
                  </p>
                </div>

                <div className="flex gap-2">
                  <Button
                    className="flex-1"
                    onClick={handleTestVoice}
                    disabled={isGenerating || !selectedVoice}
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Volume2 className="mr-2 h-4 w-4" />
                        Generate Audio
                      </>
                    )}
                  </Button>
                  <Button variant="outline" size="icon">
                    <Headphones className="h-4 w-4" />
                  </Button>
                </div>

                {/* Audio Player Placeholder */}
                <div className="rounded-lg border bg-muted/50 p-6">
                  <div className="flex items-center justify-center gap-4">
                    <Button variant="outline" size="icon" disabled>
                      <Play className="h-4 w-4" />
                    </Button>
                    <div className="flex-1 h-12 bg-muted rounded flex items-center justify-center">
                      <p className="text-sm text-muted-foreground">
                        No audio generated yet
                      </p>
                    </div>
                  </div>
                </div>

                {/* Quick Templates */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Quick Templates</label>
                  <div className="flex flex-wrap gap-2">
                    {[
                      "Hello! How can I help you today?",
                      "Thank you for calling. Please hold.",
                      "I understand. Let me check that for you.",
                      "Is there anything else I can help with?",
                    ].map((template, index) => (
                      <Button
                        key={index}
                        variant="outline"
                        size="sm"
                        className="text-xs"
                        onClick={() => setTestText(template)}
                      >
                        {template.slice(0, 25)}...
                      </Button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Custom Voices Tab */}
        <TabsContent value="custom" className="space-y-6 mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Mic className="h-5 w-5" />
                Voice Cloning
              </CardTitle>
              <CardDescription>
                Create custom voices by uploading audio samples
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div
                className="border-2 border-dashed rounded-lg p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                onClick={() => toast.info("File upload would open")}
              >
                <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <p className="font-medium">Upload Audio Samples</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Drag and drop or click to upload
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  MP3, WAV, or M4A files up to 10MB each
                </p>
              </div>

              <div className="rounded-lg bg-muted/50 p-4">
                <h4 className="font-medium mb-2">Voice Cloning Guidelines</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <Check className="h-4 w-4 text-green-500 mt-0.5" />
                    Use high-quality audio recordings (at least 44.1kHz)
                  </li>
                  <li className="flex items-start gap-2">
                    <Check className="h-4 w-4 text-green-500 mt-0.5" />
                    Upload 1-3 minutes of clean speech
                  </li>
                  <li className="flex items-start gap-2">
                    <Check className="h-4 w-4 text-green-500 mt-0.5" />
                    Avoid background noise or music
                  </li>
                  <li className="flex items-start gap-2">
                    <Check className="h-4 w-4 text-green-500 mt-0.5" />
                    Ensure you have rights to clone the voice
                  </li>
                </ul>
              </div>

              <Separator />

              <div>
                <h4 className="font-medium mb-4">Your Custom Voices</h4>
                <div className="text-center py-8 text-muted-foreground">
                  <Mic className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No custom voices yet</p>
                  <p className="text-sm">Upload audio samples to create your first custom voice</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
