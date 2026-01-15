"use client";

import React, { useState, useMemo } from "react";
import {
  Mic,
  Volume2,
  Play,
  Pause,
  Plus,
  Search,
  Settings2,
  Sliders,
  Check,
  ChevronDown,
  Star,
  Globe,
  User,
  Zap,
  Clock,
  DollarSign,
  Languages,
  Filter,
  Grid,
  List,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// Mock STT Providers
const sttProviders = [
  { id: "deepgram", name: "Deepgram", icon: "D", status: "connected", latency: 120, languages: 40 },
  { id: "openai", name: "OpenAI Whisper", icon: "O", status: "connected", latency: 180, languages: 57 },
  { id: "google", name: "Google Cloud", icon: "G", status: "available", latency: 150, languages: 125 },
  { id: "azure", name: "Azure Speech", icon: "A", status: "available", latency: 140, languages: 100 },
  { id: "assemblyai", name: "AssemblyAI", icon: "A", status: "connected", latency: 160, languages: 20 },
  { id: "aws", name: "AWS Transcribe", icon: "W", status: "available", latency: 170, languages: 37 },
  { id: "speechmatics", name: "Speechmatics", icon: "S", status: "available", latency: 130, languages: 35 },
  { id: "rev", name: "Rev AI", icon: "R", status: "available", latency: 200, languages: 15 },
];

// Mock TTS Providers
const ttsProviders = [
  { id: "elevenlabs", name: "ElevenLabs", icon: "E", status: "connected", quality: "premium", voices: 120 },
  { id: "playht", name: "PlayHT", icon: "P", status: "connected", quality: "premium", voices: 100 },
  { id: "cartesia", name: "Cartesia", icon: "C", status: "available", quality: "premium", voices: 50 },
  { id: "openai", name: "OpenAI TTS", icon: "O", status: "connected", quality: "high", voices: 6 },
  { id: "google", name: "Google Cloud", icon: "G", status: "available", quality: "high", voices: 220 },
  { id: "azure", name: "Azure Speech", icon: "A", status: "available", quality: "high", voices: 400 },
  { id: "aws", name: "AWS Polly", icon: "W", status: "available", quality: "standard", voices: 60 },
  { id: "murf", name: "Murf AI", icon: "M", status: "available", quality: "premium", voices: 120 },
  { id: "wellsaid", name: "WellSaid Labs", icon: "W", status: "available", quality: "premium", voices: 50 },
  { id: "lmnt", name: "LMNT", icon: "L", status: "available", quality: "high", voices: 20 },
];

// Mock Voice Library
const voiceLibrary = [
  { id: "1", name: "Rachel", provider: "elevenlabs", gender: "female", language: "en-US", style: "friendly", preview: true, popular: true },
  { id: "2", name: "Josh", provider: "elevenlabs", gender: "male", language: "en-US", style: "professional", preview: true, popular: true },
  { id: "3", name: "Bella", provider: "elevenlabs", gender: "female", language: "en-US", style: "warm", preview: true, popular: false },
  { id: "4", name: "Adam", provider: "elevenlabs", gender: "male", language: "en-US", style: "confident", preview: true, popular: true },
  { id: "5", name: "Dorothy", provider: "elevenlabs", gender: "female", language: "en-UK", style: "british", preview: true, popular: false },
  { id: "6", name: "Shimmer", provider: "openai", gender: "female", language: "en-US", style: "natural", preview: true, popular: true },
  { id: "7", name: "Echo", provider: "openai", gender: "male", language: "en-US", style: "neutral", preview: true, popular: false },
  { id: "8", name: "Onyx", provider: "openai", gender: "male", language: "en-US", style: "deep", preview: true, popular: false },
  { id: "9", name: "Nova", provider: "openai", gender: "female", language: "en-US", style: "energetic", preview: true, popular: true },
  { id: "10", name: "Fable", provider: "openai", gender: "neutral", language: "en-US", style: "storytelling", preview: true, popular: false },
  { id: "11", name: "Jennifer", provider: "playht", gender: "female", language: "en-US", style: "conversational", preview: true, popular: true },
  { id: "12", name: "Michael", provider: "playht", gender: "male", language: "en-US", style: "professional", preview: true, popular: false },
  { id: "13", name: "Sarah", provider: "google", gender: "female", language: "en-US", style: "natural", preview: true, popular: false },
  { id: "14", name: "James", provider: "google", gender: "male", language: "en-UK", style: "british", preview: true, popular: false },
  { id: "15", name: "Maria", provider: "azure", gender: "female", language: "es-ES", style: "spanish", preview: true, popular: false },
  { id: "16", name: "Hans", provider: "azure", gender: "male", language: "de-DE", style: "german", preview: true, popular: false },
];

// Configuration Presets
const presets = [
  { id: "low-latency", name: "Low Latency", description: "Optimized for fastest response time", icon: Zap, color: "text-yellow-500" },
  { id: "high-quality", name: "High Quality", description: "Best audio quality and clarity", icon: Star, color: "text-purple-500" },
  { id: "cost-optimized", name: "Cost Optimized", description: "Balanced quality and pricing", icon: DollarSign, color: "text-green-500" },
  { id: "multilingual", name: "Multilingual", description: "Support for 100+ languages", icon: Globe, color: "text-blue-500" },
  { id: "natural", name: "Natural Conversation", description: "Most human-like interactions", icon: User, color: "text-orange-500" },
];

type ViewMode = "grid" | "list";
type Tab = "voices" | "stt" | "tts" | "presets";

export default function VoiceConfigPage() {
  const [activeTab, setActiveTab] = useState<Tab>("voices");
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [selectedGender, setSelectedGender] = useState<string | null>(null);
  const [playingVoice, setPlayingVoice] = useState<string | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<string>("natural");

  const filteredVoices = useMemo(() => {
    return voiceLibrary.filter((voice) => {
      const matchesSearch = voice.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesProvider = !selectedProvider || voice.provider === selectedProvider;
      const matchesGender = !selectedGender || voice.gender === selectedGender;
      return matchesSearch && matchesProvider && matchesGender;
    });
  }, [searchQuery, selectedProvider, selectedGender]);

  const handlePlayVoice = (voiceId: string) => {
    if (playingVoice === voiceId) {
      setPlayingVoice(null);
    } else {
      setPlayingVoice(voiceId);
      // Simulate playback ending after 3 seconds
      setTimeout(() => setPlayingVoice(null), 3000);
    }
  };

  const getProviderColor = (provider: string) => {
    const colors: Record<string, string> = {
      elevenlabs: "bg-purple-100 text-purple-800",
      openai: "bg-green-100 text-green-800",
      playht: "bg-blue-100 text-blue-800",
      google: "bg-red-100 text-red-800",
      azure: "bg-cyan-100 text-cyan-800",
      aws: "bg-orange-100 text-orange-800",
    };
    return colors[provider] || "bg-gray-100 text-gray-800";
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Voice Configuration</h1>
          <p className="text-muted-foreground">
            Configure STT, TTS providers and select voices for your agents
          </p>
        </div>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          Create Configuration
        </Button>
      </div>

      {/* Tabs */}
      <div className="flex border-b">
        {[
          { id: "voices", label: "Voice Library", icon: Volume2 },
          { id: "stt", label: "STT Providers", icon: Mic },
          { id: "tts", label: "TTS Providers", icon: Volume2 },
          { id: "presets", label: "Presets", icon: Sliders },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as Tab)}
            className={cn(
              "flex items-center gap-2 px-4 py-3 border-b-2 transition-colors",
              activeTab === tab.id
                ? "border-primary text-primary font-medium"
                : "border-transparent text-muted-foreground hover:text-foreground"
            )}
          >
            <tab.icon className="h-4 w-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Voice Library Tab */}
      {activeTab === "voices" && (
        <div className="space-y-4">
          {/* Filters */}
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex-1 min-w-[200px] max-w-md">
              <Input
                placeholder="Search voices..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                leftIcon={<Search className="h-4 w-4" />}
              />
            </div>
            <div className="flex gap-2">
              <Button
                variant={selectedProvider === null ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedProvider(null)}
              >
                All Providers
              </Button>
              <Button
                variant={selectedProvider === "elevenlabs" ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedProvider("elevenlabs")}
              >
                ElevenLabs
              </Button>
              <Button
                variant={selectedProvider === "openai" ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedProvider("openai")}
              >
                OpenAI
              </Button>
              <Button
                variant={selectedProvider === "playht" ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedProvider("playht")}
              >
                PlayHT
              </Button>
            </div>
            <div className="flex gap-2">
              <Button
                variant={selectedGender === null ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedGender(null)}
              >
                All
              </Button>
              <Button
                variant={selectedGender === "female" ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedGender("female")}
              >
                Female
              </Button>
              <Button
                variant={selectedGender === "male" ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedGender("male")}
              >
                Male
              </Button>
            </div>
            <div className="flex gap-1 ml-auto">
              <Button
                variant={viewMode === "grid" ? "default" : "ghost"}
                size="icon-sm"
                onClick={() => setViewMode("grid")}
              >
                <Grid className="h-4 w-4" />
              </Button>
              <Button
                variant={viewMode === "list" ? "default" : "ghost"}
                size="icon-sm"
                onClick={() => setViewMode("list")}
              >
                <List className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Voice Grid */}
          {viewMode === "grid" ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {filteredVoices.map((voice) => (
                <Card key={voice.id} className="group relative overflow-hidden hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className={cn(
                          "flex h-12 w-12 items-center justify-center rounded-full",
                          voice.gender === "female" ? "bg-pink-100" :
                          voice.gender === "male" ? "bg-blue-100" : "bg-purple-100"
                        )}>
                          <Volume2 className={cn(
                            "h-6 w-6",
                            voice.gender === "female" ? "text-pink-600" :
                            voice.gender === "male" ? "text-blue-600" : "text-purple-600"
                          )} />
                        </div>
                        <div>
                          <h3 className="font-semibold flex items-center gap-1">
                            {voice.name}
                            {voice.popular && <Star className="h-3 w-3 text-yellow-500 fill-yellow-500" />}
                          </h3>
                          <Badge className={getProviderColor(voice.provider)} variant="secondary">
                            {voice.provider}
                          </Badge>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-2 mb-4">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Globe className="h-3 w-3" />
                        {voice.language}
                      </div>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <User className="h-3 w-3" />
                        {voice.style}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex-1"
                        onClick={() => handlePlayVoice(voice.id)}
                      >
                        {playingVoice === voice.id ? (
                          <>
                            <Pause className="mr-1 h-3 w-3" />
                            Stop
                          </>
                        ) : (
                          <>
                            <Play className="mr-1 h-3 w-3" />
                            Preview
                          </>
                        )}
                      </Button>
                      <Button variant="default" size="sm" className="flex-1">
                        <Check className="mr-1 h-3 w-3" />
                        Select
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="p-0">
                <table className="w-full">
                  <thead>
                    <tr className="border-b text-left text-sm text-muted-foreground">
                      <th className="p-4 font-medium">Voice</th>
                      <th className="p-4 font-medium">Provider</th>
                      <th className="p-4 font-medium">Gender</th>
                      <th className="p-4 font-medium">Language</th>
                      <th className="p-4 font-medium">Style</th>
                      <th className="p-4 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredVoices.map((voice) => (
                      <tr key={voice.id} className="border-b last:border-0 hover:bg-muted/50">
                        <td className="p-4">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{voice.name}</span>
                            {voice.popular && <Star className="h-3 w-3 text-yellow-500 fill-yellow-500" />}
                          </div>
                        </td>
                        <td className="p-4">
                          <Badge className={getProviderColor(voice.provider)} variant="secondary">
                            {voice.provider}
                          </Badge>
                        </td>
                        <td className="p-4 capitalize">{voice.gender}</td>
                        <td className="p-4">{voice.language}</td>
                        <td className="p-4 capitalize">{voice.style}</td>
                        <td className="p-4">
                          <div className="flex gap-2">
                            <Button
                              variant="ghost"
                              size="icon-sm"
                              onClick={() => handlePlayVoice(voice.id)}
                            >
                              {playingVoice === voice.id ? (
                                <Pause className="h-4 w-4" />
                              ) : (
                                <Play className="h-4 w-4" />
                              )}
                            </Button>
                            <Button variant="outline" size="sm">
                              Select
                            </Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* STT Providers Tab */}
      {activeTab === "stt" && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {sttProviders.map((provider) => (
            <Card key={provider.id} className={cn(
              "cursor-pointer transition-all hover:shadow-md",
              provider.status === "connected" && "ring-2 ring-green-500"
            )}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary font-bold">
                      {provider.icon}
                    </div>
                    <div>
                      <CardTitle className="text-base">{provider.name}</CardTitle>
                      <Badge
                        variant={provider.status === "connected" ? "success" : "secondary"}
                        className="mt-1"
                      >
                        {provider.status}
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="flex items-center gap-1 text-muted-foreground mb-1">
                      <Clock className="h-3 w-3" />
                      Latency
                    </div>
                    <span className="font-medium">{provider.latency}ms</span>
                  </div>
                  <div>
                    <div className="flex items-center gap-1 text-muted-foreground mb-1">
                      <Languages className="h-3 w-3" />
                      Languages
                    </div>
                    <span className="font-medium">{provider.languages}</span>
                  </div>
                </div>
                <Button
                  variant={provider.status === "connected" ? "outline" : "default"}
                  className="w-full mt-4"
                  size="sm"
                >
                  {provider.status === "connected" ? (
                    <>
                      <Settings2 className="mr-2 h-4 w-4" />
                      Configure
                    </>
                  ) : (
                    <>
                      <Plus className="mr-2 h-4 w-4" />
                      Connect
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* TTS Providers Tab */}
      {activeTab === "tts" && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {ttsProviders.map((provider) => (
            <Card key={provider.id} className={cn(
              "cursor-pointer transition-all hover:shadow-md",
              provider.status === "connected" && "ring-2 ring-green-500"
            )}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary font-bold">
                      {provider.icon}
                    </div>
                    <div>
                      <CardTitle className="text-base">{provider.name}</CardTitle>
                      <Badge
                        variant={provider.status === "connected" ? "success" : "secondary"}
                        className="mt-1"
                      >
                        {provider.status}
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="flex items-center gap-1 text-muted-foreground mb-1">
                      <Star className="h-3 w-3" />
                      Quality
                    </div>
                    <span className={cn(
                      "font-medium capitalize",
                      provider.quality === "premium" && "text-purple-600"
                    )}>
                      {provider.quality}
                    </span>
                  </div>
                  <div>
                    <div className="flex items-center gap-1 text-muted-foreground mb-1">
                      <Volume2 className="h-3 w-3" />
                      Voices
                    </div>
                    <span className="font-medium">{provider.voices}</span>
                  </div>
                </div>
                <Button
                  variant={provider.status === "connected" ? "outline" : "default"}
                  className="w-full mt-4"
                  size="sm"
                >
                  {provider.status === "connected" ? (
                    <>
                      <Settings2 className="mr-2 h-4 w-4" />
                      Configure
                    </>
                  ) : (
                    <>
                      <Plus className="mr-2 h-4 w-4" />
                      Connect
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Presets Tab */}
      {activeTab === "presets" && (
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {presets.map((preset) => (
              <Card
                key={preset.id}
                className={cn(
                  "cursor-pointer transition-all hover:shadow-md",
                  selectedPreset === preset.id && "ring-2 ring-primary"
                )}
                onClick={() => setSelectedPreset(preset.id)}
              >
                <CardContent className="p-6">
                  <div className="flex items-start gap-4">
                    <div className={cn(
                      "flex h-12 w-12 items-center justify-center rounded-lg bg-muted",
                      preset.color
                    )}>
                      <preset.icon className="h-6 w-6" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold">{preset.name}</h3>
                        {selectedPreset === preset.id && (
                          <Check className="h-5 w-5 text-primary" />
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {preset.description}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Preset Details */}
          <Card>
            <CardHeader>
              <CardTitle>Preset Configuration</CardTitle>
              <CardDescription>
                Settings applied by the selected preset
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-4">
                  <h4 className="font-medium">Speech-to-Text</h4>
                  <div className="rounded-lg border p-4 space-y-3">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Provider</span>
                      <span className="font-medium">Deepgram</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Model</span>
                      <span className="font-medium">Nova-2</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Language</span>
                      <span className="font-medium">Multi (auto-detect)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Punctuation</span>
                      <span className="font-medium">Enabled</span>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <h4 className="font-medium">Text-to-Speech</h4>
                  <div className="rounded-lg border p-4 space-y-3">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Provider</span>
                      <span className="font-medium">ElevenLabs</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Model</span>
                      <span className="font-medium">Eleven Turbo v2</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Voice</span>
                      <span className="font-medium">Rachel</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Stability</span>
                      <span className="font-medium">0.5</span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="flex justify-end gap-3 mt-6">
                <Button variant="outline">Reset to Defaults</Button>
                <Button>Apply Preset</Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
