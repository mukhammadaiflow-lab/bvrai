"use client";

import { useState, useMemo, useRef } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import {
  Mic,
  Play,
  Pause,
  Square,
  Volume2,
  VolumeX,
  Volume1,
  Plus,
  Search,
  Filter,
  MoreVertical,
  Edit,
  Copy,
  Trash2,
  Download,
  Upload,
  Star,
  StarOff,
  Heart,
  Globe,
  User,
  Users,
  Bot,
  Settings,
  Sparkles,
  Zap,
  ChevronRight,
  ChevronDown,
  X,
  Check,
  CheckCircle,
  AlertCircle,
  Info,
  Clock,
  RefreshCw,
  LayoutGrid,
  List,
  Wand2,
  AudioLines,
  Waves,
  Radio,
  Speaker,
  Headphones,
  Sliders,
  Music,
  Mic2,
  MicOff,
  ArrowUpRight,
  ArrowDownRight,
  Tag,
  Loader2,
  ExternalLink,
  Share2,
} from "lucide-react";

// Types
type VoiceGender = "male" | "female" | "neutral";
type VoiceStyle = "professional" | "friendly" | "calm" | "energetic" | "authoritative" | "warm" | "conversational";
type VoiceAccent = "american" | "british" | "australian" | "indian" | "neutral" | "southern" | "midwest";
type VoiceProvider = "elevenlabs" | "openai" | "google" | "azure" | "custom";
type VoiceStatus = "available" | "processing" | "premium" | "custom";

interface Voice {
  id: string;
  name: string;
  description: string;
  gender: VoiceGender;
  style: VoiceStyle[];
  accent: VoiceAccent;
  language: string;
  provider: VoiceProvider;
  status: VoiceStatus;
  sampleUrl: string;
  previewText: string;
  stats: {
    usageCount: number;
    rating: number;
    popularity: number;
  };
  settings: {
    stability: number;
    clarity: number;
    speed: number;
    pitch: number;
  };
  tags: string[];
  favorite: boolean;
  isCustom: boolean;
  isPremium: boolean;
  createdAt?: string;
}

interface VoiceCategory {
  id: string;
  name: string;
  icon: any;
  count: number;
}

// Sample Data
const sampleVoices: Voice[] = [
  {
    id: "voice_1",
    name: "Sarah",
    description: "A friendly and professional female voice, perfect for customer service and sales",
    gender: "female",
    style: ["professional", "friendly", "warm"],
    accent: "american",
    language: "en-US",
    provider: "elevenlabs",
    status: "available",
    sampleUrl: "/voices/sarah-sample.mp3",
    previewText: "Hello! Thank you for calling. How can I help you today?",
    stats: { usageCount: 15420, rating: 4.9, popularity: 98 },
    settings: { stability: 75, clarity: 85, speed: 1.0, pitch: 1.0 },
    tags: ["customer-service", "sales", "popular"],
    favorite: true,
    isCustom: false,
    isPremium: false,
  },
  {
    id: "voice_2",
    name: "James",
    description: "A calm and authoritative male voice, ideal for professional announcements",
    gender: "male",
    style: ["authoritative", "calm", "professional"],
    accent: "british",
    language: "en-GB",
    provider: "elevenlabs",
    status: "available",
    sampleUrl: "/voices/james-sample.mp3",
    previewText: "Good afternoon. I'm calling to discuss your recent inquiry.",
    stats: { usageCount: 12350, rating: 4.8, popularity: 92 },
    settings: { stability: 80, clarity: 90, speed: 0.95, pitch: 0.9 },
    tags: ["professional", "announcements", "british"],
    favorite: true,
    isCustom: false,
    isPremium: false,
  },
  {
    id: "voice_3",
    name: "Emily",
    description: "An energetic and enthusiastic female voice, great for marketing calls",
    gender: "female",
    style: ["energetic", "friendly", "conversational"],
    accent: "american",
    language: "en-US",
    provider: "openai",
    status: "available",
    sampleUrl: "/voices/emily-sample.mp3",
    previewText: "Hey there! I have some exciting news to share with you!",
    stats: { usageCount: 8920, rating: 4.7, popularity: 85 },
    settings: { stability: 65, clarity: 80, speed: 1.1, pitch: 1.05 },
    tags: ["marketing", "energetic", "upbeat"],
    favorite: false,
    isCustom: false,
    isPremium: false,
  },
  {
    id: "voice_4",
    name: "Michael",
    description: "A warm and reassuring male voice, perfect for healthcare and support",
    gender: "male",
    style: ["warm", "calm", "conversational"],
    accent: "american",
    language: "en-US",
    provider: "google",
    status: "available",
    sampleUrl: "/voices/michael-sample.mp3",
    previewText: "Hi, this is a friendly reminder about your upcoming appointment.",
    stats: { usageCount: 10450, rating: 4.8, popularity: 88 },
    settings: { stability: 85, clarity: 85, speed: 0.95, pitch: 1.0 },
    tags: ["healthcare", "support", "reminders"],
    favorite: false,
    isCustom: false,
    isPremium: false,
  },
  {
    id: "voice_5",
    name: "Olivia",
    description: "A sophisticated and elegant female voice with Australian accent",
    gender: "female",
    style: ["professional", "warm", "conversational"],
    accent: "australian",
    language: "en-AU",
    provider: "elevenlabs",
    status: "premium",
    sampleUrl: "/voices/olivia-sample.mp3",
    previewText: "G'day! Thanks for reaching out. Let me help you with that.",
    stats: { usageCount: 5680, rating: 4.9, popularity: 78 },
    settings: { stability: 75, clarity: 88, speed: 1.0, pitch: 1.02 },
    tags: ["australian", "premium", "elegant"],
    favorite: false,
    isCustom: false,
    isPremium: true,
  },
  {
    id: "voice_6",
    name: "David",
    description: "A deep and commanding male voice for important announcements",
    gender: "male",
    style: ["authoritative", "professional"],
    accent: "american",
    language: "en-US",
    provider: "azure",
    status: "available",
    sampleUrl: "/voices/david-sample.mp3",
    previewText: "Your attention please. This is an important announcement.",
    stats: { usageCount: 7230, rating: 4.6, popularity: 72 },
    settings: { stability: 90, clarity: 85, speed: 0.9, pitch: 0.85 },
    tags: ["announcements", "deep", "commanding"],
    favorite: false,
    isCustom: false,
    isPremium: false,
  },
  {
    id: "voice_7",
    name: "Priya",
    description: "A clear and articulate female voice with Indian accent",
    gender: "female",
    style: ["professional", "friendly", "calm"],
    accent: "indian",
    language: "en-IN",
    provider: "google",
    status: "available",
    sampleUrl: "/voices/priya-sample.mp3",
    previewText: "Namaste! Welcome to our support line. How may I assist you?",
    stats: { usageCount: 6890, rating: 4.7, popularity: 75 },
    settings: { stability: 78, clarity: 82, speed: 1.0, pitch: 1.0 },
    tags: ["indian", "support", "multilingual"],
    favorite: false,
    isCustom: false,
    isPremium: false,
  },
  {
    id: "voice_8",
    name: "Custom Agent Voice",
    description: "Your custom cloned voice for personalized interactions",
    gender: "male",
    style: ["conversational"],
    accent: "american",
    language: "en-US",
    provider: "custom",
    status: "custom",
    sampleUrl: "/voices/custom-sample.mp3",
    previewText: "This is your custom cloned voice speaking.",
    stats: { usageCount: 2450, rating: 0, popularity: 0 },
    settings: { stability: 70, clarity: 75, speed: 1.0, pitch: 1.0 },
    tags: ["custom", "cloned"],
    favorite: true,
    isCustom: true,
    isPremium: false,
    createdAt: "2024-01-15T10:00:00Z",
  },
];

const voiceCategories: VoiceCategory[] = [
  { id: "all", name: "All Voices", icon: Mic, count: 8 },
  { id: "favorites", name: "Favorites", icon: Star, count: 3 },
  { id: "custom", name: "My Voices", icon: User, count: 1 },
  { id: "professional", name: "Professional", icon: Bot, count: 5 },
  { id: "friendly", name: "Friendly", icon: Heart, count: 4 },
  { id: "premium", name: "Premium", icon: Sparkles, count: 1 },
];

// Utility functions
const getGenderColor = (gender: VoiceGender) => {
  switch (gender) {
    case "female":
      return "bg-pink-500/20 text-pink-400";
    case "male":
      return "bg-blue-500/20 text-blue-400";
    case "neutral":
      return "bg-purple-500/20 text-purple-400";
    default:
      return "bg-gray-500/20 text-gray-400";
  }
};

const getProviderColor = (provider: VoiceProvider) => {
  switch (provider) {
    case "elevenlabs":
      return "bg-gradient-to-r from-purple-500 to-pink-500";
    case "openai":
      return "bg-gradient-to-r from-green-500 to-teal-500";
    case "google":
      return "bg-gradient-to-r from-blue-500 to-cyan-500";
    case "azure":
      return "bg-gradient-to-r from-blue-600 to-indigo-500";
    case "custom":
      return "bg-gradient-to-r from-orange-500 to-red-500";
    default:
      return "bg-gray-500";
  }
};

const formatNumber = (num: number) => {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
};

// Components
function VoiceCard({
  voice,
  onPlay,
  onSelect,
  onToggleFavorite,
  isPlaying,
  isSelected,
}: {
  voice: Voice;
  onPlay: () => void;
  onSelect: () => void;
  onToggleFavorite: () => void;
  isPlaying: boolean;
  isSelected: boolean;
}) {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <div
      className={`bg-[#1a1a2e]/80 rounded-xl border transition-all group cursor-pointer ${
        isSelected
          ? "border-purple-500 ring-2 ring-purple-500/20"
          : "border-white/5 hover:border-white/10"
      }`}
      onClick={onSelect}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            {/* Avatar */}
            <div className={`w-12 h-12 rounded-xl ${getProviderColor(voice.provider)} flex items-center justify-center`}>
              <Mic className="h-6 w-6 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-white group-hover:text-purple-400 transition-colors">
                  {voice.name}
                </h3>
                {voice.isPremium && (
                  <span className="px-1.5 py-0.5 bg-yellow-500/20 text-yellow-400 rounded text-xs font-medium">
                    Premium
                  </span>
                )}
                {voice.isCustom && (
                  <span className="px-1.5 py-0.5 bg-orange-500/20 text-orange-400 rounded text-xs font-medium">
                    Custom
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <span className={`px-1.5 py-0.5 rounded text-xs ${getGenderColor(voice.gender)}`}>
                  {voice.gender}
                </span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  <Globe className="h-3 w-3" />
                  {voice.accent}
                </span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleFavorite();
              }}
              className="p-1.5 rounded-lg hover:bg-white/5 transition-colors"
            >
              {voice.favorite ? (
                <Star className="h-4 w-4 text-yellow-400 fill-yellow-400" />
              ) : (
                <Star className="h-4 w-4 text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity" />
              )}
            </button>
            <div className="relative">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowMenu(!showMenu);
                }}
                className="p-1.5 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
              >
                <MoreVertical className="h-4 w-4" />
              </button>
              {showMenu && (
                <div className="absolute right-0 top-full mt-1 w-40 bg-[#252542] rounded-lg border border-white/10 shadow-xl z-10 py-1">
                  <button className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2">
                    <Copy className="h-4 w-4" /> Clone Voice
                  </button>
                  <button className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2">
                    <Settings className="h-4 w-4" /> Settings
                  </button>
                  <button className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2">
                    <Share2 className="h-4 w-4" /> Share
                  </button>
                  {voice.isCustom && (
                    <>
                      <div className="border-t border-white/5 my-1" />
                      <button className="w-full px-4 py-2 text-left text-sm text-red-400 hover:bg-red-500/10 flex items-center gap-2">
                        <Trash2 className="h-4 w-4" /> Delete
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-gray-400 mb-3 line-clamp-2">{voice.description}</p>

        {/* Tags */}
        <div className="flex flex-wrap gap-1.5 mb-3">
          {voice.style.slice(0, 3).map((style) => (
            <span key={style} className="px-2 py-0.5 bg-white/5 text-gray-400 rounded text-xs capitalize">
              {style}
            </span>
          ))}
        </div>

        {/* Play Button & Stats */}
        <div className="flex items-center justify-between pt-3 border-t border-white/5">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onPlay();
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              isPlaying
                ? "bg-purple-500 text-white"
                : "bg-white/5 text-gray-300 hover:bg-white/10"
            }`}
          >
            {isPlaying ? (
              <>
                <Pause className="h-4 w-4" />
                Playing
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                Preview
              </>
            )}
          </button>
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-1 text-gray-400">
              <Star className="h-3.5 w-3.5 text-yellow-400" />
              <span>{voice.stats.rating}</span>
            </div>
            <div className="flex items-center gap-1 text-gray-400">
              <Users className="h-3.5 w-3.5" />
              <span>{formatNumber(voice.stats.usageCount)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function VoicePreviewPanel({ voice, onClose }: { voice: Voice; onClose: () => void }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [settings, setSettings] = useState(voice.settings);

  return (
    <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-6">
      <div className="flex items-start justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className={`w-16 h-16 rounded-xl ${getProviderColor(voice.provider)} flex items-center justify-center`}>
            <Mic className="h-8 w-8 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-semibold text-white">{voice.name}</h2>
              {voice.isPremium && (
                <span className="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 rounded text-xs font-medium">
                  Premium
                </span>
              )}
            </div>
            <p className="text-gray-400 mt-1">{voice.description}</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Audio Player */}
      <div className="bg-[#0d0d1a] rounded-xl p-4 mb-6">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="w-12 h-12 rounded-full bg-purple-500 text-white flex items-center justify-center hover:bg-purple-600 transition-colors"
          >
            {isPlaying ? <Pause className="h-6 w-6" /> : <Play className="h-6 w-6 ml-1" />}
          </button>
          <div className="flex-1">
            <div className="h-12 bg-white/5 rounded-lg flex items-center px-4 gap-1">
              {Array.from({ length: 40 }).map((_, i) => (
                <div
                  key={i}
                  className={`w-1 rounded-full transition-all ${
                    isPlaying ? "bg-purple-500" : "bg-gray-600"
                  }`}
                  style={{
                    height: `${20 + Math.random() * 60}%`,
                    animationDelay: `${i * 50}ms`,
                  }}
                />
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Volume2 className="h-4 w-4 text-gray-400" />
            <input
              type="range"
              min="0"
              max="100"
              defaultValue="80"
              className="w-20 accent-purple-500"
            />
          </div>
        </div>
        <p className="text-sm text-gray-400 mt-3 italic">"{voice.previewText}"</p>
      </div>

      {/* Voice Settings */}
      <div className="space-y-4 mb-6">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <Sliders className="h-4 w-4 text-purple-400" />
          Voice Settings
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-400">Stability</label>
              <span className="text-sm text-white">{settings.stability}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={settings.stability}
              onChange={(e) => setSettings({ ...settings, stability: parseInt(e.target.value) })}
              className="w-full accent-purple-500"
            />
          </div>
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-400">Clarity</label>
              <span className="text-sm text-white">{settings.clarity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={settings.clarity}
              onChange={(e) => setSettings({ ...settings, clarity: parseInt(e.target.value) })}
              className="w-full accent-purple-500"
            />
          </div>
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-400">Speed</label>
              <span className="text-sm text-white">{settings.speed.toFixed(2)}x</span>
            </div>
            <input
              type="range"
              min="50"
              max="150"
              value={settings.speed * 100}
              onChange={(e) => setSettings({ ...settings, speed: parseInt(e.target.value) / 100 })}
              className="w-full accent-purple-500"
            />
          </div>
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-400">Pitch</label>
              <span className="text-sm text-white">{settings.pitch.toFixed(2)}x</span>
            </div>
            <input
              type="range"
              min="50"
              max="150"
              value={settings.pitch * 100}
              onChange={(e) => setSettings({ ...settings, pitch: parseInt(e.target.value) / 100 })}
              className="w-full accent-purple-500"
            />
          </div>
        </div>
      </div>

      {/* Voice Info */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Language</div>
          <div className="text-white">{voice.language}</div>
        </div>
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Accent</div>
          <div className="text-white capitalize">{voice.accent}</div>
        </div>
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Provider</div>
          <div className="text-white capitalize">{voice.provider}</div>
        </div>
        <div className="p-3 bg-white/5 rounded-lg">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Usage</div>
          <div className="text-white">{formatNumber(voice.stats.usageCount)} calls</div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-3">
        <button className="flex-1 px-4 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity flex items-center justify-center gap-2">
          <Check className="h-4 w-4" />
          Use This Voice
        </button>
        <button className="px-4 py-2.5 bg-white/5 text-gray-300 rounded-xl font-medium hover:bg-white/10 transition-colors flex items-center gap-2">
          <Copy className="h-4 w-4" />
          Clone
        </button>
      </div>
    </div>
  );
}

function CreateVoiceDialog({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [step, setStep] = useState(1);
  const [uploadMethod, setUploadMethod] = useState<"record" | "upload" | "text">("upload");
  const [isRecording, setIsRecording] = useState(false);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a2e] rounded-2xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">Create Custom Voice</h2>
            <p className="text-sm text-gray-400 mt-0.5">Step {step} of 3</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {step === 1 && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Voice Name</label>
                <input
                  type="text"
                  placeholder="e.g., My Custom Voice"
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Description</label>
                <textarea
                  rows={3}
                  placeholder="Describe your voice..."
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  How would you like to create your voice?
                </label>
                <div className="grid grid-cols-3 gap-3">
                  <button
                    onClick={() => setUploadMethod("record")}
                    className={`p-4 rounded-xl border text-center transition-all ${
                      uploadMethod === "record"
                        ? "bg-purple-500/10 border-purple-500"
                        : "bg-white/5 border-white/10 hover:border-white/20"
                    }`}
                  >
                    <Mic2 className={`h-6 w-6 mx-auto mb-2 ${uploadMethod === "record" ? "text-purple-400" : "text-gray-400"}`} />
                    <div className="font-medium text-white text-sm">Record</div>
                    <div className="text-xs text-gray-500 mt-1">Use your microphone</div>
                  </button>
                  <button
                    onClick={() => setUploadMethod("upload")}
                    className={`p-4 rounded-xl border text-center transition-all ${
                      uploadMethod === "upload"
                        ? "bg-purple-500/10 border-purple-500"
                        : "bg-white/5 border-white/10 hover:border-white/20"
                    }`}
                  >
                    <Upload className={`h-6 w-6 mx-auto mb-2 ${uploadMethod === "upload" ? "text-purple-400" : "text-gray-400"}`} />
                    <div className="font-medium text-white text-sm">Upload</div>
                    <div className="text-xs text-gray-500 mt-1">Upload audio files</div>
                  </button>
                  <button
                    onClick={() => setUploadMethod("text")}
                    className={`p-4 rounded-xl border text-center transition-all ${
                      uploadMethod === "text"
                        ? "bg-purple-500/10 border-purple-500"
                        : "bg-white/5 border-white/10 hover:border-white/20"
                    }`}
                  >
                    <Wand2 className={`h-6 w-6 mx-auto mb-2 ${uploadMethod === "text" ? "text-purple-400" : "text-gray-400"}`} />
                    <div className="font-medium text-white text-sm">AI Generate</div>
                    <div className="text-xs text-gray-500 mt-1">Describe with text</div>
                  </button>
                </div>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-6">
              {uploadMethod === "record" && (
                <div className="text-center py-8">
                  <div
                    className={`w-32 h-32 rounded-full mx-auto mb-6 flex items-center justify-center transition-all ${
                      isRecording
                        ? "bg-red-500/20 animate-pulse"
                        : "bg-purple-500/20"
                    }`}
                  >
                    <button
                      onClick={() => setIsRecording(!isRecording)}
                      className={`w-20 h-20 rounded-full flex items-center justify-center transition-all ${
                        isRecording ? "bg-red-500" : "bg-purple-500"
                      }`}
                    >
                      {isRecording ? (
                        <Square className="h-8 w-8 text-white" />
                      ) : (
                        <Mic2 className="h-8 w-8 text-white" />
                      )}
                    </button>
                  </div>
                  <p className="text-lg font-medium text-white mb-2">
                    {isRecording ? "Recording..." : "Click to start recording"}
                  </p>
                  <p className="text-sm text-gray-400">
                    Record at least 30 seconds of clear speech
                  </p>
                </div>
              )}

              {uploadMethod === "upload" && (
                <div className="border-2 border-dashed border-white/10 rounded-xl p-12 text-center hover:border-purple-500/50 transition-colors cursor-pointer">
                  <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <div className="text-lg font-medium text-white mb-2">
                    Drop audio files here
                  </div>
                  <p className="text-sm text-gray-400 mb-4">
                    or click to browse from your computer
                  </p>
                  <button className="px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg text-sm font-medium hover:bg-purple-500/30 transition-colors">
                    Select Files
                  </button>
                  <div className="mt-4 text-xs text-gray-500">
                    Supported: MP3, WAV, M4A (max 25MB)
                  </div>
                </div>
              )}

              {uploadMethod === "text" && (
                <div className="space-y-4">
                  <p className="text-sm text-gray-400">
                    Describe the voice characteristics you want to create:
                  </p>
                  <textarea
                    rows={6}
                    placeholder="e.g., A warm, professional female voice with a slight British accent. The tone should be friendly and reassuring, suitable for healthcare customer service..."
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                  />
                  <div className="flex items-center gap-3">
                    <Sparkles className="h-4 w-4 text-purple-400" />
                    <span className="text-sm text-gray-400">
                      AI will generate a custom voice based on your description
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {step === 3 && (
            <div className="space-y-6">
              <div className="bg-green-500/10 rounded-xl p-5 border border-green-500/20">
                <div className="flex items-start gap-3">
                  <CheckCircle className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <div className="font-medium text-green-400">Voice Created Successfully!</div>
                    <div className="text-sm text-gray-400 mt-1">
                      Your custom voice is now being processed. This usually takes 2-5 minutes.
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-white/5 rounded-xl p-5">
                <h3 className="font-semibold text-white mb-3">What's Next?</h3>
                <ul className="space-y-2 text-sm text-gray-400">
                  <li className="flex items-center gap-2">
                    <Check className="h-4 w-4 text-green-400" />
                    Voice will appear in your library when ready
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="h-4 w-4 text-green-400" />
                    You can assign it to any agent
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="h-4 w-4 text-green-400" />
                    Fine-tune settings after creation
                  </li>
                </ul>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-white/5 flex items-center justify-between">
          <button
            onClick={() => (step > 1 ? setStep(step - 1) : onClose())}
            className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            {step === 1 ? "Cancel" : "Back"}
          </button>
          <button
            onClick={() => (step < 3 ? setStep(step + 1) : onClose())}
            className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-medium hover:opacity-90 transition-opacity flex items-center gap-2"
          >
            {step === 3 ? (
              <>
                <Check className="h-4 w-4" />
                Done
              </>
            ) : (
              <>
                Next
                <ChevronRight className="h-4 w-4" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// Main Page Component
export default function VoiceLibraryPage() {
  const [voices] = useState<Voice[]>(sampleVoices);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [genderFilter, setGenderFilter] = useState<VoiceGender | "all">("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [playingVoiceId, setPlayingVoiceId] = useState<string | null>(null);
  const [selectedVoice, setSelectedVoice] = useState<Voice | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);

  // Filter voices
  const filteredVoices = useMemo(() => {
    let filtered = voices;

    if (selectedCategory === "favorites") {
      filtered = filtered.filter((v) => v.favorite);
    } else if (selectedCategory === "custom") {
      filtered = filtered.filter((v) => v.isCustom);
    } else if (selectedCategory === "premium") {
      filtered = filtered.filter((v) => v.isPremium);
    } else if (selectedCategory !== "all") {
      filtered = filtered.filter((v) => v.style.includes(selectedCategory as VoiceStyle));
    }

    if (genderFilter !== "all") {
      filtered = filtered.filter((v) => v.gender === genderFilter);
    }

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (v) =>
          v.name.toLowerCase().includes(query) ||
          v.description.toLowerCase().includes(query) ||
          v.tags.some((t) => t.includes(query))
      );
    }

    return filtered;
  }, [voices, selectedCategory, genderFilter, searchQuery]);

  return (
    <DashboardLayout>
      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar */}
        <div className="w-64 border-r border-white/5 bg-[#0d0d1a]/50 flex flex-col">
          <div className="p-4 border-b border-white/5">
            <button
              onClick={() => setShowCreateDialog(true)}
              className="w-full px-4 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
            >
              <Plus className="h-4 w-4" />
              Create Voice
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-2">
            {voiceCategories.map((category) => (
              <button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-all ${
                  selectedCategory === category.id
                    ? "bg-purple-500/10 text-purple-400"
                    : "text-gray-400 hover:bg-white/5 hover:text-white"
                }`}
              >
                <div className="flex items-center gap-2">
                  <category.icon className="h-4 w-4" />
                  <span className="text-sm">{category.name}</span>
                </div>
                <span className="text-xs text-gray-500">{category.count}</span>
              </button>
            ))}
          </div>

          {/* Stats */}
          <div className="p-4 border-t border-white/5">
            <div className="grid grid-cols-2 gap-2">
              <div className="p-3 bg-white/5 rounded-lg">
                <div className="text-lg font-bold text-white">{voices.length}</div>
                <div className="text-xs text-gray-500">Total Voices</div>
              </div>
              <div className="p-3 bg-white/5 rounded-lg">
                <div className="text-lg font-bold text-purple-400">
                  {voices.filter((v) => v.isCustom).length}
                </div>
                <div className="text-xs text-gray-500">Custom</div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <div className="px-6 py-4 border-b border-white/5">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-xl font-semibold text-white">Voice Library</h1>
                <p className="text-sm text-gray-400">
                  {filteredVoices.length} voice{filteredVoices.length !== 1 ? "s" : ""} available
                </p>
              </div>
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
                  <input
                    type="text"
                    placeholder="Search voices..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 w-64"
                  />
                </div>
                <select
                  value={genderFilter}
                  onChange={(e) => setGenderFilter(e.target.value as VoiceGender | "all")}
                  className="px-4 py-2 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Genders</option>
                  <option value="female">Female</option>
                  <option value="male">Male</option>
                  <option value="neutral">Neutral</option>
                </select>
                <div className="flex items-center bg-white/5 rounded-xl p-1">
                  <button
                    onClick={() => setViewMode("grid")}
                    className={`p-2 rounded-lg transition-colors ${
                      viewMode === "grid" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:text-white"
                    }`}
                  >
                    <LayoutGrid className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => setViewMode("list")}
                    className={`p-2 rounded-lg transition-colors ${
                      viewMode === "list" ? "bg-purple-500/20 text-purple-400" : "text-gray-400 hover:text-white"
                    }`}
                  >
                    <List className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Voice Grid */}
              <div className={selectedVoice ? "lg:col-span-2" : "lg:col-span-3"}>
                {filteredVoices.length === 0 ? (
                  <div className="bg-[#1a1a2e]/80 rounded-xl border border-white/5 p-12 text-center">
                    <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mx-auto mb-4">
                      <Mic className="h-8 w-8 text-purple-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">No voices found</h3>
                    <p className="text-gray-400 mb-6">
                      {searchQuery || genderFilter !== "all"
                        ? "Try adjusting your filters."
                        : "Create your first custom voice to get started."}
                    </p>
                    <button
                      onClick={() => setShowCreateDialog(true)}
                      className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl text-white font-medium hover:opacity-90 transition-opacity inline-flex items-center gap-2"
                    >
                      <Plus className="h-4 w-4" />
                      Create Voice
                    </button>
                  </div>
                ) : (
                  <div className={viewMode === "grid" ? "grid grid-cols-1 md:grid-cols-2 gap-4" : "space-y-4"}>
                    {filteredVoices.map((voice) => (
                      <VoiceCard
                        key={voice.id}
                        voice={voice}
                        onPlay={() =>
                          setPlayingVoiceId(playingVoiceId === voice.id ? null : voice.id)
                        }
                        onSelect={() => setSelectedVoice(voice)}
                        onToggleFavorite={() => {}}
                        isPlaying={playingVoiceId === voice.id}
                        isSelected={selectedVoice?.id === voice.id}
                      />
                    ))}
                  </div>
                )}
              </div>

              {/* Preview Panel */}
              {selectedVoice && (
                <div className="lg:col-span-1">
                  <VoicePreviewPanel
                    voice={selectedVoice}
                    onClose={() => setSelectedVoice(null)}
                  />
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Create Voice Dialog */}
        <CreateVoiceDialog
          isOpen={showCreateDialog}
          onClose={() => setShowCreateDialog(false)}
        />
      </div>
    </DashboardLayout>
  );
}
