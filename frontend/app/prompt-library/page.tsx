"use client";

import React, { useState, useMemo } from "react";
import DashboardLayout from "../components/DashboardLayout";

// Types
type PromptCategory = "greeting" | "question" | "objection" | "closing" | "transfer" | "fallback" | "confirmation" | "upsell";
type PromptTone = "professional" | "friendly" | "empathetic" | "urgent" | "casual" | "formal";
type PromptStatus = "active" | "draft" | "archived";

interface PromptVariable {
  name: string;
  description: string;
  type: "string" | "number" | "boolean" | "date";
  defaultValue?: string;
  example: string;
}

interface PromptTemplate {
  id: string;
  title: string;
  content: string;
  category: PromptCategory;
  tone: PromptTone;
  status: PromptStatus;
  language: string;
  variables: PromptVariable[];
  tags: string[];
  usageCount: number;
  rating: number;
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  isPublic: boolean;
  isFavorite: boolean;
}

interface PromptCollection {
  id: string;
  name: string;
  description: string;
  prompts: string[];
  thumbnail: string;
  createdAt: string;
  isPublic: boolean;
}

interface PromptMetrics {
  totalPrompts: number;
  activePrompts: number;
  totalUsage: number;
  avgRating: number;
  topCategory: string;
  recentlyAdded: number;
}

// Mock Data
const mockPrompts: PromptTemplate[] = [
  {
    id: "prompt-1",
    title: "Professional Welcome Greeting",
    content: "Good {{time_of_day}}, thank you for calling {{company_name}}. My name is {{agent_name}}, and I'm delighted to assist you today. How may I help you?",
    category: "greeting",
    tone: "professional",
    status: "active",
    language: "en-US",
    variables: [
      { name: "time_of_day", description: "Morning, afternoon, or evening", type: "string", example: "morning" },
      { name: "company_name", description: "Your company name", type: "string", example: "BVRAI" },
      { name: "agent_name", description: "AI agent's name", type: "string", example: "Sarah" },
    ],
    tags: ["welcome", "professional", "customer-service"],
    usageCount: 15420,
    rating: 4.8,
    createdAt: "2024-01-01T00:00:00Z",
    updatedAt: "2024-01-15T00:00:00Z",
    createdBy: "BVRAI Team",
    isPublic: true,
    isFavorite: true,
  },
  {
    id: "prompt-2",
    title: "Friendly Casual Greeting",
    content: "Hey there! Thanks for calling {{company_name}}! I'm {{agent_name}}, and I'm super happy to help you out today. What can I do for you?",
    category: "greeting",
    tone: "friendly",
    status: "active",
    language: "en-US",
    variables: [
      { name: "company_name", description: "Your company name", type: "string", example: "BVRAI" },
      { name: "agent_name", description: "AI agent's name", type: "string", example: "Alex" },
    ],
    tags: ["welcome", "casual", "friendly"],
    usageCount: 8750,
    rating: 4.6,
    createdAt: "2024-01-05T00:00:00Z",
    updatedAt: "2024-01-18T00:00:00Z",
    createdBy: "BVRAI Team",
    isPublic: true,
    isFavorite: false,
  },
  {
    id: "prompt-3",
    title: "Account Verification Question",
    content: "For security purposes, I'll need to verify your identity. Could you please provide me with your {{verification_type}} associated with your account?",
    category: "question",
    tone: "professional",
    status: "active",
    language: "en-US",
    variables: [
      { name: "verification_type", description: "Type of verification info", type: "string", example: "phone number or email address" },
    ],
    tags: ["security", "verification", "account"],
    usageCount: 12300,
    rating: 4.7,
    createdAt: "2024-01-02T00:00:00Z",
    updatedAt: "2024-01-10T00:00:00Z",
    createdBy: "BVRAI Team",
    isPublic: true,
    isFavorite: true,
  },
  {
    id: "prompt-4",
    title: "Price Objection Handler",
    content: "I completely understand your concern about the price, {{customer_name}}. Many of our customers initially felt the same way. However, when you consider {{value_proposition}}, most find that the investment pays for itself within {{timeframe}}. Would you like me to walk you through some specific examples?",
    category: "objection",
    tone: "empathetic",
    status: "active",
    language: "en-US",
    variables: [
      { name: "customer_name", description: "Customer's name", type: "string", example: "John" },
      { name: "value_proposition", description: "Key value points", type: "string", example: "the time savings and increased efficiency" },
      { name: "timeframe", description: "ROI timeframe", type: "string", example: "3 months" },
    ],
    tags: ["sales", "objection", "price"],
    usageCount: 6540,
    rating: 4.5,
    createdAt: "2024-01-08T00:00:00Z",
    updatedAt: "2024-01-16T00:00:00Z",
    createdBy: "Sales Team",
    isPublic: true,
    isFavorite: false,
  },
  {
    id: "prompt-5",
    title: "Successful Sale Closing",
    content: "Wonderful! I'm excited to welcome you to {{company_name}}, {{customer_name}}! Let me quickly confirm your order: {{order_details}}. Your total comes to {{total_amount}}. Should I proceed with processing this for you?",
    category: "closing",
    tone: "professional",
    status: "active",
    language: "en-US",
    variables: [
      { name: "company_name", description: "Your company name", type: "string", example: "BVRAI" },
      { name: "customer_name", description: "Customer's name", type: "string", example: "Sarah" },
      { name: "order_details", description: "Summary of the order", type: "string", example: "the Professional plan with annual billing" },
      { name: "total_amount", description: "Total price", type: "string", example: "$199 per month" },
    ],
    tags: ["sales", "closing", "confirmation"],
    usageCount: 4320,
    rating: 4.9,
    createdAt: "2024-01-10T00:00:00Z",
    updatedAt: "2024-01-19T00:00:00Z",
    createdBy: "Sales Team",
    isPublic: true,
    isFavorite: true,
  },
  {
    id: "prompt-6",
    title: "Transfer to Human Agent",
    content: "I appreciate your patience, {{customer_name}}. To better assist you with this matter, I'm going to connect you with one of our specialized {{department}} representatives. They'll be with you in just a moment. Is there anything else you'd like me to relay to them?",
    category: "transfer",
    tone: "professional",
    status: "active",
    language: "en-US",
    variables: [
      { name: "customer_name", description: "Customer's name", type: "string", example: "Mike" },
      { name: "department", description: "Department name", type: "string", example: "technical support" },
    ],
    tags: ["transfer", "escalation", "handoff"],
    usageCount: 9870,
    rating: 4.6,
    createdAt: "2024-01-03T00:00:00Z",
    updatedAt: "2024-01-12T00:00:00Z",
    createdBy: "BVRAI Team",
    isPublic: true,
    isFavorite: false,
  },
  {
    id: "prompt-7",
    title: "Empathetic Fallback Response",
    content: "I want to make sure I understand you correctly. It sounds like you're asking about {{interpreted_topic}}. Is that right, or would you like to explain it a different way? I'm here to help.",
    category: "fallback",
    tone: "empathetic",
    status: "active",
    language: "en-US",
    variables: [
      { name: "interpreted_topic", description: "What the AI understood", type: "string", example: "changing your subscription plan" },
    ],
    tags: ["fallback", "clarification", "understanding"],
    usageCount: 11200,
    rating: 4.4,
    createdAt: "2024-01-04T00:00:00Z",
    updatedAt: "2024-01-14T00:00:00Z",
    createdBy: "BVRAI Team",
    isPublic: true,
    isFavorite: false,
  },
  {
    id: "prompt-8",
    title: "Appointment Confirmation",
    content: "Perfect! I've scheduled your {{appointment_type}} appointment for {{date}} at {{time}}. You'll receive a confirmation {{notification_method}} shortly. Is there anything else I can help you with today?",
    category: "confirmation",
    tone: "professional",
    status: "active",
    language: "en-US",
    variables: [
      { name: "appointment_type", description: "Type of appointment", type: "string", example: "consultation" },
      { name: "date", description: "Appointment date", type: "date", example: "January 25th" },
      { name: "time", description: "Appointment time", type: "string", example: "2:00 PM" },
      { name: "notification_method", description: "How confirmation is sent", type: "string", example: "email" },
    ],
    tags: ["appointment", "confirmation", "scheduling"],
    usageCount: 7650,
    rating: 4.7,
    createdAt: "2024-01-06T00:00:00Z",
    updatedAt: "2024-01-17T00:00:00Z",
    createdBy: "BVRAI Team",
    isPublic: true,
    isFavorite: true,
  },
  {
    id: "prompt-9",
    title: "Upsell Premium Features",
    content: "By the way, {{customer_name}}, I noticed you might benefit from our {{premium_feature}}. It's included in our {{plan_name}} and could help you {{benefit_description}}. Would you like to hear more about it?",
    category: "upsell",
    tone: "friendly",
    status: "active",
    language: "en-US",
    variables: [
      { name: "customer_name", description: "Customer's name", type: "string", example: "Lisa" },
      { name: "premium_feature", description: "Feature to upsell", type: "string", example: "advanced analytics dashboard" },
      { name: "plan_name", description: "Higher tier plan", type: "string", example: "Professional plan" },
      { name: "benefit_description", description: "Key benefit", type: "string", example: "track your ROI in real-time" },
    ],
    tags: ["sales", "upsell", "premium"],
    usageCount: 3890,
    rating: 4.3,
    createdAt: "2024-01-11T00:00:00Z",
    updatedAt: "2024-01-18T00:00:00Z",
    createdBy: "Sales Team",
    isPublic: true,
    isFavorite: false,
  },
  {
    id: "prompt-10",
    title: "Urgent Issue Acknowledgment",
    content: "I understand this is urgent and affecting your business, {{customer_name}}. I want to assure you that I'm treating this as a priority. Let me {{immediate_action}} right away, and we'll get this resolved as quickly as possible.",
    category: "fallback",
    tone: "urgent",
    status: "active",
    language: "en-US",
    variables: [
      { name: "customer_name", description: "Customer's name", type: "string", example: "David" },
      { name: "immediate_action", description: "What you'll do immediately", type: "string", example: "escalate this to our senior technical team" },
    ],
    tags: ["urgent", "support", "priority"],
    usageCount: 5240,
    rating: 4.8,
    createdAt: "2024-01-09T00:00:00Z",
    updatedAt: "2024-01-20T00:00:00Z",
    createdBy: "Support Team",
    isPublic: true,
    isFavorite: true,
  },
];

const mockCollections: PromptCollection[] = [
  {
    id: "coll-1",
    name: "Sales Essentials",
    description: "Core prompts for sales calls including greetings, objection handling, and closing",
    prompts: ["prompt-1", "prompt-4", "prompt-5", "prompt-9"],
    thumbnail: "💰",
    createdAt: "2024-01-05T00:00:00Z",
    isPublic: true,
  },
  {
    id: "coll-2",
    name: "Customer Support Pack",
    description: "Everything you need for handling support inquiries",
    prompts: ["prompt-1", "prompt-3", "prompt-6", "prompt-7", "prompt-10"],
    thumbnail: "🎧",
    createdAt: "2024-01-08T00:00:00Z",
    isPublic: true,
  },
  {
    id: "coll-3",
    name: "Appointment Scheduling",
    description: "Prompts for booking and managing appointments",
    prompts: ["prompt-2", "prompt-8"],
    thumbnail: "📅",
    createdAt: "2024-01-10T00:00:00Z",
    isPublic: true,
  },
];

const mockMetrics: PromptMetrics = {
  totalPrompts: 156,
  activePrompts: 142,
  totalUsage: 284500,
  avgRating: 4.6,
  topCategory: "greeting",
  recentlyAdded: 12,
};

const categories: { value: PromptCategory; label: string; icon: string; color: string }[] = [
  { value: "greeting", label: "Greeting", icon: "👋", color: "green" },
  { value: "question", label: "Question", icon: "❔", color: "blue" },
  { value: "objection", label: "Objection", icon: "🛡️", color: "orange" },
  { value: "closing", label: "Closing", icon: "🎯", color: "purple" },
  { value: "transfer", label: "Transfer", icon: "📲", color: "cyan" },
  { value: "fallback", label: "Fallback", icon: "🔄", color: "yellow" },
  { value: "confirmation", label: "Confirmation", icon: "✅", color: "green" },
  { value: "upsell", label: "Upsell", icon: "📈", color: "pink" },
];

const tones: { value: PromptTone; label: string }[] = [
  { value: "professional", label: "Professional" },
  { value: "friendly", label: "Friendly" },
  { value: "empathetic", label: "Empathetic" },
  { value: "urgent", label: "Urgent" },
  { value: "casual", label: "Casual" },
  { value: "formal", label: "Formal" },
];

// Components
const CategoryBadge: React.FC<{ category: PromptCategory }> = ({ category }) => {
  const cat = categories.find((c) => c.value === category);
  if (!cat) return null;

  const colorClasses: Record<string, string> = {
    green: "bg-green-500/20 text-green-400 border-green-500/30",
    blue: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    orange: "bg-orange-500/20 text-orange-400 border-orange-500/30",
    purple: "bg-purple-500/20 text-purple-400 border-purple-500/30",
    cyan: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
    yellow: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    pink: "bg-pink-500/20 text-pink-400 border-pink-500/30",
  };

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${colorClasses[cat.color]}`}>
      <span>{cat.icon}</span>
      <span>{cat.label}</span>
    </span>
  );
};

const ToneBadge: React.FC<{ tone: PromptTone }> = ({ tone }) => (
  <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-gray-700/50 text-gray-300 capitalize">
    {tone}
  </span>
);

const StatusBadge: React.FC<{ status: PromptStatus }> = ({ status }) => {
  const config: Record<PromptStatus, { color: string; label: string }> = {
    active: { color: "bg-green-500/20 text-green-400 border-green-500/30", label: "Active" },
    draft: { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", label: "Draft" },
    archived: { color: "bg-gray-500/20 text-gray-400 border-gray-500/30", label: "Archived" },
  };

  const { color, label } = config[status];

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${color}`}>{label}</span>
  );
};

const RatingStars: React.FC<{ rating: number }> = ({ rating }) => (
  <div className="flex items-center gap-1">
    {[1, 2, 3, 4, 5].map((star) => (
      <span key={star} className={star <= Math.round(rating) ? "text-yellow-400" : "text-gray-600"}>
        ★
      </span>
    ))}
    <span className="text-gray-400 text-sm ml-1">{rating.toFixed(1)}</span>
  </div>
);

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  icon: string;
  trend?: number;
  subtitle?: string;
}> = ({ title, value, icon, trend, subtitle }) => (
  <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
    <div className="flex items-start justify-between">
      <div>
        <p className="text-gray-400 text-sm">{title}</p>
        <p className="text-2xl font-bold text-white mt-1">{value}</p>
        {subtitle && <p className="text-gray-500 text-xs mt-1">{subtitle}</p>}
        {trend !== undefined && (
          <div className={`flex items-center gap-1 mt-2 text-sm ${trend >= 0 ? "text-green-400" : "text-red-400"}`}>
            <span>{trend >= 0 ? "↑" : "↓"}</span>
            <span>{Math.abs(trend)}% this month</span>
          </div>
        )}
      </div>
      <div className="w-12 h-12 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center text-2xl">
        {icon}
      </div>
    </div>
  </div>
);

const PromptCard: React.FC<{
  prompt: PromptTemplate;
  onSelect: (prompt: PromptTemplate) => void;
  onToggleFavorite: (id: string) => void;
}> = ({ prompt, onSelect, onToggleFavorite }) => {
  const highlightVariables = (content: string) => {
    return content.replace(/\{\{(\w+)\}\}/g, '<span class="text-purple-400 bg-purple-500/20 px-1 rounded">{{$1}}</span>');
  };

  return (
    <div
      className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5 hover:border-purple-500/50 transition-all duration-300 cursor-pointer"
      onClick={() => onSelect(prompt)}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2 flex-wrap">
          <CategoryBadge category={prompt.category} />
          <ToneBadge tone={prompt.tone} />
          <StatusBadge status={prompt.status} />
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggleFavorite(prompt.id);
          }}
          className={`p-1 rounded transition-colors ${
            prompt.isFavorite ? "text-yellow-400" : "text-gray-500 hover:text-yellow-400"
          }`}
        >
          {prompt.isFavorite ? "★" : "☆"}
        </button>
      </div>

      <h3 className="font-semibold text-white mb-2">{prompt.title}</h3>
      <p
        className="text-gray-400 text-sm line-clamp-3 mb-3"
        dangerouslySetInnerHTML={{ __html: highlightVariables(prompt.content) }}
      />

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4 text-sm text-gray-500">
          <span className="flex items-center gap-1">
            <span>📊</span>
            <span>{prompt.usageCount.toLocaleString()}</span>
          </span>
          <RatingStars rating={prompt.rating} />
        </div>
        <span className="text-xs text-gray-500">{prompt.variables.length} variables</span>
      </div>

      {prompt.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-3">
          {prompt.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="px-2 py-0.5 bg-gray-700/50 text-gray-400 text-xs rounded-full">
              #{tag}
            </span>
          ))}
          {prompt.tags.length > 3 && (
            <span className="text-gray-500 text-xs">+{prompt.tags.length - 3}</span>
          )}
        </div>
      )}
    </div>
  );
};

const CollectionCard: React.FC<{
  collection: PromptCollection;
  promptCount: number;
}> = ({ collection, promptCount }) => (
  <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5 hover:border-purple-500/50 transition-all duration-300 cursor-pointer">
    <div className="flex items-start gap-4">
      <div className="w-14 h-14 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center text-3xl">
        {collection.thumbnail}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <h3 className="font-semibold text-white">{collection.name}</h3>
          {collection.isPublic && (
            <span className="px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded-full">Public</span>
          )}
        </div>
        <p className="text-gray-400 text-sm">{collection.description}</p>
        <p className="text-xs text-gray-500 mt-2">{promptCount} prompts</p>
      </div>
    </div>
  </div>
);

const PromptDetailDialog: React.FC<{
  prompt: PromptTemplate | null;
  onClose: () => void;
  onUse: (prompt: PromptTemplate) => void;
}> = ({ prompt, onClose, onUse }) => {
  const [previewContent, setPreviewContent] = useState("");
  const [variableValues, setVariableValues] = useState<Record<string, string>>({});

  const generatePreview = () => {
    if (!prompt) return "";
    let content = prompt.content;
    prompt.variables.forEach((v) => {
      const value = variableValues[v.name] || v.example;
      content = content.replace(new RegExp(`\\{\\{${v.name}\\}\\}`, "g"), value);
    });
    return content;
  };

  if (!prompt) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-3xl max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <CategoryBadge category={prompt.category} />
                <ToneBadge tone={prompt.tone} />
                <StatusBadge status={prompt.status} />
              </div>
              <h2 className="text-xl font-bold text-white">{prompt.title}</h2>
              <div className="flex items-center gap-4 mt-2 text-sm text-gray-400">
                <span>By {prompt.createdBy}</span>
                <span>•</span>
                <span>{prompt.usageCount.toLocaleString()} uses</span>
                <span>•</span>
                <RatingStars rating={prompt.rating} />
              </div>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-250px)]">
          {/* Template Content */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-2">Template</h3>
            <div className="bg-gray-800/50 rounded-lg p-4">
              <p className="text-gray-300 whitespace-pre-wrap">{prompt.content}</p>
            </div>
          </div>

          {/* Variables */}
          {prompt.variables.length > 0 && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Variables</h3>
              <div className="space-y-3">
                {prompt.variables.map((variable) => (
                  <div key={variable.name} className="bg-gray-800/50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <code className="text-purple-400">{`{{${variable.name}}}`}</code>
                      <span className="text-xs text-gray-500">{variable.type}</span>
                    </div>
                    <p className="text-sm text-gray-400 mb-2">{variable.description}</p>
                    <input
                      type="text"
                      placeholder={`e.g., ${variable.example}`}
                      value={variableValues[variable.name] || ""}
                      onChange={(e) => setVariableValues({ ...variableValues, [variable.name]: e.target.value })}
                      className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 text-sm focus:outline-none focus:border-purple-500"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Preview */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-2">Preview</h3>
            <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-4">
              <p className="text-purple-100">{generatePreview()}</p>
            </div>
          </div>

          {/* Tags */}
          {prompt.tags.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-2">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {prompt.tags.map((tag) => (
                  <span key={tag} className="px-3 py-1 bg-gray-700/50 text-gray-300 text-sm rounded-full">
                    #{tag}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors flex items-center gap-2">
                <span>📋</span>
                <span>Copy</span>
              </button>
              <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors flex items-center gap-2">
                <span>✏️</span>
                <span>Edit</span>
              </button>
            </div>
            <div className="flex items-center gap-3">
              <button onClick={onClose} className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
                Cancel
              </button>
              <button
                onClick={() => onUse(prompt)}
                className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity"
              >
                Use This Prompt
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const CreatePromptDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
}> = ({ isOpen, onClose }) => {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [category, setCategory] = useState<PromptCategory>("greeting");
  const [tone, setTone] = useState<PromptTone>("professional");
  const [tags, setTags] = useState("");

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">Create New Prompt</h2>
            <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Title</label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g., Professional Welcome Greeting"
                className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">Category</label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value as PromptCategory)}
                  className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  {categories.map((cat) => (
                    <option key={cat.value} value={cat.value}>
                      {cat.icon} {cat.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">Tone</label>
                <select
                  value={tone}
                  onChange={(e) => setTone(e.target.value as PromptTone)}
                  className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  {tones.map((t) => (
                    <option key={t.value} value={t.value}>{t.label}</option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">
                Content
                <span className="text-gray-500 font-normal ml-2">Use {"{{variable_name}}"} for dynamic content</span>
              </label>
              <textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="Good {{time_of_day}}, thank you for calling {{company_name}}..."
                rows={6}
                className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Tags (comma separated)</label>
              <input
                type="text"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                placeholder="welcome, professional, customer-service"
                className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>
        </div>

        <div className="p-6 border-t border-gray-700">
          <div className="flex items-center justify-end gap-3">
            <button onClick={onClose} className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
              Cancel
            </button>
            <button className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
              Create Prompt
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Component
export default function PromptLibraryPage() {
  const [activeTab, setActiveTab] = useState<"library" | "collections" | "favorites">("library");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<PromptCategory | "all">("all");
  const [selectedTone, setSelectedTone] = useState<PromptTone | "all">("all");
  const [selectedPrompt, setSelectedPrompt] = useState<PromptTemplate | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [prompts, setPrompts] = useState(mockPrompts);

  const filteredPrompts = useMemo(() => {
    return prompts.filter((prompt) => {
      const matchesSearch =
        prompt.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        prompt.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
        prompt.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()));
      const matchesCategory = selectedCategory === "all" || prompt.category === selectedCategory;
      const matchesTone = selectedTone === "all" || prompt.tone === selectedTone;
      const matchesTab = activeTab !== "favorites" || prompt.isFavorite;
      return matchesSearch && matchesCategory && matchesTone && matchesTab;
    });
  }, [prompts, searchQuery, selectedCategory, selectedTone, activeTab]);

  const handleToggleFavorite = (id: string) => {
    setPrompts((prev) =>
      prev.map((p) => (p.id === id ? { ...p, isFavorite: !p.isFavorite } : p))
    );
  };

  const tabs = [
    { id: "library", label: "All Prompts", icon: "📚" },
    { id: "collections", label: "Collections", icon: "📁" },
    { id: "favorites", label: "Favorites", icon: "⭐" },
  ];

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-900">
        {/* Header */}
        <div className="border-b border-gray-800 bg-gray-900/95 backdrop-blur-sm sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-2xl font-bold text-white">Prompt Library</h1>
                <p className="text-gray-400 mt-1">Discover and manage conversation prompts for your AI agents</p>
              </div>
              <button
                onClick={() => setShowCreateDialog(true)}
                className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
              >
                <span>+</span>
                <span>Create Prompt</span>
              </button>
            </div>

            {/* Tabs */}
            <div className="flex items-center gap-2">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg whitespace-nowrap transition-colors ${
                    activeTab === tab.id
                      ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                      : "text-gray-400 hover:text-white hover:bg-gray-800"
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Library & Favorites */}
          {(activeTab === "library" || activeTab === "favorites") && (
            <div className="space-y-6">
              {/* Metrics */}
              {activeTab === "library" && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <MetricCard title="Total Prompts" value={mockMetrics.totalPrompts} icon="📚" trend={8} />
                  <MetricCard title="Total Usage" value={`${(mockMetrics.totalUsage / 1000).toFixed(0)}K`} icon="📊" trend={15} />
                  <MetricCard title="Avg Rating" value={mockMetrics.avgRating.toFixed(1)} icon="⭐" />
                  <MetricCard title="Recently Added" value={mockMetrics.recentlyAdded} icon="✨" subtitle="This week" />
                </div>
              )}

              {/* Filters */}
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex-1 min-w-[200px]">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search prompts..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full px-4 py-2 pl-10 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                    />
                    <svg className="w-5 h-5 text-gray-500 absolute left-3 top-1/2 -translate-y-1/2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                </div>

                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value as PromptCategory | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Categories</option>
                  {categories.map((cat) => (
                    <option key={cat.value} value={cat.value}>
                      {cat.icon} {cat.label}
                    </option>
                  ))}
                </select>

                <select
                  value={selectedTone}
                  onChange={(e) => setSelectedTone(e.target.value as PromptTone | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Tones</option>
                  {tones.map((tone) => (
                    <option key={tone.value} value={tone.value}>{tone.label}</option>
                  ))}
                </select>
              </div>

              {/* Category Quick Filters */}
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setSelectedCategory("all")}
                  className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    selectedCategory === "all"
                      ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                      : "bg-gray-800/50 text-gray-400 hover:text-white"
                  }`}
                >
                  All
                </button>
                {categories.map((cat) => (
                  <button
                    key={cat.value}
                    onClick={() => setSelectedCategory(cat.value)}
                    className={`px-3 py-1.5 text-sm rounded-lg transition-colors flex items-center gap-1 ${
                      selectedCategory === cat.value
                        ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                        : "bg-gray-800/50 text-gray-400 hover:text-white"
                    }`}
                  >
                    <span>{cat.icon}</span>
                    <span>{cat.label}</span>
                  </button>
                ))}
              </div>

              {/* Results Count */}
              <p className="text-sm text-gray-400">
                Showing {filteredPrompts.length} of {prompts.length} prompts
              </p>

              {/* Prompts Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredPrompts.map((prompt) => (
                  <PromptCard
                    key={prompt.id}
                    prompt={prompt}
                    onSelect={setSelectedPrompt}
                    onToggleFavorite={handleToggleFavorite}
                  />
                ))}
              </div>

              {filteredPrompts.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-gray-400">No prompts found matching your criteria</p>
                </div>
              )}
            </div>
          )}

          {/* Collections */}
          {activeTab === "collections" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-400">{mockCollections.length} collections</p>
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2">
                  <span>+</span>
                  <span>Create Collection</span>
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {mockCollections.map((collection) => (
                  <CollectionCard
                    key={collection.id}
                    collection={collection}
                    promptCount={collection.prompts.length}
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Dialogs */}
        <PromptDetailDialog
          prompt={selectedPrompt}
          onClose={() => setSelectedPrompt(null)}
          onUse={(prompt) => {
            console.log("Using prompt:", prompt);
            setSelectedPrompt(null);
          }}
        />
        <CreatePromptDialog isOpen={showCreateDialog} onClose={() => setShowCreateDialog(false)} />
      </div>
    </DashboardLayout>
  );
}
