"use client";

import React, { useState, useCallback, useMemo } from "react";
import DashboardLayout from "@/components/DashboardLayout";

// Types
type NodeType = "start" | "greeting" | "question" | "response" | "condition" | "transfer" | "end" | "api" | "wait" | "variable" | "loop" | "webhook";
type ConnectionType = "default" | "yes" | "no" | "timeout" | "error" | "success" | "fallback";
type AgentStatus = "draft" | "testing" | "active" | "paused";
type VoiceGender = "male" | "female" | "neutral";

interface NodeConnection {
  id: string;
  sourceNodeId: string;
  targetNodeId: string;
  type: ConnectionType;
  label?: string;
}

interface NodePosition {
  x: number;
  y: number;
}

interface FlowNode {
  id: string;
  type: NodeType;
  label: string;
  position: NodePosition;
  data: Record<string, any>;
  connections: string[];
}

interface AgentVoice {
  id: string;
  name: string;
  language: string;
  gender: VoiceGender;
  preview?: string;
  accent?: string;
}

interface AgentTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  thumbnail: string;
  nodes: FlowNode[];
  connections: NodeConnection[];
  popularity: number;
}

interface AgentConfig {
  id: string;
  name: string;
  description: string;
  status: AgentStatus;
  voice: AgentVoice;
  language: string;
  greeting: string;
  fallbackMessage: string;
  maxRetries: number;
  timeout: number;
  recordingEnabled: boolean;
  transcriptionEnabled: boolean;
  nodes: FlowNode[];
  connections: NodeConnection[];
  variables: { name: string; type: string; defaultValue: string }[];
  createdAt: string;
  updatedAt: string;
  version: number;
}

interface NodeCategory {
  id: string;
  name: string;
  icon: string;
  nodes: { type: NodeType; name: string; description: string; icon: string }[];
}

// Mock Data
const mockVoices: AgentVoice[] = [
  { id: "voice-1", name: "Sarah", language: "en-US", gender: "female", accent: "American" },
  { id: "voice-2", name: "James", language: "en-US", gender: "male", accent: "American" },
  { id: "voice-3", name: "Emma", language: "en-GB", gender: "female", accent: "British" },
  { id: "voice-4", name: "Oliver", language: "en-GB", gender: "male", accent: "British" },
  { id: "voice-5", name: "Maria", language: "es-ES", gender: "female", accent: "Spanish" },
  { id: "voice-6", name: "Hans", language: "de-DE", gender: "male", accent: "German" },
  { id: "voice-7", name: "Yuki", language: "ja-JP", gender: "female", accent: "Japanese" },
  { id: "voice-8", name: "Wei", language: "zh-CN", gender: "neutral", accent: "Mandarin" },
];

const mockTemplates: AgentTemplate[] = [
  {
    id: "template-1",
    name: "Customer Support Agent",
    description: "Handle customer inquiries, troubleshoot issues, and escalate when needed",
    category: "Support",
    thumbnail: "🎧",
    nodes: [],
    connections: [],
    popularity: 1250,
  },
  {
    id: "template-2",
    name: "Sales Outreach Agent",
    description: "Make outbound calls to prospects, qualify leads, and schedule demos",
    category: "Sales",
    thumbnail: "📞",
    nodes: [],
    connections: [],
    popularity: 980,
  },
  {
    id: "template-3",
    name: "Appointment Scheduler",
    description: "Book, reschedule, and confirm appointments with customers",
    category: "Scheduling",
    thumbnail: "📅",
    nodes: [],
    connections: [],
    popularity: 850,
  },
  {
    id: "template-4",
    name: "Survey Collector",
    description: "Conduct customer surveys and collect feedback",
    category: "Research",
    thumbnail: "📊",
    nodes: [],
    connections: [],
    popularity: 620,
  },
  {
    id: "template-5",
    name: "Order Status Agent",
    description: "Provide order status updates and tracking information",
    category: "E-commerce",
    thumbnail: "📦",
    nodes: [],
    connections: [],
    popularity: 540,
  },
  {
    id: "template-6",
    name: "Payment Reminder",
    description: "Send payment reminders and collect overdue payments",
    category: "Finance",
    thumbnail: "💳",
    nodes: [],
    connections: [],
    popularity: 480,
  },
];

const nodeCategories: NodeCategory[] = [
  {
    id: "flow",
    name: "Flow Control",
    icon: "🔀",
    nodes: [
      { type: "start", name: "Start", description: "Beginning of the conversation", icon: "▶️" },
      { type: "end", name: "End Call", description: "End the conversation", icon: "⏹️" },
      { type: "condition", name: "Condition", description: "Branch based on conditions", icon: "❓" },
      { type: "loop", name: "Loop", description: "Repeat a set of actions", icon: "🔄" },
      { type: "wait", name: "Wait", description: "Pause for a duration", icon: "⏳" },
    ],
  },
  {
    id: "conversation",
    name: "Conversation",
    icon: "💬",
    nodes: [
      { type: "greeting", name: "Greeting", description: "Greet the caller", icon: "👋" },
      { type: "question", name: "Ask Question", description: "Ask the user a question", icon: "❔" },
      { type: "response", name: "Response", description: "Speak a message", icon: "🗣️" },
    ],
  },
  {
    id: "actions",
    name: "Actions",
    icon: "⚡",
    nodes: [
      { type: "transfer", name: "Transfer", description: "Transfer to human agent", icon: "📲" },
      { type: "api", name: "API Call", description: "Make an external API call", icon: "🔌" },
      { type: "webhook", name: "Webhook", description: "Trigger a webhook", icon: "🪝" },
      { type: "variable", name: "Set Variable", description: "Store data in a variable", icon: "📝" },
    ],
  },
];

const mockAgent: AgentConfig = {
  id: "agent-1",
  name: "Customer Support Agent",
  description: "Handles customer inquiries and support requests",
  status: "draft",
  voice: mockVoices[0],
  language: "en-US",
  greeting: "Hello! Thank you for calling. How can I help you today?",
  fallbackMessage: "I'm sorry, I didn't quite understand that. Could you please repeat?",
  maxRetries: 3,
  timeout: 30,
  recordingEnabled: true,
  transcriptionEnabled: true,
  nodes: [
    { id: "node-1", type: "start", label: "Start", position: { x: 100, y: 50 }, data: {}, connections: ["node-2"] },
    { id: "node-2", type: "greeting", label: "Welcome Greeting", position: { x: 100, y: 150 }, data: { message: "Hello! Thank you for calling BVRAI support. How can I help you today?" }, connections: ["node-3"] },
    { id: "node-3", type: "question", label: "Get Issue Type", position: { x: 100, y: 250 }, data: { prompt: "Are you calling about a billing issue, technical support, or something else?", timeout: 10 }, connections: ["node-4", "node-5", "node-6"] },
    { id: "node-4", type: "response", label: "Billing Response", position: { x: -100, y: 380 }, data: { message: "I understand you have a billing question. Let me connect you with our billing team." }, connections: ["node-7"] },
    { id: "node-5", type: "response", label: "Tech Support Response", position: { x: 100, y: 380 }, data: { message: "I can help with technical issues. Can you describe the problem you're experiencing?" }, connections: ["node-8"] },
    { id: "node-6", type: "response", label: "Other Response", position: { x: 300, y: 380 }, data: { message: "I'll connect you with the appropriate team." }, connections: ["node-7"] },
    { id: "node-7", type: "transfer", label: "Transfer to Agent", position: { x: 100, y: 500 }, data: { department: "general", priority: "normal" }, connections: ["node-9"] },
    { id: "node-8", type: "api", label: "Check KB", position: { x: 100, y: 500 }, data: { endpoint: "/api/knowledge-base/search", method: "POST" }, connections: ["node-9"] },
    { id: "node-9", type: "end", label: "End Call", position: { x: 100, y: 620 }, data: { message: "Thank you for calling. Have a great day!" }, connections: [] },
  ],
  connections: [
    { id: "conn-1", sourceNodeId: "node-1", targetNodeId: "node-2", type: "default" },
    { id: "conn-2", sourceNodeId: "node-2", targetNodeId: "node-3", type: "default" },
    { id: "conn-3", sourceNodeId: "node-3", targetNodeId: "node-4", type: "yes", label: "Billing" },
    { id: "conn-4", sourceNodeId: "node-3", targetNodeId: "node-5", type: "no", label: "Technical" },
    { id: "conn-5", sourceNodeId: "node-3", targetNodeId: "node-6", type: "fallback", label: "Other" },
    { id: "conn-6", sourceNodeId: "node-4", targetNodeId: "node-7", type: "default" },
    { id: "conn-7", sourceNodeId: "node-5", targetNodeId: "node-8", type: "default" },
    { id: "conn-8", sourceNodeId: "node-6", targetNodeId: "node-7", type: "default" },
    { id: "conn-9", sourceNodeId: "node-7", targetNodeId: "node-9", type: "default" },
    { id: "conn-10", sourceNodeId: "node-8", targetNodeId: "node-9", type: "default" },
  ],
  variables: [
    { name: "customer_name", type: "string", defaultValue: "" },
    { name: "issue_type", type: "string", defaultValue: "" },
    { name: "account_id", type: "string", defaultValue: "" },
  ],
  createdAt: "2024-01-15T00:00:00Z",
  updatedAt: "2024-01-20T10:30:00Z",
  version: 3,
};

// Components
const StatusBadge: React.FC<{ status: AgentStatus }> = ({ status }) => {
  const config: Record<AgentStatus, { color: string; label: string }> = {
    draft: { color: "bg-gray-500/20 text-gray-400 border-gray-500/30", label: "Draft" },
    testing: { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", label: "Testing" },
    active: { color: "bg-green-500/20 text-green-400 border-green-500/30", label: "Active" },
    paused: { color: "bg-orange-500/20 text-orange-400 border-orange-500/30", label: "Paused" },
  };

  const { color, label } = config[status];

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${color}`}>{label}</span>
  );
};

const NodeTypeIcon: React.FC<{ type: NodeType; size?: "sm" | "md" | "lg" }> = ({ type, size = "md" }) => {
  const icons: Record<NodeType, string> = {
    start: "▶️",
    greeting: "👋",
    question: "❔",
    response: "🗣️",
    condition: "❓",
    transfer: "📲",
    end: "⏹️",
    api: "🔌",
    wait: "⏳",
    variable: "📝",
    loop: "🔄",
    webhook: "🪝",
  };

  const sizeClasses = { sm: "text-lg", md: "text-2xl", lg: "text-3xl" };

  return <span className={sizeClasses[size]}>{icons[type]}</span>;
};

const NodeTypeColors: Record<NodeType, string> = {
  start: "border-green-500/50 bg-green-500/10",
  greeting: "border-blue-500/50 bg-blue-500/10",
  question: "border-purple-500/50 bg-purple-500/10",
  response: "border-cyan-500/50 bg-cyan-500/10",
  condition: "border-yellow-500/50 bg-yellow-500/10",
  transfer: "border-orange-500/50 bg-orange-500/10",
  end: "border-red-500/50 bg-red-500/10",
  api: "border-pink-500/50 bg-pink-500/10",
  wait: "border-gray-500/50 bg-gray-500/10",
  variable: "border-indigo-500/50 bg-indigo-500/10",
  loop: "border-teal-500/50 bg-teal-500/10",
  webhook: "border-rose-500/50 bg-rose-500/10",
};

const DraggableNode: React.FC<{
  type: NodeType;
  name: string;
  description: string;
  icon: string;
}> = ({ type, name, description, icon }) => {
  const handleDragStart = (e: React.DragEvent) => {
    e.dataTransfer.setData("nodeType", type);
    e.dataTransfer.setData("nodeName", name);
  };

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      className={`p-3 rounded-lg border-2 cursor-grab active:cursor-grabbing transition-all hover:scale-105 ${NodeTypeColors[type]}`}
    >
      <div className="flex items-center gap-3">
        <span className="text-xl">{icon}</span>
        <div>
          <p className="font-medium text-white text-sm">{name}</p>
          <p className="text-xs text-gray-400">{description}</p>
        </div>
      </div>
    </div>
  );
};

const FlowNodeComponent: React.FC<{
  node: FlowNode;
  isSelected: boolean;
  onSelect: (node: FlowNode) => void;
  onDragStart: (e: React.DragEvent, node: FlowNode) => void;
}> = ({ node, isSelected, onSelect, onDragStart }) => {
  return (
    <div
      className={`absolute p-4 rounded-xl border-2 cursor-pointer transition-all min-w-[180px] ${NodeTypeColors[node.type]} ${
        isSelected ? "ring-2 ring-purple-500 ring-offset-2 ring-offset-gray-900" : ""
      }`}
      style={{ left: node.position.x, top: node.position.y }}
      onClick={() => onSelect(node)}
      draggable
      onDragStart={(e) => onDragStart(e, node)}
    >
      <div className="flex items-center gap-3 mb-2">
        <NodeTypeIcon type={node.type} size="sm" />
        <span className="font-medium text-white text-sm">{node.label}</span>
      </div>
      {node.data.message && (
        <p className="text-xs text-gray-400 line-clamp-2">{node.data.message}</p>
      )}
      {node.data.prompt && (
        <p className="text-xs text-gray-400 line-clamp-2">{node.data.prompt}</p>
      )}

      {/* Connection Points */}
      <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-gray-700 border-2 border-gray-500 rounded-full" />
      {node.type !== "start" && (
        <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-gray-700 border-2 border-gray-500 rounded-full" />
      )}
    </div>
  );
};

const TemplateCard: React.FC<{
  template: AgentTemplate;
  onSelect: (template: AgentTemplate) => void;
}> = ({ template, onSelect }) => (
  <div
    className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5 hover:border-purple-500/50 transition-all duration-300 cursor-pointer"
    onClick={() => onSelect(template)}
  >
    <div className="flex items-start gap-4">
      <div className="w-12 h-12 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center text-2xl">
        {template.thumbnail}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <h3 className="font-semibold text-white">{template.name}</h3>
          <span className="px-2 py-0.5 text-xs bg-gray-700/50 text-gray-400 rounded-full">
            {template.category}
          </span>
        </div>
        <p className="text-gray-400 text-sm">{template.description}</p>
        <p className="text-xs text-gray-500 mt-2">{template.popularity.toLocaleString()} uses</p>
      </div>
    </div>
  </div>
);

const VoiceSelector: React.FC<{
  voices: AgentVoice[];
  selectedVoice: AgentVoice;
  onSelect: (voice: AgentVoice) => void;
}> = ({ voices, selectedVoice, onSelect }) => (
  <div className="grid grid-cols-2 gap-3">
    {voices.map((voice) => (
      <div
        key={voice.id}
        onClick={() => onSelect(voice)}
        className={`p-3 rounded-lg border cursor-pointer transition-all ${
          selectedVoice.id === voice.id
            ? "border-purple-500 bg-purple-500/10"
            : "border-gray-700 hover:border-gray-600"
        }`}
      >
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg ${
            voice.gender === "female" ? "bg-pink-500/20" :
            voice.gender === "male" ? "bg-blue-500/20" :
            "bg-purple-500/20"
          }`}>
            {voice.gender === "female" ? "👩" : voice.gender === "male" ? "👨" : "🧑"}
          </div>
          <div>
            <p className="font-medium text-white">{voice.name}</p>
            <p className="text-xs text-gray-400">{voice.language} • {voice.accent}</p>
          </div>
        </div>
      </div>
    ))}
  </div>
);

const NodePropertiesPanel: React.FC<{
  node: FlowNode | null;
  onUpdate: (node: FlowNode) => void;
  onDelete: (nodeId: string) => void;
  onClose: () => void;
}> = ({ node, onUpdate, onDelete, onClose }) => {
  if (!node) return null;

  return (
    <div className="w-80 bg-gray-800/95 border-l border-gray-700 p-6 overflow-y-auto">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <NodeTypeIcon type={node.type} />
          <div>
            <h3 className="font-semibold text-white">{node.label}</h3>
            <p className="text-xs text-gray-400 capitalize">{node.type.replace("_", " ")}</p>
          </div>
        </div>
        <button onClick={onClose} className="p-1 hover:bg-gray-700 rounded">
          <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">Label</label>
          <input
            type="text"
            value={node.label}
            onChange={(e) => onUpdate({ ...node, label: e.target.value })}
            className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
          />
        </div>

        {(node.type === "greeting" || node.type === "response") && (
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Message</label>
            <textarea
              value={node.data.message || ""}
              onChange={(e) => onUpdate({ ...node, data: { ...node.data, message: e.target.value } })}
              rows={4}
              className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500 resize-none"
            />
          </div>
        )}

        {node.type === "question" && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Question Prompt</label>
              <textarea
                value={node.data.prompt || ""}
                onChange={(e) => onUpdate({ ...node, data: { ...node.data, prompt: e.target.value } })}
                rows={3}
                className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500 resize-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Timeout (seconds)</label>
              <input
                type="number"
                value={node.data.timeout || 10}
                onChange={(e) => onUpdate({ ...node, data: { ...node.data, timeout: parseInt(e.target.value) } })}
                className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
              />
            </div>
          </>
        )}

        {node.type === "condition" && (
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Condition Expression</label>
            <input
              type="text"
              value={node.data.condition || ""}
              onChange={(e) => onUpdate({ ...node, data: { ...node.data, condition: e.target.value } })}
              placeholder="e.g., {{intent}} == 'billing'"
              className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
            />
          </div>
        )}

        {node.type === "transfer" && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Department</label>
              <select
                value={node.data.department || "general"}
                onChange={(e) => onUpdate({ ...node, data: { ...node.data, department: e.target.value } })}
                className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
              >
                <option value="general">General Support</option>
                <option value="billing">Billing</option>
                <option value="technical">Technical Support</option>
                <option value="sales">Sales</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Priority</label>
              <select
                value={node.data.priority || "normal"}
                onChange={(e) => onUpdate({ ...node, data: { ...node.data, priority: e.target.value } })}
                className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
              >
                <option value="low">Low</option>
                <option value="normal">Normal</option>
                <option value="high">High</option>
                <option value="urgent">Urgent</option>
              </select>
            </div>
          </>
        )}

        {node.type === "api" && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Endpoint URL</label>
              <input
                type="text"
                value={node.data.endpoint || ""}
                onChange={(e) => onUpdate({ ...node, data: { ...node.data, endpoint: e.target.value } })}
                placeholder="https://api.example.com/endpoint"
                className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">HTTP Method</label>
              <select
                value={node.data.method || "GET"}
                onChange={(e) => onUpdate({ ...node, data: { ...node.data, method: e.target.value } })}
                className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
              >
                <option value="GET">GET</option>
                <option value="POST">POST</option>
                <option value="PUT">PUT</option>
                <option value="DELETE">DELETE</option>
              </select>
            </div>
          </>
        )}

        {node.type === "wait" && (
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Duration (seconds)</label>
            <input
              type="number"
              value={node.data.duration || 5}
              onChange={(e) => onUpdate({ ...node, data: { ...node.data, duration: parseInt(e.target.value) } })}
              className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
            />
          </div>
        )}

        {node.type === "variable" && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Variable Name</label>
              <input
                type="text"
                value={node.data.variableName || ""}
                onChange={(e) => onUpdate({ ...node, data: { ...node.data, variableName: e.target.value } })}
                placeholder="my_variable"
                className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Value</label>
              <input
                type="text"
                value={node.data.value || ""}
                onChange={(e) => onUpdate({ ...node, data: { ...node.data, value: e.target.value } })}
                placeholder="Value or expression"
                className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
              />
            </div>
          </>
        )}

        {node.type === "end" && (
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Closing Message</label>
            <textarea
              value={node.data.message || ""}
              onChange={(e) => onUpdate({ ...node, data: { ...node.data, message: e.target.value } })}
              rows={3}
              className="w-full px-3 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500 resize-none"
            />
          </div>
        )}

        <div className="pt-4 border-t border-gray-700">
          <button
            onClick={() => onDelete(node.id)}
            className="w-full px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
          >
            Delete Node
          </button>
        </div>
      </div>
    </div>
  );
};

const AgentSettingsPanel: React.FC<{
  agent: AgentConfig;
  onUpdate: (agent: AgentConfig) => void;
  voices: AgentVoice[];
}> = ({ agent, onUpdate, voices }) => {
  return (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-400 mb-2">Agent Name</label>
        <input
          type="text"
          value={agent.name}
          onChange={(e) => onUpdate({ ...agent, name: e.target.value })}
          className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-400 mb-2">Description</label>
        <textarea
          value={agent.description}
          onChange={(e) => onUpdate({ ...agent, description: e.target.value })}
          rows={2}
          className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500 resize-none"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-400 mb-2">Voice</label>
        <VoiceSelector voices={voices} selectedVoice={agent.voice} onSelect={(voice) => onUpdate({ ...agent, voice })} />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-400 mb-2">Default Greeting</label>
        <textarea
          value={agent.greeting}
          onChange={(e) => onUpdate({ ...agent, greeting: e.target.value })}
          rows={2}
          className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500 resize-none"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-400 mb-2">Fallback Message</label>
        <textarea
          value={agent.fallbackMessage}
          onChange={(e) => onUpdate({ ...agent, fallbackMessage: e.target.value })}
          rows={2}
          className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500 resize-none"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">Max Retries</label>
          <input
            type="number"
            value={agent.maxRetries}
            onChange={(e) => onUpdate({ ...agent, maxRetries: parseInt(e.target.value) })}
            className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">Timeout (sec)</label>
          <input
            type="number"
            value={agent.timeout}
            onChange={(e) => onUpdate({ ...agent, timeout: parseInt(e.target.value) })}
            className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
          />
        </div>
      </div>

      <div className="space-y-3">
        <label className="flex items-center justify-between">
          <span className="text-sm text-gray-400">Enable Recording</span>
          <button
            onClick={() => onUpdate({ ...agent, recordingEnabled: !agent.recordingEnabled })}
            className={`w-12 h-6 rounded-full relative transition-colors ${
              agent.recordingEnabled ? "bg-purple-500" : "bg-gray-700"
            }`}
          >
            <div className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
              agent.recordingEnabled ? "translate-x-6" : "translate-x-0.5"
            }`} />
          </button>
        </label>
        <label className="flex items-center justify-between">
          <span className="text-sm text-gray-400">Enable Transcription</span>
          <button
            onClick={() => onUpdate({ ...agent, transcriptionEnabled: !agent.transcriptionEnabled })}
            className={`w-12 h-6 rounded-full relative transition-colors ${
              agent.transcriptionEnabled ? "bg-purple-500" : "bg-gray-700"
            }`}
          >
            <div className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
              agent.transcriptionEnabled ? "translate-x-6" : "translate-x-0.5"
            }`} />
          </button>
        </label>
      </div>
    </div>
  );
};

// Main Component
export default function AgentBuilderPage() {
  const [activeView, setActiveView] = useState<"templates" | "builder">("builder");
  const [activePanel, setActivePanel] = useState<"nodes" | "settings" | "variables">("nodes");
  const [agent, setAgent] = useState<AgentConfig>(mockAgent);
  const [selectedNode, setSelectedNode] = useState<FlowNode | null>(null);
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const nodeType = e.dataTransfer.getData("nodeType") as NodeType;
    const nodeName = e.dataTransfer.getData("nodeName");

    if (nodeType) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = (e.clientX - rect.left - panOffset.x) / zoom;
      const y = (e.clientY - rect.top - panOffset.y) / zoom;

      const newNode: FlowNode = {
        id: `node-${Date.now()}`,
        type: nodeType,
        label: nodeName || nodeType,
        position: { x, y },
        data: {},
        connections: [],
      };

      setAgent((prev) => ({
        ...prev,
        nodes: [...prev.nodes, newNode],
      }));
    }
  }, [zoom, panOffset]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleNodeUpdate = useCallback((updatedNode: FlowNode) => {
    setAgent((prev) => ({
      ...prev,
      nodes: prev.nodes.map((n) => (n.id === updatedNode.id ? updatedNode : n)),
    }));
    setSelectedNode(updatedNode);
  }, []);

  const handleNodeDelete = useCallback((nodeId: string) => {
    setAgent((prev) => ({
      ...prev,
      nodes: prev.nodes.filter((n) => n.id !== nodeId),
      connections: prev.connections.filter((c) => c.sourceNodeId !== nodeId && c.targetNodeId !== nodeId),
    }));
    setSelectedNode(null);
  }, []);

  const handleNodeDragStart = useCallback((e: React.DragEvent, node: FlowNode) => {
    e.dataTransfer.setData("existingNodeId", node.id);
  }, []);

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-900 flex flex-col">
        {/* Header */}
        <div className="border-b border-gray-800 bg-gray-900/95 backdrop-blur-sm">
          <div className="max-w-full mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div>
                  <div className="flex items-center gap-3">
                    <h1 className="text-xl font-bold text-white">{agent.name}</h1>
                    <StatusBadge status={agent.status} />
                  </div>
                  <p className="text-gray-400 text-sm mt-0.5">{agent.description}</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 bg-gray-800 rounded-lg p-1">
                  <button
                    onClick={() => setActiveView("templates")}
                    className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                      activeView === "templates" ? "bg-purple-500 text-white" : "text-gray-400 hover:text-white"
                    }`}
                  >
                    Templates
                  </button>
                  <button
                    onClick={() => setActiveView("builder")}
                    className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                      activeView === "builder" ? "bg-purple-500 text-white" : "text-gray-400 hover:text-white"
                    }`}
                  >
                    Builder
                  </button>
                </div>

                <div className="h-6 w-px bg-gray-700" />

                <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
                  Test Agent
                </button>
                <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
                  Preview
                </button>
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
                  Publish
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        {activeView === "templates" ? (
          <div className="flex-1 p-6 overflow-y-auto">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-lg font-semibold text-white mb-4">Start from a Template</h2>
              <p className="text-gray-400 mb-6">Choose a pre-built template to get started quickly</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {mockTemplates.map((template) => (
                  <TemplateCard
                    key={template.id}
                    template={template}
                    onSelect={(t) => {
                      setAgent((prev) => ({ ...prev, name: t.name, description: t.description }));
                      setActiveView("builder");
                    }}
                  />
                ))}
              </div>

              {/* Blank Template */}
              <div
                className="mt-4 p-6 border-2 border-dashed border-gray-700 rounded-xl text-center cursor-pointer hover:border-purple-500/50 transition-colors"
                onClick={() => setActiveView("builder")}
              >
                <span className="text-3xl mb-2 block">✨</span>
                <p className="text-white font-medium">Start from Scratch</p>
                <p className="text-gray-400 text-sm">Build your agent from a blank canvas</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex overflow-hidden">
            {/* Left Panel - Node Palette / Settings */}
            <div className="w-72 bg-gray-800/50 border-r border-gray-700 flex flex-col">
              {/* Panel Tabs */}
              <div className="flex border-b border-gray-700">
                {(["nodes", "settings", "variables"] as const).map((panel) => (
                  <button
                    key={panel}
                    onClick={() => setActivePanel(panel)}
                    className={`flex-1 px-4 py-3 text-sm font-medium capitalize transition-colors ${
                      activePanel === panel
                        ? "text-purple-400 border-b-2 border-purple-400"
                        : "text-gray-400 hover:text-white"
                    }`}
                  >
                    {panel}
                  </button>
                ))}
              </div>

              <div className="flex-1 overflow-y-auto p-4">
                {activePanel === "nodes" && (
                  <div className="space-y-6">
                    {nodeCategories.map((category) => (
                      <div key={category.id}>
                        <div className="flex items-center gap-2 mb-3">
                          <span>{category.icon}</span>
                          <h3 className="text-sm font-medium text-gray-400">{category.name}</h3>
                        </div>
                        <div className="space-y-2">
                          {category.nodes.map((node) => (
                            <DraggableNode
                              key={node.type}
                              type={node.type}
                              name={node.name}
                              description={node.description}
                              icon={node.icon}
                            />
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {activePanel === "settings" && (
                  <AgentSettingsPanel agent={agent} onUpdate={setAgent} voices={mockVoices} />
                )}

                {activePanel === "variables" && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-medium text-gray-400">Variables</h3>
                      <button className="text-sm text-purple-400 hover:text-purple-300">+ Add</button>
                    </div>
                    {agent.variables.map((variable, index) => (
                      <div key={index} className="bg-gray-700/30 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <code className="text-purple-400 text-sm">{`{{${variable.name}}}`}</code>
                          <span className="text-xs text-gray-500">{variable.type}</span>
                        </div>
                        {variable.defaultValue && (
                          <p className="text-xs text-gray-400">Default: {variable.defaultValue}</p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Canvas */}
            <div
              className="flex-1 bg-gray-900 relative overflow-hidden"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              {/* Zoom Controls */}
              <div className="absolute top-4 right-4 z-10 flex items-center gap-2 bg-gray-800 rounded-lg p-1">
                <button
                  onClick={() => setZoom((z) => Math.max(0.25, z - 0.25))}
                  className="p-2 text-gray-400 hover:text-white transition-colors"
                >
                  -
                </button>
                <span className="text-sm text-gray-400 w-12 text-center">{Math.round(zoom * 100)}%</span>
                <button
                  onClick={() => setZoom((z) => Math.min(2, z + 0.25))}
                  className="p-2 text-gray-400 hover:text-white transition-colors"
                >
                  +
                </button>
              </div>

              {/* Grid Background */}
              <div
                className="absolute inset-0"
                style={{
                  backgroundImage: `radial-gradient(circle, #374151 1px, transparent 1px)`,
                  backgroundSize: `${20 * zoom}px ${20 * zoom}px`,
                  backgroundPosition: `${panOffset.x}px ${panOffset.y}px`,
                }}
              />

              {/* Nodes Container */}
              <div
                className="absolute inset-0"
                style={{
                  transform: `scale(${zoom}) translate(${panOffset.x}px, ${panOffset.y}px)`,
                  transformOrigin: "0 0",
                }}
              >
                {/* Connections */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none">
                  {agent.connections.map((conn) => {
                    const sourceNode = agent.nodes.find((n) => n.id === conn.sourceNodeId);
                    const targetNode = agent.nodes.find((n) => n.id === conn.targetNodeId);

                    if (!sourceNode || !targetNode) return null;

                    const startX = sourceNode.position.x + 90;
                    const startY = sourceNode.position.y + 80;
                    const endX = targetNode.position.x + 90;
                    const endY = targetNode.position.y;

                    const midY = (startY + endY) / 2;

                    return (
                      <g key={conn.id}>
                        <path
                          d={`M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`}
                          fill="none"
                          stroke={
                            conn.type === "yes" ? "#22c55e" :
                            conn.type === "no" ? "#ef4444" :
                            conn.type === "fallback" ? "#eab308" :
                            "#6b7280"
                          }
                          strokeWidth="2"
                          strokeDasharray={conn.type === "fallback" ? "5,5" : "none"}
                        />
                        {conn.label && (
                          <text
                            x={(startX + endX) / 2}
                            y={(startY + endY) / 2}
                            fill="#9ca3af"
                            fontSize="12"
                            textAnchor="middle"
                          >
                            {conn.label}
                          </text>
                        )}
                      </g>
                    );
                  })}
                </svg>

                {/* Flow Nodes */}
                {agent.nodes.map((node) => (
                  <FlowNodeComponent
                    key={node.id}
                    node={node}
                    isSelected={selectedNode?.id === node.id}
                    onSelect={setSelectedNode}
                    onDragStart={handleNodeDragStart}
                  />
                ))}
              </div>

              {/* Empty State */}
              {agent.nodes.length === 0 && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <span className="text-4xl mb-4 block">🎨</span>
                    <p className="text-gray-400 mb-2">Drag nodes from the left panel</p>
                    <p className="text-gray-500 text-sm">or start with a template</p>
                  </div>
                </div>
              )}
            </div>

            {/* Right Panel - Node Properties */}
            {selectedNode && (
              <NodePropertiesPanel
                node={selectedNode}
                onUpdate={handleNodeUpdate}
                onDelete={handleNodeDelete}
                onClose={() => setSelectedNode(null)}
              />
            )}
          </div>
        )}

        {/* Footer Stats */}
        <div className="border-t border-gray-800 bg-gray-900/95 px-6 py-3">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-6 text-gray-400">
              <span>{agent.nodes.length} nodes</span>
              <span>{agent.connections.length} connections</span>
              <span>{agent.variables.length} variables</span>
            </div>
            <div className="flex items-center gap-4 text-gray-500">
              <span>Version {agent.version}</span>
              <span>Last saved: {new Date(agent.updatedAt).toLocaleTimeString()}</span>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
