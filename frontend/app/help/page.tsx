"use client";

import React, { useState, useMemo } from "react";
import { DashboardLayout } from "@/components/dashboard-layout";
import {
  Search,
  Book,
  FileText,
  Video,
  MessageCircle,
  HelpCircle,
  ChevronRight,
  ChevronDown,
  ExternalLink,
  Clock,
  Star,
  ThumbsUp,
  ThumbsDown,
  Play,
  Bookmark,
  BookmarkCheck,
  Share2,
  Copy,
  ArrowLeft,
  ArrowRight,
  Home,
  Zap,
  Phone,
  Bot,
  Settings,
  Users,
  CreditCard,
  Shield,
  Code,
  Webhook,
  BarChart3,
  Headphones,
  Mail,
  Send,
  Sparkles,
  Lightbulb,
  GraduationCap,
  Target,
  Rocket,
  CheckCircle,
  AlertCircle,
  Info,
  X,
  Menu,
  Filter,
  Grid,
  List,
  TrendingUp,
  Eye,
  Calendar,
  Tag,
  Folder,
  FileQuestion,
  MessagesSquare,
  Plus,
  Minus,
} from "lucide-react";

// Types
interface Article {
  id: string;
  title: string;
  slug: string;
  excerpt: string;
  content: string;
  category: string;
  subcategory: string;
  tags: string[];
  readTime: number;
  views: number;
  helpful: number;
  notHelpful: number;
  author: string;
  publishedAt: string;
  updatedAt: string;
  featured: boolean;
  relatedArticles: string[];
}

interface Category {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  articleCount: number;
  color: string;
  subcategories: string[];
}

interface FAQ {
  id: string;
  question: string;
  answer: string;
  category: string;
  helpful: number;
}

interface VideoTutorial {
  id: string;
  title: string;
  description: string;
  thumbnail: string;
  duration: string;
  category: string;
  views: number;
  youtubeId: string;
}

// Mock data
const categories: Category[] = [
  {
    id: "getting-started",
    name: "Getting Started",
    description: "Learn the basics of setting up and using BVRAI",
    icon: <Rocket className="w-6 h-6" />,
    articleCount: 12,
    color: "purple",
    subcategories: ["Quick Start", "Account Setup", "First Agent", "Dashboard Overview"],
  },
  {
    id: "agents",
    name: "AI Agents",
    description: "Create and configure your voice AI agents",
    icon: <Bot className="w-6 h-6" />,
    articleCount: 24,
    color: "blue",
    subcategories: ["Agent Builder", "Voice Configuration", "Prompts", "Knowledge Base"],
  },
  {
    id: "phone-numbers",
    name: "Phone Numbers",
    description: "Manage phone numbers and call routing",
    icon: <Phone className="w-6 h-6" />,
    articleCount: 8,
    color: "green",
    subcategories: ["Getting Numbers", "Porting", "Configuration", "Call Routing"],
  },
  {
    id: "integrations",
    name: "Integrations",
    description: "Connect BVRAI with your favorite tools",
    icon: <Zap className="w-6 h-6" />,
    articleCount: 32,
    color: "orange",
    subcategories: ["CRM", "Calendars", "Webhooks", "API"],
  },
  {
    id: "analytics",
    name: "Analytics & Reports",
    description: "Understand your call data and performance",
    icon: <BarChart3 className="w-6 h-6" />,
    articleCount: 10,
    color: "cyan",
    subcategories: ["Dashboard", "Call Logs", "Reports", "Export"],
  },
  {
    id: "billing",
    name: "Billing & Plans",
    description: "Manage your subscription and payments",
    icon: <CreditCard className="w-6 h-6" />,
    articleCount: 6,
    color: "pink",
    subcategories: ["Plans", "Invoices", "Payment Methods", "Usage"],
  },
  {
    id: "security",
    name: "Security & Compliance",
    description: "Keep your data safe and compliant",
    icon: <Shield className="w-6 h-6" />,
    articleCount: 8,
    color: "red",
    subcategories: ["Data Protection", "Access Control", "Compliance", "Audit Logs"],
  },
  {
    id: "api",
    name: "API & Developers",
    description: "Build custom integrations with our API",
    icon: <Code className="w-6 h-6" />,
    articleCount: 18,
    color: "indigo",
    subcategories: ["REST API", "Webhooks", "SDKs", "Examples"],
  },
];

const articles: Article[] = [
  {
    id: "art_1",
    title: "Getting Started with BVRAI: Complete Guide",
    slug: "getting-started-complete-guide",
    excerpt:
      "Learn how to set up your first AI voice agent in under 10 minutes with this comprehensive guide.",
    content: `
# Getting Started with BVRAI

Welcome to BVRAI! This guide will walk you through setting up your first AI voice agent step by step.

## Prerequisites

Before you begin, make sure you have:
- A BVRAI account (sign up at bvrai.com)
- A phone number (you can purchase one or port an existing number)
- Basic understanding of your use case

## Step 1: Create Your First Agent

1. Navigate to the **Agents** section in your dashboard
2. Click **Create New Agent**
3. Choose a template or start from scratch
4. Give your agent a name and description

## Step 2: Configure Voice Settings

Select the voice that best represents your brand:
- **Aria**: Professional female voice, great for customer service
- **Marcus**: Authoritative male voice, ideal for business
- **Sofia**: Friendly female voice, perfect for appointments

## Step 3: Set Up Your Phone Number

1. Go to **Phone Numbers** in the sidebar
2. Click **Get New Number** or **Port Existing**
3. Select your country and area code
4. Complete the purchase

## Step 4: Connect Your Agent

1. Open your agent settings
2. Go to the **Phone Numbers** tab
3. Assign your new number to the agent
4. Test with a call!

## Next Steps

- Customize your agent's prompts
- Set up integrations (CRM, Calendar, etc.)
- Review analytics and optimize
    `,
    category: "getting-started",
    subcategory: "Quick Start",
    tags: ["beginner", "setup", "tutorial"],
    readTime: 8,
    views: 15420,
    helpful: 1234,
    notHelpful: 45,
    author: "BVRAI Team",
    publishedAt: "2024-01-01T00:00:00Z",
    updatedAt: "2024-01-15T00:00:00Z",
    featured: true,
    relatedArticles: ["art_2", "art_3", "art_4"],
  },
  {
    id: "art_2",
    title: "Understanding the Agent Builder",
    slug: "understanding-agent-builder",
    excerpt:
      "Master the Agent Builder to create powerful, customized AI voice agents for any use case.",
    content: "Full article content...",
    category: "agents",
    subcategory: "Agent Builder",
    tags: ["agents", "builder", "configuration"],
    readTime: 12,
    views: 8923,
    helpful: 892,
    notHelpful: 23,
    author: "BVRAI Team",
    publishedAt: "2024-01-02T00:00:00Z",
    updatedAt: "2024-01-18T00:00:00Z",
    featured: true,
    relatedArticles: ["art_1", "art_5"],
  },
  {
    id: "art_3",
    title: "Configuring Voice and Language Settings",
    slug: "voice-language-settings",
    excerpt:
      "Choose the perfect voice for your AI agent and configure language preferences.",
    content: "Full article content...",
    category: "agents",
    subcategory: "Voice Configuration",
    tags: ["voice", "language", "settings"],
    readTime: 6,
    views: 6234,
    helpful: 567,
    notHelpful: 18,
    author: "BVRAI Team",
    publishedAt: "2024-01-03T00:00:00Z",
    updatedAt: "2024-01-16T00:00:00Z",
    featured: false,
    relatedArticles: ["art_2"],
  },
  {
    id: "art_4",
    title: "Purchasing and Porting Phone Numbers",
    slug: "purchasing-porting-phone-numbers",
    excerpt:
      "Learn how to get new phone numbers or transfer existing ones to BVRAI.",
    content: "Full article content...",
    category: "phone-numbers",
    subcategory: "Getting Numbers",
    tags: ["phone", "numbers", "porting"],
    readTime: 5,
    views: 4521,
    helpful: 432,
    notHelpful: 12,
    author: "BVRAI Team",
    publishedAt: "2024-01-04T00:00:00Z",
    updatedAt: "2024-01-17T00:00:00Z",
    featured: false,
    relatedArticles: ["art_1"],
  },
  {
    id: "art_5",
    title: "Writing Effective Agent Prompts",
    slug: "writing-effective-prompts",
    excerpt:
      "Best practices for crafting prompts that make your AI agent more helpful and natural.",
    content: "Full article content...",
    category: "agents",
    subcategory: "Prompts",
    tags: ["prompts", "best-practices", "ai"],
    readTime: 10,
    views: 7823,
    helpful: 812,
    notHelpful: 34,
    author: "BVRAI Team",
    publishedAt: "2024-01-05T00:00:00Z",
    updatedAt: "2024-01-19T00:00:00Z",
    featured: true,
    relatedArticles: ["art_2", "art_6"],
  },
  {
    id: "art_6",
    title: "Connecting Your CRM (Salesforce, HubSpot, etc.)",
    slug: "connecting-crm-integration",
    excerpt:
      "Sync your AI agent with your CRM to automatically log calls and update contacts.",
    content: "Full article content...",
    category: "integrations",
    subcategory: "CRM",
    tags: ["crm", "salesforce", "hubspot", "integration"],
    readTime: 8,
    views: 5432,
    helpful: 543,
    notHelpful: 21,
    author: "BVRAI Team",
    publishedAt: "2024-01-06T00:00:00Z",
    updatedAt: "2024-01-18T00:00:00Z",
    featured: false,
    relatedArticles: ["art_7", "art_8"],
  },
  {
    id: "art_7",
    title: "Setting Up Webhooks for Real-time Events",
    slug: "setting-up-webhooks",
    excerpt:
      "Configure webhooks to receive real-time notifications for calls and agent events.",
    content: "Full article content...",
    category: "integrations",
    subcategory: "Webhooks",
    tags: ["webhooks", "api", "events", "real-time"],
    readTime: 7,
    views: 3456,
    helpful: 345,
    notHelpful: 15,
    author: "BVRAI Team",
    publishedAt: "2024-01-07T00:00:00Z",
    updatedAt: "2024-01-19T00:00:00Z",
    featured: false,
    relatedArticles: ["art_6", "art_9"],
  },
  {
    id: "art_8",
    title: "Calendar Integration: Google & Outlook",
    slug: "calendar-integration-guide",
    excerpt:
      "Enable your AI agent to schedule appointments directly in your calendar.",
    content: "Full article content...",
    category: "integrations",
    subcategory: "Calendars",
    tags: ["calendar", "google", "outlook", "scheduling"],
    readTime: 6,
    views: 4123,
    helpful: 412,
    notHelpful: 18,
    author: "BVRAI Team",
    publishedAt: "2024-01-08T00:00:00Z",
    updatedAt: "2024-01-17T00:00:00Z",
    featured: false,
    relatedArticles: ["art_6"],
  },
  {
    id: "art_9",
    title: "REST API Quick Start Guide",
    slug: "rest-api-quick-start",
    excerpt:
      "Get up and running with the BVRAI REST API to build custom integrations.",
    content: "Full article content...",
    category: "api",
    subcategory: "REST API",
    tags: ["api", "rest", "developer", "integration"],
    readTime: 15,
    views: 6789,
    helpful: 678,
    notHelpful: 32,
    author: "BVRAI Team",
    publishedAt: "2024-01-09T00:00:00Z",
    updatedAt: "2024-01-20T00:00:00Z",
    featured: true,
    relatedArticles: ["art_7", "art_10"],
  },
  {
    id: "art_10",
    title: "Understanding Your Analytics Dashboard",
    slug: "analytics-dashboard-guide",
    excerpt:
      "Learn how to use the analytics dashboard to monitor agent performance.",
    content: "Full article content...",
    category: "analytics",
    subcategory: "Dashboard",
    tags: ["analytics", "dashboard", "metrics", "performance"],
    readTime: 9,
    views: 5234,
    helpful: 523,
    notHelpful: 19,
    author: "BVRAI Team",
    publishedAt: "2024-01-10T00:00:00Z",
    updatedAt: "2024-01-19T00:00:00Z",
    featured: false,
    relatedArticles: ["art_11"],
  },
  {
    id: "art_11",
    title: "Exporting Call Recordings and Transcripts",
    slug: "exporting-recordings-transcripts",
    excerpt:
      "Download and export your call recordings and transcriptions for analysis.",
    content: "Full article content...",
    category: "analytics",
    subcategory: "Export",
    tags: ["export", "recordings", "transcripts", "data"],
    readTime: 4,
    views: 2345,
    helpful: 234,
    notHelpful: 8,
    author: "BVRAI Team",
    publishedAt: "2024-01-11T00:00:00Z",
    updatedAt: "2024-01-18T00:00:00Z",
    featured: false,
    relatedArticles: ["art_10"],
  },
  {
    id: "art_12",
    title: "Managing Your Subscription and Billing",
    slug: "subscription-billing-guide",
    excerpt:
      "Everything you need to know about plans, billing, and payment methods.",
    content: "Full article content...",
    category: "billing",
    subcategory: "Plans",
    tags: ["billing", "subscription", "plans", "payments"],
    readTime: 5,
    views: 3456,
    helpful: 345,
    notHelpful: 12,
    author: "BVRAI Team",
    publishedAt: "2024-01-12T00:00:00Z",
    updatedAt: "2024-01-17T00:00:00Z",
    featured: false,
    relatedArticles: ["art_13"],
  },
];

const faqs: FAQ[] = [
  {
    id: "faq_1",
    question: "How do I create my first AI agent?",
    answer:
      "Navigate to the Agents section, click 'Create New Agent', choose a template or start from scratch, and follow the setup wizard. You can have your first agent ready in under 5 minutes!",
    category: "getting-started",
    helpful: 234,
  },
  {
    id: "faq_2",
    question: "What phone numbers are available?",
    answer:
      "We offer local numbers in 50+ countries, toll-free numbers in the US, UK, and Canada, and mobile numbers in select regions. You can also port your existing number.",
    category: "phone-numbers",
    helpful: 189,
  },
  {
    id: "faq_3",
    question: "How much does BVRAI cost?",
    answer:
      "We offer plans starting at $29/month for individuals, $99/month for teams, and custom pricing for enterprise. All plans include a free trial. Visit our pricing page for details.",
    category: "billing",
    helpful: 312,
  },
  {
    id: "faq_4",
    question: "Can I integrate with my existing CRM?",
    answer:
      "Yes! We support integrations with Salesforce, HubSpot, Pipedrive, Freshsales, and many more. You can also use our REST API or webhooks for custom integrations.",
    category: "integrations",
    helpful: 167,
  },
  {
    id: "faq_5",
    question: "Is my data secure?",
    answer:
      "Absolutely. We use enterprise-grade encryption (AES-256), are SOC 2 Type II certified, GDPR compliant, and HIPAA ready. Your data is stored in secure data centers with 99.99% uptime.",
    category: "security",
    helpful: 256,
  },
  {
    id: "faq_6",
    question: "How do I customize my agent's voice?",
    answer:
      "Go to your agent settings, select the Voice tab, and choose from our library of natural-sounding voices. You can also adjust speech rate, pitch, and language.",
    category: "agents",
    helpful: 198,
  },
  {
    id: "faq_7",
    question: "Can I use BVRAI for outbound calls?",
    answer:
      "Yes! You can use BVRAI for both inbound and outbound calls. Set up campaigns, upload contact lists, and let your AI agent make calls on your behalf.",
    category: "agents",
    helpful: 145,
  },
  {
    id: "faq_8",
    question: "How do I access the API?",
    answer:
      "Go to Settings > API Keys to generate your API key. Then check our API documentation at docs.bvrai.com for endpoints, authentication, and examples.",
    category: "api",
    helpful: 123,
  },
  {
    id: "faq_9",
    question: "What happens if my agent can't answer a question?",
    answer:
      "You can configure fallback behaviors including transferring to a human agent, taking a message, scheduling a callback, or providing alternative contact information.",
    category: "agents",
    helpful: 178,
  },
  {
    id: "faq_10",
    question: "How do I cancel my subscription?",
    answer:
      "Go to Settings > Billing > Manage Subscription and click 'Cancel Plan'. Your service will continue until the end of your billing period. You can reactivate anytime.",
    category: "billing",
    helpful: 89,
  },
];

const videoTutorials: VideoTutorial[] = [
  {
    id: "vid_1",
    title: "BVRAI Platform Overview",
    description: "A comprehensive tour of the BVRAI platform and its features",
    thumbnail: "/thumbnails/overview.jpg",
    duration: "8:32",
    category: "getting-started",
    views: 12453,
    youtubeId: "abc123",
  },
  {
    id: "vid_2",
    title: "Creating Your First Agent",
    description: "Step-by-step guide to creating and configuring an AI agent",
    thumbnail: "/thumbnails/first-agent.jpg",
    duration: "12:45",
    category: "agents",
    views: 8923,
    youtubeId: "def456",
  },
  {
    id: "vid_3",
    title: "Voice Configuration Deep Dive",
    description: "Master the voice settings for natural-sounding conversations",
    thumbnail: "/thumbnails/voice.jpg",
    duration: "6:18",
    category: "agents",
    views: 5621,
    youtubeId: "ghi789",
  },
  {
    id: "vid_4",
    title: "Setting Up CRM Integrations",
    description: "Connect BVRAI with Salesforce, HubSpot, and more",
    thumbnail: "/thumbnails/crm.jpg",
    duration: "10:22",
    category: "integrations",
    views: 4532,
    youtubeId: "jkl012",
  },
  {
    id: "vid_5",
    title: "Understanding Analytics",
    description: "How to use analytics to improve agent performance",
    thumbnail: "/thumbnails/analytics.jpg",
    duration: "7:55",
    category: "analytics",
    views: 3421,
    youtubeId: "mno345",
  },
  {
    id: "vid_6",
    title: "API Quickstart Tutorial",
    description: "Get started with the BVRAI REST API",
    thumbnail: "/thumbnails/api.jpg",
    duration: "15:30",
    category: "api",
    views: 6234,
    youtubeId: "pqr678",
  },
];

// Color mapping
function getCategoryColor(color: string) {
  const colors: Record<string, string> = {
    purple: "from-purple-500 to-pink-500",
    blue: "from-blue-500 to-cyan-500",
    green: "from-green-500 to-emerald-500",
    orange: "from-orange-500 to-amber-500",
    cyan: "from-cyan-500 to-blue-500",
    pink: "from-pink-500 to-rose-500",
    red: "from-red-500 to-orange-500",
    indigo: "from-indigo-500 to-purple-500",
  };
  return colors[color] || colors.purple;
}

function getCategoryBgColor(color: string) {
  const colors: Record<string, string> = {
    purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    blue: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    green: "bg-green-500/10 text-green-400 border-green-500/20",
    orange: "bg-orange-500/10 text-orange-400 border-orange-500/20",
    cyan: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
    pink: "bg-pink-500/10 text-pink-400 border-pink-500/20",
    red: "bg-red-500/10 text-red-400 border-red-500/20",
    indigo: "bg-indigo-500/10 text-indigo-400 border-indigo-500/20",
  };
  return colors[color] || colors.purple;
}

// Search bar component
function SearchBar({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="relative max-w-2xl mx-auto">
      <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Search for help articles, tutorials, and more..."
        className="w-full pl-12 pr-4 py-4 bg-white/5 border border-white/10 rounded-xl text-white text-lg placeholder:text-gray-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20"
      />
      {value && (
        <button
          onClick={() => onChange("")}
          className="absolute right-4 top-1/2 -translate-y-1/2 p-1 hover:bg-white/10 rounded"
        >
          <X className="w-4 h-4 text-gray-400" />
        </button>
      )}
    </div>
  );
}

// Category card component
function CategoryCard({ category }: { category: Category }) {
  return (
    <a
      href={`/help/category/${category.id}`}
      className="group block bg-[#1a1a2e]/50 backdrop-blur-sm rounded-xl border border-white/5 hover:border-white/10 p-6 transition-all hover:transform hover:scale-[1.02]"
    >
      <div
        className={`w-12 h-12 rounded-xl bg-gradient-to-br ${getCategoryColor(
          category.color
        )} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}
      >
        {category.icon}
      </div>
      <h3 className="text-lg font-semibold text-white mb-1 group-hover:text-purple-400 transition-colors">
        {category.name}
      </h3>
      <p className="text-sm text-gray-400 mb-3">{category.description}</p>
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">
          {category.articleCount} articles
        </span>
        <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-purple-400 group-hover:translate-x-1 transition-all" />
      </div>
    </a>
  );
}

// Article card component
function ArticleCard({
  article,
  compact = false,
}: {
  article: Article;
  compact?: boolean;
}) {
  const category = categories.find((c) => c.id === article.category);

  if (compact) {
    return (
      <a
        href={`/help/article/${article.slug}`}
        className="group flex items-start gap-3 p-3 rounded-lg hover:bg-white/5 transition-colors"
      >
        <FileText className="w-5 h-5 text-gray-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium text-white group-hover:text-purple-400 transition-colors truncate">
            {article.title}
          </h4>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-gray-500">
              {article.readTime} min read
            </span>
            <span className="text-gray-600">•</span>
            <span className="text-xs text-gray-500">
              {article.views.toLocaleString()} views
            </span>
          </div>
        </div>
        <ChevronRight className="w-4 h-4 text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity" />
      </a>
    );
  }

  return (
    <a
      href={`/help/article/${article.slug}`}
      className="group block bg-[#1a1a2e]/30 hover:bg-[#1a1a2e]/60 rounded-xl border border-white/5 hover:border-white/10 p-5 transition-all"
    >
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex-1">
          {article.featured && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium bg-yellow-500/10 text-yellow-400 rounded-full mb-2">
              <Star className="w-3 h-3" />
              Featured
            </span>
          )}
          <h3 className="text-base font-semibold text-white group-hover:text-purple-400 transition-colors">
            {article.title}
          </h3>
        </div>
        <span
          className={`flex-shrink-0 px-2 py-1 text-xs font-medium rounded border ${getCategoryBgColor(
            category?.color || "purple"
          )}`}
        >
          {category?.name}
        </span>
      </div>
      <p className="text-sm text-gray-400 mb-4 line-clamp-2">{article.excerpt}</p>
      <div className="flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {article.readTime} min
          </span>
          <span className="flex items-center gap-1">
            <Eye className="w-3 h-3" />
            {article.views.toLocaleString()}
          </span>
          <span className="flex items-center gap-1">
            <ThumbsUp className="w-3 h-3" />
            {article.helpful}
          </span>
        </div>
        <span>Updated {new Date(article.updatedAt).toLocaleDateString()}</span>
      </div>
    </a>
  );
}

// FAQ accordion component
function FAQAccordion({
  faq,
  isOpen,
  onToggle,
}: {
  faq: FAQ;
  isOpen: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="border-b border-white/5">
      <button
        onClick={onToggle}
        className="flex items-center justify-between w-full py-4 text-left group"
      >
        <span className="text-white font-medium group-hover:text-purple-400 transition-colors pr-4">
          {faq.question}
        </span>
        <span
          className={`flex-shrink-0 p-1 rounded-full bg-white/5 transition-transform ${
            isOpen ? "rotate-180" : ""
          }`}
        >
          <ChevronDown className="w-4 h-4 text-gray-400" />
        </span>
      </button>
      {isOpen && (
        <div className="pb-4">
          <p className="text-gray-400 text-sm leading-relaxed">{faq.answer}</p>
          <div className="flex items-center gap-4 mt-3">
            <span className="text-xs text-gray-500">Was this helpful?</span>
            <button className="flex items-center gap-1 text-xs text-gray-400 hover:text-green-400 transition-colors">
              <ThumbsUp className="w-3 h-3" />
              Yes ({faq.helpful})
            </button>
            <button className="flex items-center gap-1 text-xs text-gray-400 hover:text-red-400 transition-colors">
              <ThumbsDown className="w-3 h-3" />
              No
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Video card component
function VideoCard({ video }: { video: VideoTutorial }) {
  return (
    <a
      href={`/help/video/${video.id}`}
      className="group block bg-[#1a1a2e]/30 hover:bg-[#1a1a2e]/60 rounded-xl border border-white/5 hover:border-white/10 overflow-hidden transition-all"
    >
      <div className="relative aspect-video bg-gray-800">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-14 h-14 rounded-full bg-purple-500/80 flex items-center justify-center group-hover:bg-purple-500 group-hover:scale-110 transition-all">
            <Play className="w-6 h-6 text-white ml-1" />
          </div>
        </div>
        <span className="absolute bottom-2 right-2 px-2 py-0.5 bg-black/70 text-white text-xs rounded">
          {video.duration}
        </span>
      </div>
      <div className="p-4">
        <h3 className="text-sm font-semibold text-white group-hover:text-purple-400 transition-colors mb-1">
          {video.title}
        </h3>
        <p className="text-xs text-gray-400 line-clamp-2 mb-2">
          {video.description}
        </p>
        <span className="text-xs text-gray-500">
          {video.views.toLocaleString()} views
        </span>
      </div>
    </a>
  );
}

// Contact support section
function ContactSupport() {
  const [message, setMessage] = useState("");
  const [email, setEmail] = useState("");

  return (
    <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 rounded-2xl border border-purple-500/20 p-6">
      <div className="flex items-start gap-4 mb-6">
        <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
          <Headphones className="w-6 h-6 text-purple-400" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white">
            Can't find what you're looking for?
          </h3>
          <p className="text-sm text-gray-400">
            Our support team is here to help 24/7
          </p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <a
          href="/help/contact"
          className="flex items-center gap-3 p-4 bg-white/5 hover:bg-white/10 rounded-xl border border-white/5 hover:border-white/10 transition-all"
        >
          <Mail className="w-5 h-5 text-blue-400" />
          <div>
            <span className="block text-sm font-medium text-white">Email</span>
            <span className="block text-xs text-gray-400">
              Get help via email
            </span>
          </div>
        </a>
        <a
          href="/help/chat"
          className="flex items-center gap-3 p-4 bg-white/5 hover:bg-white/10 rounded-xl border border-white/5 hover:border-white/10 transition-all"
        >
          <MessageCircle className="w-5 h-5 text-green-400" />
          <div>
            <span className="block text-sm font-medium text-white">
              Live Chat
            </span>
            <span className="block text-xs text-gray-400">Chat with us now</span>
          </div>
        </a>
        <a
          href="/help/schedule"
          className="flex items-center gap-3 p-4 bg-white/5 hover:bg-white/10 rounded-xl border border-white/5 hover:border-white/10 transition-all"
        >
          <Calendar className="w-5 h-5 text-purple-400" />
          <div>
            <span className="block text-sm font-medium text-white">
              Schedule Call
            </span>
            <span className="block text-xs text-gray-400">Book a meeting</span>
          </div>
        </a>
      </div>

      <div className="space-y-3">
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Your email address"
          className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500"
        />
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Describe your issue or question..."
          rows={3}
          className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-gray-500 focus:outline-none focus:border-purple-500 resize-none"
        />
        <button className="flex items-center justify-center gap-2 w-full py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors">
          <Send className="w-4 h-4" />
          Send Message
        </button>
      </div>
    </div>
  );
}

// Quick links section
function QuickLinks() {
  const links = [
    {
      icon: <Rocket className="w-5 h-5" />,
      title: "Quick Start Guide",
      href: "/help/article/getting-started-complete-guide",
    },
    {
      icon: <Bot className="w-5 h-5" />,
      title: "Create Your First Agent",
      href: "/help/article/understanding-agent-builder",
    },
    {
      icon: <Zap className="w-5 h-5" />,
      title: "Set Up Integrations",
      href: "/help/article/connecting-crm-integration",
    },
    {
      icon: <Code className="w-5 h-5" />,
      title: "API Documentation",
      href: "/help/article/rest-api-quick-start",
    },
    {
      icon: <CreditCard className="w-5 h-5" />,
      title: "Pricing & Plans",
      href: "/pricing",
    },
    {
      icon: <Shield className="w-5 h-5" />,
      title: "Security & Compliance",
      href: "/help/category/security",
    },
  ];

  return (
    <div className="bg-[#1a1a2e]/50 backdrop-blur-sm rounded-xl border border-white/5 p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Quick Links</h3>
      <div className="space-y-2">
        {links.map((link) => (
          <a
            key={link.title}
            href={link.href}
            className="flex items-center gap-3 p-3 rounded-lg hover:bg-white/5 transition-colors group"
          >
            <span className="text-gray-400 group-hover:text-purple-400 transition-colors">
              {link.icon}
            </span>
            <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
              {link.title}
            </span>
            <ChevronRight className="w-4 h-4 text-gray-600 ml-auto opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
        ))}
      </div>
    </div>
  );
}

// Main component
export default function HelpCenterPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<"all" | "articles" | "videos" | "faq">(
    "all"
  );
  const [openFaqIds, setOpenFaqIds] = useState<string[]>([]);

  // Search results
  const searchResults = useMemo(() => {
    if (!searchQuery.trim()) return { articles: [], faqs: [], videos: [] };

    const query = searchQuery.toLowerCase();

    return {
      articles: articles.filter(
        (a) =>
          a.title.toLowerCase().includes(query) ||
          a.excerpt.toLowerCase().includes(query) ||
          a.tags.some((t) => t.toLowerCase().includes(query))
      ),
      faqs: faqs.filter(
        (f) =>
          f.question.toLowerCase().includes(query) ||
          f.answer.toLowerCase().includes(query)
      ),
      videos: videoTutorials.filter(
        (v) =>
          v.title.toLowerCase().includes(query) ||
          v.description.toLowerCase().includes(query)
      ),
    };
  }, [searchQuery]);

  const toggleFaq = (id: string) => {
    setOpenFaqIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  const featuredArticles = articles.filter((a) => a.featured);
  const popularArticles = [...articles]
    .sort((a, b) => b.views - a.views)
    .slice(0, 5);

  return (
    <DashboardLayout>
      <div className="min-h-screen">
        {/* Hero section */}
        <div className="relative bg-gradient-to-b from-purple-500/10 to-transparent py-16 px-6">
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-purple-500/20 rounded-full blur-3xl" />
            <div className="absolute bottom-1/4 right-1/4 w-64 h-64 bg-pink-500/20 rounded-full blur-3xl" />
          </div>

          <div className="relative max-w-4xl mx-auto text-center">
            <div className="flex items-center justify-center gap-2 mb-4">
              <HelpCircle className="w-8 h-8 text-purple-400" />
              <h1 className="text-3xl font-bold text-white">Help Center</h1>
            </div>
            <p className="text-gray-400 mb-8">
              Find answers, tutorials, and guides to help you get the most out of
              BVRAI
            </p>
            <SearchBar value={searchQuery} onChange={setSearchQuery} />

            {/* Popular searches */}
            <div className="flex items-center justify-center gap-2 mt-4 flex-wrap">
              <span className="text-sm text-gray-500">Popular:</span>
              {[
                "getting started",
                "create agent",
                "integrations",
                "api",
                "billing",
              ].map((term) => (
                <button
                  key={term}
                  onClick={() => setSearchQuery(term)}
                  className="px-3 py-1 text-sm bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white rounded-full transition-colors"
                >
                  {term}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main content */}
        <div className="px-6 py-8 max-w-7xl mx-auto">
          {/* Search results */}
          {searchQuery && (
            <div className="mb-12">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white">
                  Search Results for "{searchQuery}"
                </h2>
                <span className="text-sm text-gray-400">
                  {searchResults.articles.length + searchResults.faqs.length + searchResults.videos.length} results
                </span>
              </div>

              {/* Tabs */}
              <div className="flex items-center gap-2 mb-6">
                {[
                  { id: "all", label: "All" },
                  { id: "articles", label: `Articles (${searchResults.articles.length})` },
                  { id: "videos", label: `Videos (${searchResults.videos.length})` },
                  { id: "faq", label: `FAQ (${searchResults.faqs.length})` },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as typeof activeTab)}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      activeTab === tab.id
                        ? "bg-purple-500/20 text-purple-400"
                        : "text-gray-400 hover:bg-white/5"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Results */}
              <div className="space-y-6">
                {(activeTab === "all" || activeTab === "articles") &&
                  searchResults.articles.length > 0 && (
                    <div>
                      {activeTab === "all" && (
                        <h3 className="text-sm font-medium text-gray-400 mb-3">
                          Articles
                        </h3>
                      )}
                      <div className="grid grid-cols-2 gap-4">
                        {searchResults.articles.map((article) => (
                          <ArticleCard key={article.id} article={article} />
                        ))}
                      </div>
                    </div>
                  )}

                {(activeTab === "all" || activeTab === "videos") &&
                  searchResults.videos.length > 0 && (
                    <div>
                      {activeTab === "all" && (
                        <h3 className="text-sm font-medium text-gray-400 mb-3">
                          Videos
                        </h3>
                      )}
                      <div className="grid grid-cols-4 gap-4">
                        {searchResults.videos.map((video) => (
                          <VideoCard key={video.id} video={video} />
                        ))}
                      </div>
                    </div>
                  )}

                {(activeTab === "all" || activeTab === "faq") &&
                  searchResults.faqs.length > 0 && (
                    <div>
                      {activeTab === "all" && (
                        <h3 className="text-sm font-medium text-gray-400 mb-3">
                          Frequently Asked Questions
                        </h3>
                      )}
                      <div className="bg-[#1a1a2e]/30 rounded-xl border border-white/5 p-4">
                        {searchResults.faqs.map((faq) => (
                          <FAQAccordion
                            key={faq.id}
                            faq={faq}
                            isOpen={openFaqIds.includes(faq.id)}
                            onToggle={() => toggleFaq(faq.id)}
                          />
                        ))}
                      </div>
                    </div>
                  )}

                {searchResults.articles.length === 0 &&
                  searchResults.faqs.length === 0 &&
                  searchResults.videos.length === 0 && (
                    <div className="text-center py-12">
                      <FileQuestion className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-white mb-2">
                        No results found
                      </h3>
                      <p className="text-gray-400 mb-4">
                        Try different keywords or browse categories below
                      </p>
                      <button
                        onClick={() => setSearchQuery("")}
                        className="text-purple-400 hover:text-purple-300"
                      >
                        Clear search
                      </button>
                    </div>
                  )}
              </div>
            </div>
          )}

          {/* Default content (no search) */}
          {!searchQuery && (
            <>
              {/* Categories */}
              <section className="mb-12">
                <h2 className="text-xl font-semibold text-white mb-6">
                  Browse by Category
                </h2>
                <div className="grid grid-cols-4 gap-4">
                  {categories.map((category) => (
                    <CategoryCard key={category.id} category={category} />
                  ))}
                </div>
              </section>

              {/* Featured + Quick Links row */}
              <div className="grid grid-cols-3 gap-6 mb-12">
                {/* Featured articles */}
                <div className="col-span-2">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-white">
                      Featured Articles
                    </h2>
                    <a
                      href="/help/articles"
                      className="text-sm text-purple-400 hover:text-purple-300"
                    >
                      View all
                    </a>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    {featuredArticles.slice(0, 4).map((article) => (
                      <ArticleCard key={article.id} article={article} />
                    ))}
                  </div>
                </div>

                {/* Quick links */}
                <div>
                  <QuickLinks />
                </div>
              </div>

              {/* Video tutorials */}
              <section className="mb-12">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-3">
                    <Video className="w-6 h-6 text-purple-400" />
                    <h2 className="text-xl font-semibold text-white">
                      Video Tutorials
                    </h2>
                  </div>
                  <a
                    href="/help/videos"
                    className="text-sm text-purple-400 hover:text-purple-300"
                  >
                    View all videos
                  </a>
                </div>
                <div className="grid grid-cols-4 gap-4">
                  {videoTutorials.slice(0, 4).map((video) => (
                    <VideoCard key={video.id} video={video} />
                  ))}
                </div>
              </section>

              {/* FAQ + Popular articles row */}
              <div className="grid grid-cols-2 gap-6 mb-12">
                {/* FAQ */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <MessagesSquare className="w-5 h-5 text-purple-400" />
                      <h2 className="text-lg font-semibold text-white">
                        Frequently Asked Questions
                      </h2>
                    </div>
                    <a
                      href="/help/faq"
                      className="text-sm text-purple-400 hover:text-purple-300"
                    >
                      View all
                    </a>
                  </div>
                  <div className="bg-[#1a1a2e]/30 rounded-xl border border-white/5 p-4">
                    {faqs.slice(0, 5).map((faq) => (
                      <FAQAccordion
                        key={faq.id}
                        faq={faq}
                        isOpen={openFaqIds.includes(faq.id)}
                        onToggle={() => toggleFaq(faq.id)}
                      />
                    ))}
                  </div>
                </div>

                {/* Popular articles */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <TrendingUp className="w-5 h-5 text-purple-400" />
                      <h2 className="text-lg font-semibold text-white">
                        Popular Articles
                      </h2>
                    </div>
                  </div>
                  <div className="bg-[#1a1a2e]/30 rounded-xl border border-white/5 p-2">
                    {popularArticles.map((article, index) => (
                      <div key={article.id} className="flex items-center gap-3">
                        <span className="w-6 h-6 flex items-center justify-center text-xs font-medium text-gray-500">
                          {index + 1}
                        </span>
                        <div className="flex-1">
                          <ArticleCard article={article} compact />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Contact support */}
              <section>
                <ContactSupport />
              </section>
            </>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
