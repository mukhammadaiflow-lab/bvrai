"use client";

import React, { useState, useMemo } from "react";
import DashboardLayout from "../components/DashboardLayout";

// Types
type CourseStatus = "not_started" | "in_progress" | "completed" | "locked";
type CourseLevel = "beginner" | "intermediate" | "advanced" | "expert";
type ModuleType = "video" | "interactive" | "quiz" | "practice" | "reading" | "simulation";
type CertificationStatus = "locked" | "available" | "in_progress" | "earned";

interface CourseModule {
  id: string;
  title: string;
  description: string;
  type: ModuleType;
  duration: number; // minutes
  completed: boolean;
  score?: number;
  maxScore?: number;
  requiredScore?: number;
}

interface Course {
  id: string;
  title: string;
  description: string;
  category: string;
  level: CourseLevel;
  status: CourseStatus;
  progress: number;
  modules: CourseModule[];
  totalDuration: number;
  completedDuration: number;
  thumbnail: string;
  instructor: string;
  rating: number;
  enrolledCount: number;
  prerequisites: string[];
  skills: string[];
  lastAccessed?: string;
  completedAt?: string;
  certificateId?: string;
}

interface LearningPath {
  id: string;
  title: string;
  description: string;
  courses: string[];
  progress: number;
  totalDuration: number;
  completedCourses: number;
  totalCourses: number;
  level: CourseLevel;
  skills: string[];
  certification?: string;
}

interface Certification {
  id: string;
  title: string;
  description: string;
  status: CertificationStatus;
  requiredCourses: string[];
  completedCourses: number;
  totalCourses: number;
  earnedAt?: string;
  expiresAt?: string;
  badgeUrl: string;
  skills: string[];
}

interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  earnedAt: string;
  rarity: "common" | "rare" | "epic" | "legendary";
  points: number;
}

interface LeaderboardEntry {
  rank: number;
  agentId: string;
  agentName: string;
  agentAvatar: string;
  points: number;
  coursesCompleted: number;
  certificationsEarned: number;
  streak: number;
}

interface TrainingStats {
  totalCoursesCompleted: number;
  totalHoursLearned: number;
  currentStreak: number;
  longestStreak: number;
  totalPoints: number;
  rank: number;
  totalUsers: number;
  certificationsEarned: number;
  skillsAcquired: number;
  averageScore: number;
}

// Mock Data
const mockCourses: Course[] = [
  {
    id: "course-1",
    title: "Voice AI Fundamentals",
    description: "Learn the basics of voice AI technology, speech recognition, and natural language processing. This course covers everything you need to get started with voice AI agents.",
    category: "Fundamentals",
    level: "beginner",
    status: "completed",
    progress: 100,
    modules: [
      { id: "m1", title: "Introduction to Voice AI", description: "Overview of voice AI technology", type: "video", duration: 15, completed: true, score: 95, maxScore: 100 },
      { id: "m2", title: "Speech Recognition Basics", description: "How speech recognition works", type: "video", duration: 20, completed: true, score: 88, maxScore: 100 },
      { id: "m3", title: "Natural Language Processing", description: "Understanding NLP concepts", type: "interactive", duration: 30, completed: true, score: 92, maxScore: 100 },
      { id: "m4", title: "Voice Synthesis", description: "Text-to-speech technology", type: "video", duration: 25, completed: true, score: 90, maxScore: 100 },
      { id: "m5", title: "Module Quiz", description: "Test your knowledge", type: "quiz", duration: 15, completed: true, score: 85, maxScore: 100, requiredScore: 70 },
    ],
    totalDuration: 105,
    completedDuration: 105,
    thumbnail: "🎙️",
    instructor: "Dr. Sarah Chen",
    rating: 4.8,
    enrolledCount: 1250,
    prerequisites: [],
    skills: ["Voice AI", "Speech Recognition", "NLP Basics"],
    completedAt: "2024-01-15T10:30:00Z",
    certificateId: "cert-vf-001",
  },
  {
    id: "course-2",
    title: "Advanced Conversation Design",
    description: "Master the art of designing natural, engaging conversations for voice AI agents. Learn dialogue flows, persona development, and user experience best practices.",
    category: "Conversation Design",
    level: "intermediate",
    status: "in_progress",
    progress: 65,
    modules: [
      { id: "m1", title: "Conversation Flow Design", description: "Creating natural dialogue flows", type: "video", duration: 25, completed: true, score: 94, maxScore: 100 },
      { id: "m2", title: "Persona Development", description: "Building agent personalities", type: "interactive", duration: 35, completed: true, score: 91, maxScore: 100 },
      { id: "m3", title: "Handling Edge Cases", description: "Managing unexpected inputs", type: "practice", duration: 45, completed: true, score: 88, maxScore: 100 },
      { id: "m4", title: "Multi-turn Conversations", description: "Context management techniques", type: "video", duration: 30, completed: false },
      { id: "m5", title: "Error Recovery", description: "Graceful error handling", type: "interactive", duration: 40, completed: false },
      { id: "m6", title: "Practical Simulation", description: "Real-world conversation practice", type: "simulation", duration: 60, completed: false },
      { id: "m7", title: "Final Assessment", description: "Comprehensive evaluation", type: "quiz", duration: 30, completed: false, requiredScore: 80 },
    ],
    totalDuration: 265,
    completedDuration: 172,
    thumbnail: "💬",
    instructor: "Michael Torres",
    rating: 4.9,
    enrolledCount: 890,
    prerequisites: ["course-1"],
    skills: ["Conversation Design", "User Experience", "Dialogue Management"],
    lastAccessed: "2024-01-20T14:45:00Z",
  },
  {
    id: "course-3",
    title: "Customer Service Excellence",
    description: "Train your agents to deliver exceptional customer service. Learn empathy, problem-solving, and resolution techniques for voice AI interactions.",
    category: "Customer Service",
    level: "intermediate",
    status: "in_progress",
    progress: 40,
    modules: [
      { id: "m1", title: "Customer Service Principles", description: "Core principles of great service", type: "video", duration: 20, completed: true, score: 96, maxScore: 100 },
      { id: "m2", title: "Empathy in AI", description: "Building empathetic responses", type: "interactive", duration: 40, completed: true, score: 89, maxScore: 100 },
      { id: "m3", title: "Problem Identification", description: "Understanding customer issues", type: "practice", duration: 35, completed: false },
      { id: "m4", title: "Resolution Strategies", description: "Effective problem-solving", type: "video", duration: 30, completed: false },
      { id: "m5", title: "De-escalation Techniques", description: "Handling frustrated customers", type: "simulation", duration: 50, completed: false },
      { id: "m6", title: "Service Recovery", description: "Turning negatives into positives", type: "interactive", duration: 35, completed: false },
      { id: "m7", title: "Practical Assessment", description: "Real scenario evaluation", type: "quiz", duration: 25, completed: false, requiredScore: 75 },
    ],
    totalDuration: 235,
    completedDuration: 94,
    thumbnail: "🎯",
    instructor: "Jennifer Lee",
    rating: 4.7,
    enrolledCount: 1100,
    prerequisites: ["course-1"],
    skills: ["Customer Service", "Empathy", "Problem Solving"],
    lastAccessed: "2024-01-19T09:20:00Z",
  },
  {
    id: "course-4",
    title: "Sales Mastery for Voice AI",
    description: "Transform your voice agents into effective sales professionals. Learn persuasion techniques, objection handling, and closing strategies.",
    category: "Sales",
    level: "advanced",
    status: "not_started",
    progress: 0,
    modules: [
      { id: "m1", title: "Sales Psychology", description: "Understanding buyer behavior", type: "video", duration: 30, completed: false },
      { id: "m2", title: "Building Rapport", description: "Creating instant connections", type: "interactive", duration: 45, completed: false },
      { id: "m3", title: "Needs Discovery", description: "Uncovering customer needs", type: "practice", duration: 40, completed: false },
      { id: "m4", title: "Value Proposition", description: "Communicating value effectively", type: "video", duration: 35, completed: false },
      { id: "m5", title: "Objection Handling", description: "Overcoming customer objections", type: "simulation", duration: 60, completed: false },
      { id: "m6", title: "Closing Techniques", description: "Sealing the deal", type: "interactive", duration: 45, completed: false },
      { id: "m7", title: "Follow-up Strategies", description: "Post-call engagement", type: "reading", duration: 20, completed: false },
      { id: "m8", title: "Sales Simulation", description: "Complete sales scenario", type: "simulation", duration: 90, completed: false },
      { id: "m9", title: "Final Certification", description: "Sales certification exam", type: "quiz", duration: 45, completed: false, requiredScore: 85 },
    ],
    totalDuration: 410,
    completedDuration: 0,
    thumbnail: "💰",
    instructor: "David Martinez",
    rating: 4.9,
    enrolledCount: 650,
    prerequisites: ["course-1", "course-2"],
    skills: ["Sales", "Persuasion", "Negotiation", "Closing"],
  },
  {
    id: "course-5",
    title: "Technical Support Specialist",
    description: "Equip your agents with technical troubleshooting skills. Learn diagnostic techniques, solution delivery, and technical communication.",
    category: "Technical Support",
    level: "advanced",
    status: "locked",
    progress: 0,
    modules: [
      { id: "m1", title: "Technical Communication", description: "Explaining complex topics simply", type: "video", duration: 25, completed: false },
      { id: "m2", title: "Diagnostic Methodology", description: "Systematic troubleshooting", type: "interactive", duration: 50, completed: false },
      { id: "m3", title: "Common Issues Database", description: "Building knowledge repositories", type: "reading", duration: 30, completed: false },
      { id: "m4", title: "Remote Assistance", description: "Guiding users through solutions", type: "practice", duration: 55, completed: false },
      { id: "m5", title: "Escalation Procedures", description: "When and how to escalate", type: "video", duration: 20, completed: false },
      { id: "m6", title: "Technical Simulation", description: "Live troubleshooting scenarios", type: "simulation", duration: 75, completed: false },
      { id: "m7", title: "Certification Exam", description: "Technical support certification", type: "quiz", duration: 40, completed: false, requiredScore: 80 },
    ],
    totalDuration: 295,
    completedDuration: 0,
    thumbnail: "🔧",
    instructor: "Alex Kim",
    rating: 4.6,
    enrolledCount: 420,
    prerequisites: ["course-1", "course-3"],
    skills: ["Technical Support", "Troubleshooting", "Documentation"],
  },
  {
    id: "course-6",
    title: "Compliance & Regulations",
    description: "Ensure your voice AI agents meet all regulatory requirements. Learn about data privacy, consent, and industry-specific regulations.",
    category: "Compliance",
    level: "intermediate",
    status: "not_started",
    progress: 0,
    modules: [
      { id: "m1", title: "Data Privacy Fundamentals", description: "GDPR, CCPA, and beyond", type: "video", duration: 35, completed: false },
      { id: "m2", title: "Consent Management", description: "Obtaining and recording consent", type: "interactive", duration: 40, completed: false },
      { id: "m3", title: "Call Recording Laws", description: "Legal requirements by jurisdiction", type: "reading", duration: 45, completed: false },
      { id: "m4", title: "PCI-DSS Compliance", description: "Handling payment information", type: "video", duration: 30, completed: false },
      { id: "m5", title: "HIPAA for Voice AI", description: "Healthcare data protection", type: "interactive", duration: 50, completed: false },
      { id: "m6", title: "Compliance Scenarios", description: "Practical application", type: "simulation", duration: 60, completed: false },
      { id: "m7", title: "Certification Exam", description: "Compliance certification", type: "quiz", duration: 35, completed: false, requiredScore: 90 },
    ],
    totalDuration: 295,
    completedDuration: 0,
    thumbnail: "📋",
    instructor: "Rachel Adams",
    rating: 4.5,
    enrolledCount: 780,
    prerequisites: ["course-1"],
    skills: ["Compliance", "Data Privacy", "Regulatory Knowledge"],
  },
  {
    id: "course-7",
    title: "Multilingual Agent Training",
    description: "Train your agents to handle multiple languages and cultural contexts. Learn localization, cultural sensitivity, and language switching.",
    category: "Localization",
    level: "advanced",
    status: "not_started",
    progress: 0,
    modules: [
      { id: "m1", title: "Multilingual NLP", description: "Language detection and processing", type: "video", duration: 40, completed: false },
      { id: "m2", title: "Cultural Sensitivity", description: "Understanding cultural contexts", type: "interactive", duration: 50, completed: false },
      { id: "m3", title: "Language Switching", description: "Seamless language transitions", type: "practice", duration: 45, completed: false },
      { id: "m4", title: "Localization Best Practices", description: "Adapting content for regions", type: "reading", duration: 35, completed: false },
      { id: "m5", title: "Accent and Dialect Handling", description: "Understanding speech variations", type: "interactive", duration: 55, completed: false },
      { id: "m6", title: "Multilingual Simulation", description: "Cross-language scenarios", type: "simulation", duration: 70, completed: false },
      { id: "m7", title: "Final Assessment", description: "Multilingual certification", type: "quiz", duration: 40, completed: false, requiredScore: 80 },
    ],
    totalDuration: 335,
    completedDuration: 0,
    thumbnail: "🌍",
    instructor: "Maria Garcia",
    rating: 4.8,
    enrolledCount: 340,
    prerequisites: ["course-1", "course-2"],
    skills: ["Multilingual", "Localization", "Cultural Awareness"],
  },
  {
    id: "course-8",
    title: "AI Ethics & Responsible Design",
    description: "Build ethical voice AI agents. Learn about bias prevention, transparency, and responsible AI development practices.",
    category: "Ethics",
    level: "intermediate",
    status: "not_started",
    progress: 0,
    modules: [
      { id: "m1", title: "Introduction to AI Ethics", description: "Foundations of ethical AI", type: "video", duration: 30, completed: false },
      { id: "m2", title: "Bias Detection", description: "Identifying and measuring bias", type: "interactive", duration: 45, completed: false },
      { id: "m3", title: "Bias Mitigation", description: "Strategies for reducing bias", type: "practice", duration: 50, completed: false },
      { id: "m4", title: "Transparency & Explainability", description: "Making AI decisions understandable", type: "video", duration: 35, completed: false },
      { id: "m5", title: "Privacy by Design", description: "Building privacy-first agents", type: "interactive", duration: 40, completed: false },
      { id: "m6", title: "Ethical Scenarios", description: "Navigating ethical dilemmas", type: "simulation", duration: 55, completed: false },
      { id: "m7", title: "Ethics Certification", description: "Ethics certification exam", type: "quiz", duration: 30, completed: false, requiredScore: 85 },
    ],
    totalDuration: 285,
    completedDuration: 0,
    thumbnail: "⚖️",
    instructor: "Dr. James Wilson",
    rating: 4.7,
    enrolledCount: 560,
    prerequisites: ["course-1"],
    skills: ["AI Ethics", "Bias Prevention", "Responsible AI"],
  },
  {
    id: "course-9",
    title: "Performance Optimization",
    description: "Optimize your voice AI agents for speed, accuracy, and efficiency. Learn latency reduction, response optimization, and scaling techniques.",
    category: "Performance",
    level: "expert",
    status: "locked",
    progress: 0,
    modules: [
      { id: "m1", title: "Latency Analysis", description: "Measuring and reducing latency", type: "video", duration: 35, completed: false },
      { id: "m2", title: "Response Optimization", description: "Faster, better responses", type: "interactive", duration: 50, completed: false },
      { id: "m3", title: "Caching Strategies", description: "Intelligent response caching", type: "practice", duration: 45, completed: false },
      { id: "m4", title: "Load Balancing", description: "Handling high traffic", type: "video", duration: 40, completed: false },
      { id: "m5", title: "Cost Optimization", description: "Reducing operational costs", type: "interactive", duration: 35, completed: false },
      { id: "m6", title: "Monitoring & Alerting", description: "Proactive performance management", type: "practice", duration: 45, completed: false },
      { id: "m7", title: "Scaling Simulation", description: "Managing scale challenges", type: "simulation", duration: 60, completed: false },
      { id: "m8", title: "Expert Certification", description: "Performance expert exam", type: "quiz", duration: 45, completed: false, requiredScore: 85 },
    ],
    totalDuration: 355,
    completedDuration: 0,
    thumbnail: "⚡",
    instructor: "Chris Anderson",
    rating: 4.9,
    enrolledCount: 280,
    prerequisites: ["course-1", "course-2", "course-4"],
    skills: ["Performance", "Optimization", "Scaling"],
  },
  {
    id: "course-10",
    title: "Integration & API Mastery",
    description: "Master integrations and API usage for voice AI agents. Learn CRM integration, webhook handling, and third-party service connections.",
    category: "Technical",
    level: "advanced",
    status: "not_started",
    progress: 0,
    modules: [
      { id: "m1", title: "API Fundamentals", description: "REST, GraphQL, and webhooks", type: "video", duration: 35, completed: false },
      { id: "m2", title: "CRM Integration", description: "Connecting to CRM systems", type: "interactive", duration: 55, completed: false },
      { id: "m3", title: "Database Connections", description: "Real-time data access", type: "practice", duration: 50, completed: false },
      { id: "m4", title: "Authentication", description: "OAuth, API keys, and security", type: "video", duration: 40, completed: false },
      { id: "m5", title: "Error Handling", description: "Graceful integration failures", type: "interactive", duration: 35, completed: false },
      { id: "m6", title: "Webhook Processing", description: "Real-time event handling", type: "practice", duration: 45, completed: false },
      { id: "m7", title: "Integration Simulation", description: "Complex integration scenarios", type: "simulation", duration: 65, completed: false },
      { id: "m8", title: "Certification Exam", description: "Integration specialist exam", type: "quiz", duration: 40, completed: false, requiredScore: 80 },
    ],
    totalDuration: 365,
    completedDuration: 0,
    thumbnail: "🔌",
    instructor: "Lisa Park",
    rating: 4.6,
    enrolledCount: 390,
    prerequisites: ["course-1", "course-2"],
    skills: ["API", "Integration", "Webhooks"],
  },
];

const mockLearningPaths: LearningPath[] = [
  {
    id: "path-1",
    title: "Voice AI Professional",
    description: "Complete path to becoming a certified Voice AI Professional. Master fundamentals, conversation design, and customer service.",
    courses: ["course-1", "course-2", "course-3"],
    progress: 68,
    totalDuration: 605,
    completedCourses: 1,
    totalCourses: 3,
    level: "intermediate",
    skills: ["Voice AI", "Conversation Design", "Customer Service"],
    certification: "Voice AI Professional Certificate",
  },
  {
    id: "path-2",
    title: "Sales Excellence",
    description: "Become a top-performing sales agent. Learn persuasion, negotiation, and closing techniques for voice AI.",
    courses: ["course-1", "course-2", "course-4"],
    progress: 55,
    totalDuration: 780,
    completedCourses: 1,
    totalCourses: 3,
    level: "advanced",
    skills: ["Sales", "Persuasion", "Negotiation"],
    certification: "Sales Excellence Certificate",
  },
  {
    id: "path-3",
    title: "Technical Support Expert",
    description: "Master technical troubleshooting and support for voice AI agents.",
    courses: ["course-1", "course-3", "course-5"],
    progress: 47,
    totalDuration: 635,
    completedCourses: 1,
    totalCourses: 3,
    level: "advanced",
    skills: ["Technical Support", "Troubleshooting", "Problem Solving"],
    certification: "Technical Support Expert Certificate",
  },
  {
    id: "path-4",
    title: "Compliance Specialist",
    description: "Ensure your agents meet all regulatory and compliance requirements.",
    courses: ["course-1", "course-6", "course-8"],
    progress: 33,
    totalDuration: 685,
    completedCourses: 1,
    totalCourses: 3,
    level: "intermediate",
    skills: ["Compliance", "Ethics", "Regulations"],
    certification: "Compliance Specialist Certificate",
  },
  {
    id: "path-5",
    title: "Global Agent Mastery",
    description: "Train agents for global deployment with multilingual and cultural expertise.",
    courses: ["course-1", "course-2", "course-7"],
    progress: 55,
    totalDuration: 705,
    completedCourses: 1,
    totalCourses: 3,
    level: "advanced",
    skills: ["Multilingual", "Localization", "Cultural Awareness"],
    certification: "Global Agent Master Certificate",
  },
];

const mockCertifications: Certification[] = [
  {
    id: "cert-1",
    title: "Voice AI Fundamentals",
    description: "Certification for completing the Voice AI Fundamentals course",
    status: "earned",
    requiredCourses: ["course-1"],
    completedCourses: 1,
    totalCourses: 1,
    earnedAt: "2024-01-15T10:30:00Z",
    badgeUrl: "🎓",
    skills: ["Voice AI", "Speech Recognition", "NLP"],
  },
  {
    id: "cert-2",
    title: "Conversation Design Professional",
    description: "Professional certification in conversation design for voice AI",
    status: "in_progress",
    requiredCourses: ["course-1", "course-2"],
    completedCourses: 1,
    totalCourses: 2,
    badgeUrl: "💬",
    skills: ["Conversation Design", "User Experience"],
  },
  {
    id: "cert-3",
    title: "Customer Service Excellence",
    description: "Excellence certification in customer service for voice AI",
    status: "in_progress",
    requiredCourses: ["course-1", "course-3"],
    completedCourses: 1,
    totalCourses: 2,
    badgeUrl: "⭐",
    skills: ["Customer Service", "Empathy", "Problem Solving"],
  },
  {
    id: "cert-4",
    title: "Sales Mastery",
    description: "Master certification in sales techniques for voice AI",
    status: "available",
    requiredCourses: ["course-1", "course-2", "course-4"],
    completedCourses: 1,
    totalCourses: 3,
    badgeUrl: "💰",
    skills: ["Sales", "Persuasion", "Negotiation"],
  },
  {
    id: "cert-5",
    title: "Technical Support Expert",
    description: "Expert certification in technical support",
    status: "locked",
    requiredCourses: ["course-1", "course-3", "course-5"],
    completedCourses: 1,
    totalCourses: 3,
    badgeUrl: "🔧",
    skills: ["Technical Support", "Troubleshooting"],
  },
  {
    id: "cert-6",
    title: "Compliance & Regulations",
    description: "Certification in regulatory compliance for voice AI",
    status: "available",
    requiredCourses: ["course-1", "course-6"],
    completedCourses: 1,
    totalCourses: 2,
    badgeUrl: "📋",
    skills: ["Compliance", "Data Privacy"],
  },
];

const mockAchievements: Achievement[] = [
  { id: "ach-1", title: "First Steps", description: "Complete your first course", icon: "🎯", earnedAt: "2024-01-15T10:30:00Z", rarity: "common", points: 50 },
  { id: "ach-2", title: "Quick Learner", description: "Complete a course in under 2 hours", icon: "⚡", earnedAt: "2024-01-15T10:30:00Z", rarity: "rare", points: 100 },
  { id: "ach-3", title: "Perfect Score", description: "Get 100% on any quiz", icon: "💯", earnedAt: "2024-01-18T14:20:00Z", rarity: "rare", points: 150 },
  { id: "ach-4", title: "Streak Master", description: "Maintain a 7-day learning streak", icon: "🔥", earnedAt: "2024-01-20T09:00:00Z", rarity: "epic", points: 200 },
  { id: "ach-5", title: "Certified Pro", description: "Earn your first certification", icon: "🎓", earnedAt: "2024-01-15T10:30:00Z", rarity: "rare", points: 250 },
];

const mockLeaderboard: LeaderboardEntry[] = [
  { rank: 1, agentId: "agent-1", agentName: "Sales Bot Pro", agentAvatar: "🤖", points: 12500, coursesCompleted: 8, certificationsEarned: 5, streak: 45 },
  { rank: 2, agentId: "agent-2", agentName: "Support Master", agentAvatar: "🎯", points: 11200, coursesCompleted: 7, certificationsEarned: 4, streak: 32 },
  { rank: 3, agentId: "agent-3", agentName: "Customer Care AI", agentAvatar: "💬", points: 9800, coursesCompleted: 6, certificationsEarned: 4, streak: 28 },
  { rank: 4, agentId: "agent-4", agentName: "Tech Helper", agentAvatar: "🔧", points: 8500, coursesCompleted: 5, certificationsEarned: 3, streak: 21 },
  { rank: 5, agentId: "current", agentName: "Your Agent", agentAvatar: "⭐", points: 7250, coursesCompleted: 4, certificationsEarned: 2, streak: 14 },
  { rank: 6, agentId: "agent-5", agentName: "Lead Gen Bot", agentAvatar: "📈", points: 6800, coursesCompleted: 4, certificationsEarned: 2, streak: 18 },
  { rank: 7, agentId: "agent-6", agentName: "Appointment Setter", agentAvatar: "📅", points: 5900, coursesCompleted: 3, certificationsEarned: 2, streak: 12 },
  { rank: 8, agentId: "agent-7", agentName: "Survey Bot", agentAvatar: "📊", points: 4500, coursesCompleted: 3, certificationsEarned: 1, streak: 8 },
  { rank: 9, agentId: "agent-8", agentName: "Welcome Bot", agentAvatar: "👋", points: 3200, coursesCompleted: 2, certificationsEarned: 1, streak: 5 },
  { rank: 10, agentId: "agent-9", agentName: "FAQ Assistant", agentAvatar: "❓", points: 2100, coursesCompleted: 2, certificationsEarned: 0, streak: 3 },
];

const mockStats: TrainingStats = {
  totalCoursesCompleted: 4,
  totalHoursLearned: 28.5,
  currentStreak: 14,
  longestStreak: 21,
  totalPoints: 7250,
  rank: 5,
  totalUsers: 156,
  certificationsEarned: 2,
  skillsAcquired: 12,
  averageScore: 91.2,
};

// Components
const StatCard: React.FC<{ title: string; value: string | number; icon: string; trend?: number; subtitle?: string }> = ({
  title,
  value,
  icon,
  trend,
  subtitle,
}) => (
  <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
    <div className="flex items-start justify-between">
      <div>
        <p className="text-gray-400 text-sm">{title}</p>
        <p className="text-2xl font-bold text-white mt-1">{value}</p>
        {subtitle && <p className="text-gray-500 text-xs mt-1">{subtitle}</p>}
        {trend !== undefined && (
          <div className={`flex items-center gap-1 mt-2 text-sm ${trend >= 0 ? "text-green-400" : "text-red-400"}`}>
            <span>{trend >= 0 ? "↑" : "↓"}</span>
            <span>{Math.abs(trend)}% from last week</span>
          </div>
        )}
      </div>
      <div className="w-12 h-12 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center text-2xl">
        {icon}
      </div>
    </div>
  </div>
);

const ProgressRing: React.FC<{ progress: number; size?: number; strokeWidth?: number }> = ({
  progress,
  size = 60,
  strokeWidth = 6,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle cx={size / 2} cy={size / 2} r={radius} stroke="#374151" strokeWidth={strokeWidth} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="url(#progressGradient)"
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-500"
        />
        <defs>
          <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#a855f7" />
            <stop offset="100%" stopColor="#ec4899" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-sm font-bold text-white">{progress}%</span>
      </div>
    </div>
  );
};

const ModuleIcon: React.FC<{ type: ModuleType }> = ({ type }) => {
  const icons: Record<ModuleType, string> = {
    video: "🎬",
    interactive: "🎮",
    quiz: "📝",
    practice: "💪",
    reading: "📚",
    simulation: "🎯",
  };
  return <span>{icons[type]}</span>;
};

const LevelBadge: React.FC<{ level: CourseLevel }> = ({ level }) => {
  const colors: Record<CourseLevel, string> = {
    beginner: "bg-green-500/20 text-green-400 border-green-500/30",
    intermediate: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    advanced: "bg-purple-500/20 text-purple-400 border-purple-500/30",
    expert: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${colors[level]} capitalize`}>
      {level}
    </span>
  );
};

const StatusBadge: React.FC<{ status: CourseStatus }> = ({ status }) => {
  const config: Record<CourseStatus, { color: string; label: string }> = {
    not_started: { color: "bg-gray-500/20 text-gray-400 border-gray-500/30", label: "Not Started" },
    in_progress: { color: "bg-blue-500/20 text-blue-400 border-blue-500/30", label: "In Progress" },
    completed: { color: "bg-green-500/20 text-green-400 border-green-500/30", label: "Completed" },
    locked: { color: "bg-red-500/20 text-red-400 border-red-500/30", label: "Locked" },
  };

  const { color, label } = config[status];

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${color}`}>
      {label}
    </span>
  );
};

const CourseCard: React.FC<{
  course: Course;
  onSelect: (course: Course) => void;
}> = ({ course, onSelect }) => {
  return (
    <div
      className={`bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl overflow-hidden hover:border-purple-500/50 transition-all duration-300 cursor-pointer group ${
        course.status === "locked" ? "opacity-60" : ""
      }`}
      onClick={() => course.status !== "locked" && onSelect(course)}
    >
      <div className="p-6">
        <div className="flex items-start gap-4">
          <div className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center text-3xl flex-shrink-0">
            {course.thumbnail}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-2">
              <LevelBadge level={course.level} />
              <StatusBadge status={course.status} />
            </div>
            <h3 className="font-semibold text-white group-hover:text-purple-400 transition-colors truncate">
              {course.title}
            </h3>
            <p className="text-gray-400 text-sm mt-1 line-clamp-2">{course.description}</p>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-4 text-sm text-gray-400">
          <span className="flex items-center gap-1">
            <span>📚</span>
            <span>{course.modules.length} modules</span>
          </span>
          <span className="flex items-center gap-1">
            <span>⏱️</span>
            <span>{Math.floor(course.totalDuration / 60)}h {course.totalDuration % 60}m</span>
          </span>
          <span className="flex items-center gap-1">
            <span>⭐</span>
            <span>{course.rating}</span>
          </span>
        </div>

        {course.status !== "not_started" && course.status !== "locked" && (
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Progress</span>
              <span className="text-sm font-medium text-white">{course.progress}%</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-500"
                style={{ width: `${course.progress}%` }}
              />
            </div>
          </div>
        )}

        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-gray-700 rounded-full flex items-center justify-center text-xs">
              👤
            </div>
            <span className="text-sm text-gray-400">{course.instructor}</span>
          </div>
          <span className="text-xs text-gray-500">{course.enrolledCount.toLocaleString()} enrolled</span>
        </div>

        {course.status === "locked" && course.prerequisites.length > 0 && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-xs text-red-400">
              <span className="font-medium">Prerequisites required:</span> Complete prerequisite courses to unlock
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

const LearningPathCard: React.FC<{ path: LearningPath; onSelect: (path: LearningPath) => void }> = ({ path, onSelect }) => (
  <div
    className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6 hover:border-purple-500/50 transition-all duration-300 cursor-pointer"
    onClick={() => onSelect(path)}
  >
    <div className="flex items-start justify-between mb-4">
      <div>
        <LevelBadge level={path.level} />
        <h3 className="font-semibold text-white mt-2">{path.title}</h3>
        <p className="text-gray-400 text-sm mt-1 line-clamp-2">{path.description}</p>
      </div>
      <ProgressRing progress={path.progress} />
    </div>

    <div className="flex items-center gap-4 text-sm text-gray-400 mb-4">
      <span>{path.completedCourses}/{path.totalCourses} courses</span>
      <span>•</span>
      <span>{Math.floor(path.totalDuration / 60)}h total</span>
    </div>

    <div className="flex flex-wrap gap-2 mb-4">
      {path.skills.slice(0, 3).map((skill) => (
        <span key={skill} className="px-2 py-1 bg-gray-700/50 text-gray-300 text-xs rounded-full">
          {skill}
        </span>
      ))}
    </div>

    {path.certification && (
      <div className="p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
        <p className="text-xs text-purple-400 flex items-center gap-2">
          <span>🎓</span>
          <span>{path.certification}</span>
        </p>
      </div>
    )}
  </div>
);

const CertificationCard: React.FC<{ certification: Certification }> = ({ certification }) => {
  const statusConfig: Record<CertificationStatus, { color: string; label: string }> = {
    locked: { color: "bg-gray-500/20 text-gray-400 border-gray-500/30", label: "Locked" },
    available: { color: "bg-blue-500/20 text-blue-400 border-blue-500/30", label: "Available" },
    in_progress: { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", label: "In Progress" },
    earned: { color: "bg-green-500/20 text-green-400 border-green-500/30", label: "Earned" },
  };

  const { color, label } = statusConfig[certification.status];

  return (
    <div className={`bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6 ${
      certification.status === "locked" ? "opacity-60" : ""
    }`}>
      <div className="flex items-start gap-4">
        <div className="w-16 h-16 bg-gradient-to-br from-yellow-500/20 to-orange-500/20 rounded-xl flex items-center justify-center text-3xl">
          {certification.badgeUrl}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${color}`}>
              {label}
            </span>
          </div>
          <h3 className="font-semibold text-white">{certification.title}</h3>
          <p className="text-gray-400 text-sm mt-1">{certification.description}</p>
        </div>
      </div>

      <div className="mt-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-400">Requirements</span>
          <span className="text-sm font-medium text-white">
            {certification.completedCourses}/{certification.totalCourses} courses
          </span>
        </div>
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full"
            style={{ width: `${(certification.completedCourses / certification.totalCourses) * 100}%` }}
          />
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {certification.skills.map((skill) => (
          <span key={skill} className="px-2 py-1 bg-gray-700/50 text-gray-300 text-xs rounded-full">
            {skill}
          </span>
        ))}
      </div>

      {certification.earnedAt && (
        <div className="mt-4 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
          <p className="text-xs text-green-400">
            Earned on {new Date(certification.earnedAt).toLocaleDateString()}
          </p>
        </div>
      )}
    </div>
  );
};

const AchievementBadge: React.FC<{ achievement: Achievement }> = ({ achievement }) => {
  const rarityColors: Record<string, string> = {
    common: "from-gray-500 to-gray-600",
    rare: "from-blue-500 to-blue-600",
    epic: "from-purple-500 to-purple-600",
    legendary: "from-yellow-500 to-orange-500",
  };

  return (
    <div className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
      <div className={`w-12 h-12 bg-gradient-to-br ${rarityColors[achievement.rarity]} rounded-xl flex items-center justify-center text-2xl`}>
        {achievement.icon}
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="font-medium text-white text-sm truncate">{achievement.title}</h4>
        <p className="text-gray-400 text-xs truncate">{achievement.description}</p>
      </div>
      <div className="text-right">
        <p className="text-sm font-bold text-yellow-400">+{achievement.points}</p>
        <p className="text-xs text-gray-500 capitalize">{achievement.rarity}</p>
      </div>
    </div>
  );
};

const LeaderboardRow: React.FC<{ entry: LeaderboardEntry; isCurrentUser: boolean }> = ({ entry, isCurrentUser }) => (
  <div className={`flex items-center gap-4 p-4 rounded-lg ${
    isCurrentUser ? "bg-purple-500/10 border border-purple-500/30" : "bg-gray-800/30"
  }`}>
    <div className={`w-8 h-8 flex items-center justify-center font-bold text-lg ${
      entry.rank === 1 ? "text-yellow-400" :
      entry.rank === 2 ? "text-gray-300" :
      entry.rank === 3 ? "text-orange-400" :
      "text-gray-500"
    }`}>
      {entry.rank <= 3 ? ["🥇", "🥈", "🥉"][entry.rank - 1] : `#${entry.rank}`}
    </div>
    <div className="w-10 h-10 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center text-xl">
      {entry.agentAvatar}
    </div>
    <div className="flex-1">
      <p className={`font-medium ${isCurrentUser ? "text-purple-400" : "text-white"}`}>
        {entry.agentName} {isCurrentUser && "(You)"}
      </p>
      <div className="flex items-center gap-3 text-xs text-gray-400 mt-1">
        <span>{entry.coursesCompleted} courses</span>
        <span>•</span>
        <span>{entry.certificationsEarned} certs</span>
        <span>•</span>
        <span>🔥 {entry.streak} day streak</span>
      </div>
    </div>
    <div className="text-right">
      <p className="font-bold text-white">{entry.points.toLocaleString()}</p>
      <p className="text-xs text-gray-500">points</p>
    </div>
  </div>
);

const CourseDetailDialog: React.FC<{
  course: Course | null;
  onClose: () => void;
}> = ({ course, onClose }) => {
  const [activeModuleTab, setActiveModuleTab] = useState<"all" | "completed" | "remaining">("all");

  if (!course) return null;

  const filteredModules = course.modules.filter((module) => {
    if (activeModuleTab === "completed") return module.completed;
    if (activeModuleTab === "remaining") return !module.completed;
    return true;
  });

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-4">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center text-3xl">
                {course.thumbnail}
              </div>
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <LevelBadge level={course.level} />
                  <StatusBadge status={course.status} />
                </div>
                <h2 className="text-xl font-bold text-white">{course.title}</h2>
                <p className="text-gray-400 text-sm mt-1">{course.description}</p>
              </div>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {/* Course Stats */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-800/50 rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-white">{course.modules.length}</p>
              <p className="text-sm text-gray-400">Modules</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-white">{Math.floor(course.totalDuration / 60)}h {course.totalDuration % 60}m</p>
              <p className="text-sm text-gray-400">Duration</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-white">⭐ {course.rating}</p>
              <p className="text-sm text-gray-400">Rating</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-white">{course.enrolledCount.toLocaleString()}</p>
              <p className="text-sm text-gray-400">Enrolled</p>
            </div>
          </div>

          {/* Progress */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Overall Progress</span>
              <span className="text-sm font-medium text-white">{course.progress}%</span>
            </div>
            <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                style={{ width: `${course.progress}%` }}
              />
            </div>
          </div>

          {/* Skills */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Skills You'll Learn</h3>
            <div className="flex flex-wrap gap-2">
              {course.skills.map((skill) => (
                <span key={skill} className="px-3 py-1 bg-purple-500/20 text-purple-400 text-sm rounded-full border border-purple-500/30">
                  {skill}
                </span>
              ))}
            </div>
          </div>

          {/* Module Tabs */}
          <div className="flex items-center gap-2 mb-4">
            {(["all", "completed", "remaining"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveModuleTab(tab)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  activeModuleTab === tab
                    ? "bg-purple-500/20 text-purple-400 border border-purple-500/30"
                    : "text-gray-400 hover:text-white hover:bg-gray-800"
                }`}
              >
                {tab === "all" ? "All Modules" : tab === "completed" ? "Completed" : "Remaining"}
              </button>
            ))}
          </div>

          {/* Modules List */}
          <div className="space-y-3">
            {filteredModules.map((module, index) => (
              <div
                key={module.id}
                className={`p-4 rounded-lg border ${
                  module.completed
                    ? "bg-green-500/5 border-green-500/20"
                    : "bg-gray-800/50 border-gray-700/50"
                }`}
              >
                <div className="flex items-start gap-4">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    module.completed
                      ? "bg-green-500/20 text-green-400"
                      : "bg-gray-700/50 text-gray-400"
                  }`}>
                    {module.completed ? "✓" : index + 1}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h4 className="font-medium text-white">{module.title}</h4>
                      <ModuleIcon type={module.type} />
                      <span className="px-2 py-0.5 text-xs bg-gray-700/50 text-gray-400 rounded-full capitalize">
                        {module.type}
                      </span>
                    </div>
                    <p className="text-sm text-gray-400 mt-1">{module.description}</p>
                    <div className="flex items-center gap-4 mt-2 text-sm text-gray-500">
                      <span>{module.duration} min</span>
                      {module.score !== undefined && (
                        <span className="text-green-400">Score: {module.score}/{module.maxScore}</span>
                      )}
                      {module.requiredScore !== undefined && !module.completed && (
                        <span className="text-yellow-400">Required: {module.requiredScore}%</span>
                      )}
                    </div>
                  </div>
                  <button
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      module.completed
                        ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                        : "bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:opacity-90"
                    }`}
                    disabled={module.completed}
                  >
                    {module.completed ? "Completed" : "Start"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="p-6 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center">
                👤
              </div>
              <div>
                <p className="text-sm font-medium text-white">{course.instructor}</p>
                <p className="text-xs text-gray-400">Course Instructor</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-white transition-colors">
                Share Course
              </button>
              {course.status === "completed" ? (
                <button className="px-6 py-2 bg-gradient-to-r from-green-500 to-emerald-500 text-white font-medium rounded-lg">
                  View Certificate
                </button>
              ) : course.status === "in_progress" ? (
                <button className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
                  Continue Learning
                </button>
              ) : (
                <button className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
                  Start Course
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const LearningPathDetailDialog: React.FC<{
  path: LearningPath | null;
  courses: Course[];
  onClose: () => void;
  onSelectCourse: (course: Course) => void;
}> = ({ path, courses, onClose, onSelectCourse }) => {
  if (!path) return null;

  const pathCourses = path.courses.map((courseId) => courses.find((c) => c.id === courseId)).filter(Boolean) as Course[];

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-3xl max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <LevelBadge level={path.level} />
                <span className="px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded-full border border-purple-500/30">
                  Learning Path
                </span>
              </div>
              <h2 className="text-xl font-bold text-white">{path.title}</h2>
              <p className="text-gray-400 text-sm mt-1">{path.description}</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {/* Path Stats */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-800/50 rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-white">{path.completedCourses}/{path.totalCourses}</p>
              <p className="text-sm text-gray-400">Courses</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-white">{Math.floor(path.totalDuration / 60)}h</p>
              <p className="text-sm text-gray-400">Total Duration</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-white">{path.progress}%</p>
              <p className="text-sm text-gray-400">Progress</p>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mb-6">
            <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                style={{ width: `${path.progress}%` }}
              />
            </div>
          </div>

          {/* Skills */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-3">Skills You'll Acquire</h3>
            <div className="flex flex-wrap gap-2">
              {path.skills.map((skill) => (
                <span key={skill} className="px-3 py-1 bg-purple-500/20 text-purple-400 text-sm rounded-full border border-purple-500/30">
                  {skill}
                </span>
              ))}
            </div>
          </div>

          {/* Certification */}
          {path.certification && (
            <div className="mb-6 p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
              <div className="flex items-center gap-3">
                <span className="text-2xl">🎓</span>
                <div>
                  <p className="font-medium text-yellow-400">Certification Available</p>
                  <p className="text-sm text-gray-400">{path.certification}</p>
                </div>
              </div>
            </div>
          )}

          {/* Courses */}
          <h3 className="text-sm font-medium text-gray-400 mb-3">Courses in this Path</h3>
          <div className="space-y-3">
            {pathCourses.map((course, index) => (
              <div
                key={course.id}
                className={`p-4 rounded-lg border cursor-pointer transition-all ${
                  course.status === "completed"
                    ? "bg-green-500/5 border-green-500/20"
                    : course.status === "locked"
                    ? "bg-gray-800/30 border-gray-700/30 opacity-60"
                    : "bg-gray-800/50 border-gray-700/50 hover:border-purple-500/50"
                }`}
                onClick={() => course.status !== "locked" && onSelectCourse(course)}
              >
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-xl ${
                    course.status === "completed"
                      ? "bg-green-500/20"
                      : "bg-gray-700/50"
                  }`}>
                    {course.thumbnail}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500">Step {index + 1}</span>
                      <StatusBadge status={course.status} />
                    </div>
                    <h4 className="font-medium text-white mt-1">{course.title}</h4>
                    <p className="text-sm text-gray-400 mt-1">
                      {course.modules.length} modules • {Math.floor(course.totalDuration / 60)}h {course.totalDuration % 60}m
                    </p>
                  </div>
                  {course.status !== "locked" && (
                    <ProgressRing progress={course.progress} size={48} strokeWidth={4} />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="p-6 border-t border-gray-700">
          <button
            onClick={onClose}
            className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity"
          >
            Continue Learning Path
          </button>
        </div>
      </div>
    </div>
  );
};

// Main Component
export default function TrainingCenterPage() {
  const [activeTab, setActiveTab] = useState<"dashboard" | "courses" | "paths" | "certifications" | "achievements" | "leaderboard">("dashboard");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedLevel, setSelectedLevel] = useState<CourseLevel | "all">("all");
  const [selectedStatus, setSelectedStatus] = useState<CourseStatus | "all">("all");
  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null);
  const [selectedPath, setSelectedPath] = useState<LearningPath | null>(null);

  const categories = useMemo(() => {
    const cats = new Set(mockCourses.map((c) => c.category));
    return ["all", ...Array.from(cats)];
  }, []);

  const filteredCourses = useMemo(() => {
    return mockCourses.filter((course) => {
      const matchesSearch =
        course.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        course.description.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesCategory = selectedCategory === "all" || course.category === selectedCategory;
      const matchesLevel = selectedLevel === "all" || course.level === selectedLevel;
      const matchesStatus = selectedStatus === "all" || course.status === selectedStatus;
      return matchesSearch && matchesCategory && matchesLevel && matchesStatus;
    });
  }, [searchQuery, selectedCategory, selectedLevel, selectedStatus]);

  const tabs = [
    { id: "dashboard", label: "Dashboard", icon: "📊" },
    { id: "courses", label: "Courses", icon: "📚" },
    { id: "paths", label: "Learning Paths", icon: "🛤️" },
    { id: "certifications", label: "Certifications", icon: "🎓" },
    { id: "achievements", label: "Achievements", icon: "🏆" },
    { id: "leaderboard", label: "Leaderboard", icon: "🏅" },
  ];

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-900">
        {/* Header */}
        <div className="border-b border-gray-800 bg-gray-900/95 backdrop-blur-sm sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-2xl font-bold text-white">Agent Training Center</h1>
                <p className="text-gray-400 mt-1">Develop your AI agent's skills and capabilities</p>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 px-4 py-2 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                  <span className="text-xl">🔥</span>
                  <span className="text-purple-400 font-medium">{mockStats.currentStreak} day streak</span>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <span className="text-xl">⭐</span>
                  <span className="text-yellow-400 font-medium">{mockStats.totalPoints.toLocaleString()} points</span>
                </div>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex items-center gap-2 overflow-x-auto pb-2">
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
          {/* Dashboard Tab */}
          {activeTab === "dashboard" && (
            <div className="space-y-8">
              {/* Stats Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard title="Courses Completed" value={mockStats.totalCoursesCompleted} icon="📚" trend={12} />
                <StatCard title="Hours Learned" value={`${mockStats.totalHoursLearned}h`} icon="⏱️" trend={8} />
                <StatCard title="Certifications" value={mockStats.certificationsEarned} icon="🎓" />
                <StatCard
                  title="Leaderboard Rank"
                  value={`#${mockStats.rank}`}
                  icon="🏅"
                  subtitle={`of ${mockStats.totalUsers} agents`}
                />
              </div>

              {/* Continue Learning */}
              <div>
                <h2 className="text-lg font-semibold text-white mb-4">Continue Learning</h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {mockCourses
                    .filter((c) => c.status === "in_progress")
                    .slice(0, 2)
                    .map((course) => (
                      <CourseCard key={course.id} course={course} onSelect={setSelectedCourse} />
                    ))}
                </div>
              </div>

              {/* Two Column Layout */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Recent Achievements */}
                <div>
                  <h2 className="text-lg font-semibold text-white mb-4">Recent Achievements</h2>
                  <div className="space-y-3">
                    {mockAchievements.slice(0, 4).map((achievement) => (
                      <AchievementBadge key={achievement.id} achievement={achievement} />
                    ))}
                  </div>
                </div>

                {/* Active Certifications */}
                <div>
                  <h2 className="text-lg font-semibold text-white mb-4">Certification Progress</h2>
                  <div className="space-y-4">
                    {mockCertifications
                      .filter((c) => c.status === "in_progress" || c.status === "available")
                      .slice(0, 3)
                      .map((cert) => (
                        <div key={cert.id} className="bg-gray-800/50 rounded-lg p-4">
                          <div className="flex items-center gap-3">
                            <div className="text-2xl">{cert.badgeUrl}</div>
                            <div className="flex-1">
                              <h4 className="font-medium text-white">{cert.title}</h4>
                              <div className="flex items-center gap-2 mt-2">
                                <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                                    style={{ width: `${(cert.completedCourses / cert.totalCourses) * 100}%` }}
                                  />
                                </div>
                                <span className="text-xs text-gray-400">
                                  {cert.completedCourses}/{cert.totalCourses}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>

              {/* Recommended Courses */}
              <div>
                <h2 className="text-lg font-semibold text-white mb-4">Recommended for You</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {mockCourses
                    .filter((c) => c.status === "not_started")
                    .slice(0, 3)
                    .map((course) => (
                      <CourseCard key={course.id} course={course} onSelect={setSelectedCourse} />
                    ))}
                </div>
              </div>
            </div>
          )}

          {/* Courses Tab */}
          {activeTab === "courses" && (
            <div className="space-y-6">
              {/* Filters */}
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex-1 min-w-[200px]">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search courses..."
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
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  {categories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat === "all" ? "All Categories" : cat}
                    </option>
                  ))}
                </select>

                <select
                  value={selectedLevel}
                  onChange={(e) => setSelectedLevel(e.target.value as CourseLevel | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Levels</option>
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                  <option value="expert">Expert</option>
                </select>

                <select
                  value={selectedStatus}
                  onChange={(e) => setSelectedStatus(e.target.value as CourseStatus | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Status</option>
                  <option value="not_started">Not Started</option>
                  <option value="in_progress">In Progress</option>
                  <option value="completed">Completed</option>
                  <option value="locked">Locked</option>
                </select>
              </div>

              {/* Course Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredCourses.map((course) => (
                  <CourseCard key={course.id} course={course} onSelect={setSelectedCourse} />
                ))}
              </div>

              {filteredCourses.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-gray-400">No courses found matching your criteria</p>
                </div>
              )}
            </div>
          )}

          {/* Learning Paths Tab */}
          {activeTab === "paths" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {mockLearningPaths.map((path) => (
                  <LearningPathCard key={path.id} path={path} onSelect={setSelectedPath} />
                ))}
              </div>
            </div>
          )}

          {/* Certifications Tab */}
          {activeTab === "certifications" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {mockCertifications.map((cert) => (
                  <CertificationCard key={cert.id} certification={cert} />
                ))}
              </div>
            </div>
          )}

          {/* Achievements Tab */}
          {activeTab === "achievements" && (
            <div className="space-y-8">
              {/* Stats */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <StatCard title="Total Achievements" value={mockAchievements.length} icon="🏆" />
                <StatCard
                  title="Total Points"
                  value={mockAchievements.reduce((sum, a) => sum + a.points, 0)}
                  icon="⭐"
                />
                <StatCard
                  title="Legendary Badges"
                  value={mockAchievements.filter((a) => a.rarity === "legendary").length}
                  icon="👑"
                />
              </div>

              {/* Achievements by Rarity */}
              {(["legendary", "epic", "rare", "common"] as const).map((rarity) => {
                const achievements = mockAchievements.filter((a) => a.rarity === rarity);
                if (achievements.length === 0) return null;

                return (
                  <div key={rarity}>
                    <h2 className="text-lg font-semibold text-white mb-4 capitalize flex items-center gap-2">
                      {rarity === "legendary" && "👑"}
                      {rarity === "epic" && "💎"}
                      {rarity === "rare" && "💫"}
                      {rarity === "common" && "⭐"}
                      {rarity} Achievements
                      <span className="text-sm font-normal text-gray-400">({achievements.length})</span>
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {achievements.map((achievement) => (
                        <AchievementBadge key={achievement.id} achievement={achievement} />
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Leaderboard Tab */}
          {activeTab === "leaderboard" && (
            <div className="space-y-6">
              {/* Top 3 Podium */}
              <div className="flex items-end justify-center gap-4 mb-8">
                {[1, 0, 2].map((index) => {
                  const entry = mockLeaderboard[index];
                  if (!entry) return null;
                  const heights = ["h-32", "h-40", "h-24"];
                  const podiumColors = ["bg-gray-400", "bg-yellow-500", "bg-orange-600"];

                  return (
                    <div key={entry.agentId} className="flex flex-col items-center">
                      <div className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center text-3xl mb-2">
                        {entry.agentAvatar}
                      </div>
                      <p className="text-white font-medium text-center mb-1">{entry.agentName}</p>
                      <p className="text-yellow-400 font-bold mb-2">{entry.points.toLocaleString()}</p>
                      <div className={`w-24 ${heights[index]} ${podiumColors[index]} rounded-t-lg flex items-start justify-center pt-4`}>
                        <span className="text-2xl font-bold text-white">#{entry.rank}</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Full Leaderboard */}
              <div className="space-y-2">
                {mockLeaderboard.map((entry) => (
                  <LeaderboardRow
                    key={entry.agentId}
                    entry={entry}
                    isCurrentUser={entry.agentId === "current"}
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Dialogs */}
        <CourseDetailDialog course={selectedCourse} onClose={() => setSelectedCourse(null)} />
        <LearningPathDetailDialog
          path={selectedPath}
          courses={mockCourses}
          onClose={() => setSelectedPath(null)}
          onSelectCourse={(course) => {
            setSelectedPath(null);
            setSelectedCourse(course);
          }}
        />
      </div>
    </DashboardLayout>
  );
}
