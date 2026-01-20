"use client";

import React, { useState, useMemo } from "react";
import DashboardLayout from "@/components/DashboardLayout";

// Types
type SecurityLevel = "critical" | "high" | "medium" | "low";
type ComplianceStatus = "compliant" | "non_compliant" | "partial" | "pending_review";
type AlertStatus = "active" | "investigating" | "resolved" | "false_positive";
type AuditAction = "login" | "logout" | "api_call" | "data_access" | "setting_change" | "permission_change" | "file_access" | "export";

interface SecurityScore {
  overall: number;
  authentication: number;
  dataProtection: number;
  accessControl: number;
  networkSecurity: number;
  monitoring: number;
}

interface SecurityAlert {
  id: string;
  title: string;
  description: string;
  level: SecurityLevel;
  status: AlertStatus;
  source: string;
  detectedAt: string;
  resolvedAt?: string;
  affectedResources: string[];
  recommendedAction: string;
}

interface ComplianceFramework {
  id: string;
  name: string;
  shortName: string;
  description: string;
  status: ComplianceStatus;
  lastAudit: string;
  nextAudit: string;
  controls: { total: number; passed: number; failed: number; pending: number };
  certificate?: string;
  validUntil?: string;
}

interface AuditLogEntry {
  id: string;
  action: AuditAction;
  user: string;
  userEmail: string;
  ipAddress: string;
  location: string;
  resource: string;
  details: string;
  timestamp: string;
  success: boolean;
  riskLevel: SecurityLevel;
}

interface SecurityPolicy {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  category: string;
  lastUpdated: string;
  enforcement: "strict" | "flexible" | "audit_only";
}

interface DataClassification {
  level: string;
  name: string;
  description: string;
  color: string;
  dataCount: number;
  retention: string;
}

interface AccessReview {
  id: string;
  user: string;
  role: string;
  department: string;
  lastReview: string;
  nextReview: string;
  status: "approved" | "pending" | "revoked" | "escalated";
  accessLevel: "admin" | "manager" | "user" | "viewer";
}

// Mock Data
const mockSecurityScore: SecurityScore = {
  overall: 87,
  authentication: 95,
  dataProtection: 88,
  accessControl: 82,
  networkSecurity: 90,
  monitoring: 80,
};

const mockAlerts: SecurityAlert[] = [
  {
    id: "alert-1",
    title: "Multiple failed login attempts detected",
    description: "15 failed login attempts from IP 192.168.1.100 in the last hour targeting user admin@company.com",
    level: "high",
    status: "investigating",
    source: "Authentication Monitor",
    detectedAt: "2024-01-20T10:30:00Z",
    affectedResources: ["admin@company.com", "Login System"],
    recommendedAction: "Review IP address and consider temporary block. Verify user account security.",
  },
  {
    id: "alert-2",
    title: "Unusual data export activity",
    description: "Large volume data export (2.5GB) initiated by john.doe@company.com outside normal working hours",
    level: "medium",
    status: "active",
    source: "Data Loss Prevention",
    detectedAt: "2024-01-20T08:15:00Z",
    affectedResources: ["Customer Database", "Export Service"],
    recommendedAction: "Contact user to verify legitimate business need for export.",
  },
  {
    id: "alert-3",
    title: "API rate limit exceeded",
    description: "API key 'prod-key-123' exceeded rate limit with 50,000 requests in 5 minutes",
    level: "low",
    status: "resolved",
    source: "API Gateway",
    detectedAt: "2024-01-19T16:45:00Z",
    resolvedAt: "2024-01-19T17:00:00Z",
    affectedResources: ["API Gateway", "Rate Limiter"],
    recommendedAction: "Review API usage patterns and consider rate limit adjustment.",
  },
  {
    id: "alert-4",
    title: "Potential SQL injection attempt",
    description: "Suspicious query pattern detected in API request from IP 10.0.0.50",
    level: "critical",
    status: "resolved",
    source: "Web Application Firewall",
    detectedAt: "2024-01-19T14:20:00Z",
    resolvedAt: "2024-01-19T14:25:00Z",
    affectedResources: ["API Endpoint /api/search", "Database"],
    recommendedAction: "IP has been blocked. Review and patch the vulnerable endpoint.",
  },
  {
    id: "alert-5",
    title: "Expired SSL certificate warning",
    description: "SSL certificate for api.example.com will expire in 7 days",
    level: "medium",
    status: "active",
    source: "Certificate Monitor",
    detectedAt: "2024-01-20T00:00:00Z",
    affectedResources: ["api.example.com"],
    recommendedAction: "Renew SSL certificate before expiration.",
  },
];

const mockComplianceFrameworks: ComplianceFramework[] = [
  {
    id: "soc2",
    name: "SOC 2 Type II",
    shortName: "SOC 2",
    description: "Service Organization Control 2 - Trust Services Criteria",
    status: "compliant",
    lastAudit: "2024-01-10T00:00:00Z",
    nextAudit: "2025-01-10T00:00:00Z",
    controls: { total: 116, passed: 116, failed: 0, pending: 0 },
    certificate: "SOC2-2024-001",
    validUntil: "2025-01-10T00:00:00Z",
  },
  {
    id: "gdpr",
    name: "General Data Protection Regulation",
    shortName: "GDPR",
    description: "EU data protection and privacy regulation",
    status: "compliant",
    lastAudit: "2024-01-05T00:00:00Z",
    nextAudit: "2024-07-05T00:00:00Z",
    controls: { total: 45, passed: 43, failed: 0, pending: 2 },
  },
  {
    id: "hipaa",
    name: "Health Insurance Portability and Accountability Act",
    shortName: "HIPAA",
    description: "US healthcare data protection standard",
    status: "partial",
    lastAudit: "2023-12-15T00:00:00Z",
    nextAudit: "2024-06-15T00:00:00Z",
    controls: { total: 54, passed: 48, failed: 3, pending: 3 },
  },
  {
    id: "pci",
    name: "Payment Card Industry Data Security Standard",
    shortName: "PCI-DSS",
    description: "Payment card data security requirements",
    status: "compliant",
    lastAudit: "2024-01-08T00:00:00Z",
    nextAudit: "2025-01-08T00:00:00Z",
    controls: { total: 78, passed: 78, failed: 0, pending: 0 },
    certificate: "PCI-DSS-2024-001",
    validUntil: "2025-01-08T00:00:00Z",
  },
  {
    id: "ccpa",
    name: "California Consumer Privacy Act",
    shortName: "CCPA",
    description: "California data privacy law",
    status: "compliant",
    lastAudit: "2024-01-12T00:00:00Z",
    nextAudit: "2024-07-12T00:00:00Z",
    controls: { total: 32, passed: 32, failed: 0, pending: 0 },
  },
  {
    id: "iso27001",
    name: "ISO/IEC 27001",
    shortName: "ISO 27001",
    description: "International information security standard",
    status: "pending_review",
    lastAudit: "2023-11-20T00:00:00Z",
    nextAudit: "2024-02-20T00:00:00Z",
    controls: { total: 114, passed: 98, failed: 5, pending: 11 },
  },
];

const mockAuditLogs: AuditLogEntry[] = [
  {
    id: "log-1",
    action: "login",
    user: "John Doe",
    userEmail: "john.doe@company.com",
    ipAddress: "192.168.1.100",
    location: "San Francisco, CA",
    resource: "Dashboard",
    details: "Successful login via SSO",
    timestamp: "2024-01-20T10:30:00Z",
    success: true,
    riskLevel: "low",
  },
  {
    id: "log-2",
    action: "data_access",
    user: "Jane Smith",
    userEmail: "jane.smith@company.com",
    ipAddress: "10.0.0.50",
    location: "New York, NY",
    resource: "Customer Database",
    details: "Accessed customer records (100 rows)",
    timestamp: "2024-01-20T10:25:00Z",
    success: true,
    riskLevel: "medium",
  },
  {
    id: "log-3",
    action: "setting_change",
    user: "Admin User",
    userEmail: "admin@company.com",
    ipAddress: "172.16.0.1",
    location: "London, UK",
    resource: "Security Settings",
    details: "Updated password policy - min length changed from 8 to 12",
    timestamp: "2024-01-20T10:20:00Z",
    success: true,
    riskLevel: "high",
  },
  {
    id: "log-4",
    action: "permission_change",
    user: "Admin User",
    userEmail: "admin@company.com",
    ipAddress: "172.16.0.1",
    location: "London, UK",
    resource: "User Permissions",
    details: "Granted admin role to mike.wilson@company.com",
    timestamp: "2024-01-20T10:15:00Z",
    success: true,
    riskLevel: "critical",
  },
  {
    id: "log-5",
    action: "export",
    user: "Sarah Johnson",
    userEmail: "sarah.johnson@company.com",
    ipAddress: "192.168.1.105",
    location: "Austin, TX",
    resource: "Call Recordings",
    details: "Exported 50 call recordings (2.5GB)",
    timestamp: "2024-01-20T10:10:00Z",
    success: true,
    riskLevel: "medium",
  },
  {
    id: "log-6",
    action: "api_call",
    user: "API Service",
    userEmail: "api@system.internal",
    ipAddress: "10.0.0.100",
    location: "Internal",
    resource: "Voice API",
    details: "Initiated outbound call to +1-555-0123",
    timestamp: "2024-01-20T10:05:00Z",
    success: true,
    riskLevel: "low",
  },
  {
    id: "log-7",
    action: "login",
    user: "Unknown",
    userEmail: "admin@company.com",
    ipAddress: "203.0.113.50",
    location: "Unknown Location",
    resource: "Login Portal",
    details: "Failed login attempt - incorrect password (attempt 5)",
    timestamp: "2024-01-20T10:00:00Z",
    success: false,
    riskLevel: "high",
  },
];

const mockPolicies: SecurityPolicy[] = [
  { id: "pol-1", name: "Multi-Factor Authentication", description: "Require MFA for all user accounts", enabled: true, category: "Authentication", lastUpdated: "2024-01-15T00:00:00Z", enforcement: "strict" },
  { id: "pol-2", name: "Password Complexity", description: "Minimum 12 characters with mixed case, numbers, and symbols", enabled: true, category: "Authentication", lastUpdated: "2024-01-20T00:00:00Z", enforcement: "strict" },
  { id: "pol-3", name: "Session Timeout", description: "Automatic logout after 30 minutes of inactivity", enabled: true, category: "Access Control", lastUpdated: "2024-01-10T00:00:00Z", enforcement: "strict" },
  { id: "pol-4", name: "Data Encryption at Rest", description: "AES-256 encryption for all stored data", enabled: true, category: "Data Protection", lastUpdated: "2024-01-05T00:00:00Z", enforcement: "strict" },
  { id: "pol-5", name: "API Rate Limiting", description: "Limit API requests to 1000/minute per key", enabled: true, category: "Network Security", lastUpdated: "2024-01-12T00:00:00Z", enforcement: "flexible" },
  { id: "pol-6", name: "Audit Logging", description: "Log all security-relevant events", enabled: true, category: "Monitoring", lastUpdated: "2024-01-08T00:00:00Z", enforcement: "strict" },
  { id: "pol-7", name: "Data Retention", description: "Automatically delete data older than 2 years", enabled: true, category: "Data Protection", lastUpdated: "2024-01-01T00:00:00Z", enforcement: "audit_only" },
  { id: "pol-8", name: "IP Allowlisting", description: "Restrict access to approved IP ranges", enabled: false, category: "Network Security", lastUpdated: "2023-12-20T00:00:00Z", enforcement: "flexible" },
];

const mockDataClassifications: DataClassification[] = [
  { level: "public", name: "Public", description: "Information that can be freely shared", color: "green", dataCount: 1250, retention: "Indefinite" },
  { level: "internal", name: "Internal", description: "Internal business information", color: "blue", dataCount: 8450, retention: "7 years" },
  { level: "confidential", name: "Confidential", description: "Sensitive business data", color: "yellow", dataCount: 3200, retention: "5 years" },
  { level: "restricted", name: "Restricted", description: "Highly sensitive data requiring strict controls", color: "red", dataCount: 890, retention: "3 years" },
];

const mockAccessReviews: AccessReview[] = [
  { id: "ar-1", user: "John Doe", role: "Developer", department: "Engineering", lastReview: "2024-01-15T00:00:00Z", nextReview: "2024-04-15T00:00:00Z", status: "approved", accessLevel: "user" },
  { id: "ar-2", user: "Jane Smith", role: "Manager", department: "Sales", lastReview: "2024-01-10T00:00:00Z", nextReview: "2024-04-10T00:00:00Z", status: "approved", accessLevel: "manager" },
  { id: "ar-3", user: "Mike Wilson", role: "Admin", department: "IT", lastReview: "2024-01-20T00:00:00Z", nextReview: "2024-04-20T00:00:00Z", status: "pending", accessLevel: "admin" },
  { id: "ar-4", user: "Sarah Johnson", role: "Analyst", department: "Analytics", lastReview: "2023-12-01T00:00:00Z", nextReview: "2024-03-01T00:00:00Z", status: "escalated", accessLevel: "user" },
];

// Components
const SecurityScoreGauge: React.FC<{ score: number; size?: "sm" | "md" | "lg" }> = ({ score, size = "lg" }) => {
  const sizeClasses = { sm: "w-24 h-24", md: "w-32 h-32", lg: "w-40 h-40" };
  const radius = size === "sm" ? 40 : size === "md" ? 56 : 72;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;
  const center = size === "sm" ? 48 : size === "md" ? 64 : 80;

  const getScoreColor = () => {
    if (score >= 90) return "#22c55e";
    if (score >= 70) return "#eab308";
    if (score >= 50) return "#f97316";
    return "#ef4444";
  };

  const getScoreLabel = () => {
    if (score >= 90) return "Excellent";
    if (score >= 70) return "Good";
    if (score >= 50) return "Fair";
    return "Poor";
  };

  return (
    <div className={`relative ${sizeClasses[size]}`}>
      <svg className="w-full h-full transform -rotate-90">
        <circle cx={center} cy={center} r={radius} stroke="#374151" strokeWidth={size === "sm" ? 6 : 8} fill="none" />
        <circle
          cx={center}
          cy={center}
          r={radius}
          stroke={getScoreColor()}
          strokeWidth={size === "sm" ? 6 : 8}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-1000"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`font-bold ${size === "sm" ? "text-xl" : size === "md" ? "text-2xl" : "text-4xl"}`} style={{ color: getScoreColor() }}>
          {score}
        </span>
        {size === "lg" && <span className="text-gray-400 text-sm mt-1">{getScoreLabel()}</span>}
      </div>
    </div>
  );
};

const AlertLevelBadge: React.FC<{ level: SecurityLevel }> = ({ level }) => {
  const config: Record<SecurityLevel, { color: string; icon: string }> = {
    critical: { color: "bg-red-500/20 text-red-400 border-red-500/30", icon: "🔴" },
    high: { color: "bg-orange-500/20 text-orange-400 border-orange-500/30", icon: "🟠" },
    medium: { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", icon: "🟡" },
    low: { color: "bg-green-500/20 text-green-400 border-green-500/30", icon: "🟢" },
  };

  const { color, icon } = config[level];

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${color} capitalize`}>
      <span>{icon}</span>
      <span>{level}</span>
    </span>
  );
};

const AlertStatusBadge: React.FC<{ status: AlertStatus }> = ({ status }) => {
  const config: Record<AlertStatus, { color: string; label: string }> = {
    active: { color: "bg-red-500/20 text-red-400 border-red-500/30", label: "Active" },
    investigating: { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", label: "Investigating" },
    resolved: { color: "bg-green-500/20 text-green-400 border-green-500/30", label: "Resolved" },
    false_positive: { color: "bg-gray-500/20 text-gray-400 border-gray-500/30", label: "False Positive" },
  };

  const { color, label } = config[status];

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${color}`}>{label}</span>
  );
};

const ComplianceStatusBadge: React.FC<{ status: ComplianceStatus }> = ({ status }) => {
  const config: Record<ComplianceStatus, { color: string; label: string; icon: string }> = {
    compliant: { color: "bg-green-500/20 text-green-400 border-green-500/30", label: "Compliant", icon: "✓" },
    non_compliant: { color: "bg-red-500/20 text-red-400 border-red-500/30", label: "Non-Compliant", icon: "✗" },
    partial: { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", label: "Partial", icon: "⚠" },
    pending_review: { color: "bg-blue-500/20 text-blue-400 border-blue-500/30", label: "Pending Review", icon: "⏳" },
  };

  const { color, label, icon } = config[status];

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${color}`}>
      <span>{icon}</span>
      <span>{label}</span>
    </span>
  );
};

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  icon: string;
  subtitle?: string;
  trend?: number;
  color?: string;
}> = ({ title, value, icon, subtitle, trend, color = "purple" }) => {
  const colorClasses: Record<string, string> = {
    purple: "from-purple-500/20 to-pink-500/20",
    green: "from-green-500/20 to-emerald-500/20",
    yellow: "from-yellow-500/20 to-orange-500/20",
    red: "from-red-500/20 to-rose-500/20",
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-gray-400 text-sm">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {subtitle && <p className="text-gray-500 text-xs mt-1">{subtitle}</p>}
          {trend !== undefined && (
            <div className={`flex items-center gap-1 mt-2 text-sm ${trend >= 0 ? "text-green-400" : "text-red-400"}`}>
              <span>{trend >= 0 ? "↑" : "↓"}</span>
              <span>{Math.abs(trend)}% vs last month</span>
            </div>
          )}
        </div>
        <div className={`w-12 h-12 bg-gradient-to-br ${colorClasses[color]} rounded-xl flex items-center justify-center text-2xl`}>
          {icon}
        </div>
      </div>
    </div>
  );
};

const AlertCard: React.FC<{ alert: SecurityAlert; onSelect: (alert: SecurityAlert) => void }> = ({ alert, onSelect }) => (
  <div
    className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5 hover:border-purple-500/50 transition-all duration-300 cursor-pointer"
    onClick={() => onSelect(alert)}
  >
    <div className="flex items-start gap-4">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
        alert.level === "critical" ? "bg-red-500/20" :
        alert.level === "high" ? "bg-orange-500/20" :
        alert.level === "medium" ? "bg-yellow-500/20" :
        "bg-green-500/20"
      }`}>
        <span className="text-xl">
          {alert.level === "critical" ? "🚨" :
           alert.level === "high" ? "⚠️" :
           alert.level === "medium" ? "⚡" : "ℹ️"}
        </span>
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap mb-2">
          <AlertLevelBadge level={alert.level} />
          <AlertStatusBadge status={alert.status} />
        </div>
        <h3 className="font-semibold text-white">{alert.title}</h3>
        <p className="text-gray-400 text-sm mt-1 line-clamp-2">{alert.description}</p>
        <div className="flex items-center gap-4 mt-3 text-xs text-gray-500">
          <span>{alert.source}</span>
          <span>•</span>
          <span>{new Date(alert.detectedAt).toLocaleString()}</span>
        </div>
      </div>
      <span className="text-gray-400">→</span>
    </div>
  </div>
);

const ComplianceCard: React.FC<{ framework: ComplianceFramework }> = ({ framework }) => {
  const passRate = Math.round((framework.controls.passed / framework.controls.total) * 100);

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <ComplianceStatusBadge status={framework.status} />
          </div>
          <h3 className="font-semibold text-white">{framework.shortName}</h3>
          <p className="text-gray-400 text-sm mt-1">{framework.name}</p>
        </div>
        {framework.certificate && (
          <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center text-2xl">
            🏆
          </div>
        )}
      </div>

      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-400">Controls</span>
          <span className="text-sm text-white font-medium">{passRate}% passed</span>
        </div>
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${
              passRate >= 90 ? "bg-green-500" : passRate >= 70 ? "bg-yellow-500" : "bg-red-500"
            }`}
            style={{ width: `${passRate}%` }}
          />
        </div>
        <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
          <span>{framework.controls.passed} passed</span>
          <span>{framework.controls.failed} failed</span>
          <span>{framework.controls.pending} pending</span>
        </div>
      </div>

      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-500">Last Audit: {new Date(framework.lastAudit).toLocaleDateString()}</span>
        <button className="text-purple-400 hover:text-purple-300">View Details</button>
      </div>
    </div>
  );
};

const AuditLogRow: React.FC<{ entry: AuditLogEntry }> = ({ entry }) => {
  const actionIcons: Record<AuditAction, string> = {
    login: "🔐",
    logout: "🚪",
    api_call: "🔌",
    data_access: "📊",
    setting_change: "⚙️",
    permission_change: "👤",
    file_access: "📁",
    export: "📤",
  };

  return (
    <div className={`flex items-center gap-4 p-4 rounded-lg ${entry.success ? "bg-gray-800/30" : "bg-red-500/5 border border-red-500/20"}`}>
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
        entry.success ? "bg-gray-700/50" : "bg-red-500/20"
      }`}>
        <span>{actionIcons[entry.action]}</span>
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-white">{entry.user}</span>
          <AlertLevelBadge level={entry.riskLevel} />
          {!entry.success && (
            <span className="px-2 py-0.5 text-xs bg-red-500/20 text-red-400 rounded-full">Failed</span>
          )}
        </div>
        <p className="text-sm text-gray-400 truncate">{entry.details}</p>
      </div>
      <div className="text-right text-sm">
        <p className="text-gray-400">{entry.ipAddress}</p>
        <p className="text-gray-500">{new Date(entry.timestamp).toLocaleTimeString()}</p>
      </div>
    </div>
  );
};

const PolicyCard: React.FC<{ policy: SecurityPolicy; onToggle: (id: string) => void }> = ({ policy, onToggle }) => (
  <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
    <div className="flex items-center gap-4">
      <div className={`w-3 h-3 rounded-full ${policy.enabled ? "bg-green-500" : "bg-gray-500"}`} />
      <div>
        <div className="flex items-center gap-2">
          <span className="font-medium text-white">{policy.name}</span>
          <span className={`px-2 py-0.5 text-xs rounded-full ${
            policy.enforcement === "strict" ? "bg-red-500/20 text-red-400" :
            policy.enforcement === "flexible" ? "bg-yellow-500/20 text-yellow-400" :
            "bg-gray-500/20 text-gray-400"
          }`}>
            {policy.enforcement.replace("_", " ")}
          </span>
        </div>
        <p className="text-sm text-gray-400 mt-1">{policy.description}</p>
      </div>
    </div>
    <button
      onClick={() => onToggle(policy.id)}
      className={`w-12 h-6 rounded-full relative transition-colors ${
        policy.enabled ? "bg-purple-500" : "bg-gray-700"
      }`}
    >
      <div className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
        policy.enabled ? "translate-x-6" : "translate-x-0.5"
      }`} />
    </button>
  </div>
);

const AlertDetailDialog: React.FC<{
  alert: SecurityAlert | null;
  onClose: () => void;
}> = ({ alert, onClose }) => {
  if (!alert) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <AlertLevelBadge level={alert.level} />
                <AlertStatusBadge status={alert.status} />
              </div>
              <h2 className="text-xl font-bold text-white">{alert.title}</h2>
              <p className="text-gray-400 text-sm mt-1">
                Detected by {alert.source} on {new Date(alert.detectedAt).toLocaleString()}
              </p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-2">Description</h3>
              <p className="text-gray-300">{alert.description}</p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-2">Affected Resources</h3>
              <div className="flex flex-wrap gap-2">
                {alert.affectedResources.map((resource) => (
                  <span key={resource} className="px-3 py-1 bg-gray-700/50 text-gray-300 text-sm rounded-full">
                    {resource}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-400 mb-2">Recommended Action</h3>
              <div className="p-4 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                <p className="text-purple-300">{alert.recommendedAction}</p>
              </div>
            </div>

            {alert.resolvedAt && (
              <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                <p className="text-green-400">
                  Resolved on {new Date(alert.resolvedAt).toLocaleString()}
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="p-6 border-t border-gray-700">
          <div className="flex items-center justify-end gap-3">
            {alert.status !== "resolved" && (
              <>
                <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
                  Mark as False Positive
                </button>
                <button className="px-4 py-2 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 transition-colors">
                  Mark Resolved
                </button>
              </>
            )}
            <button
              onClick={onClose}
              className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Component
export default function SecurityCenterPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "alerts" | "compliance" | "audit" | "policies" | "access">("overview");
  const [selectedAlert, setSelectedAlert] = useState<SecurityAlert | null>(null);
  const [alertFilter, setAlertFilter] = useState<SecurityLevel | "all">("all");
  const [alertStatusFilter, setAlertStatusFilter] = useState<AlertStatus | "all">("all");

  const filteredAlerts = useMemo(() => {
    return mockAlerts.filter((alert) => {
      const matchesLevel = alertFilter === "all" || alert.level === alertFilter;
      const matchesStatus = alertStatusFilter === "all" || alert.status === alertStatusFilter;
      return matchesLevel && matchesStatus;
    });
  }, [alertFilter, alertStatusFilter]);

  const activeAlerts = mockAlerts.filter((a) => a.status === "active" || a.status === "investigating");
  const criticalAlerts = mockAlerts.filter((a) => a.level === "critical" && a.status !== "resolved");

  const tabs = [
    { id: "overview", label: "Overview", icon: "📊" },
    { id: "alerts", label: "Alerts", icon: "🚨", badge: activeAlerts.length },
    { id: "compliance", label: "Compliance", icon: "📋" },
    { id: "audit", label: "Audit Log", icon: "📝" },
    { id: "policies", label: "Policies", icon: "⚙️" },
    { id: "access", label: "Access Control", icon: "🔐" },
  ];

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-900">
        {/* Header */}
        <div className="border-b border-gray-800 bg-gray-900/95 backdrop-blur-sm sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-2xl font-bold text-white">Security & Compliance Center</h1>
                <p className="text-gray-400 mt-1">Monitor security posture and compliance status</p>
              </div>
              <div className="flex items-center gap-3">
                {criticalAlerts.length > 0 && (
                  <div className="px-4 py-2 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2">
                    <span className="text-red-400">🚨</span>
                    <span className="text-red-400 font-medium">{criticalAlerts.length} Critical</span>
                  </div>
                )}
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
                  Run Security Scan
                </button>
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
                  {tab.badge && (
                    <span className="px-1.5 py-0.5 text-xs bg-red-500/20 text-red-400 rounded-full">
                      {tab.badge}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Overview Tab */}
          {activeTab === "overview" && (
            <div className="space-y-8">
              {/* Security Score */}
              <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-xl p-6">
                <div className="flex items-center gap-8">
                  <SecurityScoreGauge score={mockSecurityScore.overall} />
                  <div className="flex-1 grid grid-cols-5 gap-4">
                    {[
                      { name: "Authentication", score: mockSecurityScore.authentication },
                      { name: "Data Protection", score: mockSecurityScore.dataProtection },
                      { name: "Access Control", score: mockSecurityScore.accessControl },
                      { name: "Network Security", score: mockSecurityScore.networkSecurity },
                      { name: "Monitoring", score: mockSecurityScore.monitoring },
                    ].map((item) => (
                      <div key={item.name} className="text-center">
                        <SecurityScoreGauge score={item.score} size="sm" />
                        <p className="text-sm text-gray-400 mt-2">{item.name}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Quick Stats */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard title="Active Alerts" value={activeAlerts.length} icon="🚨" color={activeAlerts.length > 0 ? "red" : "green"} />
                <MetricCard title="Compliance Score" value="95%" icon="📋" trend={2} color="green" />
                <MetricCard title="Failed Logins (24h)" value="23" icon="🔐" trend={-15} color="yellow" />
                <MetricCard title="Data Exports (24h)" value="12" icon="📤" color="purple" />
              </div>

              {/* Two Column Layout */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Recent Alerts */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">Recent Alerts</h3>
                    <button
                      onClick={() => setActiveTab("alerts")}
                      className="text-sm text-purple-400 hover:text-purple-300"
                    >
                      View All →
                    </button>
                  </div>
                  <div className="space-y-4">
                    {mockAlerts.slice(0, 3).map((alert) => (
                      <AlertCard key={alert.id} alert={alert} onSelect={setSelectedAlert} />
                    ))}
                  </div>
                </div>

                {/* Compliance Overview */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">Compliance Status</h3>
                    <button
                      onClick={() => setActiveTab("compliance")}
                      className="text-sm text-purple-400 hover:text-purple-300"
                    >
                      View All →
                    </button>
                  </div>
                  <div className="space-y-4">
                    {mockComplianceFrameworks.slice(0, 4).map((framework) => (
                      <div key={framework.id} className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <ComplianceStatusBadge status={framework.status} />
                          <span className="font-medium text-white">{framework.shortName}</span>
                        </div>
                        <span className="text-sm text-gray-400">
                          {framework.controls.passed}/{framework.controls.total} controls
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Data Classification */}
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Data Classification Summary</h3>
                <div className="grid grid-cols-4 gap-4">
                  {mockDataClassifications.map((cls) => (
                    <div key={cls.level} className="text-center p-4 bg-gray-700/30 rounded-lg">
                      <div className={`w-4 h-4 rounded-full mx-auto mb-2 ${
                        cls.color === "green" ? "bg-green-500" :
                        cls.color === "blue" ? "bg-blue-500" :
                        cls.color === "yellow" ? "bg-yellow-500" :
                        "bg-red-500"
                      }`} />
                      <p className="font-medium text-white">{cls.name}</p>
                      <p className="text-2xl font-bold text-white mt-1">{cls.dataCount.toLocaleString()}</p>
                      <p className="text-xs text-gray-500">records</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Alerts Tab */}
          {activeTab === "alerts" && (
            <div className="space-y-6">
              {/* Filters */}
              <div className="flex items-center gap-4">
                <select
                  value={alertFilter}
                  onChange={(e) => setAlertFilter(e.target.value as SecurityLevel | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Levels</option>
                  <option value="critical">Critical</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                <select
                  value={alertStatusFilter}
                  onChange={(e) => setAlertStatusFilter(e.target.value as AlertStatus | "all")}
                  className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="investigating">Investigating</option>
                  <option value="resolved">Resolved</option>
                  <option value="false_positive">False Positive</option>
                </select>
              </div>

              {/* Alert List */}
              <div className="space-y-4">
                {filteredAlerts.map((alert) => (
                  <AlertCard key={alert.id} alert={alert} onSelect={setSelectedAlert} />
                ))}
              </div>

              {filteredAlerts.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-gray-400">No alerts matching your filters</p>
                </div>
              )}
            </div>
          )}

          {/* Compliance Tab */}
          {activeTab === "compliance" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {mockComplianceFrameworks.map((framework) => (
                  <ComplianceCard key={framework.id} framework={framework} />
                ))}
              </div>
            </div>
          )}

          {/* Audit Log Tab */}
          {activeTab === "audit" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-400">Showing last 24 hours of activity</p>
                <button className="px-4 py-2 text-gray-400 hover:text-white transition-colors">
                  Export Logs
                </button>
              </div>
              <div className="space-y-2">
                {mockAuditLogs.map((entry) => (
                  <AuditLogRow key={entry.id} entry={entry} />
                ))}
              </div>
            </div>
          )}

          {/* Policies Tab */}
          {activeTab === "policies" && (
            <div className="space-y-6">
              {Object.entries(
                mockPolicies.reduce((acc, policy) => {
                  if (!acc[policy.category]) acc[policy.category] = [];
                  acc[policy.category].push(policy);
                  return acc;
                }, {} as Record<string, SecurityPolicy[]>)
              ).map(([category, policies]) => (
                <div key={category}>
                  <h3 className="text-lg font-semibold text-white mb-4">{category}</h3>
                  <div className="space-y-3">
                    {policies.map((policy) => (
                      <PolicyCard
                        key={policy.id}
                        policy={policy}
                        onToggle={(id) => console.log("Toggle policy:", id)}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Access Control Tab */}
          {activeTab === "access" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">Access Reviews</h3>
                <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity">
                  Start Review
                </button>
              </div>
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left text-sm text-gray-400 font-medium p-4">User</th>
                      <th className="text-left text-sm text-gray-400 font-medium p-4">Role</th>
                      <th className="text-left text-sm text-gray-400 font-medium p-4">Department</th>
                      <th className="text-left text-sm text-gray-400 font-medium p-4">Access Level</th>
                      <th className="text-left text-sm text-gray-400 font-medium p-4">Status</th>
                      <th className="text-left text-sm text-gray-400 font-medium p-4">Next Review</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mockAccessReviews.map((review) => (
                      <tr key={review.id} className="border-b border-gray-700/50 last:border-0 hover:bg-gray-700/20">
                        <td className="p-4 font-medium text-white">{review.user}</td>
                        <td className="p-4 text-gray-400">{review.role}</td>
                        <td className="p-4 text-gray-400">{review.department}</td>
                        <td className="p-4">
                          <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                            review.accessLevel === "admin" ? "bg-red-500/20 text-red-400" :
                            review.accessLevel === "manager" ? "bg-yellow-500/20 text-yellow-400" :
                            "bg-gray-500/20 text-gray-400"
                          }`}>
                            {review.accessLevel}
                          </span>
                        </td>
                        <td className="p-4">
                          <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                            review.status === "approved" ? "bg-green-500/20 text-green-400" :
                            review.status === "pending" ? "bg-yellow-500/20 text-yellow-400" :
                            review.status === "escalated" ? "bg-red-500/20 text-red-400" :
                            "bg-gray-500/20 text-gray-400"
                          }`}>
                            {review.status}
                          </span>
                        </td>
                        <td className="p-4 text-gray-400">{new Date(review.nextReview).toLocaleDateString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Alert Detail Dialog */}
        <AlertDetailDialog alert={selectedAlert} onClose={() => setSelectedAlert(null)} />
      </div>
    </DashboardLayout>
  );
}
