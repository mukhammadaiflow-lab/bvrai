"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Bot,
  Phone,
  BarChart3,
  Settings,
  Mic,
  Webhook,
  CreditCard,
  Key,
  Users,
  LayoutDashboard,
  ChevronLeft,
  ChevronRight,
  LogOut,
  HelpCircle,
  MessageSquare,
  Sparkles,
  Rocket,
  Radio,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface NavItem {
  title: string;
  href: string;
  icon: React.ReactNode;
  badge?: string | number;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

const navGroups: NavGroup[] = [
  {
    title: "Overview",
    items: [
      { title: "Dashboard", href: "/dashboard", icon: <LayoutDashboard className="h-5 w-5" /> },
      { title: "Getting Started", href: "/getting-started", icon: <Rocket className="h-5 w-5" />, badge: "New" },
    ],
  },
  {
    title: "Voice Agents",
    items: [
      { title: "Agents", href: "/agents", icon: <Bot className="h-5 w-5" /> },
      { title: "Templates", href: "/templates", icon: <Sparkles className="h-5 w-5" />, badge: "New" },
      { title: "Voice Config", href: "/settings/voice", icon: <Mic className="h-5 w-5" /> },
      { title: "Live Calls", href: "/calls/live", icon: <Radio className="h-5 w-5" /> },
      { title: "Call History", href: "/calls", icon: <Phone className="h-5 w-5" /> },
      { title: "Conversations", href: "/conversations", icon: <MessageSquare className="h-5 w-5" />, badge: 3 },
    ],
  },
  {
    title: "Analytics",
    items: [
      { title: "Analytics", href: "/analytics", icon: <BarChart3 className="h-5 w-5" /> },
    ],
  },
  {
    title: "Integration",
    items: [
      { title: "Webhooks", href: "/webhooks", icon: <Webhook className="h-5 w-5" /> },
      { title: "API Keys", href: "/settings/api-keys", icon: <Key className="h-5 w-5" /> },
    ],
  },
  {
    title: "Settings",
    items: [
      { title: "Team", href: "/team", icon: <Users className="h-5 w-5" /> },
      { title: "Billing", href: "/billing", icon: <CreditCard className="h-5 w-5" /> },
      { title: "Settings", href: "/settings", icon: <Settings className="h-5 w-5" /> },
    ],
  },
];

interface SidebarProps {
  collapsed?: boolean;
  onToggle?: () => void;
}

export function Sidebar({ collapsed = false, onToggle }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 h-screen border-r border-border/50 bg-sidebar-background/95 backdrop-blur-xl transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}
    >
      {/* Logo */}
      <div className="flex h-16 items-center justify-between border-b border-border/50 px-4">
        {!collapsed && (
          <Link href="/dashboard" className="flex items-center gap-3 group">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent shadow-lg shadow-primary/25 group-hover:shadow-primary/40 transition-shadow">
              <Bot className="h-5 w-5 text-white" />
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">BVRAI</span>
              <span className="text-[10px] text-muted-foreground -mt-0.5">Voice AI Platform</span>
            </div>
          </Link>
        )}
        {collapsed && (
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent mx-auto shadow-lg shadow-primary/25">
            <Bot className="h-5 w-5 text-white" />
          </div>
        )}
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onToggle}
          className={cn("rounded-lg hover:bg-primary/10", collapsed && "mx-auto mt-2")}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-1 p-3 h-[calc(100vh-8rem)] overflow-y-auto scrollbar-thin">
        {navGroups.map((group) => (
          <div key={group.title} className="mb-3">
            {!collapsed && (
              <p className="mb-2 px-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground/70">
                {group.title}
              </p>
            )}
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const isActive = pathname === item.href || pathname?.startsWith(`${item.href}/`);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all",
                      isActive
                        ? "bg-gradient-to-r from-primary/15 to-accent/10 text-primary"
                        : "text-muted-foreground hover:text-foreground hover:bg-primary/5",
                      collapsed && "justify-center px-2"
                    )}
                    title={collapsed ? item.title : undefined}
                  >
                    {isActive && (
                      <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 rounded-r-full bg-gradient-to-b from-primary to-accent" />
                    )}
                    <span className={cn(isActive && "text-primary")}>{item.icon}</span>
                    {!collapsed && (
                      <>
                        <span className="flex-1">{item.title}</span>
                        {item.badge && (
                          <span className={cn(
                            "rounded-full px-2 py-0.5 text-[10px] font-semibold",
                            typeof item.badge === "number"
                              ? "bg-gradient-to-r from-primary to-accent text-white min-w-[20px] text-center"
                              : "bg-accent/20 text-accent"
                          )}>
                            {item.badge}
                          </span>
                        )}
                      </>
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="absolute bottom-0 left-0 right-0 border-t border-border/50 bg-sidebar-background/80 backdrop-blur-sm p-3">
        <div className={cn("flex gap-1", collapsed ? "flex-col items-center" : "")}>
          <Button
            variant="ghost"
            size={collapsed ? "icon-sm" : "sm"}
            className={cn("rounded-xl hover:bg-primary/10", !collapsed && "flex-1 justify-start")}
            asChild
          >
            <Link href="/help">
              <HelpCircle className="h-4 w-4" />
              {!collapsed && <span className="ml-2">Help & Support</span>}
            </Link>
          </Button>
          <Button
            variant="ghost"
            size={collapsed ? "icon-sm" : "sm"}
            className={cn("rounded-xl hover:bg-destructive/10 hover:text-destructive", !collapsed && "flex-1 justify-start")}
          >
            <LogOut className="h-4 w-4" />
            {!collapsed && <span className="ml-2">Sign Out</span>}
          </Button>
        </div>
      </div>
    </aside>
  );
}
