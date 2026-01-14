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
    ],
  },
  {
    title: "Voice Agents",
    items: [
      { title: "Agents", href: "/agents", icon: <Bot className="h-5 w-5" /> },
      { title: "Voice Config", href: "/voice-config", icon: <Mic className="h-5 w-5" /> },
      { title: "Calls", href: "/calls", icon: <Phone className="h-5 w-5" /> },
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
      { title: "API Keys", href: "/api-keys", icon: <Key className="h-5 w-5" /> },
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
        "fixed left-0 top-0 z-40 h-screen border-r bg-card transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}
    >
      {/* Logo */}
      <div className="flex h-16 items-center justify-between border-b px-4">
        {!collapsed && (
          <Link href="/dashboard" className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Bot className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="font-bold text-lg">Builder AI</span>
          </Link>
        )}
        {collapsed && (
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary mx-auto">
            <Bot className="h-5 w-5 text-primary-foreground" />
          </div>
        )}
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onToggle}
          className={cn(collapsed && "mx-auto")}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-2 p-4 h-[calc(100vh-8rem)] overflow-y-auto scrollbar-hide">
        {navGroups.map((group) => (
          <div key={group.title} className="mb-2">
            {!collapsed && (
              <p className="mb-2 px-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {group.title}
              </p>
            )}
            <div className="space-y-1">
              {group.items.map((item) => {
                const isActive = pathname === item.href || pathname?.startsWith(`${item.href}/`);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                      isActive
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                      collapsed && "justify-center px-2"
                    )}
                    title={collapsed ? item.title : undefined}
                  >
                    {item.icon}
                    {!collapsed && (
                      <>
                        <span className="flex-1">{item.title}</span>
                        {item.badge && (
                          <span className="rounded-full bg-primary/20 px-2 py-0.5 text-xs">
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
      <div className="absolute bottom-0 left-0 right-0 border-t bg-card p-4">
        <div className={cn("flex gap-2", collapsed ? "flex-col items-center" : "")}>
          <Button
            variant="ghost"
            size={collapsed ? "icon-sm" : "sm"}
            className={cn(!collapsed && "flex-1 justify-start")}
            asChild
          >
            <Link href="/help">
              <HelpCircle className="h-4 w-4" />
              {!collapsed && <span className="ml-2">Help</span>}
            </Link>
          </Button>
          <Button
            variant="ghost"
            size={collapsed ? "icon-sm" : "sm"}
            className={cn(!collapsed && "flex-1 justify-start text-destructive hover:text-destructive")}
          >
            <LogOut className="h-4 w-4" />
            {!collapsed && <span className="ml-2">Logout</span>}
          </Button>
        </div>
      </div>
    </aside>
  );
}
