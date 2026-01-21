"use client";

import React, { useState } from "react";
import { usePathname } from "next/navigation";
import { Bell, Search, User } from "lucide-react";
import { Sidebar } from "./sidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const pathname = usePathname();

  // Get page title from pathname
  const getPageTitle = () => {
    const segments = pathname?.split("/").filter(Boolean) || [];
    if (segments.length === 0) return "Dashboard";
    return segments[segments.length - 1]
      .split("-")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  return (
    <div className="min-h-screen bg-background gradient-mesh">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <div
        className={cn(
          "min-h-screen transition-all duration-300",
          sidebarCollapsed ? "ml-16" : "ml-64"
        )}
      >
        {/* Header */}
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border/50 bg-background/80 px-6 backdrop-blur-xl">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-semibold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">{getPageTitle()}</h1>
          </div>

          <div className="flex items-center gap-4">
            {/* Search */}
            <div className="hidden md:block">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search anything..."
                  className="w-72 pl-9 rounded-xl border-border/50 bg-muted/50 focus:bg-background transition-colors"
                />
              </div>
            </div>

            {/* Notifications */}
            <Button variant="ghost" size="icon" className="relative rounded-xl hover:bg-primary/10">
              <Bell className="h-5 w-5" />
              <span className="absolute -right-0.5 -top-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-gradient-to-r from-primary to-accent text-[10px] font-bold text-white">
                3
              </span>
            </Button>

            {/* User Menu */}
            <Button variant="ghost" size="icon" className="rounded-xl p-0.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent">
                <User className="h-4 w-4 text-white" />
              </div>
            </Button>
          </div>
        </header>

        {/* Main Content */}
        <main className="p-6 md:p-8">{children}</main>
      </div>
    </div>
  );
}
