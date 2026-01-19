"use client";

import React from "react";
import Link from "next/link";
import {
  Bot,
  Phone,
  Calendar,
  ShoppingCart,
  HeadphonesIcon,
  Building2,
  Stethoscope,
  GraduationCap,
  Briefcase,
  Utensils,
  Plane,
  Home,
  Car,
  Heart,
  Star,
  Clock,
  Zap,
  Users,
  ChevronRight,
  Play,
  Copy,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export interface AgentTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
  industry: string;
  features: string[];
  tools: string[];
  complexity: "basic" | "intermediate" | "advanced";
  estimated_setup_time: string;
  popularity: number;
  is_new: boolean;
  is_featured: boolean;
  sample_greeting: string;
  sample_prompt: string;
  use_cases: string[];
}

const iconMap: Record<string, React.ElementType> = {
  phone: Phone,
  calendar: Calendar,
  shopping: ShoppingCart,
  headphones: HeadphonesIcon,
  building: Building2,
  stethoscope: Stethoscope,
  graduation: GraduationCap,
  briefcase: Briefcase,
  utensils: Utensils,
  plane: Plane,
  home: Home,
  car: Car,
  heart: Heart,
  bot: Bot,
};

const categoryColors: Record<string, string> = {
  "Customer Support": "bg-blue-100 text-blue-700 border-blue-200",
  "Sales": "bg-green-100 text-green-700 border-green-200",
  "Scheduling": "bg-purple-100 text-purple-700 border-purple-200",
  "Healthcare": "bg-red-100 text-red-700 border-red-200",
  "Real Estate": "bg-amber-100 text-amber-700 border-amber-200",
  "E-commerce": "bg-pink-100 text-pink-700 border-pink-200",
  "Travel": "bg-cyan-100 text-cyan-700 border-cyan-200",
  "Education": "bg-indigo-100 text-indigo-700 border-indigo-200",
  "Financial": "bg-emerald-100 text-emerald-700 border-emerald-200",
  "Hospitality": "bg-orange-100 text-orange-700 border-orange-200",
};

const complexityColors: Record<string, string> = {
  basic: "bg-green-100 text-green-700",
  intermediate: "bg-yellow-100 text-yellow-700",
  advanced: "bg-red-100 text-red-700",
};

interface TemplateCardProps {
  template: AgentTemplate;
  variant?: "default" | "compact" | "featured";
  onUse?: (template: AgentTemplate) => void;
  onPreview?: (template: AgentTemplate) => void;
}

export function TemplateCard({
  template,
  variant = "default",
  onUse,
  onPreview,
}: TemplateCardProps) {
  const Icon = iconMap[template.icon] || Bot;

  if (variant === "compact") {
    return (
      <Card
        className="cursor-pointer hover:shadow-md transition-all hover:border-primary/50 group"
        onClick={() => onUse?.(template)}
      >
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
                categoryColors[template.category] || "bg-gray-100 text-gray-700"
              )}
            >
              <Icon className="h-5 w-5" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="font-medium truncate">{template.name}</h3>
                {template.is_new && (
                  <Badge variant="secondary" className="text-xs shrink-0">
                    New
                  </Badge>
                )}
              </div>
              <p className="text-sm text-muted-foreground truncate">
                {template.description}
              </p>
            </div>
            <ChevronRight className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (variant === "featured") {
    return (
      <Card className="overflow-hidden border-2 border-primary/20 bg-gradient-to-br from-primary/5 to-transparent">
        <CardContent className="p-6">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-4">
              <div
                className={cn(
                  "flex h-14 w-14 items-center justify-center rounded-xl",
                  categoryColors[template.category] || "bg-gray-100 text-gray-700"
                )}
              >
                <Icon className="h-7 w-7" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-xl font-semibold">{template.name}</h3>
                  <Badge className="bg-primary/10 text-primary border-primary/20">
                    <Star className="mr-1 h-3 w-3 fill-primary" />
                    Featured
                  </Badge>
                </div>
                <p className="text-muted-foreground mt-1">{template.description}</p>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-2 mt-4">
            {template.features.slice(0, 4).map((feature) => (
              <Badge key={feature} variant="outline" className="text-xs">
                {feature}
              </Badge>
            ))}
          </div>

          <div className="grid grid-cols-3 gap-4 mt-6">
            <div className="text-center p-3 rounded-lg bg-background/50">
              <p className="text-2xl font-bold text-primary">{template.popularity}+</p>
              <p className="text-xs text-muted-foreground">Uses</p>
            </div>
            <div className="text-center p-3 rounded-lg bg-background/50">
              <p className="text-2xl font-bold text-primary">{template.tools.length}</p>
              <p className="text-xs text-muted-foreground">Tools</p>
            </div>
            <div className="text-center p-3 rounded-lg bg-background/50">
              <p className="text-2xl font-bold text-primary">{template.estimated_setup_time}</p>
              <p className="text-xs text-muted-foreground">Setup</p>
            </div>
          </div>

          <div className="flex gap-3 mt-6">
            <Button className="flex-1" onClick={() => onUse?.(template)}>
              <Zap className="mr-2 h-4 w-4" />
              Use Template
            </Button>
            <Button variant="outline" onClick={() => onPreview?.(template)}>
              <Play className="mr-2 h-4 w-4" />
              Preview
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Default variant
  return (
    <Card className="hover:shadow-md transition-all hover:border-primary/50 h-full flex flex-col">
      <CardContent className="p-5 flex flex-col h-full">
        <div className="flex items-start justify-between">
          <div
            className={cn(
              "flex h-12 w-12 items-center justify-center rounded-xl",
              categoryColors[template.category] || "bg-gray-100 text-gray-700"
            )}
          >
            <Icon className="h-6 w-6" />
          </div>
          <div className="flex gap-1">
            {template.is_new && (
              <Badge variant="secondary" className="text-xs">
                New
              </Badge>
            )}
            {template.is_featured && (
              <Badge className="text-xs bg-amber-100 text-amber-700">
                <Star className="mr-1 h-3 w-3 fill-amber-500" />
                Popular
              </Badge>
            )}
          </div>
        </div>

        <div className="mt-4">
          <h3 className="font-semibold text-lg">{template.name}</h3>
          <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
            {template.description}
          </p>
        </div>

        <div className="flex flex-wrap gap-1.5 mt-4">
          <Badge variant="outline" className={cn("text-xs", categoryColors[template.category])}>
            {template.category}
          </Badge>
          <Badge variant="outline" className={cn("text-xs capitalize", complexityColors[template.complexity])}>
            {template.complexity}
          </Badge>
        </div>

        <div className="flex items-center gap-4 mt-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {template.estimated_setup_time}
          </span>
          <span className="flex items-center gap-1">
            <Users className="h-3 w-3" />
            {template.popularity}+ uses
          </span>
        </div>

        <div className="flex-1" />

        <div className="flex gap-2 mt-4 pt-4 border-t">
          <Button
            className="flex-1"
            size="sm"
            onClick={() => onUse?.(template)}
          >
            <Copy className="mr-2 h-4 w-4" />
            Use
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPreview?.(template)}
          >
            <Play className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export function TemplateCardSkeleton() {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div className="h-12 w-12 rounded-xl bg-muted animate-pulse" />
          <div className="h-5 w-12 rounded-full bg-muted animate-pulse" />
        </div>
        <div className="mt-4 space-y-2">
          <div className="h-6 w-3/4 rounded bg-muted animate-pulse" />
          <div className="h-4 w-full rounded bg-muted animate-pulse" />
          <div className="h-4 w-2/3 rounded bg-muted animate-pulse" />
        </div>
        <div className="flex gap-2 mt-4">
          <div className="h-5 w-20 rounded-full bg-muted animate-pulse" />
          <div className="h-5 w-16 rounded-full bg-muted animate-pulse" />
        </div>
        <div className="flex gap-2 mt-4 pt-4 border-t">
          <div className="h-9 flex-1 rounded bg-muted animate-pulse" />
          <div className="h-9 w-9 rounded bg-muted animate-pulse" />
        </div>
      </CardContent>
    </Card>
  );
}
