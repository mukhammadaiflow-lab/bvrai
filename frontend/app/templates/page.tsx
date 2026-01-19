"use client";

import React, { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  Search,
  Filter,
  Grid3X3,
  List,
  Sparkles,
  Bot,
  Zap,
  Star,
  TrendingUp,
  Clock,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import {
  TemplateCard,
  TemplateCardSkeleton,
  TemplatePreview,
  AgentTemplate,
  agentTemplates,
  templateCategories,
  templateIndustries,
} from "@/components/templates";

type ViewMode = "grid" | "list";
type SortOption = "popular" | "newest" | "alphabetical";

export default function TemplatesPage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [selectedIndustry, setSelectedIndustry] = useState("All Industries");
  const [selectedComplexity, setSelectedComplexity] = useState("all");
  const [sortBy, setSortBy] = useState<SortOption>("popular");
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [previewTemplate, setPreviewTemplate] = useState<AgentTemplate | null>(null);

  // Filter and sort templates
  const filteredTemplates = useMemo(() => {
    let result = [...agentTemplates];

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (t) =>
          t.name.toLowerCase().includes(query) ||
          t.description.toLowerCase().includes(query) ||
          t.category.toLowerCase().includes(query) ||
          t.features.some((f) => f.toLowerCase().includes(query))
      );
    }

    // Category filter
    if (selectedCategory !== "all") {
      result = result.filter((t) => t.category === selectedCategory);
    }

    // Industry filter
    if (selectedIndustry !== "All Industries") {
      result = result.filter((t) => t.industry === selectedIndustry);
    }

    // Complexity filter
    if (selectedComplexity !== "all") {
      result = result.filter((t) => t.complexity === selectedComplexity);
    }

    // Sort
    switch (sortBy) {
      case "popular":
        result.sort((a, b) => b.popularity - a.popularity);
        break;
      case "newest":
        result.sort((a, b) => (b.is_new ? 1 : 0) - (a.is_new ? 1 : 0));
        break;
      case "alphabetical":
        result.sort((a, b) => a.name.localeCompare(b.name));
        break;
    }

    return result;
  }, [searchQuery, selectedCategory, selectedIndustry, selectedComplexity, sortBy]);

  // Featured templates (top 3 by popularity)
  const featuredTemplates = agentTemplates
    .filter((t) => t.is_featured)
    .sort((a, b) => b.popularity - a.popularity)
    .slice(0, 3);

  const handleUseTemplate = (template: AgentTemplate) => {
    router.push(`/agents/new?template=${template.id}`);
  };

  const handlePreview = (template: AgentTemplate) => {
    setPreviewTemplate(template);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Agent Templates</h1>
          <p className="text-muted-foreground">
            Pre-built voice agents ready to deploy in minutes
          </p>
        </div>
        <Button onClick={() => router.push("/agents/new")}>
          <Bot className="mr-2 h-4 w-4" />
          Build from Scratch
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Sparkles className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold">{agentTemplates.length}</p>
                <p className="text-sm text-muted-foreground">Templates</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-green-100">
                <TrendingUp className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {agentTemplates.reduce((sum, t) => sum + t.popularity, 0).toLocaleString()}+
                </p>
                <p className="text-sm text-muted-foreground">Total Uses</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-purple-100">
                <Star className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{templateCategories.length - 1}</p>
                <p className="text-sm text-muted-foreground">Categories</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-amber-100">
                <Clock className="h-5 w-5 text-amber-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">5-15</p>
                <p className="text-sm text-muted-foreground">Min Setup</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Featured Templates */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <Zap className="h-5 w-5 text-amber-500" />
          <h2 className="text-xl font-semibold">Featured Templates</h2>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {featuredTemplates.map((template) => (
            <TemplateCard
              key={template.id}
              template={template}
              variant="featured"
              onUse={handleUseTemplate}
              onPreview={handlePreview}
            />
          ))}
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center">
            <div className="flex-1">
              <Input
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                leftIcon={<Search className="h-4 w-4" />}
                className="max-w-md"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="Category" />
                </SelectTrigger>
                <SelectContent>
                  {templateCategories.map((cat) => (
                    <SelectItem key={cat.id} value={cat.id}>
                      {cat.name} ({cat.count})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={selectedIndustry} onValueChange={setSelectedIndustry}>
                <SelectTrigger className="w-40">
                  <SelectValue placeholder="Industry" />
                </SelectTrigger>
                <SelectContent>
                  {templateIndustries.map((industry) => (
                    <SelectItem key={industry} value={industry}>
                      {industry}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={selectedComplexity} onValueChange={setSelectedComplexity}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Complexity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="basic">Basic</SelectItem>
                  <SelectItem value="intermediate">Intermediate</SelectItem>
                  <SelectItem value="advanced">Advanced</SelectItem>
                </SelectContent>
              </Select>

              <Select value={sortBy} onValueChange={(v) => setSortBy(v as SortOption)}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="popular">Most Popular</SelectItem>
                  <SelectItem value="newest">Newest First</SelectItem>
                  <SelectItem value="alphabetical">A-Z</SelectItem>
                </SelectContent>
              </Select>

              <div className="flex rounded-lg border p-1">
                <Button
                  variant={viewMode === "grid" ? "default" : "ghost"}
                  size="icon-sm"
                  onClick={() => setViewMode("grid")}
                >
                  <Grid3X3 className="h-4 w-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "default" : "ghost"}
                  size="icon-sm"
                  onClick={() => setViewMode("list")}
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Category Tabs */}
      <Tabs value={selectedCategory} onValueChange={setSelectedCategory}>
        <TabsList className="flex-wrap h-auto gap-1 p-1">
          {templateCategories.slice(0, 8).map((cat) => (
            <TabsTrigger key={cat.id} value={cat.id} className="text-sm">
              {cat.name}
              <Badge variant="secondary" className="ml-2 text-xs">
                {cat.count}
              </Badge>
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      {/* Results */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <p className="text-sm text-muted-foreground">
            Showing {filteredTemplates.length} templates
          </p>
          {(searchQuery || selectedCategory !== "all" || selectedIndustry !== "All Industries") && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setSearchQuery("");
                setSelectedCategory("all");
                setSelectedIndustry("All Industries");
                setSelectedComplexity("all");
              }}
            >
              Clear filters
            </Button>
          )}
        </div>

        {filteredTemplates.length === 0 ? (
          <Card className="p-12 text-center">
            <Bot className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
            <h3 className="text-lg font-medium">No templates found</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Try adjusting your filters or search query
            </p>
            <Button
              variant="outline"
              className="mt-4"
              onClick={() => {
                setSearchQuery("");
                setSelectedCategory("all");
                setSelectedIndustry("All Industries");
                setSelectedComplexity("all");
              }}
            >
              Clear all filters
            </Button>
          </Card>
        ) : viewMode === "grid" ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {filteredTemplates.map((template) => (
              <TemplateCard
                key={template.id}
                template={template}
                onUse={handleUseTemplate}
                onPreview={handlePreview}
              />
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {filteredTemplates.map((template) => (
              <TemplateCard
                key={template.id}
                template={template}
                variant="compact"
                onUse={handleUseTemplate}
                onPreview={handlePreview}
              />
            ))}
          </div>
        )}
      </div>

      {/* Request Template CTA */}
      <Card className="bg-gradient-to-r from-primary/5 to-primary/10 border-primary/20">
        <CardContent className="py-8 text-center">
          <Sparkles className="h-10 w-10 mx-auto mb-4 text-primary" />
          <h3 className="text-xl font-semibold">Can't find what you need?</h3>
          <p className="text-muted-foreground mt-1 mb-4">
            Request a custom template or build your own agent from scratch
          </p>
          <div className="flex justify-center gap-3">
            <Button variant="outline">Request Template</Button>
            <Button onClick={() => router.push("/agents/new")}>
              <Bot className="mr-2 h-4 w-4" />
              Build Custom Agent
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Preview Modal */}
      <TemplatePreview
        template={previewTemplate}
        open={!!previewTemplate}
        onOpenChange={(open) => !open && setPreviewTemplate(null)}
        onUse={handleUseTemplate}
      />
    </div>
  );
}
