"use client";

import dynamic from "next/dynamic";
import { Suspense, ComponentType, ReactNode } from "react";
import { Skeleton } from "@/components/ui/skeleton";

// Loading skeleton for tables
export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-4 w-20" />
        <Skeleton className="h-4 flex-1" />
        <Skeleton className="h-4 w-16" />
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 py-2">
          <Skeleton className="h-10 w-24" />
          <Skeleton className="h-10 w-32" />
          <Skeleton className="h-10 w-20" />
          <Skeleton className="h-10 flex-1" />
          <Skeleton className="h-8 w-16" />
        </div>
      ))}
    </div>
  );
}

// Loading skeleton for charts
export function ChartSkeleton({ height = 300 }: { height?: number }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Skeleton className="h-6 w-32" />
        <Skeleton className="h-8 w-24" />
      </div>
      <Skeleton className="w-full" style={{ height }} />
    </div>
  );
}

// Loading skeleton for cards
export function CardsSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="p-4 border rounded-lg space-y-2">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-8 w-24" />
          <Skeleton className="h-3 w-16" />
        </div>
      ))}
    </div>
  );
}

// Loading skeleton for dialogs/modals
export function DialogSkeleton() {
  return (
    <div className="space-y-4 p-4">
      <Skeleton className="h-6 w-48" />
      <div className="space-y-3">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
      </div>
      <div className="flex justify-end gap-2">
        <Skeleton className="h-10 w-20" />
        <Skeleton className="h-10 w-20" />
      </div>
    </div>
  );
}

// Loading skeleton for full page content
export function PageContentSkeleton() {
  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-10 w-32" />
      </div>
      <CardsSkeleton count={4} />
      <div className="flex gap-4">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-10 w-32" />
      </div>
      <TableSkeleton rows={10} />
    </div>
  );
}

// Generic lazy component wrapper
export function withLazyLoading<P extends object>(
  importFn: () => Promise<{ default: ComponentType<P> }>,
  LoadingSkeleton: ComponentType = PageContentSkeleton
) {
  const LazyComponent = dynamic(importFn, {
    loading: () => <LoadingSkeleton />,
    ssr: false,
  });

  return function LazyWrapper(props: P) {
    return (
      <Suspense fallback={<LoadingSkeleton />}>
        <LazyComponent {...props} />
      </Suspense>
    );
  };
}

// Export common lazy components
export const LazyDataTable = dynamic(
  () => import("@/components/ui/data-table").then((mod) => mod.DataTable || mod),
  {
    loading: () => <TableSkeleton />,
    ssr: false,
  }
);

// Export dynamic import helper for pages
export function createLazyPage<P extends object>(
  pagePath: string,
  LoadingSkeleton: ComponentType = PageContentSkeleton
) {
  return dynamic<P>(
    () => import(`@/app/${pagePath}/content`),
    {
      loading: () => <LoadingSkeleton />,
      ssr: false,
    }
  );
}
