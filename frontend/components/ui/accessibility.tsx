"use client";

import React, { useEffect, useRef, useCallback } from "react";
import { cn } from "@/lib/utils";

// Skip Link - allows keyboard users to skip navigation
interface SkipLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
}

export function SkipLink({ href, children, className }: SkipLinkProps) {
  return (
    <a
      href={href}
      className={cn(
        "skip-link",
        "fixed top-4 left-4 z-[100] px-4 py-2 rounded-md",
        "bg-primary text-primary-foreground font-medium",
        "focus:not-sr-only sr-only",
        "focus:ring-2 focus:ring-ring focus:ring-offset-2",
        className
      )}
    >
      {children}
    </a>
  );
}

// Focus Trap - keeps focus within a container (for modals, dialogs)
interface FocusTrapProps {
  children: React.ReactNode;
  active?: boolean;
  className?: string;
}

export function FocusTrap({ children, active = true, className }: FocusTrapProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const getFocusableElements = useCallback(() => {
    if (!containerRef.current) return [];
    return Array.from(
      containerRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )
    ).filter((el) => !el.hasAttribute("disabled") && el.offsetParent !== null);
  }, []);

  useEffect(() => {
    if (!active) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;

      const focusable = getFocusableElements();
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [active, getFocusableElements]);

  // Focus first element when trap becomes active
  useEffect(() => {
    if (active) {
      const focusable = getFocusableElements();
      if (focusable.length > 0) {
        focusable[0].focus();
      }
    }
  }, [active, getFocusableElements]);

  return (
    <div ref={containerRef} className={className}>
      {children}
    </div>
  );
}

// Screen Reader Only - visually hidden but accessible to screen readers
interface SrOnlyProps {
  children: React.ReactNode;
  as?: keyof JSX.IntrinsicElements;
}

export function SrOnly({ children, as: Component = "span" }: SrOnlyProps) {
  return <Component className="sr-only">{children}</Component>;
}

// Live Region - announces dynamic content changes to screen readers
interface LiveRegionProps {
  children: React.ReactNode;
  "aria-live"?: "polite" | "assertive" | "off";
  "aria-atomic"?: boolean;
  className?: string;
}

export function LiveRegion({
  children,
  "aria-live": ariaLive = "polite",
  "aria-atomic": ariaAtomic = true,
  className,
}: LiveRegionProps) {
  return (
    <div
      role="status"
      aria-live={ariaLive}
      aria-atomic={ariaAtomic}
      className={cn("sr-only", className)}
    >
      {children}
    </div>
  );
}

// Announce - dynamically announce messages to screen readers
interface AnnounceProps {
  message: string;
  "aria-live"?: "polite" | "assertive";
}

export function Announce({ message, "aria-live": ariaLive = "polite" }: AnnounceProps) {
  const [announcement, setAnnouncement] = React.useState("");

  useEffect(() => {
    if (message) {
      // Clear first to ensure announcement is made even if same message
      setAnnouncement("");
      const timer = setTimeout(() => setAnnouncement(message), 100);
      return () => clearTimeout(timer);
    }
  }, [message]);

  return (
    <div
      role="status"
      aria-live={ariaLive}
      aria-atomic
      className="sr-only"
    >
      {announcement}
    </div>
  );
}

// Keyboard Navigation List - handles arrow key navigation in lists
interface KeyboardNavListProps {
  children: React.ReactNode;
  className?: string;
  orientation?: "horizontal" | "vertical" | "both";
  loop?: boolean;
  onSelect?: (index: number) => void;
}

export function KeyboardNavList({
  children,
  className,
  orientation = "vertical",
  loop = true,
  onSelect,
}: KeyboardNavListProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const [focusIndex, setFocusIndex] = React.useState(0);

  const getItems = useCallback(() => {
    if (!listRef.current) return [];
    return Array.from(
      listRef.current.querySelectorAll<HTMLElement>('[role="option"], [role="menuitem"], [data-nav-item]')
    );
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const items = getItems();
      if (items.length === 0) return;

      let nextIndex = focusIndex;

      const isVertical = orientation === "vertical" || orientation === "both";
      const isHorizontal = orientation === "horizontal" || orientation === "both";

      switch (e.key) {
        case "ArrowDown":
          if (isVertical) {
            e.preventDefault();
            nextIndex = loop
              ? (focusIndex + 1) % items.length
              : Math.min(focusIndex + 1, items.length - 1);
          }
          break;
        case "ArrowUp":
          if (isVertical) {
            e.preventDefault();
            nextIndex = loop
              ? (focusIndex - 1 + items.length) % items.length
              : Math.max(focusIndex - 1, 0);
          }
          break;
        case "ArrowRight":
          if (isHorizontal) {
            e.preventDefault();
            nextIndex = loop
              ? (focusIndex + 1) % items.length
              : Math.min(focusIndex + 1, items.length - 1);
          }
          break;
        case "ArrowLeft":
          if (isHorizontal) {
            e.preventDefault();
            nextIndex = loop
              ? (focusIndex - 1 + items.length) % items.length
              : Math.max(focusIndex - 1, 0);
          }
          break;
        case "Home":
          e.preventDefault();
          nextIndex = 0;
          break;
        case "End":
          e.preventDefault();
          nextIndex = items.length - 1;
          break;
        case "Enter":
        case " ":
          e.preventDefault();
          onSelect?.(focusIndex);
          return;
      }

      if (nextIndex !== focusIndex) {
        setFocusIndex(nextIndex);
        items[nextIndex]?.focus();
      }
    },
    [focusIndex, getItems, loop, onSelect, orientation]
  );

  return (
    <div
      ref={listRef}
      className={className}
      onKeyDown={handleKeyDown}
      role="listbox"
      aria-activedescendant={`item-${focusIndex}`}
    >
      {React.Children.map(children, (child, index) => {
        if (React.isValidElement(child)) {
          return React.cloneElement(child as React.ReactElement<any>, {
            id: `item-${index}`,
            tabIndex: index === focusIndex ? 0 : -1,
            "data-nav-item": true,
            "aria-selected": index === focusIndex,
          });
        }
        return child;
      })}
    </div>
  );
}

// Roving Tab Index - alternative to KeyboardNavList with roving tabindex pattern
interface RovingTabIndexProps {
  children: React.ReactNode;
  className?: string;
  currentIndex?: number;
  onChange?: (index: number) => void;
}

export function RovingTabIndex({
  children,
  className,
  currentIndex = 0,
  onChange,
}: RovingTabIndexProps) {
  const [activeIndex, setActiveIndex] = React.useState(currentIndex);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
    const items = containerRef.current?.querySelectorAll<HTMLElement>("[data-roving]");
    if (!items) return;

    let nextIndex = index;

    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
        e.preventDefault();
        nextIndex = (index + 1) % items.length;
        break;
      case "ArrowLeft":
      case "ArrowUp":
        e.preventDefault();
        nextIndex = (index - 1 + items.length) % items.length;
        break;
      case "Home":
        e.preventDefault();
        nextIndex = 0;
        break;
      case "End":
        e.preventDefault();
        nextIndex = items.length - 1;
        break;
    }

    if (nextIndex !== index) {
      setActiveIndex(nextIndex);
      onChange?.(nextIndex);
      items[nextIndex]?.focus();
    }
  };

  return (
    <div ref={containerRef} className={className}>
      {React.Children.map(children, (child, index) => {
        if (React.isValidElement(child)) {
          return React.cloneElement(child as React.ReactElement<any>, {
            tabIndex: index === activeIndex ? 0 : -1,
            "data-roving": true,
            onKeyDown: (e: React.KeyboardEvent) => handleKeyDown(e, index),
            onClick: () => {
              setActiveIndex(index);
              onChange?.(index);
            },
          });
        }
        return child;
      })}
    </div>
  );
}

// Reduce Motion - respects user's motion preferences
interface ReduceMotionProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export function ReduceMotion({ children, fallback }: ReduceMotionProps) {
  const [prefersReducedMotion, setPrefersReducedMotion] = React.useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefersReducedMotion(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => setPrefersReducedMotion(e.matches);
    mediaQuery.addEventListener("change", handler);
    return () => mediaQuery.removeEventListener("change", handler);
  }, []);

  if (prefersReducedMotion && fallback) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
}

// useReducedMotion hook
export function useReducedMotion(): boolean {
  const [prefersReducedMotion, setPrefersReducedMotion] = React.useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefersReducedMotion(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => setPrefersReducedMotion(e.matches);
    mediaQuery.addEventListener("change", handler);
    return () => mediaQuery.removeEventListener("change", handler);
  }, []);

  return prefersReducedMotion;
}

// Focus Visible - wrapper that only shows focus ring on keyboard navigation
interface FocusVisibleProps {
  children: React.ReactNode;
  className?: string;
}

export function FocusVisible({ children, className }: FocusVisibleProps) {
  return (
    <div
      className={cn(
        "focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 focus-within:ring-offset-background",
        className
      )}
    >
      {children}
    </div>
  );
}
