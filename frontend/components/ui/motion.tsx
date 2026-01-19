"use client";

import React, { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

// Fade In Animation
interface FadeInProps {
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  className?: string;
  direction?: "up" | "down" | "left" | "right" | "none";
  distance?: number;
}

export function FadeIn({
  children,
  delay = 0,
  duration = 300,
  className,
  direction = "up",
  distance = 20,
}: FadeInProps) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  const getTransform = () => {
    if (!isVisible) {
      switch (direction) {
        case "up":
          return `translateY(${distance}px)`;
        case "down":
          return `translateY(-${distance}px)`;
        case "left":
          return `translateX(${distance}px)`;
        case "right":
          return `translateX(-${distance}px)`;
        default:
          return "none";
      }
    }
    return "none";
  };

  return (
    <div
      className={className}
      style={{
        opacity: isVisible ? 1 : 0,
        transform: getTransform(),
        transition: `opacity ${duration}ms ease-out, transform ${duration}ms ease-out`,
      }}
    >
      {children}
    </div>
  );
}

// Staggered Children Animation
interface StaggerProps {
  children: React.ReactNode;
  staggerDelay?: number;
  initialDelay?: number;
  className?: string;
}

export function Stagger({
  children,
  staggerDelay = 50,
  initialDelay = 0,
  className,
}: StaggerProps) {
  return (
    <div className={className}>
      {React.Children.map(children, (child, index) => (
        <FadeIn delay={initialDelay + index * staggerDelay} key={index}>
          {child}
        </FadeIn>
      ))}
    </div>
  );
}

// Scale In Animation
interface ScaleInProps {
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  className?: string;
}

export function ScaleIn({
  children,
  delay = 0,
  duration = 200,
  className,
}: ScaleInProps) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  return (
    <div
      className={className}
      style={{
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? "scale(1)" : "scale(0.95)",
        transition: `opacity ${duration}ms ease-out, transform ${duration}ms ease-out`,
      }}
    >
      {children}
    </div>
  );
}

// Slide In Animation
interface SlideInProps {
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  className?: string;
  from?: "left" | "right" | "top" | "bottom";
}

export function SlideIn({
  children,
  delay = 0,
  duration = 300,
  className,
  from = "left",
}: SlideInProps) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  const getTransform = () => {
    if (!isVisible) {
      switch (from) {
        case "left":
          return "translateX(-100%)";
        case "right":
          return "translateX(100%)";
        case "top":
          return "translateY(-100%)";
        case "bottom":
          return "translateY(100%)";
      }
    }
    return "translate(0)";
  };

  return (
    <div
      className={className}
      style={{
        transform: getTransform(),
        transition: `transform ${duration}ms cubic-bezier(0.16, 1, 0.3, 1)`,
      }}
    >
      {children}
    </div>
  );
}

// Pulse Animation
interface PulseProps {
  children: React.ReactNode;
  className?: string;
  active?: boolean;
}

export function Pulse({ children, className, active = true }: PulseProps) {
  return (
    <div className={cn(active && "animate-pulse", className)}>{children}</div>
  );
}

// Bounce Animation
interface BounceProps {
  children: React.ReactNode;
  className?: string;
  active?: boolean;
}

export function Bounce({ children, className, active = true }: BounceProps) {
  return (
    <div className={cn(active && "animate-bounce", className)}>{children}</div>
  );
}

// Spin Animation
interface SpinProps {
  children: React.ReactNode;
  className?: string;
}

export function Spin({ children, className }: SpinProps) {
  return <div className={cn("animate-spin", className)}>{children}</div>;
}

// Number Counter Animation
interface CounterProps {
  value: number;
  duration?: number;
  className?: string;
  formatter?: (value: number) => string;
}

export function Counter({
  value,
  duration = 1000,
  className,
  formatter = (v) => v.toLocaleString(),
}: CounterProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    let startTime: number;
    let animationFrame: number;

    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);

      // Easing function for smooth animation
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      setDisplayValue(Math.floor(easeOutQuart * value));

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };

    animationFrame = requestAnimationFrame(animate);

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [value, duration]);

  return <span className={className}>{formatter(displayValue)}</span>;
}

// Typing Animation
interface TypingProps {
  text: string;
  speed?: number;
  delay?: number;
  className?: string;
  cursor?: boolean;
}

export function Typing({
  text,
  speed = 50,
  delay = 0,
  className,
  cursor = true,
}: TypingProps) {
  const [displayText, setDisplayText] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    const startDelay = setTimeout(() => {
      setIsTyping(true);
      let index = 0;
      const interval = setInterval(() => {
        if (index < text.length) {
          setDisplayText(text.slice(0, index + 1));
          index++;
        } else {
          clearInterval(interval);
          setIsTyping(false);
        }
      }, speed);

      return () => clearInterval(interval);
    }, delay);

    return () => clearTimeout(startDelay);
  }, [text, speed, delay]);

  return (
    <span className={className}>
      {displayText}
      {cursor && isTyping && (
        <span className="animate-blink ml-0.5">|</span>
      )}
    </span>
  );
}

// Shimmer Loading Effect
interface ShimmerProps {
  className?: string;
  width?: string | number;
  height?: string | number;
}

export function Shimmer({ className, width, height }: ShimmerProps) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded bg-muted",
        className
      )}
      style={{ width, height }}
    >
      <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
    </div>
  );
}

// Page Transition Wrapper
interface PageTransitionProps {
  children: React.ReactNode;
  className?: string;
}

export function PageTransition({ children, className }: PageTransitionProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div
      className={cn(
        "transition-all duration-300 ease-out",
        mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4",
        className
      )}
    >
      {children}
    </div>
  );
}

// Presence Animation (for conditional rendering)
interface PresenceProps {
  children: React.ReactNode;
  present: boolean;
  className?: string;
}

export function Presence({ children, present, className }: PresenceProps) {
  const [shouldRender, setShouldRender] = useState(present);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    if (present) {
      setShouldRender(true);
      requestAnimationFrame(() => setIsAnimating(true));
    } else {
      setIsAnimating(false);
      const timer = setTimeout(() => setShouldRender(false), 200);
      return () => clearTimeout(timer);
    }
  }, [present]);

  if (!shouldRender) return null;

  return (
    <div
      className={cn(
        "transition-all duration-200",
        isAnimating ? "opacity-100 scale-100" : "opacity-0 scale-95",
        className
      )}
    >
      {children}
    </div>
  );
}

// Hover Card Effect
interface HoverCardEffectProps {
  children: React.ReactNode;
  className?: string;
}

export function HoverCardEffect({ children, className }: HoverCardEffectProps) {
  return (
    <div
      className={cn(
        "transition-all duration-200",
        "hover:-translate-y-1 hover:shadow-lg",
        "active:translate-y-0 active:shadow-md",
        className
      )}
    >
      {children}
    </div>
  );
}

// Focus Ring Component (for accessibility)
interface FocusRingProps {
  children: React.ReactNode;
  className?: string;
}

export function FocusRing({ children, className }: FocusRingProps) {
  return (
    <div
      className={cn(
        "focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2",
        "rounded-md transition-shadow",
        className
      )}
    >
      {children}
    </div>
  );
}
