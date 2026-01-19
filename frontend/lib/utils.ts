/**
 * Utility functions for the Builder Voice AI Platform frontend.
 */

import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { formatDistanceToNow, format, parseISO } from "date-fns";

/**
 * Merge Tailwind CSS classes with clsx support.
 * Handles class conflicts intelligently.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Format a duration in seconds to human-readable format.
 * @param seconds - Duration in seconds
 * @returns Formatted string (e.g., "2m 30s", "1h 15m")
 */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || seconds < 0) {
    return "0s";
  }

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  const parts: string[] = [];

  if (hours > 0) {
    parts.push(`${hours}h`);
  }
  if (minutes > 0) {
    parts.push(`${minutes}m`);
  }
  if (secs > 0 || parts.length === 0) {
    parts.push(`${secs}s`);
  }

  return parts.join(" ");
}

/**
 * Format a number with commas for readability.
 * @param num - Number to format
 * @returns Formatted string (e.g., "1,234,567")
 */
export function formatNumber(num: number | null | undefined): string {
  if (num === null || num === undefined) {
    return "0";
  }
  return num.toLocaleString("en-US");
}

/**
 * Format a number as currency (USD).
 * @param cents - Amount in cents
 * @returns Formatted string (e.g., "$12.34")
 */
export function formatCurrency(cents: number | null | undefined): string {
  if (cents === null || cents === undefined) {
    return "$0.00";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(cents / 100);
}

/**
 * Format a date string to relative time.
 * @param dateString - ISO date string
 * @returns Relative time string (e.g., "5 minutes ago")
 */
export function formatRelativeTime(dateString: string | null | undefined): string {
  if (!dateString) {
    return "Never";
  }

  try {
    const date = parseISO(dateString);
    return formatDistanceToNow(date, { addSuffix: true });
  } catch {
    return "Invalid date";
  }
}

/**
 * Format a date string to a specific format.
 * @param dateString - ISO date string
 * @param formatStr - Date format string (default: "MMM d, yyyy HH:mm")
 * @returns Formatted date string
 */
export function formatDate(
  dateString: string | null | undefined,
  formatStr: string = "MMM d, yyyy HH:mm"
): string {
  if (!dateString) {
    return "-";
  }

  try {
    const date = parseISO(dateString);
    return format(date, formatStr);
  } catch {
    return "Invalid date";
  }
}

/**
 * Format a percentage value.
 * @param value - Decimal value (e.g., 0.85 for 85%)
 * @param decimals - Number of decimal places (default: 1)
 * @returns Formatted percentage string (e.g., "85.0%")
 */
export function formatPercentage(
  value: number | null | undefined,
  decimals: number = 1
): string {
  if (value === null || value === undefined) {
    return "0%";
  }
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format phone number to display format.
 * @param phone - Phone number string
 * @returns Formatted phone number (e.g., "+1 (555) 123-4567")
 */
export function formatPhoneNumber(phone: string | null | undefined): string {
  if (!phone) {
    return "-";
  }

  // Remove all non-digit characters except +
  const cleaned = phone.replace(/[^\d+]/g, "");

  // Handle US numbers
  if (cleaned.startsWith("+1") && cleaned.length === 12) {
    const match = cleaned.match(/^\+1(\d{3})(\d{3})(\d{4})$/);
    if (match) {
      return `+1 (${match[1]}) ${match[2]}-${match[3]}`;
    }
  }

  return phone;
}

/**
 * Truncate a string to a maximum length with ellipsis.
 * @param str - String to truncate
 * @param maxLength - Maximum length (default: 50)
 * @returns Truncated string
 */
export function truncate(
  str: string | null | undefined,
  maxLength: number = 50
): string {
  if (!str) {
    return "";
  }
  if (str.length <= maxLength) {
    return str;
  }
  return `${str.slice(0, maxLength - 3)}...`;
}

/**
 * Generate a random ID.
 * @param prefix - Optional prefix for the ID
 * @returns Random ID string
 */
export function generateId(prefix: string = ""): string {
  const random = Math.random().toString(36).substring(2, 15);
  return prefix ? `${prefix}_${random}` : random;
}

/**
 * Debounce a function.
 * @param fn - Function to debounce
 * @param delay - Delay in milliseconds
 * @returns Debounced function
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout;

  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Sleep for a specified duration.
 * @param ms - Duration in milliseconds
 * @returns Promise that resolves after the duration
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Check if a value is empty (null, undefined, empty string, or empty array).
 * @param value - Value to check
 * @returns True if the value is empty
 */
export function isEmpty(value: unknown): boolean {
  if (value === null || value === undefined) {
    return true;
  }
  if (typeof value === "string") {
    return value.trim() === "";
  }
  if (Array.isArray(value)) {
    return value.length === 0;
  }
  if (typeof value === "object") {
    return Object.keys(value).length === 0;
  }
  return false;
}

/**
 * Get initials from a name.
 * @param name - Full name
 * @returns Initials (e.g., "JD" for "John Doe")
 */
export function getInitials(name: string | null | undefined): string {
  if (!name) {
    return "?";
  }

  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) {
    return parts[0].charAt(0).toUpperCase();
  }

  return (
    parts[0].charAt(0).toUpperCase() +
    parts[parts.length - 1].charAt(0).toUpperCase()
  );
}

/**
 * Copy text to clipboard.
 * @param text - Text to copy
 * @returns Promise that resolves when copied
 */
export async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard) {
    await navigator.clipboard.writeText(text);
  } else {
    // Fallback for older browsers
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-9999px";
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    document.body.removeChild(textArea);
  }
}

/**
 * Get status color class based on status string.
 * @param status - Status string
 * @returns Tailwind CSS color class
 */
export function getStatusColor(status: string): string {
  const statusColors: Record<string, string> = {
    // Call statuses
    initiated: "bg-blue-100 text-blue-800",
    ringing: "bg-yellow-100 text-yellow-800",
    in_progress: "bg-green-100 text-green-800",
    completed: "bg-gray-100 text-gray-800",
    failed: "bg-red-100 text-red-800",
    busy: "bg-orange-100 text-orange-800",
    no_answer: "bg-orange-100 text-orange-800",

    // Agent statuses
    draft: "bg-gray-100 text-gray-800",
    active: "bg-green-100 text-green-800",
    paused: "bg-yellow-100 text-yellow-800",
    archived: "bg-gray-100 text-gray-600",

    // Webhook statuses
    pending: "bg-yellow-100 text-yellow-800",
    success: "bg-green-100 text-green-800",
    error: "bg-red-100 text-red-800",

    // Default
    default: "bg-gray-100 text-gray-800",
  };

  return statusColors[status.toLowerCase()] || statusColors.default;
}
