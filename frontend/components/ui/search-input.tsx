"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { Search, X, Loader2 } from "lucide-react";
import { Input } from "./input";
import { Button } from "./button";

interface SearchInputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "onChange"> {
  value?: string;
  onChange?: (value: string) => void;
  onSearch?: (value: string) => void;
  loading?: boolean;
  debounceMs?: number;
  showClear?: boolean;
  className?: string;
}

function SearchInput({
  value: controlledValue,
  onChange,
  onSearch,
  loading = false,
  debounceMs = 300,
  showClear = true,
  className,
  placeholder = "Search...",
  ...props
}: SearchInputProps) {
  const [internalValue, setInternalValue] = React.useState(controlledValue || "");
  const debounceRef = React.useRef<NodeJS.Timeout>();

  const value = controlledValue !== undefined ? controlledValue : internalValue;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;

    if (controlledValue === undefined) {
      setInternalValue(newValue);
    }
    onChange?.(newValue);

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      onSearch?.(newValue);
    }, debounceMs);
  };

  const handleClear = () => {
    if (controlledValue === undefined) {
      setInternalValue("");
    }
    onChange?.("");
    onSearch?.("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      onSearch?.(value);
    }
    if (e.key === "Escape") {
      handleClear();
    }
  };

  React.useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  return (
    <div className={cn("relative", className)}>
      <Input
        type="text"
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        leftIcon={
          loading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Search className="h-4 w-4" />
          )
        }
        rightIcon={
          showClear && value ? (
            <Button
              variant="ghost"
              size="icon-sm"
              className="h-6 w-6 -mr-1"
              onClick={handleClear}
              type="button"
            >
              <X className="h-3 w-3" />
            </Button>
          ) : undefined
        }
        {...props}
      />
    </div>
  );
}

interface SearchWithSuggestionsProps extends SearchInputProps {
  suggestions?: string[];
  onSuggestionSelect?: (suggestion: string) => void;
  maxSuggestions?: number;
}

function SearchWithSuggestions({
  suggestions = [],
  onSuggestionSelect,
  maxSuggestions = 5,
  ...props
}: SearchWithSuggestionsProps) {
  const [showSuggestions, setShowSuggestions] = React.useState(false);
  const [selectedIndex, setSelectedIndex] = React.useState(-1);
  const containerRef = React.useRef<HTMLDivElement>(null);

  const filteredSuggestions = React.useMemo(() => {
    if (!props.value) return [];
    return suggestions
      .filter((s) =>
        s.toLowerCase().includes((props.value || "").toLowerCase())
      )
      .slice(0, maxSuggestions);
  }, [suggestions, props.value, maxSuggestions]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showSuggestions || filteredSuggestions.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) =>
        prev < filteredSuggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
    } else if (e.key === "Enter" && selectedIndex >= 0) {
      e.preventDefault();
      onSuggestionSelect?.(filteredSuggestions[selectedIndex]);
      setShowSuggestions(false);
    }
  };

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div ref={containerRef} className="relative">
      <SearchInput
        {...props}
        onFocus={() => setShowSuggestions(true)}
        onKeyDown={handleKeyDown}
      />
      {showSuggestions && filteredSuggestions.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 rounded-md border bg-popover p-1 shadow-md z-50">
          {filteredSuggestions.map((suggestion, index) => (
            <button
              key={suggestion}
              type="button"
              className={cn(
                "w-full rounded-sm px-2 py-1.5 text-left text-sm hover:bg-accent",
                index === selectedIndex && "bg-accent"
              )}
              onClick={() => {
                onSuggestionSelect?.(suggestion);
                setShowSuggestions(false);
              }}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export { SearchInput, SearchWithSuggestions };
