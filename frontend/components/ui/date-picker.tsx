"use client";

import * as React from "react";
import { format } from "date-fns";
import { Calendar as CalendarIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./button";
import { Calendar } from "./calendar";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";

interface DatePickerProps {
  date?: Date;
  onDateChange?: (date: Date | undefined) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

function DatePicker({
  date,
  onDateChange,
  placeholder = "Pick a date",
  className,
  disabled,
}: DatePickerProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start text-left font-normal",
            !date && "text-muted-foreground",
            className
          )}
          disabled={disabled}
        >
          <CalendarIcon className="mr-2 h-4 w-4" />
          {date ? format(date, "PPP") : placeholder}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <Calendar
          mode="single"
          selected={date}
          onSelect={(selected) => onDateChange?.(selected as Date)}
        />
      </PopoverContent>
    </Popover>
  );
}

interface DateRangePickerProps {
  dateRange?: { from?: Date; to?: Date };
  onDateRangeChange?: (range: { from?: Date; to?: Date } | undefined) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  numberOfMonths?: number;
}

function DateRangePicker({
  dateRange,
  onDateRangeChange,
  placeholder = "Pick a date range",
  className,
  disabled,
  numberOfMonths = 2,
}: DateRangePickerProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start text-left font-normal",
            !dateRange?.from && "text-muted-foreground",
            className
          )}
          disabled={disabled}
        >
          <CalendarIcon className="mr-2 h-4 w-4" />
          {dateRange?.from ? (
            dateRange.to ? (
              <>
                {format(dateRange.from, "LLL dd, y")} -{" "}
                {format(dateRange.to, "LLL dd, y")}
              </>
            ) : (
              format(dateRange.from, "LLL dd, y")
            )
          ) : (
            placeholder
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <Calendar
          mode="range"
          selected={dateRange}
          onSelect={(selected) =>
            onDateRangeChange?.(selected as { from?: Date; to?: Date })
          }
          numberOfMonths={numberOfMonths}
        />
      </PopoverContent>
    </Popover>
  );
}

interface DateRangePresetsProps {
  dateRange?: { from?: Date; to?: Date };
  onDateRangeChange?: (range: { from?: Date; to?: Date }) => void;
  className?: string;
}

function DateRangePresets({
  dateRange,
  onDateRangeChange,
  className,
}: DateRangePresetsProps) {
  const today = new Date();

  const presets = [
    {
      label: "Today",
      getValue: () => ({ from: today, to: today }),
    },
    {
      label: "Yesterday",
      getValue: () => {
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);
        return { from: yesterday, to: yesterday };
      },
    },
    {
      label: "Last 7 days",
      getValue: () => {
        const from = new Date(today);
        from.setDate(from.getDate() - 6);
        return { from, to: today };
      },
    },
    {
      label: "Last 30 days",
      getValue: () => {
        const from = new Date(today);
        from.setDate(from.getDate() - 29);
        return { from, to: today };
      },
    },
    {
      label: "This month",
      getValue: () => {
        const from = new Date(today.getFullYear(), today.getMonth(), 1);
        return { from, to: today };
      },
    },
    {
      label: "Last month",
      getValue: () => {
        const from = new Date(today.getFullYear(), today.getMonth() - 1, 1);
        const to = new Date(today.getFullYear(), today.getMonth(), 0);
        return { from, to };
      },
    },
  ];

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start text-left font-normal",
            !dateRange?.from && "text-muted-foreground",
            className
          )}
        >
          <CalendarIcon className="mr-2 h-4 w-4" />
          {dateRange?.from ? (
            dateRange.to ? (
              <>
                {format(dateRange.from, "LLL dd, y")} -{" "}
                {format(dateRange.to, "LLL dd, y")}
              </>
            ) : (
              format(dateRange.from, "LLL dd, y")
            )
          ) : (
            "Pick a date range"
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <div className="flex">
          <div className="border-r p-3 space-y-1">
            {presets.map((preset) => (
              <Button
                key={preset.label}
                variant="ghost"
                size="sm"
                className="w-full justify-start"
                onClick={() => onDateRangeChange?.(preset.getValue())}
              >
                {preset.label}
              </Button>
            ))}
          </div>
          <Calendar
            mode="range"
            selected={dateRange}
            onSelect={(selected) =>
              onDateRangeChange?.(selected as { from?: Date; to?: Date })
            }
            numberOfMonths={2}
          />
        </div>
      </PopoverContent>
    </Popover>
  );
}

export { DatePicker, DateRangePicker, DateRangePresets };
