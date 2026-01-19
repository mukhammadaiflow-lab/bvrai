"use client";

import * as React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import {
  format,
  addMonths,
  subMonths,
  startOfMonth,
  endOfMonth,
  startOfWeek,
  endOfWeek,
  eachDayOfInterval,
  isSameMonth,
  isSameDay,
  isToday,
} from "date-fns";
import { cn } from "@/lib/utils";
import { Button } from "./button";

export interface CalendarProps {
  mode?: "single" | "range";
  selected?: Date | { from?: Date; to?: Date };
  onSelect?: (date: Date | { from?: Date; to?: Date } | undefined) => void;
  disabled?: (date: Date) => boolean;
  className?: string;
  initialFocus?: boolean;
  numberOfMonths?: number;
}

function Calendar({
  mode = "single",
  selected,
  onSelect,
  disabled,
  className,
  numberOfMonths = 1,
}: CalendarProps) {
  const [currentMonth, setCurrentMonth] = React.useState(new Date());
  const [rangeStart, setRangeStart] = React.useState<Date | undefined>(
    mode === "range" && selected && typeof selected === "object" && "from" in selected
      ? selected.from
      : undefined
  );

  const weekDays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

  const getDaysInMonth = (date: Date) => {
    const start = startOfWeek(startOfMonth(date));
    const end = endOfWeek(endOfMonth(date));
    return eachDayOfInterval({ start, end });
  };

  const handleDateClick = (date: Date) => {
    if (disabled?.(date)) return;

    if (mode === "single") {
      onSelect?.(date);
    } else if (mode === "range") {
      if (!rangeStart) {
        setRangeStart(date);
        onSelect?.({ from: date, to: undefined });
      } else {
        if (date < rangeStart) {
          onSelect?.({ from: date, to: rangeStart });
        } else {
          onSelect?.({ from: rangeStart, to: date });
        }
        setRangeStart(undefined);
      }
    }
  };

  const isSelected = (date: Date) => {
    if (mode === "single" && selected instanceof Date) {
      return isSameDay(date, selected);
    }
    if (mode === "range" && selected && typeof selected === "object" && "from" in selected) {
      if (selected.from && selected.to) {
        return date >= selected.from && date <= selected.to;
      }
      if (selected.from) {
        return isSameDay(date, selected.from);
      }
    }
    return false;
  };

  const isRangeStart = (date: Date) => {
    if (mode === "range" && selected && typeof selected === "object" && "from" in selected) {
      return selected.from && isSameDay(date, selected.from);
    }
    return false;
  };

  const isRangeEnd = (date: Date) => {
    if (mode === "range" && selected && typeof selected === "object" && "from" in selected) {
      return selected.to && isSameDay(date, selected.to);
    }
    return false;
  };

  const renderMonth = (monthOffset: number) => {
    const month = addMonths(currentMonth, monthOffset);
    const days = getDaysInMonth(month);

    return (
      <div key={monthOffset} className="p-3">
        <div className="flex items-center justify-between mb-4">
          {monthOffset === 0 && (
            <Button
              variant="outline"
              size="icon-sm"
              onClick={() => setCurrentMonth(subMonths(currentMonth, 1))}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
          )}
          {monthOffset !== 0 && <div className="w-8" />}
          <h2 className="text-sm font-semibold">
            {format(month, "MMMM yyyy")}
          </h2>
          {monthOffset === numberOfMonths - 1 && (
            <Button
              variant="outline"
              size="icon-sm"
              onClick={() => setCurrentMonth(addMonths(currentMonth, 1))}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          )}
          {monthOffset !== numberOfMonths - 1 && <div className="w-8" />}
        </div>

        <div className="grid grid-cols-7 gap-1 mb-2">
          {weekDays.map((day) => (
            <div
              key={day}
              className="text-center text-xs font-medium text-muted-foreground py-1"
            >
              {day}
            </div>
          ))}
        </div>

        <div className="grid grid-cols-7 gap-1">
          {days.map((date, index) => {
            const isCurrentMonth = isSameMonth(date, month);
            const isSelectedDate = isSelected(date);
            const isDisabled = disabled?.(date);
            const isTodayDate = isToday(date);
            const isStart = isRangeStart(date);
            const isEnd = isRangeEnd(date);

            return (
              <button
                key={index}
                type="button"
                disabled={isDisabled}
                onClick={() => handleDateClick(date)}
                className={cn(
                  "h-9 w-9 text-sm rounded-md transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
                  !isCurrentMonth && "text-muted-foreground/50",
                  isSelectedDate && "bg-primary text-primary-foreground",
                  isSelectedDate && !isStart && !isEnd && mode === "range" && "rounded-none bg-primary/20 text-foreground",
                  isStart && "rounded-l-md",
                  isEnd && "rounded-r-md",
                  isTodayDate && !isSelectedDate && "border border-primary",
                  isDisabled && "opacity-50 cursor-not-allowed hover:bg-transparent"
                )}
              >
                {format(date, "d")}
              </button>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className={cn("bg-background rounded-md border", className)}>
      <div className={cn("flex", numberOfMonths > 1 && "divide-x")}>
        {Array.from({ length: numberOfMonths }).map((_, i) => renderMonth(i))}
      </div>
    </div>
  );
}

Calendar.displayName = "Calendar";

export { Calendar };
