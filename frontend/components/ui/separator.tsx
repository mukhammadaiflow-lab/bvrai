"use client";

import * as React from "react";
import * as SeparatorPrimitive from "@radix-ui/react-separator";
import { cn } from "@/lib/utils";

const Separator = React.forwardRef<
  React.ElementRef<typeof SeparatorPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SeparatorPrimitive.Root> & {
    label?: string;
  }
>(
  (
    { className, orientation = "horizontal", decorative = true, label, ...props },
    ref
  ) => {
    if (label) {
      return (
        <div className="relative flex items-center">
          <SeparatorPrimitive.Root
            ref={ref}
            decorative={decorative}
            orientation={orientation}
            className={cn(
              "shrink-0 bg-border",
              orientation === "horizontal" ? "h-[1px] flex-1" : "h-full w-[1px]",
              className
            )}
            {...props}
          />
          <span className="px-3 text-xs text-muted-foreground uppercase tracking-wider">
            {label}
          </span>
          <SeparatorPrimitive.Root
            decorative={decorative}
            orientation={orientation}
            className={cn(
              "shrink-0 bg-border",
              orientation === "horizontal" ? "h-[1px] flex-1" : "h-full w-[1px]"
            )}
          />
        </div>
      );
    }

    return (
      <SeparatorPrimitive.Root
        ref={ref}
        decorative={decorative}
        orientation={orientation}
        className={cn(
          "shrink-0 bg-border",
          orientation === "horizontal" ? "h-[1px] w-full" : "h-full w-[1px]",
          className
        )}
        {...props}
      />
    );
  }
);
Separator.displayName = SeparatorPrimitive.Root.displayName;

export { Separator };
