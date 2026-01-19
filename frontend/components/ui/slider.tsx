"use client";

import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { cn } from "@/lib/utils";

interface SliderProps
  extends React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root> {
  showValue?: boolean;
  valueFormat?: (value: number) => string;
  size?: "sm" | "default" | "lg";
}

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  SliderProps
>(
  (
    {
      className,
      showValue,
      valueFormat = (v) => v.toString(),
      size = "default",
      value,
      defaultValue,
      ...props
    },
    ref
  ) => {
    const trackSizes = {
      sm: "h-1",
      default: "h-2",
      lg: "h-3",
    };

    const thumbSizes = {
      sm: "h-3 w-3",
      default: "h-5 w-5",
      lg: "h-6 w-6",
    };

    const currentValue = value ?? defaultValue ?? [0];

    return (
      <div className="relative">
        <SliderPrimitive.Root
          ref={ref}
          className={cn(
            "relative flex w-full touch-none select-none items-center",
            className
          )}
          value={value}
          defaultValue={defaultValue}
          {...props}
        >
          <SliderPrimitive.Track
            className={cn(
              "relative w-full grow overflow-hidden rounded-full bg-secondary",
              trackSizes[size]
            )}
          >
            <SliderPrimitive.Range className="absolute h-full bg-primary" />
          </SliderPrimitive.Track>
          {(value ?? defaultValue ?? [0]).map((_, index) => (
            <SliderPrimitive.Thumb
              key={index}
              className={cn(
                "block rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
                thumbSizes[size]
              )}
            />
          ))}
        </SliderPrimitive.Root>
        {showValue && (
          <div className="mt-2 flex justify-between">
            {Array.isArray(currentValue) && currentValue.length > 1 ? (
              <>
                <span className="text-xs text-muted-foreground">
                  {valueFormat(currentValue[0])}
                </span>
                <span className="text-xs text-muted-foreground">
                  {valueFormat(currentValue[currentValue.length - 1])}
                </span>
              </>
            ) : (
              <span className="text-xs text-muted-foreground">
                {valueFormat(Array.isArray(currentValue) ? currentValue[0] : currentValue)}
              </span>
            )}
          </div>
        )}
      </div>
    );
  }
);
Slider.displayName = SliderPrimitive.Root.displayName;

interface RangeSliderProps
  extends Omit<SliderProps, "value" | "defaultValue" | "onValueChange"> {
  value?: [number, number];
  defaultValue?: [number, number];
  onValueChange?: (value: [number, number]) => void;
}

const RangeSlider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  RangeSliderProps
>(({ ...props }, ref) => (
  <Slider
    ref={ref}
    {...props}
    value={props.value}
    defaultValue={props.defaultValue}
    onValueChange={props.onValueChange as (value: number[]) => void}
  />
));
RangeSlider.displayName = "RangeSlider";

export { Slider, RangeSlider };
