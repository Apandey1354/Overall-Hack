import * as React from "react";
import { cn } from "../../lib/utils";

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "neon" | "outline";
}

export const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, variant = "neon", ...props }, ref) => {
    const variants = {
      neon: "bg-gradient-to-r from-neon-red to-neon-orange text-white shadow-[0_2px_12px_rgba(255,51,88,0.65)]",
      outline: "border border-white/30 text-white/80",
    };
    return (
      <div
        ref={ref}
        className={cn(
          "inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide",
          variants[variant],
          className,
        )}
        {...props}
      />
    );
  },
);
Badge.displayName = "Badge";

