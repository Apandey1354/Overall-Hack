import * as React from "react";
import { cn } from "../../lib/utils";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost";
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", ...props }, ref) => {
    const variants = {
      primary:
        "bg-gradient-to-r from-neon-red via-neon-orange to-neon-red text-white shadow-neon hover:scale-[1.01]",
      ghost: "border border-white/15 text-white/80 hover:border-neon-red/40 hover:text-white",
    };
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center rounded-full px-4 py-2 text-sm font-semibold uppercase tracking-wide transition-all duration-300",
          variants[variant],
          className,
        )}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

