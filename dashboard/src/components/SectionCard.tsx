import { PropsWithChildren, ReactNode } from "react";
import clsx from "clsx";

interface Props extends PropsWithChildren {
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

export function SectionCard({ title, description, action, className, children }: Props) {
  return (
    <section
      className={clsx(
        "section-card",
        className,
      )}
    >
      <div className="section-card__header">
        <div>
          <h2>{title}</h2>
          {description && <p className="section-card__subtitle">{description}</p>}
        </div>
        {action && <div className="section-card__action">{action}</div>}
      </div>
      <div className="section-card__body">{children}</div>
    </section>
  );
}

