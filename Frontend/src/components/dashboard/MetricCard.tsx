import { LucideIcon } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

export const MetricCard = ({ title, value, unit, icon: Icon, trend }: MetricCardProps) => {
  return (
    <div className="glass-card p-5 group hover:cyber-glow transition-all duration-300">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
            {title}
          </p>
          <div className="flex items-baseline gap-1">
            <span className="text-3xl font-bold font-mono text-foreground group-hover:text-primary transition-colors">
              {value}
            </span>
            {unit && (
              <span className="text-sm text-muted-foreground font-mono">
                {unit}
              </span>
            )}
          </div>
          {trend && (
            <p className={`text-xs mt-2 font-mono ${trend.isPositive ? "text-success" : "text-destructive"}`}>
              {trend.isPositive ? "+" : "-"}{Math.abs(trend.value)}% from avg
            </p>
          )}
        </div>
        <div className="p-3 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
          <Icon className="w-5 h-5 text-primary" />
        </div>
      </div>
    </div>
  );
};
