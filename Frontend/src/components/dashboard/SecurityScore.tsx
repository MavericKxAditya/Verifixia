import { ShieldCheck, TrendingUp, TrendingDown } from "lucide-react";

interface SecurityScoreProps {
  score: number;
  trend: "up" | "down" | "stable";
}

export const SecurityScore = ({ score, trend }: SecurityScoreProps) => {
  const getScoreColor = () => {
    if (score >= 80) return "text-success";
    if (score >= 50) return "text-warning";
    return "text-destructive";
  };

  const getScoreLabel = () => {
    if (score >= 80) return "Excellent";
    if (score >= 50) return "Moderate";
    return "Critical";
  };

  const getTrendIcon = () => {
    if (trend === "up") return <TrendingUp className="w-4 h-4 text-success" />;
    if (trend === "down") return <TrendingDown className="w-4 h-4 text-destructive" />;
    return null;
  };

  return (
    <div className="glass-card p-4 cyber-glow">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          Security Score
        </h3>
        <ShieldCheck className="w-5 h-5 text-primary" />
      </div>

      <div className="flex items-end gap-3">
        <span className={`text-4xl font-bold font-mono ${getScoreColor()}`}>
          {score}
        </span>
        <div className="flex items-center gap-1 mb-1">
          <span className={`text-sm ${getScoreColor()}`}>{getScoreLabel()}</span>
          {getTrendIcon()}
        </div>
      </div>

      {/* Score bar */}
      <div className="mt-4 h-2 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-700 ease-out rounded-full ${
            score >= 80 ? "bg-success" : score >= 50 ? "bg-warning" : "bg-destructive"
          }`}
          style={{ width: `${score}%` }}
        />
      </div>

      <div className="flex justify-between mt-2 text-xs text-muted-foreground font-mono">
        <span>0</span>
        <span>100</span>
      </div>
    </div>
  );
};
