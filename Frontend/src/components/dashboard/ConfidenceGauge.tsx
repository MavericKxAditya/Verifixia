import { useEffect, useState } from "react";

interface ConfidenceGaugeProps {
  value: number; // 0-100
  isActive: boolean;
}

export const ConfidenceGauge = ({ value, isActive }: ConfidenceGaugeProps) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (!isActive) {
      setDisplayValue(0);
      return;
    }
    
    const timer = setTimeout(() => {
      setDisplayValue(value);
    }, 300);
    return () => clearTimeout(timer);
  }, [value, isActive]);

  // Semi-circle gauge calculations
  const radius = 80;
  const strokeWidth = 12;
  const circumference = Math.PI * radius;
  const progress = (displayValue / 100) * circumference;
  
  const getColor = () => {
    if (displayValue < 30) return "hsl(var(--success))";
    if (displayValue < 70) return "hsl(var(--warning))";
    return "hsl(var(--destructive))";
  };

  const getLabel = () => {
    if (displayValue < 30) return "AUTHENTIC";
    if (displayValue < 70) return "SUSPICIOUS";
    return "LIKELY FAKE";
  };

  const getLabelColor = () => {
    if (displayValue < 30) return "text-success";
    if (displayValue < 70) return "text-warning";
    return "text-destructive text-glow-red";
  };

  return (
    <div className="glass-card p-6">
      <h3 className="text-sm font-medium text-muted-foreground mb-4 uppercase tracking-wider">
        Deepfake Confidence Score
      </h3>
      
      <div className="relative flex flex-col items-center">
        {/* SVG Gauge */}
        <svg 
          width="200" 
          height="120" 
          viewBox="0 0 200 120"
          className="overflow-visible"
        >
          {/* Background arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="hsl(var(--muted))"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          
          {/* Progress arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={getColor()}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference - progress}
            className="transition-all duration-700 ease-out"
            style={{
              filter: displayValue > 70 ? "drop-shadow(0 0 8px hsl(var(--destructive) / 0.6))" : "none"
            }}
          />
          
          {/* Center text */}
          <text
            x="100"
            y="85"
            textAnchor="middle"
            className="font-mono font-bold fill-foreground"
            style={{ fontSize: "36px" }}
          >
            {displayValue}%
          </text>
        </svg>

        {/* Status Label */}
        <div className={`mt-2 font-semibold text-lg ${getLabelColor()} transition-colors duration-300`}>
          {isActive ? getLabel() : "STANDBY"}
        </div>

        {/* Scale markers */}
        <div className="flex justify-between w-full mt-4 px-2 text-xs font-mono text-muted-foreground">
          <span>0%</span>
          <span className="text-success">SAFE</span>
          <span className="text-warning">WARN</span>
          <span className="text-destructive">FAKE</span>
          <span>100%</span>
        </div>
      </div>
    </div>
  );
};
