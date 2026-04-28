import { Shield, Wifi } from "lucide-react";
import { useEffect, useState } from "react";

export const Header = () => {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <header className="glass-card px-6 py-4 flex items-center justify-between">
      {/* Logo & Title */}
      <div className="flex items-center gap-4">
        <div className="relative">
          <Shield className="w-10 h-10 text-primary" />
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-primary rounded-full animate-pulse-glow" />
        </div>
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            <span className="text-primary text-glow-cyan">DeepShield</span>
            <span className="text-foreground"> AI</span>
          </h1>
          <p className="text-xs text-muted-foreground tracking-widest uppercase">
            Live Monitoring
          </p>
        </div>
      </div>

      {/* Status Indicators */}
      <div className="flex items-center gap-8">
        {/* System Status */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center gap-2">
            <div className="pulse-indicator">
              <div className="w-2.5 h-2.5 bg-success rounded-full cyber-glow-green" />
            </div>
            <span className="text-sm font-medium text-success">System Online</span>
          </div>
          <div className="w-px h-6 bg-border" />
          <div className="flex items-center gap-2 text-muted-foreground">
            <Wifi className="w-4 h-4 text-primary" />
            <span className="text-sm font-mono">128ms</span>
          </div>
        </div>

        {/* Digital Clock */}
        <div className="text-right">
          <div className="font-mono text-2xl font-semibold text-primary text-glow-cyan tracking-wider">
            {formatTime(time)}
          </div>
          <div className="text-xs text-muted-foreground font-mono">
            {formatDate(time)}
          </div>
        </div>
      </div>
    </header>
  );
};
