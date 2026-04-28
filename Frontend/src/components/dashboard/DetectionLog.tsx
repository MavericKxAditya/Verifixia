import { AlertTriangle, CheckCircle, Info, XCircle } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

export interface LogEntry {
  id: string;
  timestamp: string;
  message: string;
  type: "info" | "warning" | "error" | "success";
}

interface DetectionLogProps {
  logs: LogEntry[];
}

export const DetectionLog = ({ logs }: DetectionLogProps) => {
  const getIcon = (type: LogEntry["type"]) => {
    switch (type) {
      case "error":
        return <XCircle className="w-4 h-4 text-destructive" />;
      case "warning":
        return <AlertTriangle className="w-4 h-4 text-warning" />;
      case "success":
        return <CheckCircle className="w-4 h-4 text-success" />;
      default:
        return <Info className="w-4 h-4 text-primary" />;
    }
  };

  const getBgColor = (type: LogEntry["type"]) => {
    switch (type) {
      case "error":
        return "bg-destructive/10 border-l-destructive";
      case "warning":
        return "bg-warning/10 border-l-warning";
      case "success":
        return "bg-success/10 border-l-success";
      default:
        return "bg-primary/5 border-l-primary";
    }
  };

  return (
    <div className="glass-card p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          Detection Log
        </h3>
        <span className="text-xs font-mono text-primary">
          {logs.length} events
        </span>
      </div>

      <ScrollArea className="flex-1 -mx-4 px-4 cyber-scrollbar">
        <div className="space-y-2">
          {logs.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-sm">
              No detection events yet
            </div>
          ) : (
            logs.map((log, index) => (
              <div
                key={log.id}
                className={`p-3 rounded-lg border-l-2 ${getBgColor(log.type)} animate-fade-in-up`}
                style={{ animationDelay: `${index * 50}ms` }}
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5">{getIcon(log.type)}</div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-foreground leading-tight">
                      {log.message}
                    </p>
                    <p className="text-xs font-mono text-muted-foreground mt-1">
                      {log.timestamp}
                    </p>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
};
