import { X, AlertTriangle, CheckCircle, Info, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface Notification {
  id: string;
  type: "alert" | "success" | "info" | "warning";
  title: string;
  description: string;
  timestamp: string;
  read: boolean;
}

const mockNotifications: Notification[] = [
  {
    id: "1",
    type: "alert",
    title: "High-risk deepfake detected",
    description: "Stream #402 flagged with 94% confidence score",
    timestamp: "2 min ago",
    read: false,
  },
  {
    id: "2",
    type: "warning",
    title: "Unusual activity pattern",
    description: "Multiple failed authentication attempts from IP 192.168.1.45",
    timestamp: "15 min ago",
    read: false,
  },
  {
    id: "3",
    type: "success",
    title: "System scan completed",
    description: "All security protocols verified and operational",
    timestamp: "1 hour ago",
    read: true,
  },
  {
    id: "4",
    type: "info",
    title: "Model update available",
    description: "Verifixia v2.5 detection model ready for deployment",
    timestamp: "3 hours ago",
    read: true,
  },
  {
    id: "5",
    type: "alert",
    title: "API rate limit warning",
    description: "External integration approaching usage threshold",
    timestamp: "5 hours ago",
    read: true,
  },
];

const iconMap = {
  alert: AlertTriangle,
  warning: AlertTriangle,
  success: CheckCircle,
  info: Info,
};

const colorMap = {
  alert: "text-destructive",
  warning: "text-warning",
  success: "text-success",
  info: "text-primary",
};

interface NotificationPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const NotificationPanel = ({ isOpen, onClose }: NotificationPanelProps) => {
  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 z-40 bg-background/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Panel */}
      <div className="fixed right-0 top-0 z-50 h-full w-full max-w-md glass-card border-l border-border/50 animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border/50">
          <div>
            <h2 className="text-lg font-semibold">Notifications</h2>
            <p className="text-sm text-muted-foreground">
              {mockNotifications.filter(n => !n.read).length} unread
            </p>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Notifications List */}
        <ScrollArea className="h-[calc(100vh-80px)]">
          <div className="p-4 space-y-3">
            {mockNotifications.map((notification) => {
              const Icon = iconMap[notification.type];
              return (
                <div
                  key={notification.id}
                  className={cn(
                    "p-4 rounded-lg border transition-all cursor-pointer hover:bg-secondary/30",
                    notification.read
                      ? "border-border/30 bg-transparent"
                      : "border-primary/30 bg-primary/5"
                  )}
                >
                  <div className="flex items-start gap-3">
                    <div className={cn("mt-0.5", colorMap[notification.type])}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <p className="font-medium text-sm truncate">
                          {notification.title}
                        </p>
                        {!notification.read && (
                          <Badge variant="secondary" className="bg-primary/20 text-primary text-[10px]">
                            NEW
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {notification.description}
                      </p>
                      <div className="flex items-center gap-1 mt-2 text-xs text-muted-foreground">
                        <Clock className="w-3 h-3" />
                        {notification.timestamp}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </div>
    </>
  );
};
