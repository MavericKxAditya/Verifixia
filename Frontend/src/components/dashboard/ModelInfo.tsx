import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Cpu, Layers, Zap, HardDrive, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

interface ModelInfoProps {
  modelData?: {
    model_name?: string;
    version?: string;
    architecture?: string;
    framework?: string;
    device?: string;
    total_parameters?: number;
    input_size?: string;
    status?: string;
  };
  processingTime?: {
    preprocessing_ms?: number;
    inference_ms?: number;
    total_ms?: number;
  };
  isLoaded?: boolean;
}

export const ModelInfo = ({ modelData, processingTime, isLoaded = false }: ModelInfoProps) => {
  const formatNumber = (num: number | undefined) => {
    if (!num) return "N/A";
    if (num >= 1000000) return `${(num / 1000000).toFixed(2)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(2)}K`;
    return num.toString();
  };

  const getStatusColor = (status?: string) => {
    switch (status) {
      case "loaded":
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/40";
      case "not_found":
        return "bg-destructive/10 text-destructive border-destructive/40";
      default:
        return "bg-muted text-muted-foreground border-border/60";
    }
  };

  return (
    <Card className="border border-border/60 bg-card/80 backdrop-blur-sm">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-xs font-medium text-muted-foreground tracking-[0.18em] uppercase">
          Model Information
        </CardTitle>
        <Badge
          className={cn(
            "text-[10px] font-semibold border",
            getStatusColor(modelData?.status)
          )}
        >
          {isLoaded ? "ACTIVE" : "STANDBY"}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {/* Model Name & Version */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Cpu className="w-4 h-4 text-primary" />
            <span className="text-xs font-medium">Model</span>
          </div>
          <div className="text-right">
            <div className="text-xs font-mono text-foreground">
              {modelData?.model_name || "Verifixia"}
            </div>
            <div className="text-[10px] text-muted-foreground">
              v{modelData?.version || "1.2.0"}
            </div>
          </div>
        </div>

        {/* Architecture */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Layers className="w-4 h-4 text-primary" />
            <span className="text-xs font-medium">Architecture</span>
          </div>
          <span className="text-xs font-mono text-foreground">
            {modelData?.architecture || "EfficientNet-B0 + LSTM"}
          </span>
        </div>

        {/* Framework & Device */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-muted-foreground">
            <HardDrive className="w-4 h-4 text-primary" />
            <span className="text-xs font-medium">Runtime</span>
          </div>
          <div className="text-right">
            <div className="text-xs font-mono text-foreground">
              {modelData?.framework || "PyTorch"}
            </div>
            <div className="text-[10px] text-muted-foreground uppercase">
              {modelData?.device || "CPU"}
            </div>
          </div>
        </div>

        {/* Parameters */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Activity className="w-4 h-4 text-primary" />
            <span className="text-xs font-medium">Parameters</span>
          </div>
          <span className="text-xs font-mono text-foreground">
            {formatNumber(modelData?.total_parameters)}
          </span>
        </div>

        {/* Processing Time */}
        {processingTime && processingTime.total_ms !== undefined && processingTime.total_ms > 0 && (
          <div className="pt-2 border-t border-border/40">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Zap className="w-4 h-4 text-primary" />
                <span className="text-xs font-medium">Processing Time</span>
              </div>
              <span className="text-xs font-mono text-primary font-semibold">
                {processingTime.total_ms.toFixed(1)}ms
              </span>
            </div>
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-muted-foreground">Preprocessing</span>
                <span className="font-mono text-foreground">
                  {processingTime.preprocessing_ms?.toFixed(1) || "0.0"}ms
                </span>
              </div>
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-muted-foreground">Inference</span>
                <span className="font-mono text-foreground">
                  {processingTime.inference_ms?.toFixed(1) || "0.0"}ms
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Input Size */}
        <div className="pt-2 border-t border-border/40">
          <div className="flex items-center justify-between text-[10px]">
            <span className="text-muted-foreground">Input Size</span>
            <span className="font-mono text-foreground">
              {modelData?.input_size || "224x224"}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
