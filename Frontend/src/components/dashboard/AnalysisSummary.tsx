import { ShieldAlert, ShieldCheck, ImageIcon, Film, Info } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type PredictionType = "Real" | "Fake" | "Unknown";

interface AnalysisSummaryProps {
  prediction?: PredictionType;
  confidence?: number | null;
  filename?: string;
  isVideo?: boolean;
  threatLevel?: string;
  modelUsed?: string;
  analysis?: {
    level?: string;
    description?: string;
    recommendation?: string;
  };
}

export const AnalysisSummary = ({
  prediction = "Unknown",
  confidence,
  filename,
  isVideo,
  threatLevel,
  modelUsed,
  analysis,
}: AnalysisSummaryProps) => {
  const normalizedConfidence =
    typeof confidence === "number" && !Number.isNaN(confidence)
      ? Math.round(confidence <= 1 ? confidence * 100 : confidence)
      : null;

  const isFake = prediction === "Fake";
  const isReal = prediction === "Real";

  const getThreatBadgeColor = (level?: string) => {
    switch (level) {
      case "high":
        return "bg-destructive/10 text-destructive border-destructive/40";
      case "medium":
        return "bg-warning/10 text-warning border-warning/40";
      case "low":
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/40";
      default:
        return "bg-muted text-muted-foreground border-border/60";
    }
  };

  return (
    <Card className="border border-border/60 bg-card/80 backdrop-blur-sm">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-xs font-medium text-muted-foreground tracking-[0.18em] uppercase">
          Last Analysis
        </CardTitle>
        <span
          className={cn(
            "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-semibold border",
            isFake &&
              "bg-destructive/10 text-destructive border-destructive/40 shadow-[0_0_18px_rgba(248,113,113,0.35)]",
            isReal && "bg-emerald-500/10 text-emerald-400 border-emerald-500/40",
            !isFake && !isReal && "bg-muted text-muted-foreground border-border/60",
          )}
        >
          {isFake ? (
            <>
              <ShieldAlert className="w-3 h-3 mr-1" /> Deepfake
            </>
          ) : isReal ? (
            <>
              <ShieldCheck className="w-3 h-3 mr-1" /> Authentic
            </>
          ) : (
            "No analysis yet"
          )}
        </span>
      </CardHeader>
      <CardContent className="space-y-3 pt-0 text-xs">
        {/* File Info */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-muted-foreground">
            {isVideo ? (
              <Film className="w-4 h-4 text-primary" />
            ) : (
              <ImageIcon className="w-4 h-4 text-primary" />
            )}
            <span className="truncate max-w-[160px] font-mono text-[11px]">
              {filename ?? "Awaiting upload"}
            </span>
          </div>

          {normalizedConfidence !== null && (
            <span className="font-mono text-[11px] text-primary font-semibold">
              {normalizedConfidence.toString().padStart(2, "0")}%
            </span>
          )}
        </div>

        {/* Threat Level */}
        {threatLevel && (
          <div className="flex items-center justify-between gap-2 pt-1">
            <span className="text-[11px] text-muted-foreground">Threat Level</span>
            <Badge className={cn("text-[10px] font-semibold border", getThreatBadgeColor(threatLevel))}>
              {threatLevel.toUpperCase()}
            </Badge>
          </div>
        )}

        {/* Model Used */}
        {modelUsed && (
          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px] text-muted-foreground">Model</span>
            <span className="text-[10px] font-mono text-foreground">{modelUsed}</span>
          </div>
        )}

        {/* Analysis Details */}
        {analysis && analysis.description && (
          <div className="pt-2 border-t border-border/40 space-y-2">
            <div className="flex items-start gap-2">
              <Info className="w-3 h-3 text-primary mt-0.5 flex-shrink-0" />
              <div className="space-y-1">
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {analysis.description}
                </p>
                {analysis.recommendation && (
                  <p className="text-[10px] leading-relaxed text-primary/80 italic">
                    {analysis.recommendation}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Default Description */}
        {!analysis?.description && (
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Upload any face image or short clip. Verifixia AI estimates the likelihood of{" "}
            <span className="font-semibold text-primary/90">synthetic manipulation</span> and feeds
            results into your forensic logs.
          </p>
        )}
      </CardContent>
    </Card>
  );
};
