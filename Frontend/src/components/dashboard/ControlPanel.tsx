import { useRef, useState, type ChangeEvent } from "react";
import { Play, Square, Settings, Upload, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";

interface ControlPanelProps {
  isMonitoring: boolean;
  sensitivity: number;
  onStartStop: () => void;
  onSensitivityChange: (value: number) => void;
  onUploadMedia: (file: File) => Promise<void> | void;
  hasUploadedMedia?: boolean;
  onClearMedia?: () => void;
}

export const ControlPanel = ({
  isMonitoring,
  sensitivity,
  onStartStop,
  onSensitivityChange,
  onUploadMedia,
  hasUploadedMedia = false,
  onClearMedia,
}: ControlPanelProps) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const getSensitivityLabel = () => {
    if (sensitivity < 30) return "Low";
    if (sensitivity < 70) return "Medium";
    return "High";
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      await onUploadMedia(file);
    } finally {
      setIsUploading(false);
      // Allow selecting the same file again
      event.target.value = "";
    }
  };

  return (
    <div className="glass-card p-5">
      <div className="flex items-center gap-2 mb-5">
        <Settings className="w-4 h-4 text-primary" />
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          Controls
        </h3>
      </div>

      {/* Start/Stop + Upload Controls */}
      <div className="mb-6 space-y-3">
        {/* Start/Stop Button */}
        <div>
          <Button
            onClick={onStartStop}
            size="lg"
            className={`w-full font-semibold text-base transition-all duration-300 ${
              isMonitoring
                ? "bg-destructive hover:bg-destructive/90 text-destructive-foreground cyber-glow-red"
                : "bg-primary hover:bg-primary/90 text-primary-foreground cyber-glow"
            }`}
          >
            {isMonitoring ? (
              <>
                <Square className="w-5 h-5 mr-2 fill-current" />
                Stop Monitoring
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2 fill-current" />
                Start Monitoring
              </>
            )}
          </Button>
        </div>

        {/* Upload Media Button */}
        <div>
          <Button
            type="button"
            variant="outline"
            className="w-full font-medium text-sm justify-center"
            onClick={handleUploadClick}
            disabled={isUploading}
          >
            <Upload className="w-4 h-4 mr-2" />
            {isUploading ? "Analyzing..." : "Upload Media"}
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>

        {hasUploadedMedia && (
          <div>
            <Button
              type="button"
              variant="destructive"
              className="w-full font-medium text-sm justify-center"
              onClick={onClearMedia}
              disabled={isUploading}
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Remove Current Media
            </Button>
          </div>
        )}
      </div>

      {/* Sensitivity Slider */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Label className="text-sm text-muted-foreground">
            Detection Sensitivity
          </Label>
          <span className="text-sm font-mono text-primary">
            {sensitivity}% ({getSensitivityLabel()})
          </span>
        </div>
        
        <Slider
          value={[sensitivity]}
          onValueChange={(v) => onSensitivityChange(v[0])}
          max={100}
          step={1}
          className="w-full"
        />

        <div className="flex justify-between text-xs text-muted-foreground font-mono">
          <span>Less Strict</span>
          <span>More Strict</span>
        </div>
      </div>

      {/* Quick Info */}
      <div className="mt-6 pt-4 border-t border-border">
        <p className="text-xs text-muted-foreground leading-relaxed">
          Higher sensitivity may increase false positives but catches more subtle manipulations.
        </p>
      </div>
    </div>
  );
};
