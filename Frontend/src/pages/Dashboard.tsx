import { useState, useEffect, useCallback, useRef } from "react";
import { Clock, Gauge, ScanFace } from "lucide-react";
import { toast } from "sonner";

import { VideoFeed } from "@/components/dashboard/VideoFeed";
import { ConfidenceGauge } from "@/components/dashboard/ConfidenceGauge";
import { DetectionLog, LogEntry } from "@/components/dashboard/DetectionLog";
import { SecurityScore } from "@/components/dashboard/SecurityScore";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { ControlPanel } from "@/components/dashboard/ControlPanel";
import { AnalysisSummary } from "@/components/dashboard/AnalysisSummary";
import { ModelInfo } from "@/components/dashboard/ModelInfo";
import { uploadImage, fetchDetectionLogs, fetchModelInfo, logLiveEvent } from "../../api";

export const Dashboard = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [sensitivity, setSensitivity] = useState(65);
  const [confidenceScore, setConfidenceScore] = useState(0);
  const [securityScore, setSecurityScore] = useState(100);
  const [securityTrend, setSecurityTrend] = useState<"up" | "down" | "stable">("stable");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [latency, setLatency] = useState(0);
  const [fps, setFps] = useState(0);
  const [facesScanned, setFacesScanned] = useState(0);
  const [threatLevel, setThreatLevel] = useState<"safe" | "warning" | "danger">("safe");
  const [lastPrediction, setLastPrediction] = useState<"Real" | "Fake" | "Unknown">("Unknown");
  const [lastConfidence, setLastConfidence] = useState<number | null>(null);
  const [lastFilename, setLastFilename] = useState<string | undefined>(undefined);
  const [lastIsVideo, setLastIsVideo] = useState<boolean | undefined>(undefined);
  const [lastThreatLevel, setLastThreatLevel] = useState<string | undefined>(undefined);
  const [lastModelUsed, setLastModelUsed] = useState<string | undefined>(undefined);
  const [lastAnalysis, setLastAnalysis] = useState<Record<string, string> | undefined>(undefined);
  const [lastModelInfo, setLastModelInfo] = useState<Record<string, string> | undefined>(undefined);
  const [lastProcessingTime, setLastProcessingTime] = useState<Record<string, number> | undefined>(undefined);
  const [mediaSrc, setMediaSrc] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<"image" | "video" | null>(null);
  const [currentObjectUrl, setCurrentObjectUrl] = useState<string | null>(null);
  const [liveSessionId, setLiveSessionId] = useState<string>(() => crypto.randomUUID());
  
  // Smoothing state for live monitoring
  const [smoothedFakeScore, setSmoothedFakeScore] = useState(0);
  const EMA_ALPHA = 0.2; // Smoothing factor (lower = smoother/slower, higher = faster/jitterier)
  const SMOOTHING_WINDOW = 10; // Number of frames for initial average
  
  const previousThreatLevel = useRef<"safe" | "warning" | "danger">("safe");
  const hasAnalysisResult = lastConfidence !== null || lastPrediction !== "Unknown";

  const addLogEntry = useCallback((entry: Omit<LogEntry, "id" | "timestamp">) => {
    const now = new Date();
    const timestamp = now.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }) + "." + now.getMilliseconds().toString().padStart(3, "0");

    const newEntry: LogEntry = {
      id: crypto.randomUUID(),
      timestamp,
      ...entry,
    };

    setLogs((prev) => [newEntry, ...prev].slice(0, 50));
  }, []);

  const clearCurrentObjectUrl = useCallback(() => {
    if (!currentObjectUrl) return;
    URL.revokeObjectURL(currentObjectUrl);
    setCurrentObjectUrl(null);
  }, [currentObjectUrl]);

  const clearUploadedMedia = useCallback(() => {
    clearCurrentObjectUrl();
    setMediaSrc(null);
    setMediaType(null);
    setLastFilename(undefined);
    setLastIsVideo(undefined);
    setLastThreatLevel(undefined);
    setLastPrediction("Unknown");
    setLastConfidence(null);
    setLastModelUsed(undefined);
    setLastAnalysis(undefined);
    setLastModelInfo(undefined);
    setLastProcessingTime(undefined);
  }, [clearCurrentObjectUrl]);

  const hydratePersistedLogs = useCallback(async () => {
    try {
      const data = await fetchDetectionLogs();
      const serverLogs = Array.isArray(data?.items) ? data.items : [];
      const mappedLogs: LogEntry[] = serverLogs
        .slice()
        .reverse()
        .map((log: { timestamp?: string; filename?: string; prediction?: string }) => {
          const parsedTs = new Date(log.timestamp);
          const timestamp = Number.isNaN(parsedTs.getTime())
            ? String(log.timestamp ?? "")
            : parsedTs.toLocaleTimeString("en-US", {
                hour12: false,
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              });

          const prediction = String(log.prediction || "Unknown");
          const message = `${log.filename || "Uploaded media"}: ${prediction}`;
          const type: LogEntry["type"] =
            prediction.toLowerCase() === "fake"
              ? "error"
              : prediction.toLowerCase() === "real"
              ? "success"
              : "info";

          return {
            id: crypto.randomUUID(),
            timestamp,
            message,
            type,
          };
        });

      setLogs((prev) => {
        const combined = [...prev, ...mappedLogs];
        const seen = new Set<string>();
        const deduped = combined.filter((entry) => {
          const key = `${entry.timestamp}|${entry.message}|${entry.type}`;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
        return deduped.slice(0, 50);
      });
    } catch (error) {
      console.warn("Could not fetch persisted detection logs", error);
    }
  }, []);

  // Real-time frame capture for live monitoring
  useEffect(() => {
    if (!isMonitoring || !!mediaSrc) {
      if (!mediaSrc && !isMonitoring) {
        setThreatLevel("safe");
      }
      return;
    }

    let captureInterval: NodeJS.Timeout | null = null;
    let frameCount = 0;

    const captureFrame = async () => {
      // Internal logging removed to reduce console spam
      try {
        // Find the video element more specifically
        const video = document.querySelector('[data-testid="webcam-video"]') as HTMLVideoElement ||
                     document.querySelector('.absolute.inset-0.w-full.h-full.object-cover') as HTMLVideoElement ||
                     document.querySelector('video') as HTMLVideoElement;

        if (!video || !video.videoWidth || !video.videoHeight) {
          console.log('Video element not ready:', { video: !!video, width: video?.videoWidth, height: video?.videoHeight, readyState: video?.readyState, paused: video?.paused });
          return;
        }

        // Ensure video is actually playing
        if (video.paused || video.readyState < 2) {
          return;
        }

        // Create a canvas to capture the current frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          console.error('Could not get canvas context');
          return;
        }

        // Draw the current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to blob
        canvas.toBlob(async (blob) => {
          if (!blob) {
            console.error('Could not create blob from canvas');
            return;
          }

          // Create a File object from the blob
          const file = new File([blob], `frame_${frameCount++}.jpg`, { type: 'image/jpeg' });

          try {
            // Send frame to backend for analysis
            console.log('Sending frame to backend for analysis...');
            const result = await uploadImage(file);

            console.log('Frame analysis result:', result);

            // Update UI with real prediction results
            const rawFakeProb = result.probabilities?.Fake ?? 
                               (result.prediction?.toLowerCase() === 'fake' ? result.confidence : 100 - (result.confidence || 100));
                               
            // Use Exponential Moving Average (EMA) for premium smoothing
            setSmoothedFakeScore(prev => {
              const nextSmoothed = prev === 0 ? rawFakeProb : (EMA_ALPHA * rawFakeProb + (1 - EMA_ALPHA) * prev);
              
              const isFake = nextSmoothed > sensitivity;
              const displayConfidence = isFake ? nextSmoothed : (100 - nextSmoothed);
              
              setLastPrediction(isFake ? 'Fake' : 'Real');
              setLastConfidence(displayConfidence);
              
              // CRITICAL: Gauge always shows Fake Probability (0 = Safe, 100 = Fake)
              setConfidenceScore(Math.round(nextSmoothed)); 
              
              setLastThreatLevel(isFake ? "high" : "low");
              setThreatLevel(isFake ? "danger" : "safe");
              
              // Update security score
              setSecurityScore(s => {
                const step = isFake ? (nextSmoothed / 20) : 0.5;
                const nextS = isFake ? Math.max(0, s - step) : Math.min(100, s + step);
                setSecurityTrend(isFake ? "down" : (nextS === 100 ? "stable" : "up"));
                return Math.round(nextS);
              });
              
              return nextSmoothed;
            });

            setLastModelUsed(result.model_used);
            setLastAnalysis(result.analysis);
            setLastModelInfo(result.model_info);
            setLastProcessingTime(result.processing_time);

            // Update latency with real processing time
            if (result.processing_time?.total_ms) {
              setLatency(Math.round(result.processing_time.total_ms));
            }

            // Log the live event using the RAW result for telemetry
            await logLiveEvent({
              prediction: result.prediction,
              confidence: result.confidence,
              source_type: 'live',
              filename: file.name,
              threat_level: result.threat_level,
              model_used: result.model_used,
              processing_time_ms: result.processing_time?.total_ms || 0
            });

          } catch (error) {
            console.error('Frame analysis failed:', error);
            // Fallback to safe state on error
            setThreatLevel("safe");
          }
        }, 'image/jpeg', 0.8); // 80% quality

      } catch (error) {
        console.error('Frame capture failed:', error);
      }
    };

    // Capture frames every 1 second for more responsive monitoring
    captureInterval = setInterval(captureFrame, 1000);

    // Initial capture after 1 second
    setTimeout(captureFrame, 1000);

    return () => {
      if (captureInterval) {
        clearInterval(captureInterval);
      }
    };
  }, [isMonitoring, mediaSrc, addLogEntry, sensitivity]);

  // Update metrics based on real processing times
  useEffect(() => {
    if (!isMonitoring || !!mediaSrc) return;

    const metricInterval = setInterval(() => {
      // Update FPS (simulate 1 FPS for frame capture rate)
      setFps(1);

      // Update faces scanned
      setFacesScanned((prev) => prev + 1);

      // Latency will be updated from actual processing times in the frame capture effect
    }, 500);

    return () => clearInterval(metricInterval);
  }, [isMonitoring, mediaSrc]);

  // Trigger alert toast when deepfake detected
  useEffect(() => {
    if (threatLevel === "danger" && isMonitoring) {
      toast.error("⚠️ DEEPFAKE DETECTED", {
        description: "High probability of synthetic media manipulation detected in video feed.",
        duration: 5000,
        className: "cyber-glow-red",
      });
    }
  }, [threatLevel, isMonitoring]);

  useEffect(() => {
    fetchModelInfo()
      .then((data) => {
        const info = data?.info || data;
        if (!info) return;
        setLastModelInfo(info);
        setLastModelUsed(info.model_name || data?.type || "Verifixia AI");
      })
      .catch((error) => {
        console.warn("Could not fetch model info", error);
      });
  }, []);

  useEffect(() => {
    hydratePersistedLogs();
  }, [hydratePersistedLogs]);

  useEffect(() => {
    return () => {
      if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
      }
    };
  }, [currentObjectUrl]);

  const handleStartStop = () => {
    setIsMonitoring(!isMonitoring);
    if (!isMonitoring) {
      const nextSessionId = crypto.randomUUID();
      setLiveSessionId(nextSessionId);
      if (mediaSrc) {
        clearUploadedMedia();
        addLogEntry({ message: "Uploaded media removed. Switched to live monitoring feed.", type: "info" });
      }
      addLogEntry({ message: "Monitoring session started", type: "info" });
      logLiveEvent({
        session_id: nextSessionId,
        source: "Live Monitoring",
        event_name: "monitoring_started",
        prediction: "Unknown",
        threat_level: "low",
        latency_ms: latency,
      }).catch((err) => console.warn("Failed to persist live start event", err));
      toast.success("Monitoring Started", {
        description: "Verifixia is now analyzing the video feed.",
      });
    } else {
      addLogEntry({ message: "Monitoring session ended", type: "info" });
      logLiveEvent({
        session_id: liveSessionId,
        source: "Live Monitoring",
        event_name: "monitoring_stopped",
        prediction: lastPrediction,
        threat_level: threatLevel,
        confidence: lastConfidence ?? 0,
        latency_ms: latency,
      }).catch((err) => console.warn("Failed to persist live stop event", err));
      setConfidenceScore(0);
      setSmoothedFakeScore(0);
      toast.info("Monitoring Stopped", {
        description: "Video analysis has been paused.",
      });
    }
  };

  const handleUploadMedia = async (file: File) => {
    if (mediaSrc) {
      clearUploadedMedia();
    }

    addLogEntry({
      message: `Upload received: "${file.name}". Running deepfake analysis...`,
      type: "info",
    });

    const localUrlTemp = URL.createObjectURL(file);
    setCurrentObjectUrl(localUrlTemp);
    setMediaSrc(localUrlTemp);
    setMediaType(file.type.startsWith("video/") ? "video" : "image");

    let isProcessing = true;
    // Removed artificial random fluctuation to prevent user confusion and ensure a premium feel.

    try {
      const result = await uploadImage(file);
      isProcessing = false;

      const rawConfidence = typeof result?.confidence === "string"
        ? parseFloat(result.confidence)
        : result?.confidence;
        
      const percentScore = (typeof rawConfidence === "number" && !Number.isNaN(rawConfidence)) 
        ? (rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence)
        : 50;

      // Ensure sensitivity applies to uploads too!
      const rawFakeProb = result?.probabilities?.Fake ?? 
                          (result?.prediction?.toLowerCase() === 'fake' ? percentScore : 100 - percentScore);
                          
      const isFake = rawFakeProb > sensitivity;
      const finalPrediction = isFake ? "Fake" : "Real";
      const finalConfidence = isFake ? rawFakeProb : (100 - rawFakeProb);

      // CRITICAL: Gauge always shows Fake Probability (0 = Safe, 100 = Fake)
      setConfidenceScore(Math.round(rawFakeProb));
      setLastConfidence(Math.round(finalConfidence));
      setLastPrediction(finalPrediction);
      setLastFilename(result?.filename ?? file.name);
      setLastIsVideo(result?.isVideo === true);
      setLastThreatLevel(isFake ? "high" : "low");
      setThreatLevel(isFake ? "danger" : "safe");
      
      // Update Security Score heavily on manual uploads
      setSecurityScore(prev => {
        const next = isFake ? Math.max(0, prev - 15) : Math.min(100, prev + 5);
        setSecurityTrend(isFake ? "down" : "up");
        return Math.round(next);
      });

      setLastModelUsed(result?.model_used);
      setLastAnalysis(result?.analysis);
      setLastModelInfo(result?.model_info);
      setLastProcessingTime(result?.processing_time);

      // The media source and type are already set at the start of handleUploadMedia
      // to ensure immediate visibility during processing. No need to reset them here.

      // Add detailed log entry
      const logMessage = result?.model_used 
        ? `Analysis complete using ${result.model_used}: ${finalPrediction} (${Math.round(finalConfidence || 0)}%)`
        : `Analysis complete for "${file.name}": ${finalPrediction}`;

      addLogEntry({
        message: logMessage,
        type: finalPrediction === "Fake" ? "error" : finalPrediction === "Real" ? "success" : "info",
      });

      // Show detailed toast
      const toastDescription = result?.processing_time?.total_ms
        ? `Confidence: ${Math.round(finalConfidence || 0)}% | Processing: ${result.processing_time.total_ms.toFixed(1)}ms`
        : `Confidence: ${Math.round(finalConfidence)}%`;

      toast.success(`Media analysis complete (${finalPrediction})`, {
        description: toastDescription,
      });

      // Pull persisted logs so upload history survives refresh and appears in forensic views.
      hydratePersistedLogs();
    } catch (error) {
      isProcessing = false;
      console.error("Upload failed", error);
      setThreatLevel("safe");
      setConfidenceScore(0);
      addLogEntry({
        message: `Analysis failed for "${file.name}". Please try again.`,
        type: "error",
      });
      toast.error("Media analysis failed", {
        description: "Could not analyze the uploaded file. Please try again.",
      });
    }
  };

  useEffect(() => {
    const threatRaised = previousThreatLevel.current !== "danger" && threatLevel === "danger";
    previousThreatLevel.current = threatLevel;
    if (!isMonitoring || !threatRaised) return;

    logLiveEvent({
      session_id: liveSessionId,
      source: "Live Monitoring",
      event_name: "deepfake_alert",
      prediction: "Fake",
      threat_level: "high",
      confidence: confidenceScore,
      latency_ms: latency,
      message: "High probability synthetic manipulation detected in live feed.",
    }).catch((err) => console.warn("Failed to persist live alert event", err));
  }, [threatLevel, isMonitoring, liveSessionId, confidenceScore, latency]);

  const handleClearMedia = () => {
    if (!mediaSrc) return;
    clearUploadedMedia();
    addLogEntry({ message: "Uploaded media removed by user.", type: "info" });
    toast.info("Media removed", {
      description: "You can upload another file or continue with live monitoring.",
    });
  };

  return (
    <div className="max-w-[1600px] mx-auto space-y-4 md:space-y-6">
      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6">
        {/* Left Column - Video Feed & Gauge */}
        <div className="lg:col-span-2 space-y-4 md:space-y-6">
          <VideoFeed isMonitoring={isMonitoring} threatLevel={threatLevel} mediaSrc={mediaSrc} mediaType={mediaType} />
          <ConfidenceGauge value={confidenceScore} isActive={isMonitoring || hasAnalysisResult} />
        </div>

        {/* Right Column - Sidebar */}
        <div className="space-y-4 md:space-y-6">
          <SecurityScore score={securityScore} trend={securityTrend} />
          <ControlPanel
            isMonitoring={isMonitoring}
            sensitivity={sensitivity}
            onStartStop={handleStartStop}
            onSensitivityChange={setSensitivity}
            onUploadMedia={handleUploadMedia}
            hasUploadedMedia={Boolean(mediaSrc)}
            onClearMedia={handleClearMedia}
          />
          <ModelInfo
            modelData={lastModelInfo}
            processingTime={lastProcessingTime}
            isLoaded={!!lastModelUsed}
          />
          <AnalysisSummary
            prediction={lastPrediction}
            confidence={lastConfidence}
            filename={lastFilename}
            isVideo={lastIsVideo}
            threatLevel={lastThreatLevel}
            modelUsed={lastModelUsed}
            analysis={lastAnalysis}
          />
          <div className="h-[300px]">
            <DetectionLog logs={logs} />
          </div>
        </div>
      </div>

      {/* Bottom Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          title="Average Latency"
          value={latency}
          unit="ms"
          icon={Clock}
          trend={{ value: 12, isPositive: true }}
        />
        <MetricCard
          title="Frame Analysis Rate"
          value={fps}
          unit="FPS"
          icon={Gauge}
          trend={{ value: 5, isPositive: true }}
        />
        <MetricCard
          title="Total Faces Scanned"
          value={facesScanned.toLocaleString()}
          icon={ScanFace}
        />
      </div>
    </div>
  );
};

export default Dashboard;
