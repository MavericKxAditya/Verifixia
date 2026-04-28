import { useEffect, useMemo, useState } from "react";
import {
  Moon,
  Sun,
  Monitor,
  Shield,
  Archive,
  Bell,
  Gauge,
  Database,
  RefreshCw,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { useTheme } from "next-themes";
import { fetchHealth, fetchModelInfo } from "../../api";

export const Settings = () => {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [sensitivity, setSensitivity] = useState(65);
  const [autoArchive, setAutoArchive] = useState(true);
  const [threatAlerts, setThreatAlerts] = useState(true);
  const [summaryEmails, setSummaryEmails] = useState(false);
  const [retentionDays, setRetentionDays] = useState("30");
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [modelInfo, setModelInfo] = useState<Record<string, unknown> | null>(null);
  const activeTheme = useMemo(() => {
    if (theme === "system") return "system";
    if (theme === "light" || theme === "dark") return theme;
    return resolvedTheme === "light" ? "light" : "dark";
  }, [theme, resolvedTheme]);

  useEffect(() => {
    fetchHealth()
      .then((data) => setHealth(data))
      .catch((error) => console.warn("Could not load backend health", error));

    fetchModelInfo()
      .then((data) => setModelInfo(data?.info || null))
      .catch((error) => console.warn("Could not load model info", error));
  }, []);

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Security Settings</h1>
        <p className="text-muted-foreground">
          Configure system behavior and preferences
        </p>
      </div>

      {/* Appearance */}
      <Card className="glass-card border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Monitor className="w-5 h-5" />
            Appearance
          </CardTitle>
          <CardDescription>Customize the interface appearance</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <Label>Theme Mode</Label>
            <div className="grid grid-cols-3 gap-3">
              <Button
                variant={activeTheme === "light" ? "default" : "outline"}
                className="w-full"
                onClick={() => setTheme("light")}
              >
                <Sun className="w-4 h-4 mr-2" />
                Light
              </Button>
              <Button
                variant={activeTheme === "dark" ? "default" : "outline"}
                className="w-full"
                onClick={() => setTheme("dark")}
              >
                <Moon className="w-4 h-4 mr-2" />
                Dark
              </Button>
              <Button
                variant={activeTheme === "system" ? "default" : "outline"}
                className="w-full"
                onClick={() => setTheme("system")}
              >
                <Monitor className="w-4 h-4 mr-2" />
                System
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detection Settings */}
      <Card className="glass-card border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5" />
            Detection Configuration
          </CardTitle>
          <CardDescription>Adjust detection sensitivity and thresholds</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <Label>System Sensitivity</Label>
                <p className="text-sm text-muted-foreground">
                  Higher values increase detection but may cause false positives
                </p>
              </div>
              <Badge variant="outline" className="font-mono">
                {sensitivity}%
              </Badge>
            </div>
            <Slider
              value={[sensitivity]}
              onValueChange={(v) => setSensitivity(v[0])}
              max={100}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Low (Fast)</span>
              <span>Balanced</span>
              <span>High (Accurate)</span>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Real-time Processing</Label>
              <p className="text-sm text-muted-foreground">
                Enable live frame-by-frame analysis
              </p>
            </div>
            <Switch defaultChecked />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>GPU Acceleration</Label>
              <p className="text-sm text-muted-foreground">
                Use hardware acceleration for faster processing
              </p>
            </div>
            <Switch defaultChecked />
          </div>
        </CardContent>
      </Card>

      {/* Data Management */}
      <Card className="glass-card border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="w-5 h-5" />
            Data Management
          </CardTitle>
          <CardDescription>Configure log retention and archiving</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Auto-Archive Logs</Label>
              <p className="text-sm text-muted-foreground">
                Automatically archive old detection logs
              </p>
            </div>
            <Switch checked={autoArchive} onCheckedChange={setAutoArchive} />
          </div>

          <div className="space-y-3">
            <Label>Data Retention Period</Label>
            <Select value={retentionDays} onValueChange={setRetentionDays}>
              <SelectTrigger>
                <SelectValue placeholder="Select retention period" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="7">7 days</SelectItem>
                <SelectItem value="14">14 days</SelectItem>
                <SelectItem value="30">30 days</SelectItem>
                <SelectItem value="60">60 days</SelectItem>
                <SelectItem value="90">90 days</SelectItem>
                <SelectItem value="365">1 year</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex gap-3">
            <Button variant="outline">
              <Archive className="w-4 h-4 mr-2" />
              Export All Logs
            </Button>
            <Button variant="outline" className="text-destructive hover:text-destructive">
              Clear Old Logs
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Notifications */}
      <Card className="glass-card border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="w-5 h-5" />
            Notifications
          </CardTitle>
          <CardDescription>Configure alert preferences</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Threat Detection Alerts</Label>
              <p className="text-sm text-muted-foreground">
                Get notified when deepfakes are detected
              </p>
            </div>
            <Switch checked={threatAlerts} onCheckedChange={setThreatAlerts} />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Daily Summary Emails</Label>
              <p className="text-sm text-muted-foreground">
                Receive daily detection summary reports
              </p>
            </div>
            <Switch checked={summaryEmails} onCheckedChange={setSummaryEmails} />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>System Health Alerts</Label>
              <p className="text-sm text-muted-foreground">
                Get notified of performance issues
              </p>
            </div>
            <Switch defaultChecked />
          </div>
        </CardContent>
      </Card>

      {/* System */}
      <Card className="glass-card border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <RefreshCw className="w-5 h-5" />
            System
          </CardTitle>
          <CardDescription>System information and actions</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 rounded-lg border border-border/50 bg-secondary/20">
            <div>
              <p className="font-medium">Verifixia AI Engine</p>
              <p className="text-sm text-muted-foreground">
                Backend port 3001 • Frontend port 8086
              </p>
            </div>
            <Badge
              variant="secondary"
              className={
                health?.status === "healthy"
                  ? "bg-success/20 text-success"
                  : "bg-warning/20 text-warning"
              }
            >
              {health?.status === "healthy" ? "Healthy" : "Unavailable"}
            </Badge>
          </div>

          <div className="flex items-center justify-between p-4 rounded-lg border border-border/50 bg-secondary/20">
            <div>
              <p className="font-medium">Detection Model</p>
              <p className="text-sm text-muted-foreground">
                {String(modelInfo?.model_name || modelInfo?.architecture || "Heuristic fallback")}
              </p>
            </div>
            <Badge variant="outline" size="sm">
              {String(health?.active_model || "unknown")}
            </Badge>
          </div>

          <div className="flex gap-3">
            <Button variant="outline" disabled>
              <RefreshCw className="w-4 h-4 mr-2" />
              Restart Services
            </Button>
            <Button variant="outline" disabled>
              Run Diagnostics
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Settings;
