import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  User,
  Shield,
  Key,
  Camera,
  Smartphone,
  Monitor,
  Globe,
  Copy,
  Eye,
  EyeOff,
  Plus,
  Trash2,
  Check,
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { fetchUserProfile, updateUserProfile } from "../../api";
import { getCurrentUser } from "@/lib/auth";

interface Session {
  id: string;
  device: string;
  browser: string;
  ip: string;
  location: string;
  lastActive: string;
  current: boolean;
}

interface ApiKey {
  id: string;
  name: string;
  key: string;
  created: string;
  lastUsed: string;
}

const mockSessions: Session[] = [
  {
    id: "1",
    device: "Desktop",
    browser: "Current Browser",
    ip: "127.0.0.1",
    location: "Local Session",
    lastActive: "Now",
    current: true,
  },
];

const mockApiKeys: ApiKey[] = [
  {
    id: "1",
    name: "Frontend Session",
    key: "Managed by Firebase / backend auth",
    created: "Current session",
    lastUsed: "Now",
  },
];

export const Profile = () => {
  const [searchParams] = useSearchParams();
  const defaultTab = searchParams.get("tab") || "account";
  const authUser = getCurrentUser();

  const [biometricLogin, setBiometricLogin] = useState(true);
  const [activityAlerts, setActivityAlerts] = useState(true);
  const [showApiKey, setShowApiKey] = useState<string | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(true);
  const [savingProfile, setSavingProfile] = useState(false);
  const [profile, setProfile] = useState<Record<string, unknown> | null>(null);
  const [form, setForm] = useState({
    firstName: "",
    lastName: "",
    email: authUser?.email || "",
    displayName: authUser?.displayName || "",
    role: "",
    phone: "",
    organization: "",
  });

  useEffect(() => {
    let isMounted = true;

    const loadProfile = async () => {
      try {
        const data = await fetchUserProfile();
        if (!isMounted) return;
        const backendProfile = (data?.profile || {}) as Record<string, unknown>;
        setProfile(backendProfile);

        const displayName = String(
          backendProfile.display_name ||
          backendProfile.displayName ||
          authUser?.displayName ||
          ""
        );
        const [firstName = "", ...rest] = displayName.split(" ").filter(Boolean);
        const lastName = rest.join(" ");

        setForm({
          firstName,
          lastName,
          email: String(backendProfile.email || authUser?.email || ""),
          displayName,
          role: String(backendProfile.role || ""),
          phone: String(backendProfile.phone || ""),
          organization: String(backendProfile.organization || ""),
        });
      } catch (error) {
        console.warn("Could not load backend profile", error);
        if (!isMounted) return;
        setForm((prev) => ({
          ...prev,
          email: authUser?.email || prev.email,
          displayName: authUser?.displayName || prev.displayName,
        }));
      } finally {
        if (isMounted) setLoadingProfile(false);
      }
    };

    loadProfile();
    return () => {
      isMounted = false;
    };
  }, [authUser?.displayName, authUser?.email]);

  const avatarFallback = useMemo(() => {
    const source = form.displayName || authUser?.displayName || form.email || "VF";
    const parts = source.split(/\s+/).filter(Boolean);
    if (parts.length >= 2) {
      return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    }
    return source.slice(0, 2).toUpperCase();
  }, [authUser?.displayName, form.displayName, form.email]);

  const handleCopyKey = (key: string) => {
    navigator.clipboard.writeText(key);
    toast.success("API key copied to clipboard");
  };

  const getDeviceIcon = (device: string) => {
    if (device === "Mobile") return Smartphone;
    return Monitor;
  };

  const handleSaveProfile = async () => {
    const displayName = [form.firstName, form.lastName].filter(Boolean).join(" ").trim();

    try {
      setSavingProfile(true);
      await updateUserProfile({
        display_name: displayName || form.displayName || null,
        role: form.role || null,
        phone: form.phone || null,
        organization: form.organization || null,
      });
      setForm((prev) => ({
        ...prev,
        displayName: displayName || prev.displayName,
      }));
      toast.success("Profile updated");
    } catch (error) {
      console.error(error);
      toast.error("Could not update profile");
    } finally {
      setSavingProfile(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Profile Settings</h1>
        <p className="text-muted-foreground">
          Manage your account settings and security preferences
        </p>
      </div>

      <Tabs defaultValue={defaultTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 max-w-md">
          <TabsTrigger value="account" className="flex items-center gap-2">
            <User className="w-4 h-4" />
            Account
          </TabsTrigger>
          <TabsTrigger value="security" className="flex items-center gap-2">
            <Shield className="w-4 h-4" />
            Security
          </TabsTrigger>
          <TabsTrigger value="api" className="flex items-center gap-2">
            <Key className="w-4 h-4" />
            API Keys
          </TabsTrigger>
        </TabsList>

        <TabsContent value="account" className="space-y-6">
          <Card className="glass-card border-border/50">
            <CardHeader>
              <CardTitle>Profile Information</CardTitle>
              <CardDescription>Update your personal details</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center gap-6">
                <div className="relative">
                  <Avatar className="w-20 h-20 border-2 border-primary/30">
                    <AvatarImage src={String(profile?.photoURL || "")} />
                    <AvatarFallback className="bg-primary/10 text-primary text-xl">
                      {avatarFallback}
                    </AvatarFallback>
                  </Avatar>
                  <Button
                    size="icon"
                    variant="secondary"
                    className="absolute -bottom-1 -right-1 w-8 h-8 rounded-full"
                    disabled
                  >
                    <Camera className="w-4 h-4" />
                  </Button>
                </div>
                <div>
                  <p className="font-medium">Profile Photo</p>
                  <p className="text-sm text-muted-foreground">
                    Managed by your authenticated account provider.
                  </p>
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="firstName">First Name</Label>
                  <Input
                    id="firstName"
                    value={form.firstName}
                    onChange={(e) => setForm((prev) => ({ ...prev, firstName: e.target.value }))}
                    disabled={loadingProfile}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lastName">Last Name</Label>
                  <Input
                    id="lastName"
                    value={form.lastName}
                    onChange={(e) => setForm((prev) => ({ ...prev, lastName: e.target.value }))}
                    disabled={loadingProfile}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <Input id="email" type="email" value={form.email} disabled />
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="role">Role</Label>
                  <Input
                    id="role"
                    value={form.role}
                    onChange={(e) => setForm((prev) => ({ ...prev, role: e.target.value }))}
                    disabled={loadingProfile}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="phone">Phone</Label>
                  <Input
                    id="phone"
                    value={form.phone}
                    onChange={(e) => setForm((prev) => ({ ...prev, phone: e.target.value }))}
                    disabled={loadingProfile}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="organization">Organization</Label>
                <Input
                  id="organization"
                  value={form.organization}
                  onChange={(e) => setForm((prev) => ({ ...prev, organization: e.target.value }))}
                  disabled={loadingProfile}
                />
              </div>

              <Button onClick={handleSaveProfile} disabled={loadingProfile || savingProfile}>
                {savingProfile ? "Saving..." : "Save Changes"}
              </Button>
            </CardContent>
          </Card>

          <Card className="glass-card border-border/50">
            <CardHeader>
              <CardTitle>Change Password</CardTitle>
              <CardDescription>
                Password changes are managed by your authentication provider.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Input value="Use Firebase or your identity provider to update your password." disabled />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-6">
          <Card className="glass-card border-border/50">
            <CardHeader>
              <CardTitle>Security Settings</CardTitle>
              <CardDescription>Configure your security preferences</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Biometric Login</Label>
                  <p className="text-sm text-muted-foreground">
                    Use fingerprint or face recognition to sign in
                  </p>
                </div>
                <Switch checked={biometricLogin} onCheckedChange={setBiometricLogin} />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Activity Alerts</Label>
                  <p className="text-sm text-muted-foreground">
                    Get notified of suspicious account activity
                  </p>
                </div>
                <Switch checked={activityAlerts} onCheckedChange={setActivityAlerts} />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Two-Factor Authentication</Label>
                  <p className="text-sm text-muted-foreground">
                    Add an extra layer of security
                  </p>
                </div>
                <Badge variant="secondary" className="bg-success/20 text-success">
                  <Check className="w-3 h-3 mr-1" />
                  Auth Provider Managed
                </Badge>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card border-border/50">
            <CardHeader>
              <CardTitle>Active Sessions</CardTitle>
              <CardDescription>Manage devices where you're signed in</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockSessions.map((session) => {
                  const DeviceIcon = getDeviceIcon(session.device);
                  return (
                    <div
                      key={session.id}
                      className="flex items-center justify-between p-4 rounded-lg border border-border/50 bg-secondary/20"
                    >
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                          <DeviceIcon className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <p className="font-medium">{session.browser}</p>
                            {session.current && (
                              <Badge variant="secondary" className="bg-success/20 text-success text-xs">
                                Current
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Globe className="w-3 h-3" />
                            {session.ip} • {session.location}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-muted-foreground">{session.lastActive}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api" className="space-y-6">
          <Card className="glass-card border-border/50">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>API Keys</CardTitle>
                  <CardDescription>
                    Backend APIs are authenticated through the current signed-in session.
                  </CardDescription>
                </div>
                <Button disabled>
                  <Plus className="w-4 h-4 mr-2" />
                  Generate New Key
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockApiKeys.map((apiKey) => (
                  <div
                    key={apiKey.id}
                    className="p-4 rounded-lg border border-border/50 bg-secondary/20"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Key className="w-4 h-4 text-primary" />
                        <span className="font-medium">{apiKey.name}</span>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-destructive hover:text-destructive"
                        disabled
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                    <div className="flex items-center gap-2 mb-3">
                      <code className="flex-1 px-3 py-2 rounded bg-background font-mono text-sm">
                        {showApiKey === apiKey.id ? apiKey.key : "•".repeat(32)}
                      </code>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setShowApiKey(showApiKey === apiKey.id ? null : apiKey.id)}
                      >
                        {showApiKey === apiKey.id ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </Button>
                      <Button variant="ghost" size="icon" onClick={() => handleCopyKey(apiKey.key)}>
                        <Copy className="w-4 h-4" />
                      </Button>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>Created: {apiKey.created}</span>
                      <span>Last used: {apiKey.lastUsed}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Profile;
