import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { createContext, useContext, useEffect, useState } from "react";
import { AppLayout } from "@/components/layout/AppLayout";
import Dashboard from "./pages/Dashboard";
// Login page may be omitted when testing without auth
import Login from "./pages/Login";
import Profile from "./pages/Profile";
import ForensicLogs from "./pages/ForensicLogs";
import Analytics from "./pages/Analytics";
import Settings from "./pages/Settings";
import Support from "./pages/Support";
import NotFound from "./pages/NotFound";
import { watchAuthState, firebaseEnabled } from "@/lib/auth";

const queryClient = new QueryClient();

// evaluation of environment variable outside components for simplicity
const bypassLogin = String(import.meta.env.VITE_BYPASS_LOGIN || "false") === "true";

// ── Auth context – single Firebase listener for the entire app ───────────────
interface AuthState {
  isReady: boolean;
  isAuthed: boolean;
}

const AuthContext = createContext<AuthState>({
  isReady: !firebaseEnabled,
  isAuthed: false,
});

function AuthProvider({ children }: { children: React.ReactNode }) {
  const bypassLogin = String(import.meta.env.VITE_BYPASS_LOGIN || "false") === "true";

  const [state, setState] = useState<AuthState>(
    bypassLogin
      ? { isReady: true, isAuthed: true }
      : { isReady: !firebaseEnabled, isAuthed: false }
  );

  useEffect(() => {
    if (bypassLogin) {
      // skip any Firebase initialization
      return;
    }
    if (!firebaseEnabled) {
      setState({ isReady: true, isAuthed: true });
      return;
    }
    const unsub = watchAuthState((user) => {
      setState({ isReady: true, isAuthed: !!user });
    });
    return () => unsub();
  }, [bypassLogin]);

  return <AuthContext.Provider value={state}>{children}</AuthContext.Provider>;
}

function useAuth() {
  return useContext(AuthContext);
}

// ── Loading spinner shown while Firebase resolves auth state ─────────────────
const AuthLoadingScreen = () => (
  <div className="min-h-screen bg-background flex items-center justify-center">
    <div className="flex flex-col items-center gap-4">
      <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin" />
      <p className="text-muted-foreground text-sm">Verifying session…</p>
    </div>
  </div>
);

// ── Guards ────────────────────────────────────────────────────────────────────

// when login is bypassed we don't need any redirect logic at all
const ProtectedRoute = ({ children }: { children: JSX.Element }) => children;
const PublicOnlyRoute = ({ children }: { children: JSX.Element }) => <>{children}</>;

// ── App ───────────────────────────────────────────────────────────────────────
const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            {/* When bypassing login we don't expose the login page at all */}
            {!bypassLogin && (
              <Route path="/login" element={<PublicOnlyRoute><Login /></PublicOnlyRoute>} />
            )}

            {/* Protected routes – guards are no-ops when bypassLogin is true */}
            <Route element={<ProtectedRoute><AppLayout /></ProtectedRoute>}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/forensic-logs" element={<ForensicLogs />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/support" element={<Support />} />
              <Route path="/profile" element={<Profile />} />
            </Route>

            <Route path="*" element={<NotFound />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
