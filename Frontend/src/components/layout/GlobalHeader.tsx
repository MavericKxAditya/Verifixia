import { useState } from "react";
import { Search, Bell, User, ChevronDown, Settings, LogOut, Shield } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useNavigate } from "react-router-dom";
import { getCurrentUser, logoutUser } from "@/lib/auth";
import { toast } from "sonner";

interface GlobalHeaderProps {
  onOpenNotifications: () => void;
  notificationCount?: number;
}

export const GlobalHeader = ({ onOpenNotifications, notificationCount = 3 }: GlobalHeaderProps) => {
  const [searchQuery, setSearchQuery] = useState("");
  const navigate = useNavigate();
  const user = getCurrentUser();
  const userName = user?.displayName || user?.email?.split("@")[0] || "User";
  const userRole = user?.email ? "Authenticated User" : "Guest";

  return (
    <header className="glass-card px-6 py-3 flex items-center justify-between border-b border-border/50">
      {/* Universal Search */}
      <div className="relative w-full max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search logs, UIDs, or sessions..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10 bg-secondary/50 border-border/50 focus:border-primary/50 h-9"
        />
      </div>

      {/* Right Side Actions */}
      <div className="flex items-center gap-4">
        {/* Notification Bell */}
        <Button
          variant="ghost"
          size="icon"
          className="relative"
          onClick={onOpenNotifications}
        >
          <Bell className="w-5 h-5" />
          {notificationCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-destructive text-destructive-foreground text-[10px] font-bold rounded-full flex items-center justify-center">
              {notificationCount > 9 ? "9+" : notificationCount}
            </span>
          )}
        </Button>

        {/* User Profile Dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="flex items-center gap-2 px-2">
              <div className="w-8 h-8 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center">
                <User className="w-4 h-4 text-primary" />
              </div>
              <div className="hidden md:block text-left">
                <p className="text-sm font-medium">{userName}</p>
                <p className="text-xs text-muted-foreground">{userRole}</p>
              </div>
              <ChevronDown className="w-4 h-4 text-muted-foreground hidden md:block" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>My Account</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => navigate("/profile")}>
              <User className="w-4 h-4 mr-2" />
              Profile Settings
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => navigate("/settings")}>
              <Settings className="w-4 h-4 mr-2" />
              Security Settings
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => navigate("/profile?tab=api")}>
              <Shield className="w-4 h-4 mr-2" />
              API Keys
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem 
              className="text-destructive focus:text-destructive"
              onClick={async () => {
                await logoutUser();
                toast.success("Logged out");
                navigate("/login");
              }}
            >
              <LogOut className="w-4 h-4 mr-2" />
              Log Out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
};
