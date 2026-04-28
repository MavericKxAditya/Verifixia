import { useState } from "react";
import { Outlet } from "react-router-dom";
import { AppSidebar } from "./AppSidebar";
import { GlobalHeader } from "./GlobalHeader";
import { NotificationPanel } from "./NotificationPanel";
import { cn } from "@/lib/utils";

export const AppLayout = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background cyber-grid">
      {/* Sidebar */}
      <AppSidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content Area */}
      <div
        className={cn(
          "transition-all duration-300",
          sidebarCollapsed ? "ml-16" : "ml-64"
        )}
      >
        {/* Global Header */}
        <GlobalHeader
          onOpenNotifications={() => setNotificationsOpen(true)}
          notificationCount={2}
        />

        {/* Page Content */}
        <main className="p-4 md:p-6">
          <Outlet />
        </main>
      </div>

      {/* Notification Panel */}
      <NotificationPanel
        isOpen={notificationsOpen}
        onClose={() => setNotificationsOpen(false)}
      />
    </div>
  );
};
