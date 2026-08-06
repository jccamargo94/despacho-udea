"use client";

import { AppSidebar } from "@/components/app-sidebar";
import { RequireAuth } from "@/components/require-auth";
import type { ReactNode } from "react";

export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <RequireAuth>
      <div className="min-h-screen bg-background">
        <AppSidebar />
        <main className="md:pl-60">
          <div className="mx-auto max-w-6xl px-6 py-8">{children}</div>
        </main>
      </div>
    </RequireAuth>
  );
}
