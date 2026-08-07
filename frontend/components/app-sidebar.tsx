"use client";

import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";
import { Gauge, GitCompareArrows, Layers, LogOut, Zap } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/runs", label: "Ejecuciones", icon: Gauge },
  { href: "/scenarios", label: "Escenarios", icon: Layers },
  { href: "/compare", label: "Comparar", icon: GitCompareArrows },
] as const;

export function AppSidebar() {
  const pathname = usePathname();
  const { signOut } = useAuth();

  return (
    <aside className="fixed inset-y-0 left-0 hidden w-60 flex-col border-r border-border bg-card px-4 py-6 md:flex">
      <div className="mb-8 flex items-center gap-2">
        <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Zap className="size-4" />
        </div>
        <div>
          <p className="font-heading text-sm font-bold leading-tight">GridForge</p>
          <p className="text-xs text-muted-foreground">Technical Dispatch Modeler</p>
        </div>
      </div>
      <nav className="flex flex-1 flex-col gap-1">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname?.startsWith(href) ?? false;
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <Icon className="size-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      <Button variant="ghost" className="justify-start gap-2" onClick={() => signOut()}>
        <LogOut className="size-4" />
        Salir
      </Button>
    </aside>
  );
}
