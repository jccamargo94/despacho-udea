"use client";

import { RequireAuth } from "@/components/require-auth";
import { useAuth } from "@/lib/auth-context";
import Link from "next/link";
import type { ReactNode } from "react";

function Nav() {
  const { signOut } = useAuth();
  return (
    <header>
      <nav>
        <Link href="/runs">Ejecuciones</Link>
        <Link href="/scenarios">Escenarios</Link>
        <Link href="/compare">Comparar</Link>
        <button onClick={() => signOut()}>Salir</button>
      </nav>
    </header>
  );
}

export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <RequireAuth>
      <Nav />
      <main>{children}</main>
    </RequireAuth>
  );
}
