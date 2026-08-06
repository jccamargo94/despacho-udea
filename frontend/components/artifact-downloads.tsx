"use client";

import { downloadRunArtifact } from "@/lib/api-client";
import type { RunArtifacts } from "@/lib/types";
import { useState } from "react";

type ArtifactKey = keyof RunArtifacts;

const ARTIFACT_LABELS: Record<ArtifactKey, string> = {
  dispatch: "Despacho",
  prices: "Precios",
  bess: "BESS",
};

export function ArtifactDownloads({
  runId,
  artifacts,
}: {
  runId: string;
  artifacts: RunArtifacts;
}) {
  const [error, setError] = useState<string | null>(null);

  async function handleDownload(artifact: ArtifactKey) {
    setError(null);
    try {
      const blob = await downloadRunArtifact(runId, artifact);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${artifact}-${runId}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      setError("No se pudo descargar el artefacto.");
    }
  }

  const available = (Object.keys(artifacts) as ArtifactKey[]).filter((key) => artifacts[key]);

  if (available.length === 0) {
    return <p>No hay artefactos disponibles todavia.</p>;
  }

  return (
    <div>
      {available.map((artifact) => (
        <button key={artifact} type="button" onClick={() => handleDownload(artifact)}>
          Descargar {ARTIFACT_LABELS[artifact]}
        </button>
      ))}
      {error && <p role="alert">{error}</p>}
    </div>
  );
}
