import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ArtifactDownloads } from "./artifact-downloads";

vi.mock("@/lib/api-client", () => ({
  downloadRunArtifact: vi.fn(),
}));

import { downloadRunArtifact } from "@/lib/api-client";

beforeEach(() => {
  vi.mocked(downloadRunArtifact).mockReset();
  vi.stubGlobal("URL", {
    ...URL,
    createObjectURL: vi.fn(() => "blob:mock-url"),
    revokeObjectURL: vi.fn(),
  });
});

describe("ArtifactDownloads", () => {
  it("shows an empty-state message when no artifacts are available", () => {
    render(<ArtifactDownloads runId="run-1" artifacts={{ dispatch: false, prices: false, bess: false }} />);

    expect(screen.getByText(/no hay artefactos disponibles/i)).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("renders one button per available artifact and none for unavailable ones", () => {
    render(<ArtifactDownloads runId="run-1" artifacts={{ dispatch: true, prices: true, bess: false }} />);

    expect(screen.getByRole("button", { name: /despacho/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /precios/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /bess/i })).not.toBeInTheDocument();
  });

  it("clicking a download button fetches the blob and triggers a synthetic anchor click", async () => {
    const fakeBlob = new Blob(["csv,data"]);
    vi.mocked(downloadRunArtifact).mockResolvedValue(fakeBlob);
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    render(<ArtifactDownloads runId="run-1" artifacts={{ dispatch: true, prices: false, bess: false }} />);
    fireEvent.click(screen.getByRole("button", { name: /despacho/i }));

    await vi.waitFor(() => {
      expect(downloadRunArtifact).toHaveBeenCalledWith("run-1", "dispatch");
    });
    expect(clickSpy).toHaveBeenCalled();

    clickSpy.mockRestore();
  });

  it("shows an error message when the download fails", async () => {
    vi.mocked(downloadRunArtifact).mockRejectedValue(new Error("500"));

    render(<ArtifactDownloads runId="run-1" artifacts={{ dispatch: true, prices: false, bess: false }} />);
    fireEvent.click(screen.getByRole("button", { name: /despacho/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/no se pudo descargar/i);
  });
});
