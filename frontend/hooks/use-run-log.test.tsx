import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useRunLog } from "./use-run-log";

vi.mock("@/lib/api-client", () => ({
  getRunLog: vi.fn(),
}));

import { getRunLog } from "@/lib/api-client";

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useRunLog", () => {
  it("returns the log text once loaded", async () => {
    vi.mocked(getRunLog).mockResolvedValue("line 1");

    const { result } = renderHook(() => useRunLog("run-1"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBe("line 1");
  });

  it("returns null (not an error) when there is no log yet", async () => {
    vi.mocked(getRunLog).mockResolvedValue(null);

    const { result } = renderHook(() => useRunLog("run-1"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBeNull();
  });
});
