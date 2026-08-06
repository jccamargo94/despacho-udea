import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useRunDetail } from "./use-run-detail";

vi.mock("@/lib/api-client", () => ({ getRun: vi.fn() }));
import { getRun } from "@/lib/api-client";

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient();
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useRunDetail", () => {
  it("polls (refetchInterval truthy) while status is pending", async () => {
    vi.mocked(getRun).mockResolvedValue({ status: "pending" } as never);
    const { result } = renderHook(() => useRunDetail("r1"), { wrapper });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data?.status).toBe("pending");
  });

  it("stops polling once status is done", async () => {
    vi.mocked(getRun).mockResolvedValue({ status: "done" } as never);
    const { result } = renderHook(() => useRunDetail("r1"), { wrapper });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data?.status).toBe("done");
  });
});
