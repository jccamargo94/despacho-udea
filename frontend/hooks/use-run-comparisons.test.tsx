import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useRunComparisons } from "./use-run-comparisons";

vi.mock("@/lib/api-client", () => ({ getRun: vi.fn() }));
import { getRun } from "@/lib/api-client";

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient();
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useRunComparisons", () => {
  it("returns one query result per id, in order", async () => {
    vi.mocked(getRun).mockImplementation((id: string) =>
      Promise.resolve({ run_id: id, status: "done" } as never)
    );

    const { result } = renderHook(() => useRunComparisons(["r1", "r2"]), { wrapper });

    await waitFor(() => expect(result.current).toHaveLength(2));
    await waitFor(() => expect(result.current[0].data?.run_id).toBe("r1"));
    await waitFor(() => expect(result.current[1].data?.run_id).toBe("r2"));
  });

  it("returns an empty array for an empty id list", () => {
    const { result } = renderHook(() => useRunComparisons([]), { wrapper });
    expect(result.current).toEqual([]);
  });
});
