import { getRun } from "@/lib/api-client";
import { isTerminalStatus } from "@/lib/run-status";
import type { RunDetail } from "@/lib/types";
import { useQuery } from "@tanstack/react-query";

export function useRunDetail(id: string) {
  return useQuery({
    queryKey: ["run", id],
    queryFn: () => getRun(id),
    refetchInterval: (query) => {
      const data = query.state.data as RunDetail | undefined;
      if (!data || !isTerminalStatus(data.status)) return 3000;
      return false;
    },
  });
}
