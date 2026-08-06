import { getRun } from "@/lib/api-client";
import { useQueries } from "@tanstack/react-query";

export function useRunComparisons(ids: string[]) {
  return useQueries({
    queries: ids.map((id) => ({
      queryKey: ["run", id],
      queryFn: () => getRun(id),
    })),
  });
}
