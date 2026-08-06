import { getRunLog } from "@/lib/api-client";
import { useQuery } from "@tanstack/react-query";

export function useRunLog(runId: string) {
  return useQuery({
    queryKey: ["run-log", runId],
    queryFn: () => getRunLog(runId),
  });
}
