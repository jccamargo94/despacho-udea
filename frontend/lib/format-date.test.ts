import { describe, expect, it } from "vitest";
import { formatBogotaTime, formatDuration } from "./format-date";

describe("formatBogotaTime", () => {
  it("renders a UTC ISO timestamp in America/Bogota (UTC-5, no DST)", () => {
    // 2024-04-18T05:00:00Z -> 2024-04-18 00:00 in Bogota
    const result = formatBogotaTime("2024-04-18T05:00:00Z");
    expect(result).toContain("2024-04-18");
    expect(result).toContain("00:00");
  });

  it("returns a dash for null", () => {
    expect(formatBogotaTime(null)).toBe("—");
  });
});

describe("formatDuration", () => {
  it("formats the difference between started_at and finished_at as Xm Ys", () => {
    expect(formatDuration("2024-04-18T05:00:00Z", "2024-04-18T05:04:12Z")).toBe("4m 12s");
  });

  it("returns a dash when either timestamp is missing", () => {
    expect(formatDuration(null, "2024-04-18T05:04:12Z")).toBe("--");
    expect(formatDuration("2024-04-18T05:00:00Z", null)).toBe("--");
  });
});
