import { describe, expect, it } from "vitest";
import { formatBogotaTime } from "./format-date";

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
