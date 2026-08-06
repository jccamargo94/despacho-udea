import { describe, expect, it } from "vitest";
import { toHourlyDispatchSeries } from "./dispatch-chart-data";
import type { DispatchRow } from "./types";

describe("toHourlyDispatchSeries", () => {
  it("groups rows by hour-of-day parsed from the datetime string, not a Date object", () => {
    const rows: DispatchRow[] = [
      { generador: "A", datetime: "2024-04-18 00:00:00", dispatch: 10 },
      { generador: "A", datetime: "2024-04-18 01:00:00", dispatch: 20 },
    ];

    const { data } = toHourlyDispatchSeries(rows);

    expect(data).toEqual([
      { hour: 0, A: 10 },
      { hour: 1, A: 20 },
    ]);
  });

  it("keeps every generador as its own series when there are 6 or fewer", () => {
    const rows: DispatchRow[] = [
      { generador: "A", datetime: "2024-04-18 00:00:00", dispatch: 10 },
      { generador: "B", datetime: "2024-04-18 00:00:00", dispatch: 5 },
    ];

    const { data, seriesKeys } = toHourlyDispatchSeries(rows);

    expect(seriesKeys.sort()).toEqual(["A", "B"]);
    expect(data).toEqual([{ hour: 0, A: 10, B: 5 }]);
  });

  it("buckets everything past the top 6 (by total dispatch) into Otros", () => {
    const rows: DispatchRow[] = [
      { generador: "G1", datetime: "2024-04-18 00:00:00", dispatch: 100 },
      { generador: "G2", datetime: "2024-04-18 00:00:00", dispatch: 90 },
      { generador: "G3", datetime: "2024-04-18 00:00:00", dispatch: 80 },
      { generador: "G4", datetime: "2024-04-18 00:00:00", dispatch: 70 },
      { generador: "G5", datetime: "2024-04-18 00:00:00", dispatch: 60 },
      { generador: "G6", datetime: "2024-04-18 00:00:00", dispatch: 50 },
      { generador: "G7", datetime: "2024-04-18 00:00:00", dispatch: 5 },
      { generador: "G8", datetime: "2024-04-18 00:00:00", dispatch: 3 },
    ];

    const { data, seriesKeys } = toHourlyDispatchSeries(rows);

    expect(seriesKeys).toEqual(["G1", "G2", "G3", "G4", "G5", "G6", "Otros"]);
    expect(data[0].Otros).toBe(8);
    expect(data[0].G7).toBeUndefined();
  });
});
