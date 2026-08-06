"use client";

import { toHourlyDispatchSeries } from "@/lib/dispatch-chart-data";
import type { DispatchRow } from "@/lib/types";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const COLORS = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#6b7280"];

export function DispatchChart({ rows }: { rows: DispatchRow[] }) {
  if (rows.length === 0) {
    return <p>No hay datos de despacho todavia.</p>;
  }

  const { data, seriesKeys } = toHourlyDispatchSeries(rows);

  return (
    <ResponsiveContainer width="100%" height={320}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="hour" label={{ value: "Hora del dia", position: "insideBottom", offset: -5 }} />
        <YAxis label={{ value: "MW", angle: -90, position: "insideLeft" }} />
        <Tooltip />
        <Legend />
        {seriesKeys.map((key, i) => (
          <Area
            key={key}
            type="monotone"
            dataKey={key}
            stackId="dispatch"
            stroke={COLORS[i % COLORS.length]}
            fill={COLORS[i % COLORS.length]}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
