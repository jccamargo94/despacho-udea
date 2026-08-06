import type { Scenario } from "@/lib/types";

export function ScenariosTable({ scenarios }: { scenarios: Scenario[] }) {
  if (scenarios.length === 0) return <p>Sin escenarios todavia.</p>;

  return (
    <table>
      <thead>
        <tr>
          <th>Nivel de penetracion</th>
          <th>Modo</th>
          <th>Unidades</th>
        </tr>
      </thead>
      <tbody>
        {scenarios.map((s) => (
          <tr key={s.id}>
            <td>{s.penetration_level}</td>
            <td>{s.mode}</td>
            <td>{s.units.length}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
