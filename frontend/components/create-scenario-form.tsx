"use client";

import { createScenario } from "@/lib/api-client";
import type { BessUnit, CreateScenarioRequest } from "@/lib/types";
import { useMutation } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

function emptyUnit(): BessUnit {
  return {
    name: "",
    mwh_nom: 0,
    hours_to_deplete: 1,
    initial_soc: 0,
    min_soc: 0,
    max_soc: 0,
    efficiency: 0,
    charge_bid: null,
    discharge_bid: null,
  };
}

export function CreateScenarioForm({ onCreated }: { onCreated: () => void }) {
  const [mode, setMode] = useState<"arbitrage" | "grid_asset">("arbitrage");
  const [penetrationLevel, setPenetrationLevel] = useState("");
  const [units, setUnits] = useState<BessUnit[]>([emptyUnit()]);

  // TanStack Query v5 calls mutationFn(variables, context); the wrapper
  // keeps createScenario's own single-argument signature intact.
  const mutation = useMutation({
    mutationFn: (variables: CreateScenarioRequest) => createScenario(variables),
    onSuccess: onCreated,
  });

  function updateUnit(index: number, patch: Partial<BessUnit>) {
    setUnits((prev) => prev.map((u, i) => (i === index ? { ...u, ...patch } : u)));
  }

  function addUnit() {
    setUnits((prev) => [...prev, emptyUnit()]);
  }

  function removeUnit(index: number) {
    setUnits((prev) => prev.filter((_, i) => i !== index));
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    mutation.mutate({ mode, penetration_level: penetrationLevel, units });
  }

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="mode">Modo</label>
      <select
        id="mode"
        value={mode}
        onChange={(e) => setMode(e.target.value as "arbitrage" | "grid_asset")}
      >
        <option value="arbitrage">arbitrage</option>
        <option value="grid_asset">grid_asset</option>
      </select>

      <label htmlFor="penetration_level">Nivel de penetracion</label>
      <input
        id="penetration_level"
        value={penetrationLevel}
        onChange={(e) => setPenetrationLevel(e.target.value)}
        required
      />

      {units.map((unit, i) => (
        <fieldset key={i}>
          <legend>Unidad {i + 1}</legend>

          <label htmlFor={`unit-${i}-name`}>Nombre</label>
          <input
            id={`unit-${i}-name`}
            value={unit.name}
            onChange={(e) => updateUnit(i, { name: e.target.value })}
            required
          />

          <label htmlFor={`unit-${i}-mwh_nom`}>Capacidad (MWh)</label>
          <input
            id={`unit-${i}-mwh_nom`}
            type="number"
            step="any"
            value={unit.mwh_nom}
            onChange={(e) => updateUnit(i, { mwh_nom: Number(e.target.value) })}
            required
          />

          <label htmlFor={`unit-${i}-hours_to_deplete`}>Horas para agotar</label>
          <input
            id={`unit-${i}-hours_to_deplete`}
            type="number"
            step="any"
            min="0.01"
            value={unit.hours_to_deplete}
            onChange={(e) => updateUnit(i, { hours_to_deplete: Number(e.target.value) })}
            required
          />

          <label htmlFor={`unit-${i}-initial_soc`}>SOC inicial (fraccion 0-1)</label>
          <input
            id={`unit-${i}-initial_soc`}
            type="number"
            step="any"
            min="0"
            max="1"
            value={unit.initial_soc}
            onChange={(e) => updateUnit(i, { initial_soc: Number(e.target.value) })}
            required
          />

          <label htmlFor={`unit-${i}-min_soc`}>SOC minimo (fraccion 0-1)</label>
          <input
            id={`unit-${i}-min_soc`}
            type="number"
            step="any"
            min="0"
            max="1"
            value={unit.min_soc}
            onChange={(e) => updateUnit(i, { min_soc: Number(e.target.value) })}
            required
          />

          <label htmlFor={`unit-${i}-max_soc`}>SOC maximo (fraccion 0-1)</label>
          <input
            id={`unit-${i}-max_soc`}
            type="number"
            step="any"
            min="0"
            max="1"
            value={unit.max_soc}
            onChange={(e) => updateUnit(i, { max_soc: Number(e.target.value) })}
            required
          />

          <label htmlFor={`unit-${i}-efficiency`}>Eficiencia (fraccion 0-1)</label>
          <input
            id={`unit-${i}-efficiency`}
            type="number"
            step="any"
            min="0"
            max="1"
            value={unit.efficiency}
            onChange={(e) => updateUnit(i, { efficiency: Number(e.target.value) })}
            required
          />

          {mode === "arbitrage" && (
            <>
              <label htmlFor={`unit-${i}-charge_bid`}>Oferta de carga</label>
              <input
                id={`unit-${i}-charge_bid`}
                type="number"
                step="any"
                value={unit.charge_bid ?? ""}
                onChange={(e) => updateUnit(i, { charge_bid: Number(e.target.value) })}
                required
              />

              <label htmlFor={`unit-${i}-discharge_bid`}>Oferta de descarga</label>
              <input
                id={`unit-${i}-discharge_bid`}
                type="number"
                step="any"
                value={unit.discharge_bid ?? ""}
                onChange={(e) => updateUnit(i, { discharge_bid: Number(e.target.value) })}
                required
              />
            </>
          )}

          <button type="button" onClick={() => removeUnit(i)}>
            Quitar unidad
          </button>
        </fieldset>
      ))}

      <button type="button" onClick={addUnit}>
        Agregar unidad
      </button>

      <button type="submit" disabled={mutation.isPending}>
        Crear escenario
      </button>

      {mutation.isError && <p role="alert">{(mutation.error as Error).message}</p>}
    </form>
  );
}
