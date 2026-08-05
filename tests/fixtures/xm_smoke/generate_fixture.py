"""One-off generator for the Fase 2B smoke-test fixture.

Run once (`uv run python tests/fixtures/xm_smoke/generate_fixture.py` from
repo root) and commit the output alongside this script. Not run at test
time — the fixture files it produces are the actual test input.
"""

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

FECHA = date(2024, 4, 18)
MMDD = "0418"
BASE = Path(__file__).parent
HOURS = [datetime(2024, 4, 18) + timedelta(hours=h) for h in range(24)]

GENERATORS = [
    {
        "name": "TERMO1",
        "dispo_kw": 300_000,
        "bid_cop_kwh": 150,
        "pap_cop": 1_500_000,
        "mo": 10,
        "gpini": 150,
        "conf": "CONF1",
        "tconf": 5,
    },
    {
        "name": "TERMO2",
        "dispo_kw": 200_000,
        "bid_cop_kwh": 180,
        "pap_cop": 1_500_000,
        "mo": 5,
        "gpini": 0,
        "conf": "CONF0",
        "tconf": 0,
    },
]

with open(BASE / "dispo_declarada.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "resource_name", "dispo", "gen_type"])
    for g in GENERATORS:
        for h in HOURS:
            w.writerow([h.isoformat(sep=" "), g["name"], g["dispo_kw"], "TERMICA"])

with open(BASE / "ofertas.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Date", "resource_name", "Value"])
    for g in GENERATORS:
        w.writerow([FECHA.isoformat(), g["name"], g["bid_cop_kwh"]])

with open(BASE / "demaCome.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "dema"])
    for h in HOURS:
        w.writerow([h.isoformat(sep=" "), 350_000])

with open(BASE / "agc_asignado.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "recurso", "agc"])
    w.writerow([HOURS[0].isoformat(sep=" "), "TERMO1", 0])

with open(BASE / "parametros_plantas.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["generador", "TMG"])
    for g in GENERATORS:
        w.writerow([g["name"], 1])

(BASE / "precio_bolsa").mkdir(exist_ok=True)
with open(BASE / "precio_bolsa" / "precio_bolsa_2024.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "precio_bolsa"])
    for h in HOURS:
        w.writerow([h.isoformat(sep=" "), 200])

(BASE / "ramps.json").write_text("{}")
(BASE / "preideal_dispatch_map.json").write_text("{}")

flat_dir = BASE / str(FECHA)
flat_dir.mkdir(exist_ok=True)
ci_dir = BASE / "condicion_inicial" / str(FECHA)
ci_dir.mkdir(parents=True, exist_ok=True)

ofei_lines = []
for g in GENERATORS:
    ofei_lines.append(f"{g['name']},C PAPC,{g['pap_cop']}")
for g in GENERATORS:
    mo_vals = ",".join(str(g["mo"]) for _ in range(24))
    ofei_lines.append(f"{g['name']}, MO,{mo_vals}")
(flat_dir / f"OFEI{MMDD}.txt").write_text("\n".join(ofei_lines) + "\n")

prid_row = ["TOTAL"] + ["350"] * 24
(flat_dir / f"PrId{MMDD}_NAL.txt").write_text(",".join(prid_row) + "\n", encoding="latin1")

with open(ci_dir / f"dCondIniP{MMDD}.txt", "w") as f:
    f.write("Recurso,Tipo,Gpini-1,Conf_Pini-1,T_CONF_Pini-1\n")
    for g in GENERATORS:
        f.write(f"{g['name']},T,{g['gpini']},{g['conf']},{g['tconf']}\n")

(ci_dir / f"dCondIniU{MMDD}.txt").write_text("Recurso,Tipo,Gini-1,Cini-1\n")

print("fixture written to", BASE)
