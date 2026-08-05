"""Parse the XM OFEI text file (initial offers).

Extracts, in one pass over the file, the same artifacts the original scripts
built inline: PAP start-up prices, minimum-operative (MO) profiles, combined
cycle configurations/prices/availabilities, and per-resource bid prices.
"""

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

PRICE_PATTERN = r"P(\d+)"
DISPO_PATTERN = r"DISCONF(\d+)"


@dataclass
class OfeiData:
    precio_arranque: pd.DataFrame  # columns: resource, type, price
    minimo_operativo: pd.DataFrame  # columns: resource, type, hour, minimo_operativo, datetime
    cc: dict  # plant -> [plant_conf, ...]
    cc_price: dict  # plant_conf -> price
    cc_dispo: dict  # plant_conf -> 24 hourly availabilities
    prices: dict  # resource -> bid price (already * 1e-3)


def parse_ofei(path: str, dispatch_date: date) -> OfeiData:
    output = []
    MO = []
    CC: dict = {}
    cc_price: dict = {}
    cc_dispo: dict = {}
    prices: dict = {}

    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if "PAP" in line:
                output.append(line)
            if "MO" in line:
                mo_line = line.split(",")
                if len(mo_line) > 2 and "MO" in mo_line[1]:
                    MO.append(mo_line)
            if (conf := re.findall(PRICE_PATTERN, line)) and "CC" in line:
                fline = line.split(",")
                cc_price[f"{fline[0].strip()}_{conf[0]}"] = float(fline[2])
                if CC.get(fline[0].strip()):
                    CC[fline[0].strip()].append(f"{fline[0].strip()}_{conf[0]}")
                else:
                    CC[fline[0].strip()] = [f"{fline[0].strip()}_{conf[0]}"]
            # Disponibilidad CC
            if (conf := re.findall(DISPO_PATTERN, line)) and "CC" in line:
                fline = line.split(",")
                cc_dispo[f"{fline[0].strip()}_{conf[0]}"] = [int(disp) for disp in fline[2:]]

            # Extract prices
            if "P" in line:
                pri = line.split(",")
                if (
                    len(pri) == 3
                    and " P" in pri[1]
                    and "u" not in pri[1].lower()
                    and "a" not in pri[1].lower()
                ):
                    prices[pri[0]] = float(pri[2]) * 1e-3

    precio_arranque = pd.DataFrame(
        [line.split(",") for line in output if "usd" not in line.lower()],
        columns=["resource", "type", "price"],
    )
    precio_arranque["price"] = precio_arranque["price"].astype(float)

    # Minimo operativo
    minimo_operativo = pd.DataFrame(
        MO,
        columns=["resource", "type"] + list(range(24)),
    )
    minimo_operativo = minimo_operativo.set_index(["resource", "type"]).stack().reset_index()
    minimo_operativo.columns = ["resource", "type", "hour", "minimo_operativo"]
    minimo_operativo["datetime"] = pd.to_datetime(dispatch_date) + pd.to_timedelta(
        minimo_operativo["hour"], unit="h"
    )
    minimo_operativo["minimo_operativo"] = minimo_operativo["minimo_operativo"].astype(float)

    return OfeiData(
        precio_arranque=precio_arranque,
        minimo_operativo=minimo_operativo,
        cc=CC,
        cc_price=cc_price,
        cc_dispo=cc_dispo,
        prices=prices,
    )
