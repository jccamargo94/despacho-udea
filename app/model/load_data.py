from typing import Iterable, Iterator

from datetime import date

from thefuzz import process, fuzz
import pandas as pd


class DataLoader:
    def __init__(self, folder: str, dispatch_date: date) -> None:
        self._folder = folder
        self._dispatch_date = dispatch_date

    def load_csv(self) -> None:
        self.dispo = pd.read_csv(
            f"{self._folder}/dispo_declarada.csv",
            parse_dates=["datetime"],
            engine="pyarrow",
        )
        self.ofertas = pd.read_csv(
            f"{self._folder}/ofertas.csv", parse_dates=["Date"], engine="pyarrow"
        )
        self.demanda = pd.read_csv(
            f"{self._folder}/demaCome.csv", parse_dates=["datetime"], engine="pyarrow"
        )
        self.agc_asignado = pd.read_csv(
            f"{self._folder}/agc_asignado.csv",
            parse_dates=["datetime"],
            engine="pyarrow",
        )
        return None

    def load_txt_files(self) -> None:
        output = []
        MO = []
        with open(
            f"{self._folder}/OFEI{self._dispatch_date.month:0>2}{self._dispatch_date.day:0>2}.txt",
            "r",
        ) as file:
            for line in file.readlines():
                line = line.strip()
                if "PAP" in line:
                    output.append(line)
                if "MO" in line:
                    mo_line = line.split(",")
                    if len(mo_line) > 2 and "MO" in mo_line[1]:
                        MO.append(mo_line)
        # Precio de arranque
        self.precio_arranque = pd.DataFrame(
            [line.split(",") for line in output if not "usd" in line.lower()],
            columns=["resource", "type", "price"],
        )
        self.precio_arranque["price"] = self.precio_arranque["price"].astype(float)

        # Minimo operativo
        self.minimo_operativo = pd.DataFrame(
            MO,
            columns=[
                "resource",
                "type",
            ]
            + list(range(24)),
        )
        self.minimo_operativo = (
            self.minimo_operativo.set_index(["resource", "type"]).stack().reset_index()
        )
        self.minimo_operativo.columns = [
            "resource",
            "type",
            "hour",
            "self.minimo_operativo",
        ]
        self.minimo_operativo["datetime"] = pd.to_datetime(
            self._dispatch_date
        ) + pd.to_timedelta(self.minimo_operativo["hour"], unit="h")
        self.minimo_operativo["minimo_operativo"] = self.minimo_operativo[
            "minimo_operativo"
        ].astype(float)
        # Condiciones Iniciales
        with open(
            f"data/dCondIniP{self._dispatch_date.month:0>2}{self._dispatch_date.day:0>2}.txt",
            "r",
        ) as file:
            self.initial_conditions = file.readlines()
            self.initial_conditions = [
                line.strip().split(",") for line in self.initial_conditions
            ]
            headers = self.initial_conditions.pop(0)
        self.initial_conditions = pd.DataFrame(self.initial_conditions, columns=headers)
        return None

    def filter_data(
        self,
    ) -> None:
        self.dispo = self.dispo[
            (self.dispo["datetime"].dt.date == self._dispatch_date)
            & (self.dispo["resource_name"].notnull())
        ]
        self.dispo = self.dispo.drop_duplicates(subset=["resource_name", "datetime"])
        self.ofertas = self.ofertas[self.ofertas["Date"].dt.date == self._dispatch_date]
        self.agc_asignado = self.agc_asignado[
            self.agc_asignado["datetime"].dt.date == self._dispatch_date
        ]
        self.demanda = self.demanda[
            self.demanda["datetime"].dt.date == self._dispatch_date
        ]
        return None

    def create_model_sets(self) -> None:
        self.major_generators = self.ofertas.resource_name.unique()
        self.generators = self.dispo.resource_name.unique()
        self.timestamps = self.demanda["datetime"].to_dict().values()
        self.fuel_generators = self.dispo[
            (self.dispo["resource_name"].isin(self.major_generators))
            & (self.dispo["gen_type"] == "TERMICA")
        ].resource_name.unique()
        return None

    def load_operational_minimum() -> pd.DataFrame:
        MO_map = {
            gen: results[0]
            for gen in minimo_operativo.resource.unique()
            if (
                results := process.extractOne(
                    query=gen.lower(),
                    choices=generators,
                    scorer=fuzz.token_sort_ratio,
                    processor=lambda x: x.lower().replace(" ", ""),
                    score_cutoff=70,
                )
            )
        }
        minimo_operativo["resource"] = minimo_operativo["resource"].apply(
            lambda x: MO_map.get(x, x)
        )
        return minimo_operativo

    def load_cold_start_costs() -> tuple[dict, dict]:
        generators_pap_map = {
            gen: process.extractOne(
                query=gen.lower(),
                choices=precio_arranque.resource.unique(),
                scorer=fuzz.partial_token_sort_ratio,
                processor=lambda x: x.lower().replace(" ", ""),
                score_cutoff=70,
            )[0]
            for gen in fuel_generators
        }

        cold_start = {}
        for gen in fuel_generators:
            gen_name_mapped = generators_pap_map[gen]
            gen_pap = precio_arranque[
                (precio_arranque["resource"] == gen_name_mapped)
                & (precio_arranque.type.str.contains("F"))
            ]["price"].values[0]
            cold_start[gen] = float(gen_pap) * 1e-3
        return generators_pap_map, cold_start

    def create_model_params():
        # Valores en MWh
        Pmax = (
            dispo.query("resource_name in @generators")
            .set_index(["resource_name", "datetime"])
            .sort_index()["dispo"]
            * 1e-3
        )
        Pmin = minimo_operativo.set_index(["resource", "datetime"]).sort_index()[
            "minimo_operativo"
        ]
        beta = (
            ofertas.query("resource_name in @generators")
            .set_index(["resource_name"])
            .sort_index()["Value"]
            * 1e3
        )
        agc_indexed = agc_asignado.set_index(["recurso", "datetime"])["agc"] * 1e-3
        return Pmax, Pmin, beta, agc_indexed
