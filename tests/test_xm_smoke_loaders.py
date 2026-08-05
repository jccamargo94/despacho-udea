"""Layer-1 check: every raw fixture file parses through its real loader/parser
with no exceptions, before build_case's own logic gets involved. Isolates
format bugs (encoding, column names, delimiter) from case_builder logic bugs."""

from datetime import date
from pathlib import Path

from app.data import loaders
from app.data.download import ensure_data_for_date
from app.data.ofei import parse_ofei
from app.data.paths import resolve_input

DD = str(Path(__file__).parent / "fixtures" / "xm_smoke")
FECHA = date(2024, 4, 18)


def test_ensure_data_for_date_is_a_noop(monkeypatch):
    def _no_network(*a, **kw):
        raise AssertionError(f"unexpected network call: {a} {kw}")

    monkeypatch.setattr("app.data.download.requests.get", _no_network)
    ensure_data_for_date(FECHA, data_dir=DD)


def test_root_csvs_load():
    dispo = loaders.load_dispo(DD)
    assert len(dispo[dispo["datetime"].dt.date == FECHA]) == 48  # 2 generators x 24h

    ofertas = loaders.load_ofertas(DD)
    assert len(ofertas[ofertas["Date"].dt.date == FECHA]) == 2

    demanda = loaders.load_demanda(DD)
    assert len(demanda[demanda["datetime"].dt.date == FECHA]) == 24

    agc = loaders.load_agc(DD)
    assert "agc" in agc.columns

    params = loaders.load_parametros_plantas(DD)
    assert set(params["generador"]) == {"TERMO1", "TERMO2"}

    precio_bolsa = loaders.load_precio_bolsa(DD)
    assert len(precio_bolsa[precio_bolsa["datetime"].dt.date == FECHA]) == 24


def test_ofei_parses():
    ofei_path = resolve_input("OFEI", FECHA, DD)
    ofei = parse_ofei(ofei_path, FECHA)
    assert set(ofei.precio_arranque["resource"]) == {"TERMO1", "TERMO2"}
    assert all(ofei.precio_arranque["type"].str.contains("C"))
    assert set(ofei.minimo_operativo["resource"]) == {"TERMO1", "TERMO2"}
    assert ofei.cc == {}


def test_condicion_inicial_files_readable():
    p_path = resolve_input("dCondIniP", FECHA, DD)
    with open(p_path) as f:
        lines = f.readlines()
    assert len(lines) == 3  # header + 2 generators

    u_path = resolve_input("dCondIniU", FECHA, DD)
    with open(u_path) as f:
        assert f.readline().strip() == "Recurso,Tipo,Gini-1,Cini-1"


def test_prid_readable_latin1():
    import pandas as pd

    prid_path = resolve_input("PrId", FECHA, DD)
    df = pd.read_csv(prid_path, header=None, encoding="latin1")
    assert df.shape == (1, 25)  # 1 generator row, name + 24 hours
