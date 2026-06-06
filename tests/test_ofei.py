from datetime import date
from pathlib import Path

from app.data.ofei import parse_ofei

FIX = Path(__file__).parent / "fixtures" / "OFEI_sample.txt"


def test_parses_each_section():
    d = parse_ofei(str(FIX), date(2024, 4, 18))
    # PAP start-up prices
    assert not d.precio_arranque.empty
    assert d.precio_arranque["price"].iloc[0] == 1500000.0
    # minimo operativo reshaped to long form
    assert list(d.minimo_operativo.columns) == [
        "resource",
        "type",
        "hour",
        "minimo_operativo",
        "datetime",
    ]
    assert len(d.minimo_operativo) == 24  # one MO resource x 24 hours
    # combined cycle availability: 24 hourly values
    assert d.cc == {"TERMOCC": ["TERMOCC_1"]}
    assert d.cc_price == {"TERMOCC_1": 800000.0}
    assert all(len(v) == 24 for v in d.cc_dispo.values())
    # bid prices scaled by 1e-3
    assert d.prices == {"GEN1": 45.0}
