import pandas as pd

from app.data.loaders import load_precio_bolsa


def test_precio_bolsa_scaled(tmp_path):
    (tmp_path / "precio_bolsa").mkdir()
    pd.DataFrame(
        {"datetime": ["2024-04-18 00:00"], "precio_bolsa": [0.1]}
    ).to_csv(tmp_path / "precio_bolsa" / "precio_bolsa_2024.csv", index=False)
    out = load_precio_bolsa(str(tmp_path))
    assert abs(out["precio_bolsa"].iloc[0] - 100.0) < 1e-9
