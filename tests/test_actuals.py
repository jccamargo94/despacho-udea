from datetime import date

from app.data.actuals import load_actual_price


def test_load_actual_price(tmp_path):
    (tmp_path / "preideal_price").mkdir()
    # one label column + 24 hourly prices
    row = "MPO," + ",".join(str(float(i)) for i in range(24))
    (tmp_path / "preideal_price" / "2024-04-18.txt").write_text(row + "\n")
    vals = load_actual_price(date(2024, 4, 18), data_dir=str(tmp_path))
    assert len(vals) == 24
    assert vals[0] == 0.0 and vals[23] == 23.0
