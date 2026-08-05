from datetime import date

from app.data.actuals import load_actual_dispatch, load_actual_price


def test_load_actual_price(tmp_path):
    (tmp_path / "preideal_price").mkdir()
    # one label column + 24 hourly prices
    row = "MPO," + ",".join(str(float(i)) for i in range(24))
    (tmp_path / "preideal_price" / "2024-04-18.txt").write_text(row + "\n")
    vals = load_actual_price(date(2024, 4, 18), data_dir=str(tmp_path))
    assert len(vals) == 24
    assert vals[0] == 0.0 and vals[23] == 23.0


def test_load_actual_dispatch_decodes_latin1(tmp_path):
    # Real XM plant names contain latin1-only bytes (e.g. GUATAPE with an
    # accented E, 0xC9 in latin1 -- not valid UTF-8). Must not raise, and the
    # accented character must round-trip correctly.
    (tmp_path / "preideal_dispatch").mkdir()
    (tmp_path / "preideal_dispatch" / "2024-04-18.txt").write_bytes(b"GUATAP\xc9,1,2\n")
    df = load_actual_dispatch(date(2024, 4, 18), data_dir=str(tmp_path))
    assert df.iloc[0, 0] == "GUATAPÉ"
