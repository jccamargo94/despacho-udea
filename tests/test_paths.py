from datetime import date

import pytest

from app.data.paths import resolve_input


def test_prefers_offline_then_falls_back(tmp_path):
    d = date(2024, 4, 18)
    # live per-date location only
    live = tmp_path / "2024-04-18"
    live.mkdir()
    (live / "OFEI0418.txt").write_text("x")
    assert resolve_input("OFEI", d, str(tmp_path)) == str(live / "OFEI0418.txt")

    # organized location is preferred when present
    offline = tmp_path / "oferta_inicial"
    offline.mkdir()
    (offline / "OFEI0418.txt").write_text("x")
    assert resolve_input("OFEI", d, str(tmp_path)) == str(offline / "OFEI0418.txt")


def test_condicion_inicial_layout(tmp_path):
    d = date(2024, 4, 18)
    ci = tmp_path / "condicion_inicial" / "2024-04-18"
    ci.mkdir(parents=True)
    (ci / "dCondIniP0418.txt").write_text("x")
    assert resolve_input("dCondIniP", d, str(tmp_path)) == str(ci / "dCondIniP0418.txt")


def test_prid_nal_suffix(tmp_path):
    d = date(2024, 4, 18)
    live = tmp_path / "2024-04-18"
    live.mkdir()
    (live / "PrId0418_NAL.txt").write_text("x")
    assert resolve_input("PrId", d, str(tmp_path)).endswith("PrId0418_NAL.txt")


def test_missing_lists_tried_paths(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        resolve_input("OFEI", date(2024, 4, 18), str(tmp_path))
    assert "oferta_inicial" in str(e.value)
