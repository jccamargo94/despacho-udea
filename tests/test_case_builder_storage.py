"""Regression test: case_builder.py must read every input file through
Storage, not plain open() — otherwise a GCS-backed data_dir silently
can't find these files. See docs/superpowers/specs/2026-08-06-ingesta-storage-xm-design.md
section 5."""

from datetime import date
from pathlib import Path

from app.pipeline.case_builder import build_case
from app.schemas.case import DispatchCase, DispatchLevel
from app.schemas.input_pack import InputPack, InputSource
from app.storage.local import LocalStorage

DD = str(Path(__file__).parent / "fixtures" / "xm_smoke")
FECHA = date(2024, 4, 18)


def test_build_case_reads_ramps_and_preideal_map_through_storage(monkeypatch):
    calls = []
    original_open = LocalStorage.open

    def spy_open(self, path, mode="r", encoding=None):
        calls.append(path)
        return original_open(self, path, mode, encoding)

    monkeypatch.setattr(LocalStorage, "open", spy_open)

    case = DispatchCase(dispatch_date=FECHA, level=DispatchLevel.preideal, solver="cbc")
    inputs = InputPack(dispatch_date=FECHA, source=InputSource.historical, data_dir=DD)
    build_case(case, inputs, ders=None)

    assert "ramps.json" in calls
    assert "preideal_dispatch_map.json" in calls
