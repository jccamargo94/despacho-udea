from datetime import date

from app.schemas.input_pack import InputPack, InputSource


def test_input_pack_defaults():
    p = InputPack(dispatch_date=date(2024, 4, 18), source=InputSource.historical, data_dir="data")
    assert p.checksum is None
    assert p.downloaded_at is None


def test_input_pack_serializes_source_as_string():
    p = InputPack(dispatch_date=date(2024, 4, 18), source="live", data_dir="data")
    assert p.source == InputSource.live
    assert p.dict()["source"] == "live"
