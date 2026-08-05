import json
from datetime import date

from app.data.download import ensure_data_for_date, save_file
from app.storage import LocalStorage


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.content = payload if isinstance(payload, bytes) else b""

    def json(self):
        return self._payload


def test_save_file_writes_via_storage(monkeypatch, tmp_path):
    calls = iter([
        _FakeResponse({"ficheros": [{"nombre": "OFEI0418.txt"}]}),
        _FakeResponse({"url": "https://example.invalid/OFEI0418.txt"}),
        _FakeResponse(b"file-contents"),
    ])
    monkeypatch.setattr(
        "app.data.download.requests.get", lambda *a, **k: next(calls)
    )
    storage = LocalStorage(str(tmp_path))
    save_file(file_type="OFEI", file_date=date(2024, 4, 18), storage=storage)
    assert (tmp_path / "2024-04-18" / "OFEI0418.txt").read_text() == "file-contents"


def test_ensure_data_for_date_skips_when_folder_exists(monkeypatch, tmp_path):
    (tmp_path / "2024-04-18").mkdir()
    (tmp_path / "2024-04-18" / "marker.txt").write_text("already here")

    def _boom(*a, **k):
        raise AssertionError("save_file should not be called when folder exists")

    monkeypatch.setattr("app.data.download.save_file", _boom)
    folder = ensure_data_for_date(date(2024, 4, 18), data_dir=str(tmp_path))
    assert folder == tmp_path / "2024-04-18"
