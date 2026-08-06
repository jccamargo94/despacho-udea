from datetime import date

from app.data.download import ensure_data_for_date, save_file
from app.storage import LocalStorage


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content


def test_save_file_writes_via_storage(monkeypatch, tmp_path):
    captured = {}

    def _fake_get(url, params=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse("file-contents-áéí".encode("utf-8"))

    monkeypatch.setattr("app.data.download.requests.get", _fake_get)
    storage = LocalStorage(str(tmp_path))
    save_file(file_type="OFEI", file_date=date(2024, 4, 18), storage=storage)

    assert captured["url"] == (
        "https://api-portalxm.xm.com.co/administracion-archivos/ficheros/descarga-archivo"
    )
    assert captured["params"] == {
        "ruta": "M:/InformacionAgentes/Usuarios/Publico/OFERTAS/INICIAL/2024-04/OFEI0418.txt",
        "nombreBlobContainer": "storageportalxm",
    }
    assert (tmp_path / "2024-04-18" / "OFEI0418.txt").read_text() == "file-contents-áéí"


def test_ensure_data_for_date_skips_when_folder_exists(monkeypatch, tmp_path):
    (tmp_path / "2024-04-18").mkdir()
    (tmp_path / "2024-04-18" / "marker.txt").write_text("already here")

    def _boom(*a, **k):
        raise AssertionError("save_file should not be called when folder exists")

    monkeypatch.setattr("app.data.download.save_file", _boom)
    folder = ensure_data_for_date(date(2024, 4, 18), data_dir=str(tmp_path))
    assert folder == tmp_path / "2024-04-18"
