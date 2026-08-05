from datetime import date

import pytest

from app.storage import LocalStorage, get_storage


def test_exists_false_for_missing_file(tmp_path):
    storage = LocalStorage(str(tmp_path))
    assert storage.exists("missing.txt") is False


def test_open_write_then_read_roundtrip(tmp_path):
    storage = LocalStorage(str(tmp_path))
    with storage.open("a.txt", "w") as f:
        f.write("hello")
    assert storage.exists("a.txt") is True
    with storage.open("a.txt", "r") as f:
        assert f.read() == "hello"


def test_open_write_creates_parent_dirs(tmp_path):
    storage = LocalStorage(str(tmp_path))
    with storage.open("nested/dir/b.txt", "w") as f:
        f.write("x")
    assert (tmp_path / "nested" / "dir" / "b.txt").read_text() == "x"


def test_list_dir_returns_entry_names(tmp_path):
    (tmp_path / "condicion_inicial").mkdir()
    (tmp_path / "condicion_inicial" / "2024-04-18").mkdir()
    (tmp_path / "condicion_inicial" / "2024-04-19").mkdir()
    storage = LocalStorage(str(tmp_path))
    assert sorted(storage.list_dir("condicion_inicial")) == ["2024-04-18", "2024-04-19"]


def test_list_dir_missing_returns_empty(tmp_path):
    storage = LocalStorage(str(tmp_path))
    assert storage.list_dir("does-not-exist") == []


def test_get_storage_returns_local_for_plain_path(tmp_path):
    storage = get_storage(str(tmp_path))
    assert isinstance(storage, LocalStorage)


def test_get_storage_raises_not_implemented_for_gcs():
    with pytest.raises(NotImplementedError):
        get_storage("gs://some-bucket/prefix")
