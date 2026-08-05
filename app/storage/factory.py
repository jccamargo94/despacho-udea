from app.storage.base import Storage
from app.storage.local import LocalStorage


def get_storage(root: str) -> Storage:
    if root.startswith("gs://"):
        raise NotImplementedError("GCS backend not implemented yet")
    return LocalStorage(root)
