import os
from datetime import date
from pathlib import Path

import requests

from app.storage import Storage, get_storage

PARAMS = {
    "OFEI": {
        "initial_path": "M:/InformacionAgentes/Usuarios/Publico/OFERTAS/INICIAL",
    },
    "dCondIniU": {
        "initial_path": "M:/InformacionAgentes/Usuarios/Publico/DESPACHO",
    },
    "dCondIniP": {
        "initial_path": "M:/InformacionAgentes/Usuarios/Publico/DESPACHO",
    },
    "PrId": {
        "initial_path": "M:/InformacionAgentes/Usuarios/Publico/PredespachoIdeal",
    },
    "iMAR": {
        "initial_path": "M:/InformacionAgentes/Usuarios/Publico/PredespachoIdeal",
    },
}

XM_DOWNLOAD_URL = "https://api-portalxm.xm.com.co/administracion-archivos/ficheros/descarga-archivo"
XM_BLOB_CONTAINER = "storageportalxm"


def save_file(file_type: str, file_date: date, storage: Storage) -> None:
    init_path = PARAMS[file_type]["initial_path"]
    path = os.path.join(init_path, f"{file_date.year}-{file_date.month:0>2}")
    complement = "_NAL" if file_type in {"PrId", "iMAR"} else ""
    filename_ = f"{file_type}{file_date.month:0>2}{file_date.day:0>2}{complement}"

    print(f"...Downloading file {filename_}.txt")
    response = requests.get(
        XM_DOWNLOAD_URL,
        params={
            "ruta": f"{path}/{filename_}.txt",
            "nombreBlobContainer": XM_BLOB_CONTAINER,
        },
    )
    with storage.open(f"{file_date}/{filename_}.txt", "w") as file:
        file.write(response.content.decode("utf-8"))


def ensure_data_for_date(dispatch_date: date, data_dir: str = "data") -> Path:
    """Download the per-day XM files into data/{date}/ if the folder is absent."""
    storage = get_storage(data_dir)
    folder_rel = str(dispatch_date)
    if storage.list_dir(folder_rel):
        print("... files already downloaded. Skipping download")
        return Path(data_dir) / folder_rel
    for file_type in PARAMS:
        save_file(file_type=file_type, file_date=dispatch_date, storage=storage)
    return Path(data_dir) / folder_rel
