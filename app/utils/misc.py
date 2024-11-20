from datetime import date
from pathlib import Path
import os

import requests

# file_name = "dCondIniP"
# file_name = "dCondIniU"
# file_name = "OFEI"
# file_name = "PrId"
# file_name = "iMAR"
# file_date = date(2024,2,17)


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

XM_DOWNLOAD_URL = "https://app-portalxmcore01.azurewebsites.net/administracion-archivos/ficheros/descarga-archivo"


def save_file(file_type: str, file_date: date) -> None:
    init_path = PARAMS[file_type]["initial_path"]
    path = os.path.join(init_path, f"{file_date.year}-{file_date.month:0>2}")
    complement = "_NAL" if file_type in {"PrId", "iMAR"} else ""
    filename_ = f"{file_type}{file_date.month:0>2}{file_date.day:0>2}{complement}"

    # Fetch to get file name in bucket

    container_name: str = ("storageportalxm",)
    ordenarPor: str = ("nombre",)
    orden: str = ("DESC",)
    pagina: int = (1,)
    resultadosPorPagina: int = (10,)
    response = requests.get(
        url="https://app-portalxmcore01.azurewebsites.net/administracion-archivos/ficheros",
        params={
            "nombre": f"{filename_}.txt",
            "ruta": f"/{path}",
            "contenedor": container_name,
            "ordenarPor": ordenarPor,
            "orden": orden,
            "pagina": pagina,
            "resultadosPorPagina": resultadosPorPagina,
        },
    )
    filename = response.json()["ficheros"][0]["nombre"]
    # Fetch url to download
    print(f"...DOwnloading file {filename}")
    r = requests.get(
        XM_DOWNLOAD_URL,
        params={
            "ruta": f"{path}/{filename}",
            "fileName": filename,
        },
    )
    url = r.json()["url"]
    file_byte = requests.get(url).content
    output_path = Path("data").joinpath(f"{file_date}", f"{filename_}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path.as_posix(), "w") as file:
        file.write(file_byte.decode("latin-1"))
