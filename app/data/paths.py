"""Input-file path resolution across the two data-layout conventions.

Historical/offline layout uses organized directories (oferta_inicial/,
condicion_inicial/{date}/, ...). The live single-date layout puts the freshly
downloaded files under data/{date}/. The resolver tries the organized location
first, then falls back to the per-date download folder, so both work.
"""

from datetime import date
from pathlib import Path

# Per file kind: ordered list of candidate subdirectories (relative to data_dir).
# "{date}" is substituted with the dispatch date. "" means data_dir itself.
CANDIDATE_SUBDIRS = {
    "OFEI": ["oferta_inicial", "{date}"],
    "dCondIniP": ["condicion_inicial/{date}", "{date}"],
    "dCondIniU": ["condicion_inicial/{date}", "{date}"],
    "PrId": ["predespacho_ideal", "{date}"],
    "iMAR": ["predespacho_ideal", "{date}"],
}


def _filename(kind: str, dispatch_date: date) -> str:
    mmdd = f"{dispatch_date.month:0>2}{dispatch_date.day:0>2}"
    complement = "_NAL" if kind in {"PrId", "iMAR"} else ""
    return f"{kind}{mmdd}{complement}.txt"


def resolve_input(kind: str, dispatch_date: date, data_dir: str = "data") -> str:
    """Return the first existing path for `kind` on `dispatch_date`.

    Raises FileNotFoundError listing every candidate that was tried.
    """
    filename = _filename(kind, dispatch_date)
    tried = []
    for sub in CANDIDATE_SUBDIRS[kind]:
        sub = sub.format(date=dispatch_date)
        p = Path(data_dir) / sub / filename
        tried.append(str(p))
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Could not find {kind} file for {dispatch_date}. Tried: {tried}"
    )
