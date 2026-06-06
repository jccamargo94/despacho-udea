"""Legacy batch runner: run every available date x dispatch type.

The duplicated data-load + case-build body is gone; this now delegates to
app.pipeline.runner. Equivalent to `python -m app run -t all` with a skip list.
Kept as a script for backwards compatibility.
"""

from datetime import date
from pathlib import Path
import os

from app.model import DispatchConfig, DispatchOptions
from app.pipeline.runner import run_many


SKIP_DATES = [
    # Fechas malas
    date(2024, 10, 3),
    date(2024, 10, 9),
    date(2024, 10, 30),
    date(2024, 10, 17),
    date(2024, 2, 15),
    # Ya ejecutadas
    date(2024, 3, 23),
    date(2024, 3, 29),
    date(2024, 4, 22),
    date(2024, 4, 25),
    date(2024, 4, 14),
    date(2024, 4, 18),
    date(2024, 4, 19),
    date(2024, 5, 25),
    date(2024, 6, 9),
    date(2024, 7, 2),
    date(2024, 8, 10),
    date(2024, 8, 29),
]


def discover_dates(data_dir: str = "data") -> list[date]:
    path = Path(os.path.join(data_dir, "condicion_inicial"))
    dates_ = []
    for folder in path.glob("*"):
        if folder.is_dir():
            y, m, d = (int(x) for x in folder.stem.split("-"))
            dates_.append(date(y, m, d))
    return sorted(dates_)


def main():
    dates_ = [d for d in discover_dates() if d not in SKIP_DATES]
    configs = [DispatchConfig(dispatch_type=t) for t in DispatchOptions._member_names_]
    results = run_many(dates_, configs)
    failed = [r for r in results if not r.ok]
    print(f"\nDone: {len(results) - len(failed)} ok, {len(failed)} failed.")
    for r in failed:
        print(f"  FAIL {r.dispatch_date} [{r.dispatch_type}]: {r.error}")


if __name__ == "__main__":
    main()
