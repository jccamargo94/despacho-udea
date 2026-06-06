"""Typer CLI for the dispatch model.

    python -m app run 2024-04-18 -t preideal
    python -m app run 2024-04-18:2024-04-30 -t all
    python -m app run 2024-04 --no-eval
"""

from datetime import date, datetime
from pathlib import Path

import typer

from app.model import DispatchConfig, DispatchOptions
from app.dates import parse_dates_arg
from app.pipeline.runner import run_many

app = typer.Typer(add_completion=False, help="Colombian dispatch model runner.")


@app.callback()
def _main():
    """Colombian dispatch model runner."""


def _available_dates(data_dir: str) -> list[date]:
    root = Path(data_dir) / "condicion_inicial"
    out: list[date] = []
    if root.exists():
        for f in root.glob("*"):
            if f.is_dir():
                y, m, d = (int(x) for x in f.stem.split("-"))
                out.append(date(y, m, d))
    return out


def _parse_skip(skip_dates: str) -> set[date]:
    return {
        datetime.strptime(tok.strip(), "%Y-%m-%d").date()
        for tok in skip_dates.split(",")
        if tok.strip()
    }


@app.command()
def run(
    dates: str = typer.Argument(
        None, help="YYYY-MM-DD | range a:b | YYYY-MM | omit = all available"
    ),
    type: list[str] = typer.Option(
        ["preideal"], "--type", "-t", help="dispatch type, repeatable, or 'all'"
    ),
    solver: str = typer.Option("cbc", help="pyomo solver name"),
    eval: bool = typer.Option(True, "--eval/--no-eval", help="evaluate vs XM actuals"),
    prices: bool = typer.Option(
        True, "--prices/--no-prices", help="fix-integers LP pricing re-solve"
    ),
    skip_dates: str = typer.Option("", help="comma-separated YYYY-MM-DD to skip"),
    out: str = typer.Option("data/results", help="results directory"),
    data_dir: str = typer.Option("data", help="input data directory"),
):
    avail = _available_dates(data_dir)
    selected = parse_dates_arg(dates, avail)
    skip = _parse_skip(skip_dates)
    selected = [d for d in selected if d not in skip]

    types = DispatchOptions._member_names_ if "all" in type else type
    configs = [DispatchConfig(dispatch_type=t) for t in types]

    if not selected:
        typer.echo("No dates selected.")
        raise typer.Exit(code=1)

    typer.echo(
        f"Running {len(selected)} date(s) x {len(configs)} type(s) with solver={solver}"
    )
    results = run_many(
        selected,
        configs,
        solver=solver,
        compute_prices=prices,
        evaluate=eval,
        out=out,
        data_dir=data_dir,
    )
    failed = [r for r in results if not r.ok]
    typer.echo(f"\nDone: {len(results) - len(failed)} ok, {len(failed)} failed.")
    for r in failed:
        typer.echo(f"  FAIL {r.dispatch_date} [{r.dispatch_type}]: {r.error}")
    raise typer.Exit(code=1 if failed else 0)
