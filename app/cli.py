"""Typer CLI for the dispatch model.

    python -m app run 2024-04-18 -t preideal
    python -m app run 2024-04-18:2024-04-30 -t all
    python -m app run 2024-04 --no-eval
"""

import calendar
from datetime import date, datetime, timedelta

import typer

from app.schemas import DispatchCase, DispatchLevel
from app.dates import parse_dates_arg
from app.data.download import ensure_data_for_date
from app.pipeline.evaluate import evaluate_saved_run
from app.pipeline.runner import run_many
from app.pipeline.scenarios import load_bess_scenario
from app.storage import get_storage

app = typer.Typer(add_completion=False, help="Colombian dispatch model runner.")


@app.callback()
def _main():
    """Colombian dispatch model runner."""


def _available_dates(data_dir: str) -> list[date]:
    storage = get_storage(data_dir)
    out: list[date] = []
    for name in storage.list_dir("condicion_inicial"):
        parts = name.split("-")
        if len(parts) != 3:
            continue
        try:
            y, m, d = (int(x) for x in parts)
        except ValueError:
            continue
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
        ["preideal"], "--type", "-t", help="dispatch level (preideal/ideal), repeatable, or 'all'"
    ),
    solver: str = typer.Option("cbc", help="pyomo solver name"),
    eval: bool = typer.Option(True, "--eval/--no-eval", help="evaluate vs XM actuals"),
    prices: bool = typer.Option(
        True, "--prices/--no-prices", help="fix-integers LP pricing re-solve"
    ),
    bess_scenario: str = typer.Option(
        None, "--bess-scenario", help="named scenario (scenarios/bess/<name>.yaml) or path to a scenario YAML"
    ),
    skip_dates: str = typer.Option("", help="comma-separated YYYY-MM-DD to skip"),
    out: str = typer.Option("data/results", help="results directory"),
    data_dir: str = typer.Option("data", help="input data directory"),
):
    avail = _available_dates(data_dir)
    selected = parse_dates_arg(dates, avail)
    skip = _parse_skip(skip_dates)
    selected = [d for d in selected if d not in skip]

    scenario = load_bess_scenario(bess_scenario) if bess_scenario else None

    levels = list(DispatchLevel) if "all" in type else [DispatchLevel(t) for t in type]
    cases = [
        DispatchCase(
            dispatch_date=d, level=lvl, solver=solver, compute_prices=prices,
            bess_scenario=scenario,
        )
        for d in selected
        for lvl in levels
    ]

    if not cases:
        typer.echo("No dates selected.")
        raise typer.Exit(code=1)

    typer.echo(
        f"Running {len(selected)} date(s) x {len(levels)} level(s) with solver={solver}"
    )
    results = run_many(cases, evaluate=eval, out=out, data_dir=data_dir)
    failed = [r for r in results if not r.ok]
    typer.echo(f"\nDone: {len(results) - len(failed)} ok, {len(failed)} failed.")
    for r in failed:
        typer.echo(f"  FAIL {r.case.dispatch_date} [{r.case.level.value}]: {r.error}")
    raise typer.Exit(code=1 if failed else 0)


def _enumerate_dates(token: str) -> list[date]:
    """Enumerate every date in `token`, with no filtering against what's
    already on disk (unlike parse_dates_arg) — fetch's whole point is to
    reach dates that aren't available locally yet."""
    token = token.strip()
    if ":" in token:
        lo, hi = (datetime.strptime(p.strip(), "%Y-%m-%d").date() for p in token.split(":", 1))
        return [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
    parts = token.split("-")
    if len(parts) == 2:
        year, month = int(parts[0]), int(parts[1])
        days_in_month = calendar.monthrange(year, month)[1]
        return [date(year, month, d) for d in range(1, days_in_month + 1)]
    return [datetime.strptime(token, "%Y-%m-%d").date()]


@app.command()
def fetch(
    dates: str = typer.Argument(..., help="YYYY-MM-DD | range a:b | YYYY-MM"),
    data_dir: str = typer.Option("data", help="input data directory"),
):
    """Download raw XM inputs for the given date(s) without running the model."""
    selected = _enumerate_dates(dates)
    for d in selected:
        typer.echo(f"==> fetching {d}")
        ensure_data_for_date(d, data_dir=data_dir)
    typer.echo(f"Done: fetched {len(selected)} date(s).")


@app.command()
def evaluate(
    dates: str = typer.Argument(
        None, help="YYYY-MM-DD | range a:b | YYYY-MM | omit = all available"
    ),
    type: list[str] = typer.Option(
        ["preideal"], "--type", "-t", help="dispatch level, repeatable, or 'all'"
    ),
    out: str = typer.Option("data/results", help="results directory"),
    data_dir: str = typer.Option("data", help="input data directory"),
):
    """Re-score saved runs against XM actuals without re-solving the model."""
    avail = _available_dates(data_dir)
    selected = parse_dates_arg(dates, avail)
    levels = list(DispatchLevel) if "all" in type else [DispatchLevel(t) for t in type]

    evaluated = 0
    for d in selected:
        for lvl in levels:
            try:
                evaluate_saved_run(d, lvl, out=out, data_dir=data_dir)
            except FileNotFoundError as e:
                typer.echo(f"  ! {d} [{lvl.value}]: {e}")
                continue
            typer.echo(f"==> evaluated {d} [{lvl.value}]")
            evaluated += 1

    if evaluated == 0:
        typer.echo("No runs evaluated.")
        raise typer.Exit(code=1)
    typer.echo(f"Done: {evaluated} run(s) evaluated.")
