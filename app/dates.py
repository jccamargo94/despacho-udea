from datetime import date, datetime


def _d(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def parse_dates_arg(token: str | None, available: list[date]) -> list[date]:
    """Resolve a CLI date token to a sorted list of dates, filtered to `available`.

    Forms:
      'YYYY-MM-DD'            -> single date (returned even if not in available)
      'YYYY-MM-DD:YYYY-MM-DD' -> inclusive range, intersected with available
      'YYYY-MM'               -> whole month, intersected with available
      None                    -> all available
    """
    avail = sorted(available)
    if token is None:
        return avail
    token = token.strip()
    if ":" in token:
        lo, hi = (_d(p) for p in token.split(":", 1))
        return [d for d in avail if lo <= d <= hi]
    parts = token.split("-")
    if len(parts) == 2:
        year, month = int(parts[0]), int(parts[1])
        return [d for d in avail if d.year == year and d.month == month]
    if len(parts) == 3:
        return [_d(token)]
    raise ValueError(f"Unrecognized date token: {token!r}")
