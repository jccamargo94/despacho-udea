from datetime import date

import pytest

from app.dates import parse_dates_arg

AVAIL = [date(2024, 4, 1), date(2024, 4, 18), date(2024, 4, 30), date(2024, 5, 2)]


def test_single():
    assert parse_dates_arg("2024-04-18", AVAIL) == [date(2024, 4, 18)]


def test_range_inclusive():
    assert parse_dates_arg("2024-04-18:2024-04-30", AVAIL) == [
        date(2024, 4, 18),
        date(2024, 4, 30),
    ]


def test_month():
    assert parse_dates_arg("2024-04", AVAIL) == [
        date(2024, 4, 1),
        date(2024, 4, 18),
        date(2024, 4, 30),
    ]


def test_all_when_none():
    assert parse_dates_arg(None, AVAIL) == sorted(AVAIL)


def test_bad_token():
    with pytest.raises(ValueError):
        parse_dates_arg("not-a-date", AVAIL)
