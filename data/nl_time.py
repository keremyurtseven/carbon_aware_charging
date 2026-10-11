"""Dutch local time (CET/CEST) without a time-zone database. Standard library only.

The Netherlands follows the EU rule: summer time (UTC+2) runs from the last
Sunday of March 01:00 UTC to the last Sunday of October 01:00 UTC; otherwise
UTC+1. Using the rule directly keeps the data scripts free of the `tzdata`
package, which Windows Python does not ship.
"""
from datetime import date, datetime, timedelta, timezone


def _last_sunday_0100_utc(year: int, month: int) -> float:
    d = date(year, month + 1, 1) - timedelta(days=1)  # last day of the month (month is 3 or 10)
    d -= timedelta(days=(d.weekday() - 6) % 7)
    return datetime(d.year, d.month, d.day, 1, tzinfo=timezone.utc).timestamp()


def utc_offset_hours(t_unix: float) -> int:
    """UTC offset (1 or 2) of Dutch local time at the instant t_unix."""
    year = datetime.fromtimestamp(t_unix, timezone.utc).year
    return 2 if _last_sunday_0100_utc(year, 3) <= t_unix < _last_sunday_0100_utc(year, 10) else 1


def local_datetime(t_unix: float) -> datetime:
    """Dutch local wall-clock time of t_unix, as a timezone-aware datetime."""
    off = utc_offset_hours(t_unix)
    return datetime.fromtimestamp(t_unix, timezone(timedelta(hours=off)))


def local_midnight_unix(day: date) -> int:
    """Unix seconds of 00:00 Dutch local time on `day` (clock changes happen later in the night)."""
    utc_midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    for off in (2, 1):
        t = utc_midnight - off * 3600
        if utc_offset_hours(t) == off:
            return int(t)
    raise ValueError(f"no consistent local midnight for {day}")
