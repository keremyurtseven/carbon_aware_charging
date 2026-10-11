"""Loading the Dutch hourly grid dataset (carbon intensity + day-ahead price).

The CSV (``data/nl_grid_hourly.csv``) has one row per local hour with columns
``timestamp_local, timestamp_utc, co2_g_per_kwh, price_eur_per_mwh``.
Data licence: CC BY 4.0 (Energy-Charts.info, Fraunhofer ISE; SMARD.de).
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

LOCAL_TZ = "Europe/Amsterdam"
# Written as one string literal on purpose: Qollab loads a data file only if the code names it in a string.
DATA_FILE = "data/nl_grid_hourly.csv"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / DATA_FILE

#: Night window: starts at 20:00 on day D and spans 12 hourly rows (20:00 ... 07:00).
NIGHT_START_HOUR = 20
NIGHT_HOURS = 12


def load_hourly(path: str | Path | None = None) -> pd.DataFrame:
    """Load the hourly dataset.

    Returns a DataFrame with columns ``co2_g_per_kwh`` and ``price_eur_per_mwh``
    indexed by a tz-aware (Europe/Amsterdam) hourly DatetimeIndex, sorted and
    de-duplicated. The index is built from ``timestamp_utc`` so it is
    unambiguous even across a DST change.
    """
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    raw = pd.read_csv(path)
    idx = pd.DatetimeIndex(pd.to_datetime(raw["timestamp_utc"], utc=True)).tz_convert(LOCAL_TZ)
    idx.name = "hour_local"
    df = pd.DataFrame(
        {
            "co2_g_per_kwh": raw["co2_g_per_kwh"].astype(float).to_numpy(),
            "price_eur_per_mwh": raw["price_eur_per_mwh"].astype(float).to_numpy(),
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def night_frame(df: pd.DataFrame, night: date) -> pd.DataFrame | None:
    """Rows of the night window D 20:00 -> D+1 08:00 (12 hours), or None if incomplete.

    Matching is done on local wall-clock time, so a night containing a DST
    change (11 or 13 real hours) is reported as incomplete rather than guessed.
    Rows with missing values also make the night incomplete.
    """
    naive = df.index.tz_localize(None)
    start = pd.Timestamp(night) + pd.Timedelta(hours=NIGHT_START_HOUR)
    wanted = pd.date_range(start, periods=NIGHT_HOURS, freq="h")
    sub = df.loc[naive.isin(wanted)]
    if len(sub) != NIGHT_HOURS:
        return None
    if sub.isna().any().any():
        return None
    if not (sub.index.tz_localize(None) == wanted).all():
        return None
    return sub


def list_nights(df: pd.DataFrame) -> list[date]:
    """Dates D (sorted) such that the whole 20:00 (D) -> 08:00 (D+1) window is present."""
    days = {ts.date() for ts in df.index.tz_localize(None)}
    candidates = days | {d - timedelta(days=1) for d in days}
    return [d for d in sorted(candidates) if night_frame(df, d) is not None]
