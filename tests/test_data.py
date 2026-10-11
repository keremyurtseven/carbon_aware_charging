from datetime import date

import pandas as pd

from carbon_ev.data import DEFAULT_DATA_PATH, list_nights, load_hourly, night_frame


def test_load_hourly_shape_and_tz():
    df = load_hourly()
    raw = pd.read_csv(DEFAULT_DATA_PATH)
    assert len(df) == len(raw)
    assert df.index.tz is not None
    assert df.index.is_monotonic_increasing
    assert {"co2_g_per_kwh", "price_eur_per_mwh"} <= set(df.columns)
    # local wall-clock time of the index equals the CSV's timestamp_local column
    assert list(df.index.strftime("%Y-%m-%d %H:%M")) == list(raw["timestamp_local"])


def test_list_nights_bundled_snapshot():
    """Data-agnostic: works for the 3-night snapshot and the full 30-day file alike."""
    df = load_hourly()
    nights = list_nights(df)
    assert {date(2026, 10, 1), date(2026, 10, 7), date(2026, 10, 8)} <= set(nights)
    assert nights == sorted(set(nights))  # sorted, no duplicates
    last_day = df.index.max().date()
    assert last_day not in nights  # last data day has no following morning
    for n in nights:
        assert len(night_frame(df, n)) == 12


def test_list_nights_generic_synthetic():
    idx = pd.date_range("2030-03-04 00:00", periods=24 * 3, freq="h", tz="Europe/Amsterdam")
    df = pd.DataFrame({"co2_g_per_kwh": 1.0, "price_eur_per_mwh": 2.0}, index=idx)
    assert list_nights(df) == [date(2030, 3, 4), date(2030, 3, 5)]
    # a hole inside the window removes night Mar 4 only (hole at 2030-03-05 03:00)
    df2 = df.drop(df.index[df.index.hour == 3][1])
    assert list_nights(df2) == [date(2030, 3, 5)]


def test_night_frame_incomplete_is_none():
    df = load_hourly()
    assert night_frame(df, df.index.max().date()) is None  # no following morning
    assert len(night_frame(df, date(2026, 10, 8))) == 12
