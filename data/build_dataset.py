"""Build the hourly Dutch grid dataset used by the optimizer.

Reads the raw 15-minute snapshots in data/raw/ (fetched from the Energy-Charts API by
fetch_data.py), packs them into one file, data/nl_grid_15min.csv, and writes
data/nl_grid_hourly.csv with one row per local hour:

    timestamp_local, timestamp_utc, co2_g_per_kwh, price_eur_per_mwh

Hourly values are the mean of the sub-hourly values in that hour (all must be present).
Local time is Dutch CET/CEST (see nl_time.py).
Hours where either series is missing are dropped.

If data/raw/ is absent (e.g. the published project, which keeps only the packed
file to stay within the platform's file-count limit), the 15-minute values are
read from data/nl_grid_15min.csv instead; the hourly output is identical.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from nl_time import local_datetime

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "nl_grid_hourly.csv"
PACKED = HERE / "nl_grid_15min.csv"   # every raw 15-minute value, one row per timestamp


def load_series(pattern):
    """Merge all raw files matching pattern into {unix_seconds: (value, step_s)}."""
    series = {}
    for path in sorted(RAW.glob(pattern)):
        d = json.loads(path.read_text())
        for i, v in enumerate(d["values"]):
            if v is not None:
                series[d["first_unix"] + i * d["step_s"]] = (float(v), d["step_s"])
    return series


def to_hourly(series):
    """Mean per full hour: {hour_start_unix: value}; hours with missing sub-hourly values are dropped."""
    buckets = {}
    for t, (v, step) in series.items():
        buckets.setdefault(t - t % 3600, []).append((v, step))
    return {h: sum(v for v, _ in vs) / len(vs) for h, vs in buckets.items()
            if len(vs) == 3600 // vs[0][1]}


def write_packed(co2, price):
    """All raw 15-minute values in one CSV (exact floats, empty where a series has no value)."""
    with PACKED.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_utc", "step_s", "co2_g_per_kwh", "price_eur_per_mwh"])
        for t in sorted(set(co2) | set(price)):
            step = (co2.get(t) or price.get(t))[1]
            w.writerow([
                datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                step,
                repr(co2[t][0]) if t in co2 else "",
                repr(price[t][0]) if t in price else "",
            ])


def read_packed():
    """Read the packed 15-minute CSV back into the {unix_seconds: (value, step_s)} dicts."""
    co2, price = {}, {}
    with PACKED.open(newline="") as f:
        for row in csv.DictReader(f):
            t = int(datetime.strptime(row["timestamp_utc"], "%Y-%m-%dT%H:%MZ")
                    .replace(tzinfo=timezone.utc).timestamp())
            step = int(row["step_s"])
            if row["co2_g_per_kwh"]:
                co2[t] = (float(row["co2_g_per_kwh"]), step)
            if row["price_eur_per_mwh"]:
                price[t] = (float(row["price_eur_per_mwh"]), step)
    return co2, price


def main():
    """Rebuild data/nl_grid_hourly.csv from data/raw/ (or from the packed file if raw/ is absent)."""
    if any(RAW.glob("*.json")):
        co2_raw, price_raw = load_series("co2eq_*.json"), load_series("price_*.json")
        write_packed(co2_raw, price_raw)
        print(f"packed {len(co2_raw) + len(price_raw)} raw values into {PACKED.name}")
    else:
        co2_raw, price_raw = read_packed()
        print(f"data/raw/ not present; read {PACKED.name}")
    co2 = to_hourly(co2_raw)
    price = to_hourly(price_raw)
    hours = sorted(set(co2) & set(price))
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_local", "timestamp_utc", "co2_g_per_kwh", "price_eur_per_mwh"])
        for h in hours:
            w.writerow([
                local_datetime(h).strftime("%Y-%m-%d %H:%M"),
                datetime.fromtimestamp(h, timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                round(co2[h], 1),
                round(price[h], 2),
            ])
    print(f"wrote {len(hours)} hourly rows to {OUT.name}")


if __name__ == "__main__":
    main()
