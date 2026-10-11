"""Download raw Dutch grid data from the Energy-Charts API (run on your own machine).

Usage:
    python data/fetch_data.py 2026-09-10 2026-10-09     # inclusive date range
    python data/build_dataset.py

Fetches 15-minute grid carbon intensity (country=nl) and day-ahead prices
(bidding zone NL) and saves one file per series per day in data/raw/, in the
same format as the bundled snapshots. Standard library only.

Robustness:
- Requests go out in multi-day chunks. If a chunk fails, it is retried with
  smaller chunks (some single-day requests fail on the server while a request
  covering the same day plus a neighbour works).
- HTTP 429 (rate limit) is respected via the Retry-After header.
- A day is only saved if it has all 96 quarter-hour values.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from nl_time import local_midnight_unix

API = "https://api.energy-charts.info"
RAW = Path(__file__).resolve().parent / "raw"
SERIES = {
    "co2eq": ("co2eq?country=nl", "co2eq", "g CO2eq / kWh"),
    "price": ("price?bzn=NL", "price", "EUR / MWh"),
}


def get(url, tries=4):
    """GET a JSON document; retries with backoff and honours HTTP 429 Retry-After."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", 30))
                print(f"  rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(2 * (attempt + 1))


def fetch_range(series, start, end):
    """Return {unix_seconds: value} for start..end inclusive, splitting on failure."""
    path, key, _ = SERIES[series]
    url = f"{API}/{path}&start={start}&end={end}"
    try:
        d = get(url)
        out = {t: v for t, v in zip(d["unix_seconds"], d[key]) if v is not None}
        lic = d.get("license_info") or d.get("license")
        return out, lic
    except Exception as e:  # noqa: BLE001
        n = (end - start).days
        if n == 0:
            # a lone day failing: retry it as part of a two-day window
            print(f"  {series} {start} failed alone ({e}); retrying as {start}..{start + timedelta(days=1)}")
            try:
                d = get(f"{API}/{path}&start={start}&end={start + timedelta(days=1)}")
                return {t: v for t, v in zip(d["unix_seconds"], d[key]) if v is not None}, \
                       d.get("license_info") or d.get("license")
            except Exception as e2:  # noqa: BLE001
                print(f"  {series} {start} still failing ({e2}); skipped")
                return {}, None
        mid = start + timedelta(days=n // 2)
        a, la = fetch_range(series, start, mid)
        b, lb = fetch_range(series, mid + timedelta(days=1), end)
        a.update(b)
        return a, la or lb


def save_days(series, values, lic, start, end):
    """Save one file per local day. A day has 92, 96 or 100 quarter-hours (clock changes)."""
    _, _, unit = SERIES[series]
    times = sorted(values)
    step = min(b - a for a, b in zip(times, times[1:])) if len(times) > 1 else 900
    day = start
    saved = 0
    while day <= end:
        t0 = local_midnight_unix(day)
        t1 = local_midnight_unix(day + timedelta(days=1))
        vals = [values.get(t) for t in range(t0, t1, step)]
        if vals and all(v is not None for v in vals):
            (RAW / f"{series}_nl_{day}.json").write_text(json.dumps({
                "source": f"{API}/{SERIES[series][0]}",
                "unit": unit,
                "license": lic or "CC BY 4.0, Energy-Charts.info (API default)",
                "first_unix": t0,
                "step_s": step,
                "values": vals,
            }))
            saved += 1
        else:
            missing = sum(v is None for v in vals)
            print(f"  {series} {day}: {missing}/{len(vals)} values missing, not saved")
        day += timedelta(days=1)
    return saved


def main(argv):
    """Fetch both series for the inclusive date range in argv (default 2026-09-10 .. 2026-10-09)."""
    start = date.fromisoformat(argv[0]) if argv else date(2026, 9, 10)
    end = date.fromisoformat(argv[1]) if len(argv) > 1 else date(2026, 10, 9)
    RAW.mkdir(exist_ok=True)
    days = (end - start).days + 1
    for series in SERIES:
        values = {}
        lic = None
        chunk = start
        while chunk <= end:
            chunk_end = min(chunk + timedelta(days=6), end)
            print(f"{series}: {chunk}..{chunk_end}")
            v, l = fetch_range(series, chunk, chunk_end)
            values.update(v)
            lic = lic or l
            chunk = chunk_end + timedelta(days=1)
            time.sleep(1)
        n = save_days(series, values, lic, start, end)
        print(f"{series}: saved {n}/{days} complete days")


if __name__ == "__main__":
    main(sys.argv[1:])
