"""Check every value in data/raw/ against the Energy-Charts API directly.

Run this before fetch_data.py (which overwrites files in data/raw/):

    python data/verify_snapshot.py

It downloads the same days straight from the API and reports any saved value
that differs. Standard library only.
"""
import json
import time
import urllib.request
from datetime import timedelta
from pathlib import Path

from nl_time import local_datetime

API = "https://api.energy-charts.info"
RAW = Path(__file__).resolve().parent / "raw"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def main():
    saved = {"co2eq": {}, "price": {}}
    for p in sorted(RAW.glob("*.json")):
        series = "co2eq" if p.name.startswith("co2eq") else "price"
        d = json.loads(p.read_text())
        for i, v in enumerate(d["values"]):
            saved[series][d["first_unix"] + i * d["step_s"]] = v

    total = bad = 0
    for series, path, key in [("co2eq", "co2eq?country=nl", "co2eq"), ("price", "price?bzn=NL", "price")]:
        days = sorted({local_datetime(t).date() for t in saved[series]})
        # two-day windows: some single-day requests fail on the server
        for day in days:
            d = get(f"{API}/{path}&start={day}&end={day + timedelta(days=1)}")
            api = dict(zip(d["unix_seconds"], d[key]))
            for t, v in saved[series].items():
                if local_datetime(t).date() != day:
                    continue
                total += 1
                if api.get(t) != v:
                    bad += 1
                    print(f"MISMATCH {series} {local_datetime(t)}: saved {v}, API {api.get(t)}")
            time.sleep(1)
    print(f"checked {total} values, {bad} mismatches")
    if bad:
        print("Note: the source can revise recent values slightly; re-fetch with fetch_data.py.")


if __name__ == "__main__":
    main()
