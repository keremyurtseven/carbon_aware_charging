# Data

## What's here

| File | Contents |
|---|---|
| `nl_grid_hourly.csv` | **The dataset the optimizer uses.** 8,760 hourly rows for the Dutch grid, 2025-10-10 to 2026-10-09 (one full year): carbon intensity (g CO2eq/kWh) and day-ahead price (EUR/MWh). Times in Dutch local time (CET or CEST) and UTC. |
| `nl_grid_15min.csv` | Every original 15-minute value exactly as returned by the API (70,080 values, exact floats), packed into one file. Sources: `https://api.energy-charts.info/co2eq?country=nl` and `https://api.energy-charts.info/price?bzn=NL`. |
| `raw/*.json` | Local only (not in the published project, which allows at most 100 files): the same values as one JSON file per series per day, as written by `fetch_data.py`. |
| `build_dataset.py` | Turns `raw/` (if present) or `nl_grid_15min.csv` into `nl_grid_hourly.csv` (hourly mean of the four 15-minute values); both paths give a byte-identical file. When `raw/` is present it also rewrites `nl_grid_15min.csv`. |
| `fetch_data.py` | Downloads any other day from the API into `raw/`, so anyone can extend or refresh the data. |
| `nl_time.py` | Dutch local time (CET/CEST) from the EU daylight-saving rule, standard library only (no `tzdata` needed, which Windows Python does not ship). Used by `fetch_data.py` and `build_dataset.py`. |

## Coverage

The dataset covers **2025-10-10 to 2026-10-09**: 365 days x 24 hours = 8,760 hourly rows, no gaps (checked: no missing values in any column). Each row is the mean of four 15-minute API values.

| Quantity | Minimum | Maximum |
|---|---|---|
| Carbon intensity (g CO2eq/kWh) | 92.3 | 748.1 |
| Day-ahead price (EUR/MWh) | -498.44 | 799.36 |

Negative prices occur in the data and are kept as they are.

**Nights.** A "night" is 20:00 on day D to 08:00 on day D+1. `list_nights` in `carbon_ev/data.py` returns the nights with complete 12-hour coverage. That gives **362 nights**. The year contains 2 clock-change nights, which are excluded because they do not have 12 local hours: the night of the autumn change (Oct 25 to 26, 2025) has 13 local hours and the night of the spring change (Mar 28 to 29, 2026) has 11. The last day (Oct 9, 2026) has no following morning in the data, so no night starts on it. So 364 nights start inside the year, and 364 - 2 = 362 remain.

**Daylight-saving handling.** Dutch local time is UTC+1 (CET) in winter and UTC+2 (CEST) from the last Sunday of March 01:00 UTC to the last Sunday of October 01:00 UTC. `nl_time.py` implements exactly this rule, so local timestamps in the CSV are correct across both changes, and the two nights above are recognised by their length rather than by guessing.

**Windy and calm nights.** Windy nights have a large carbon swing within the night. The README defines the threshold (carbon swing of at least 1.3, highest divided by lowest 2-hour block) and reports both kinds. The calm nights are part of every result: they show how the savings change when the carbon signal is flat.

## The main showcase night

The main scenario is an **overnight charging window from Oct 8, 20:00 to Oct 9, 08:00, split into 6 blocks of 2 h**. Carbon intensity falls from about 470 g/kWh at 23:00 to about 150 g/kWh at 03:00–04:00. That is a roughly 3× swing in carbon intensity within one night. Price and carbon mostly move together but not always: 23:00 is cheaper than 19:00–22:00 yet is the dirtiest hour. That mismatch gives the cost-versus-carbon trade-off its meaning.

## Provenance (read this)

No values were invented or modelled. Every number in `raw/` was returned by the Energy-Charts API through `fetch_data.py`, which writes one file per series per day and only saves a day if it has all 96 quarter-hour values.

History, for transparency. The first snapshot (Oct 1, 2, 7, 8 and 9, 2026) was retrieved through an AI assistant's web-fetch tool, which hands the response to a summarizing model. To remove that step, the team re-downloaded those days directly with `fetch_data.py` and compared: **960 of 960 values were identical** to the first snapshot. The three redundant multi-day snapshot files were then deleted, so `raw/` now contains only files written by `fetch_data.py`. The remaining days of the year were fetched with the same script (335 of 335 new days, both series, all 96 quarter-hours per day). `data/verify_snapshot.py` is an additional independent check against the API; it has no retry logic and may hit HTTP 429 (rate limit) on a long run, which is why we used `fetch_data.py` (it respects `Retry-After`) for the comparison.

`nl_grid_hourly.csv` is derived (hourly mean of four quarter-hours) by `build_dataset.py`, not edited by hand.

## Sources and license

All data comes from the **Energy-Charts API** (Fraunhofer ISE), https://api.energy-charts.info/.

- **Carbon intensity:** `/co2eq?country=nl`. The API's own specification states that its data is licensed **CC BY 4.0** unless stated otherwise, and that Energy-Charts.info must be credited as the source. This endpoint states nothing otherwise.
- **Day-ahead prices:** `/price?bzn=NL`. Each response carries the license field *CC BY 4.0 from Bundesnetzagentur | SMARD.de*. Note: the API restricts prices for some other bidding zones to private and internal use. The NL zone is not one of them.

**Required attribution** (keep this in the project README):

> Grid carbon intensity and day-ahead prices: Energy-Charts.info (Fraunhofer ISE), CC BY 4.0. Price data: Bundesnetzagentur | SMARD.de, CC BY 4.0.

**MIT vs CC BY.** The hackathon requires code and documentation under MIT. The files `nl_grid_15min.csv`, `nl_grid_hourly.csv` and `data/raw/` are third-party data and stay under CC BY 4.0. Say so in the README. CC BY 4.0 allows redistribution and modification with attribution, so bundling the data is fine.

Fetched on 2026-10-10. Values for recent days can be revised slightly by the source; `nl_grid_15min.csv` holds the exact values this project used.
