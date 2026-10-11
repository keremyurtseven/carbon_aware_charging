"""e7: a Dutch e-bus depot over a full year of real grid data (classical).

Question a depot manager asks: compared with how the depot charges today, how
much CO2 and electricity cost does carbon-aware scheduling save per year, and
how much grid connection does it need? The fleet is modelled on
public figures (``carbon_ev.depot.DEPOT_SOURCES``); only carbon intensity and
prices are measured data.

Three ways to charge the same fleet, every night:
  * unmanaged: every bus charges as soon as it is back (``on_arrival_capacity``
    without a cap); the connection must cover its peak
  * FCFS cap: the same first-come-first-served charging, capped at the smallest
    connection on which it still charges every bus (``min_cap_on_arrival_kw``)
  * smart: the cost/CO2 optimum (``milp_depot``, w = 0.7 and w = 1.0, proven
    gap <= 0.2 %), on the unmanaged connection and on the FCFS-cap connection
plus the smallest connection smart scheduling needs (``min_peak_kw``, 0.5 % gap).
The fleet is the same every night, so the three connection sizes are computed
once. Annual totals are sums over the nights present (also scaled to 365).

Savings in EUR are on the wholesale day-ahead energy price only (no grid fees,
taxes or capacity charges). CO2 uses average, not marginal, intensity. Charging
is modelled in 1-hour blocks at constant power, interruptible, no battery taper.

Usage:
    python experiments/e7_depot_year.py                 # all nights -> results/e7_*
    python experiments/e7_depot_year.py --max-nights 10 --out-dir results/scratch
    python experiments/e7_depot_year.py --replot        # figure + summary from the CSV
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _style as S  # noqa: E402
from carbon_ev.data import load_hourly, list_nights  # noqa: E402
from carbon_ev.depot import (  # noqa: E402
    block_load_kw,
    depot_fleet,
    is_feasible_kw,
    milp_depot,
    min_cap_on_arrival_kw,
    min_peak_kw,
    on_arrival_capacity,
)
from carbon_ev.problem import build_instance, evaluate  # noqa: E402

WINDY_SWING = 1.3  # same threshold as e2 (carbon swing of the 2-h blocks)
BIG = 1e12


def run(args) -> pd.DataFrame:
    df = load_hourly()
    fleet = depot_fleet(n_buses=args.buses, charger_kw=args.charger_kw, seed=args.seed)
    nights = list_nights(df)
    if args.max_nights:
        nights = nights[: args.max_nights]

    # connection sizes: the fleet is identical every night, so these are night-independent
    ref = build_instance(df, nights[0], fleet=fleet, n_blocks=12, p_max_kw=BIG)
    x_ref = on_arrival_capacity(ref)
    cap_unmanaged = float(block_load_kw(ref, x_ref).max())
    cap_fcfs = min_cap_on_arrival_kw(ref)
    cap_smart = min_peak_kw(ref)
    energy_mwh = sum(ev.k_blocks * ev.power_kw for ev in fleet) / 1000.0
    print(f"fleet: {len(fleet)} buses, {energy_mwh:.1f} MWh/night; connection needed: unmanaged "
          f"{cap_unmanaged / 1000:.2f} MW, FCFS cap {cap_fcfs / 1000:.2f} MW, smart {cap_smart / 1000:.2f} MW")

    rows = []
    t_start = time.time()
    for k, night in enumerate(nights):
        blocks2h = build_instance(df, night).co2
        swing = float(blocks2h.max() / blocks2h.min())
        row = {
            "night": night.isoformat(), "month": night.strftime("%Y-%m"),
            "windy": swing >= WINDY_SWING, "carbon_swing": swing, "energy_mwh": energy_mwh,
            "cap_unmanaged_kw": cap_unmanaged, "cap_fcfs_kw": cap_fcfs, "cap_smart_min_kw": cap_smart,
        }
        for tag, cap in (("unmanaged", cap_unmanaged), ("fcfs", cap_fcfs)):
            inst = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=cap)
            base = evaluate(inst, on_arrival_capacity(inst))
            row[f"{tag}_co2_kg"] = base["co2_kg"]
            row[f"{tag}_cost_eur"] = base["cost_eur"]
            for w in (0.7, 1.0):
                inst_w = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=cap, w=w)
                x, gap = milp_depot(inst_w)
                assert is_feasible_kw(inst_w, x)
                ev_ = evaluate(inst_w, x)
                row[f"smart_{tag}cap_w{w}_co2_kg"] = ev_["co2_kg"]
                row[f"smart_{tag}cap_w{w}_cost_eur"] = ev_["cost_eur"]
                row[f"smart_{tag}cap_w{w}_mip_gap"] = gap
        rows.append(row)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(nights)} nights, {time.time() - t_start:.0f} s", flush=True)
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    """Savings of smart vs the baseline on the SAME connection, per group."""
    scale = 365.0 / len(res)
    out = []
    for base in ("unmanaged", "fcfs"):
        for w in (0.7, 1.0):
            sm = f"smart_{base}cap_w{w}"
            co2 = res[f"{base}_co2_kg"] - res[f"{sm}_co2_kg"]
            eur = res[f"{base}_cost_eur"] - res[f"{sm}_cost_eur"]
            for group, mask in (("all", pd.Series(True, index=res.index)), ("windy", res["windy"]),
                                ("calm", ~res["windy"])):
                c, e = co2[mask], eur[mask]
                out.append({
                    "baseline": base, "connection_kw": float(res[f"cap_{base}_kw"].iloc[0]), "w": w,
                    "group": group, "n_nights": int(mask.sum()),
                    "baseline_co2_t_sum": res.loc[mask, f"{base}_co2_kg"].sum() / 1000.0,
                    "baseline_cost_eur_sum": res.loc[mask, f"{base}_cost_eur"].sum(),
                    "co2_saved_t_sum": c.sum() / 1000.0,
                    "eur_saved_sum": e.sum(),
                    "co2_saved_t_per_365": c.sum() / 1000.0 * scale if group == "all" else np.nan,
                    "eur_saved_per_365": e.sum() * scale if group == "all" else np.nan,
                    "co2_saved_pct": 100 * c.sum() / res.loc[mask, f"{base}_co2_kg"].sum(),
                    "eur_saved_pct": 100 * e.sum() / res.loc[mask, f"{base}_cost_eur"].sum(),
                    "nights_more_co2_than_baseline": int((c < -1e-6).sum()),
                    "max_mip_gap": float(res.loc[mask, f"{sm}_mip_gap"].max()),
                })
    return pd.DataFrame(out)


def plot(res: pd.DataFrame, out: Path, summ: pd.DataFrame) -> None:
    """Monthly CO2 and cost savings of smart charging and the connection each strategy needs."""
    S.apply()
    sm = "smart_unmanagedcap_w1.0"  # headline: carbon-first, w = 1
    m = res.groupby("month").agg(
        un_co2=("unmanaged_co2_kg", "sum"), sm_co2=(f"{sm}_co2_kg", "sum"),
        un_eur=("unmanaged_cost_eur", "sum"), sm_eur=(f"{sm}_cost_eur", "sum"),
        nights=("night", "count"),
    )
    co2_saved = (m.un_co2 - m.sm_co2) / 1000.0
    eur_saved = (m.un_eur - m.sm_eur) / 1000.0
    x = np.arange(len(m))
    month_labels = [pd.Timestamp(s + "-01").strftime("%b %y") + ("" if n >= 28 else "*") for s, n in zip(m.index, m.nights)]
    partial = [f"{pd.Timestamp(s + '-01').strftime('%b %y')} ({n} nights)" for s, n in zip(m.index, m.nights) if n < 28]
    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_WIDTH, 5.4), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1.0]))
    ax = axes[0]
    ax.bar(x, co2_saved, color=S.EXACT, edgecolor="black", hatch="//")
    ax.set_xticks(x, month_labels, rotation=90)
    ax.set_ylabel("CO₂ saved (t per month)")
    S.panel_title(ax, "(a) CO₂ saved")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.bar(x, eur_saved, color=S.EXACT, edgecolor="black", hatch="..")
    ax.set_xticks(x, month_labels, rotation=90)
    ax.set_ylabel("Energy cost saved (k€ per month)")
    S.panel_title(ax, "(b) Energy cost saved")
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    caps = [res.cap_unmanaged_kw.iloc[0] / 1000, res.cap_fcfs_kw.iloc[0] / 1000, res.cap_smart_min_kw.iloc[0] / 1000]
    labels = ["unmanaged\n(on arrival)", "FCFS\ncap", "smart\nschedule"]
    bars = ax.bar(labels, caps, color=[S.GREY, S.LIGHT_GREY, S.EXACT], edgecolor="black", hatch=["xx", "\\\\", "//"])
    for b_, v in zip(bars, caps):
        ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.1f} MW", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Smallest grid connection (MW)")
    S.panel_title(ax, "(c) Connection needed")
    ax.set_ylim(0, caps[0] * 1.15)
    ax.grid(axis="x", visible=False)
    hl = summ[(summ.baseline == "unmanaged") & (summ.w == 1.0) & (summ.group == "all")].iloc[0]
    title = (f"Smart charging saves {co2_saved.sum():.0f} t CO₂ and €{eur_saved.sum() * 1000:,.0f} over a year "
             f"on the same grid connection")
    caption = (f"100-bus depot, {len(res)} nights of Dutch grid data ({res.night.min()} to {res.night.max()}); carbon-first "
               f"schedule (w = 1) against unmanaged charging on the same {caps[0]:.1f} MW connection: {hl.co2_saved_pct:.1f} % less "
               f"CO₂, {hl.eur_saved_pct:.1f} % lower wholesale energy cost. The smart schedule also fits on {caps[2]:.1f} MW "
               f"instead of {caps[1]:.1f} MW with first-come-first-served (FCFS) capping. Fleet modelled on public figures; CO₂ at average grid intensity."
               + (f" * Partial months: {', '.join(partial)}." if partial else ""))
    top = S.heading(fig, title, caption)
    fig.subplots_adjust(top=top - 0.06, bottom=0.17, left=0.065, right=0.985, wspace=0.32)
    S.save(fig, out / "e7_depot_year")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buses", type=int, default=100)
    ap.add_argument("--charger-kw", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nights", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--replot", action="store_true", help="figure + summary from the existing per-night CSV")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.replot:
        res = pd.read_csv(out / "e7_depot_year.csv")
    else:
        res = run(args)
        res.to_csv(out / "e7_depot_year.csv", index=False)
    summ = summarize(res)
    summ.to_csv(out / "e7_summary.csv", index=False)
    plot(res, out, summ)
    print(f"{len(res)} nights ({res.night.min()} .. {res.night.max()}), {int(res.windy.sum())} windy")
    for base in ("unmanaged", "fcfs"):
        a = summ[(summ.baseline == base) & (summ.w == 0.7) & (summ.group == "all")].iloc[0]
        print(f"vs {base} on {a.connection_kw / 1000:.2f} MW, w=0.7: {a.co2_saved_t_sum:.0f} t CO2 "
              f"({a.co2_saved_pct:.1f} %), EUR {a.eur_saved_sum:,.0f} ({a.eur_saved_pct:.1f} %); "
              f"max MIP gap {a.max_mip_gap:.4f}")
    print(f"done in {time.time() - t0:.0f} s -> {out / 'e7_summary.csv'}")
    if not args.replot:  # runtime of a full computation, citable from results/
        (out / "e7_runtime.txt").write_text(
            f"total_runtime_s={time.time() - t0:.1f}\nnights={len(res)}\nbuses={args.buses}\n"
            "milp_depot=scipy.optimize.milp (HiGHS), mip_rel_gap 0.2 %, 4 solves per night\n", encoding="utf-8")


if __name__ == "__main__":
    main()
