"""E2: all methods on every night present in the data -> savings distribution.

For each night in ``list_nights`` (nothing hardcoded) we solve the default 3-EV
instance (w = 0.7) with

  classical : on-arrival, greedy, simulated annealing (seed 0), MILP (exact),
              plus the median and the worst feasible schedule (brute force),
  QAOA      : penalty/X mixer and XY mixer, default reps = 3, seed 0, trained with
              carbon_ev.qaoa.TRAIN_CONFIG (CVaR 0.25, ramp init, maxiter 1000; training is
              deterministic, seed only changes shot sampling),
  p = 0     : per-night untrained start-state sampling (uniform for X, Dicke for XY) from
              random_feasible_baseline, so the QAOA gain over doing nothing is measured per night.

Nights are split into WINDY and CALM by the carbon swing
(max block CO2 / min block CO2 within the night's six 2-h blocks):
windy if swing >= ``--swing-threshold`` (default 1.3), else calm. The threshold
is a documented, arbitrary cut; it is a CLI argument and recorded in the CSV.

Scope: the classical part (on-arrival, greedy, SA, MILP, median / worst feasible, night
descriptors, windy / calm) runs on EVERY night in the data (a full year, ``list_nights``). The QAOA part
(+ the p = 0 baselines) is ~70 s per night per mixer, so it only runs on the nights inside
``--qaoa-from`` ... ``--qaoa-to`` (default 2026-09-10 ... 2026-10-08, the 29-night subset).
QAOA / p = 0 rows already present in an existing results/e2_all_nights.csv with the same configuration
(reps, n_restarts = 1, TRAIN_CONFIG, seeds) are re-used instead of re-run (``--no-reuse`` forces a re-run).
They live in the SAME CSV (column ``nights_scope`` tells which rows are full-year and which subset).

Outputs (results/, or results/scratch/ with --quick):
  e2_all_nights.csv   one row per (night, method, seed); classical rows for all nights, QAOA/p0 rows for the subset
  e2_summary.csv      median / IQR / min / max across nights, per group, column ``nights_scope``
                      (full_year_N = classical, subset_FROM_TO = classical on the subset + QAOA)
  e2_monthly.csv      per month: nights, windy nights, median saving (classical optimum vs on-arrival)
  e2_savings_all_nights.png / .svg   (full-year scatter + monthly panel)

    .venv/Scripts/python experiments/e2_all_nights.py
    .venv/Scripts/python experiments/e2_all_nights.py --quick
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
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

from carbon_ev.classical import brute_force, greedy, milp, on_arrival, simulated_annealing  # noqa: E402
from carbon_ev.data import list_nights, load_hourly  # noqa: E402
from carbon_ev.problem import build_instance, evaluate  # noqa: E402
from carbon_ev.qaoa import TRAIN_CONFIG, random_feasible_baseline, run_qaoa  # noqa: E402
from carbon_ev.qubo import default_penalties, to_qubo  # noqa: E402

FIG_NAME = "e2_savings_all_nights"
SEED_CLASSICAL = 0
QAOA_FROM, QAOA_TO = "2026-09-10", "2026-10-08"  # default per-night QAOA subset (29 nights)
CLASSICAL = ["on_arrival", "greedy", "sa", "milp", "median_feasible", "worst_feasible"]


# ----------------------------------------------------------------------------- helpers
def set_alt_text(results_dir: Path, filename: str, text: str) -> None:
    """Add or replace the entry for ``filename`` in results/figures_alt_text.md (other entries untouched)."""
    path = results_dir / "figures_alt_text.md"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else ["# Figure alt text", ""]
    entry = f"- `{filename}`: {text}"
    for k, ln in enumerate(lines):
        if ln.startswith(f"- `{filename}`"):
            lines[k] = entry
            break
    else:
        lines.append(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iqr(a) -> tuple[float, float, float, float, float]:
    """(median, q25, q75, min, max), nan-aware; all nan if empty."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return (np.nan,) * 5
    return (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)),
            float(a.min()), float(a.max()))


# ----------------------------------------------------------------------------- per night
def solve_night(df, night, seeds: list[int], reps: int, shots: int, maxiter: int, threshold: float,
                do_qaoa: bool = True, reuse: pd.DataFrame | None = None) -> tuple[list[dict], bool]:
    """All rows for one night: classical always; p0 + QAOA if ``do_qaoa`` (re-used from ``reuse`` if possible).

    Returns (rows, qaoa_reused).
    """
    inst = build_instance(df, night)
    swing = float(inst.co2.max() / inst.co2.min())
    pmin = float(inst.price.min())
    price_swing = float(inst.price.max() / pmin) if pmin > 0 else float("nan")
    desc = dict(
        night=str(night), windy=bool(swing >= threshold), swing_threshold=threshold,
        carbon_swing=swing, price_swing=price_swing,
        price_range_eur_mwh=float(inst.price.max() - inst.price.min()),
        co2_min_g_kwh=float(inst.co2.min()), co2_max_g_kwh=float(inst.co2.max()),
    )
    best_x, feas = brute_force(inst)
    opt_obj = feas[0][1]
    med_x = feas[len(feas) // 2][0]
    worst_x = feas[-1][0]
    Q, off = to_qubo(inst, *default_penalties(inst))

    rows: list[dict] = []

    def add(method, family, x, seed=-1, wall=np.nan, **extra):
        ev = evaluate(inst, x)
        rows.append(dict(
            **desc, method=method, family=family, seed=seed, mixer=extra.pop("mixer", ""),
            objective=ev["objective"], co2_kg=ev["co2_kg"], cost_eur=ev["cost_eur"],
            feasible=ev["feasible"], approx_ratio=(opt_obj / ev["objective"]) if ev["feasible"] else np.nan,
            n_feasible_schedules=len(feas), wall_s=wall, **extra))

    t0 = time.perf_counter(); x = on_arrival(inst); add("on_arrival", "classical", x, wall=time.perf_counter() - t0)
    t0 = time.perf_counter(); x = greedy(inst); add("greedy", "classical", x, wall=time.perf_counter() - t0)
    t0 = time.perf_counter(); x = simulated_annealing(Q, off, seed=SEED_CLASSICAL)
    add("sa", "classical", x, seed=SEED_CLASSICAL, wall=time.perf_counter() - t0)
    t0 = time.perf_counter(); x = milp(inst); add("milp", "classical", x, wall=time.perf_counter() - t0)
    add("median_feasible", "classical", med_x)
    add("worst_feasible", "classical", worst_x)
    if not do_qaoa:
        return rows, False

    if reuse is not None:  # QAOA / p0 rows from an earlier run with the same configuration
        old = reuse[reuse.night == str(night)]
        q_old = old[old.family == "qaoa"]
        ok = (len(old[old.family == "p0"]) == 2
              and all(len(q_old[(q_old.mixer == m) & (q_old.seed.isin(seeds))]) == len(seeds) for m in ("x", "xy"))
              and (q_old.reps == reps).all() and (q_old.n_restarts == 1).all())
        if ok:
            keep = old[(old.family == "p0") | ((old.family == "qaoa") & old.seed.isin(seeds))].copy()
            for k, v in desc.items():  # descriptors (windy flag / threshold) follow the current run
                keep[k] = v
            rows += keep.to_dict("records")
            return rows, True

    b0 = random_feasible_baseline(inst, shots=shots, seed=SEED_CLASSICAL, opt_obj=opt_obj)
    for mixer, key, name in (("x", "uniform", "p0_uniform"), ("xy", "demand_satisfying", "p0_dicke")):
        m0 = b0[key]
        be = m0["best_eval"]
        rows.append(dict(
            **desc, method=name, family="p0", seed=SEED_CLASSICAL, mixer=mixer, reps=0,
            objective=be["objective"] if be else np.nan, co2_kg=be["co2_kg"] if be else np.nan,
            cost_eur=be["cost_eur"] if be else np.nan, feasible=be is not None,
            approx_ratio=m0["approx_ratio"], n_feasible_schedules=len(feas), wall_s=0.0,
            feasible_prob=m0["feasible_prob"], approx_ratio_expected=m0["approx_ratio_expected"],
            p_opt=m0["p_opt"], p_within_1pct=m0["p_within_1pct"], depth=0, n_2q=0, n_evals=0))
    for mixer in ("x", "xy"):
        for seed in seeds:
            cfg = dict(TRAIN_CONFIG, **({"maxiter": maxiter} if maxiter else {}))
            r = run_qaoa(inst, mixer, reps, shots=shots, seed=seed, opt_obj=opt_obj, n_restarts=1, **cfg)
            if r.best_x is not None:
                add(f"qaoa_{mixer}", "qaoa", r.best_x, seed=seed, wall=r.wall_s, mixer=mixer, reps=reps,
                    feasible_prob=r.feasible_prob, approx_ratio_expected=r.approx_ratio_expected,
                    p_opt=r.p_opt, p_within_1pct=r.p_within_1pct, depth=r.depth, n_2q=r.n_2q,
                    n_evals=r.n_evals)
            else:  # no feasible sample: keep the row, objective metrics are nan
                rows.append(dict(**desc, method=f"qaoa_{mixer}", family="qaoa", seed=seed, mixer=mixer,
                                 objective=np.nan, co2_kg=np.nan, cost_eur=np.nan, feasible=False,
                                 approx_ratio=np.nan, n_feasible_schedules=len(feas), wall_s=r.wall_s,
                                 reps=reps, feasible_prob=r.feasible_prob, approx_ratio_expected=np.nan,
                                 p_opt=r.p_opt, p_within_1pct=r.p_within_1pct, depth=r.depth,
                                 n_2q=r.n_2q, n_evals=r.n_evals))
    return rows, False


def ensure_n_restarts(rows: pd.DataFrame) -> pd.DataFrame:
    """Add the ``n_restarts`` column (training restarts of the QAOA rows; empty for non-QAOA rows).

    Every QAOA run in this script uses ``run_qaoa(..., n_restarts=1)`` (restart 0 only, deterministic), so
    the seeds are sampling replicates of one training run. Old CSVs written before this column existed
    get it here (value 1) without any recomputation.
    """
    if "n_restarts" not in rows.columns:
        rows = rows.copy()
        rows["n_restarts"] = np.where(rows["family"] == "qaoa", 1.0, np.nan)
    return rows


# ----------------------------------------------------------------------------- summary
def per_night_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per night: CO2 / EUR of each method (QAOA = median over seeds of the best feasible sample)."""
    g = rows.groupby(["night", "method"])[["co2_kg", "cost_eur"]].median().unstack("method")
    g.columns = [f"{m}_{k}" for k, m in g.columns]
    meta = rows.groupby("night")[["windy", "carbon_swing", "price_swing"]].first()
    return meta.join(g)


def build_summary_scope(rows: pd.DataFrame, threshold: float, scope: str) -> pd.DataFrame:
    """Summary of the nights present in ``rows``; QAOA / p0 statistics only if such rows exist."""
    pn = per_night_table(rows)
    out: list[dict] = []
    groups = {"all": pn.index == pn.index, "windy": pn["windy"].to_numpy(), "calm": ~pn["windy"].to_numpy()}

    def emit(group, quantity, values, comparison=""):
        med, q25, q75, lo, hi = iqr(values)
        out.append(dict(nights_scope=scope, group=group, n_nights=int(np.sum(groups[group])),
                        comparison=comparison, quantity=quantity,
                        n_valid=int(np.sum(~np.isnan(np.asarray(values, float)))),
                        median=med, q25=q25, q75=q75, min=lo, max=hi, swing_threshold=threshold))

    refs = {"on_arrival": "vs_on_arrival", "median_feasible": "vs_median_feasible"}
    methods = [m for m in ("milp", "greedy", "sa", "qaoa_x", "qaoa_xy") if f"{m}_co2_kg" in pn.columns]
    for gname, mask in groups.items():
        sub = pn[mask]
        for method in methods:
            for ref, rlabel in refs.items():
                if ref != "on_arrival" and method != "milp":
                    continue  # median-feasible comparison only for the exact optimum
                for unit, key in (("co2_kg", "co2_kg"), ("eur", "cost_eur")):
                    base = sub[f"{ref}_{key}"]
                    save = base - sub[f"{method}_{key}"]
                    emit(gname, f"saving_{unit}", save.to_numpy(), f"{method}_{rlabel}")
                    emit(gname, f"saving_{unit}_pct", (100 * save / base).to_numpy(), f"{method}_{rlabel}")
        # nights where the optimum emits MORE CO2 than on-arrival
        worse = (sub["milp_co2_kg"] > sub["on_arrival_co2_kg"] + 1e-9).astype(float)
        emit(gname, "frac_nights_opt_emits_more_co2", [float(worse.mean())] if len(sub) else [np.nan],
             "milp_vs_on_arrival")
        out[-1]["n_valid"] = int(worse.sum())  # count of such nights (the fraction is in 'median')
        emit(gname, "n_nights_opt_emits_more_co2", [float(worse.sum())], "milp_vs_on_arrival")

    q = rows[rows["family"] == "qaoa"]
    if len(q):
        for gname, mask in groups.items():
            nights = pn.index[mask]
            for mixer in ("x", "xy"):
                s = q[(q["mixer"] == mixer) & q["night"].isin(nights)]
                per = s.groupby("night")[["feasible_prob", "approx_ratio_expected", "approx_ratio", "p_opt",
                                          "p_within_1pct"]].median()
                p0name = "p0_uniform" if mixer == "x" else "p0_dicke"
                p0 = rows[(rows.method == p0name) & rows.night.isin(nights)].set_index("night")
                for metric in per.columns:
                    emit(gname, metric, per[metric].to_numpy(), f"qaoa_{mixer}_sampling_replicates_median")
                    if metric in p0.columns:
                        emit(gname, metric, p0.loc[per.index, metric].to_numpy(), p0name)
                    if metric in ("feasible_prob", "approx_ratio_expected", "p_within_1pct", "p_opt", "approx_ratio"):
                        gain = per[metric].to_numpy() - p0.loc[per.index, metric].to_numpy()
                        emit(gname, f"gain_{metric}", gain, f"qaoa_{mixer}_minus_{p0name}")
                        out[-1]["n_positive"] = int(np.sum(gain > 1e-12))
    return pd.DataFrame(out)


def build_summary(rows: pd.DataFrame, threshold: float, qaoa_scope: str | None = None) -> pd.DataFrame:
    """Full-year classical summary + subset summary (classical on the subset AND QAOA / p0 gains).

    ``nights_scope`` is ``full_year_<N>`` for the first block and ``qaoa_scope`` (e.g.
    ``subset_2026-09-10_2026-10-08``) for the second; the QAOA rows exist only in the second.
    """
    n_all = rows.night.nunique()
    cl = rows[rows.family == "classical"]
    parts = [build_summary_scope(cl, threshold, f"full_year_{n_all}")]
    qn = rows.loc[rows.family == "qaoa", "night"].unique()
    if len(qn):
        sub = rows[rows.night.isin(qn)]
        parts.append(build_summary_scope(sub, threshold, qaoa_scope or f"subset_{min(qn)}_{max(qn)}"))
    return pd.concat(parts, ignore_index=True)


def monthly_table(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per calendar month of the evening date: nights, windy nights, median saving of the exact optimum."""
    pn = per_night_table(rows[rows.family == "classical"]).reset_index()
    pn["month"] = pn["night"].astype(str).str[:7]
    pn["sav_co2"] = pn["on_arrival_co2_kg"] - pn["milp_co2_kg"]
    pn["sav_co2_pct"] = 100 * pn["sav_co2"] / pn["on_arrival_co2_kg"]
    pn["sav_eur"] = pn["on_arrival_cost_eur"] - pn["milp_cost_eur"]
    pn["sav_eur_pct"] = 100 * pn["sav_eur"] / pn["on_arrival_cost_eur"]
    out = []
    for month, g in pn.groupby("month"):
        w = g[g.windy]
        out.append(dict(
            month=month, n_nights=len(g), n_windy_nights=len(w), windy_share=len(w) / len(g),
            median_carbon_swing=float(g.carbon_swing.median()),
            median_saving_co2_kg=float(g.sav_co2.median()), q25_saving_co2_kg=float(g.sav_co2.quantile(.25)),
            q75_saving_co2_kg=float(g.sav_co2.quantile(.75)),
            median_saving_co2_pct=float(g.sav_co2_pct.median()),
            median_saving_eur=float(g.sav_eur.median()), median_saving_eur_pct=float(g.sav_eur_pct.median()),
            median_saving_co2_kg_windy=float(w.sav_co2.median()) if len(w) else np.nan,
            median_saving_co2_kg_calm=float(g[~g.windy].sav_co2.median()) if len(w) < len(g) else np.nan,
            n_nights_opt_emits_more_co2=int((g.sav_co2 < -1e-9).sum()),
            swing_threshold=threshold, comparison="milp_vs_on_arrival", nights_scope=f"full_year_{len(pn)}"))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- figure
def make_figure(rows: pd.DataFrame, summary: pd.DataFrame, monthly: pd.DataFrame, out_base: Path,
                threshold: float) -> str:
    """Savings of the exact optimum vs charge-on-arrival: per night against the carbon swing, and per month."""
    S.apply()
    pn = per_night_table(rows[rows.family == "classical"])
    sav_co2 = pn["on_arrival_co2_kg"] - pn["milp_co2_kg"]
    sav_eur = pn["on_arrival_cost_eur"] - pn["milp_cost_eur"]
    x = pn["carbon_swing"]
    windy = pn["windy"].to_numpy()
    n_w, n_c = int(windy.sum()), int((~windy).sum())
    scope = f"full_year_{len(pn)}"

    def med(group, q, cmp="milp_vs_on_arrival"):
        r = summary[(summary.nights_scope == scope) & (summary.group == group) & (summary.quantity == q)
                    & (summary.comparison == cmp)]
        return float(r["median"].iloc[0]) if len(r) else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_WIDTH, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    ticks = [t for t in (1.0, 1.2, 1.5, 2.0, 3.0) if t <= float(x.max()) * 1.15]
    for ax, y, ylabel, ttl in ((axes[0], sav_co2, "CO₂ saved (kg per night)", "(a) CO₂ saved per night"),
                               (axes[1], sav_eur, "Cost saved (€ per night)", "(b) Cost saved per night")):
        ax.axhline(0, color="black", lw=1)
        ax.axvline(threshold, color=S.GREY, ls="--", lw=1)
        ax.scatter(x[~windy], y[~windy], marker="s", s=30, facecolors="white", edgecolors=S.CALM,
                   linewidths=1.2, alpha=0.9, label=f"calm nights (n = {n_c})", zorder=3)
        ax.scatter(x[windy], y[windy], marker="o", s=34, c=S.WINDY, edgecolors="black", linewidths=0.5,
                   alpha=0.9, label=f"windy nights (n = {n_w})", zorder=4)
        ax.set_xlabel("Carbon swing (max / min CO₂)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.minorticks_off()
        S.panel_title(ax, ttl)
    axes[0].legend(loc="upper left", fontsize=11, handletextpad=0.3)

    ax = axes[2]  # monthly view
    m = monthly.reset_index(drop=True)
    pos = np.arange(len(m))
    ax.axhline(0, color="black", lw=1)
    ax.bar(pos, m.median_saving_co2_kg, color=S.EXACT, edgecolor="black", hatch="//", lw=0.8, zorder=2, alpha=0.9,
           label="median CO₂ saved per night (bar), IQR (whisker)")
    ax.errorbar(pos, m.median_saving_co2_kg, yerr=[m.median_saving_co2_kg - m.q25_saving_co2_kg,
                m.q75_saving_co2_kg - m.median_saving_co2_kg], fmt="none", ecolor="black", capsize=3, lw=1, zorder=3)
    ax.set_ylabel("Median CO₂ saved (kg per night)")
    ax.set_xticks(pos)
    ax.set_xticklabels([pd.Timestamp(mo + "-01").strftime("%b %y") for mo in m.month], rotation=90)
    S.panel_title(ax, "(c) By month")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax2.plot(pos, m.n_windy_nights, color=S.WINDY, marker="o", ls="-", lw=1.8, ms=6, mec="black", mew=0.6, zorder=4,
             label="windy nights in the month (right axis)")
    ax2.set_ylabel("Windy nights per month")
    ax2.set_ylim(bottom=0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    mw, mc = med("windy", "saving_co2_kg"), med("calm", "saving_co2_kg")
    eur_pct, co2_pct_w = med("all", "saving_eur_pct"), med("windy", "saving_co2_kg_pct")
    n_neg = int(round(med("all", "n_nights_opt_emits_more_co2")))
    title = f"Smart charging saves a median {eur_pct:.0f} % of charging cost, and {co2_pct_w:.0f} % of CO₂ on windy nights"
    caption = (f"Exact optimum (w = 0.7) vs charge-on-arrival for the 3-EV fleet on {len(pn)} nights "
               f"({n_w} windy with carbon swing >= {threshold:g}, {n_c} calm). Median CO₂ saving: {mw:.1f} kg per windy "
               f"night, {mc:.1f} kg per calm night. At w = 0.7 the optimum emits more CO₂ than on-arrival on {n_neg} "
               f"nights, mostly calm ones; a higher carbon weight removes this (e5_pareto).")
    top = S.heading(fig, title, caption)
    fig.subplots_adjust(left=0.065, right=0.925, top=top - 0.07, bottom=0.27, wspace=0.42)
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    S.save(fig, out_base)
    return (f"{title}. Left and middle: one point per night against the carbon swing (log scale), CO₂ and cost saved versus "
            f"charge-on-arrival; windy nights are orange filled circles, calm nights hollow sky-blue squares, and savings grow "
            f"with the swing. Right: median CO₂ saving per month as hatched bars with IQR whiskers, and the number of windy "
            f"nights per month as an orange line. {len(pn)} nights, exact optimum at w = 0.7.")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3 nights, 1 seed, small settings; writes to results/scratch/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="QAOA seeds (default 0; training is deterministic)")
    ap.add_argument("--reps", type=int, default=3, help="QAOA depth (default 3; use 2 if the run is too slow)")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--maxiter", type=int, default=None, help="override TRAIN_CONFIG maxiter (quick mode)")
    ap.add_argument("--swing-threshold", type=float, default=1.3,
                    help="windy if max/min block CO2 >= this (default 1.3)")
    ap.add_argument("--qaoa-from", default=QAOA_FROM, help="first night (YYYY-MM-DD) of the per-night QAOA runs")
    ap.add_argument("--qaoa-to", default=QAOA_TO, help="last night (YYYY-MM-DD, inclusive) of the QAOA runs")
    ap.add_argument("--no-reuse", action="store_true",
                    help="do not re-use QAOA / p0 rows of an existing e2_all_nights.csv; re-run them")
    ap.add_argument("--max-nights", type=int, default=None, help="use only N nights spread over the data (debug)")
    ap.add_argument("--replot", action="store_true", help="redraw the figure from results/e2_*.csv (no runs)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (ROOT / "results" / ("scratch" if args.quick else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    q_scope = f"subset_{args.qaoa_from}_{args.qaoa_to}"
    if args.replot:
        rows = ensure_n_restarts(pd.read_csv(out_dir / "e2_all_nights.csv"))
        rows["nights_scope"] = np.where(rows.night.isin(rows.loc[rows.family == "qaoa", "night"].unique()),
                                        q_scope, "full_year_only")
        rows.to_csv(out_dir / "e2_all_nights.csv", index=False)
        summary = build_summary(rows, args.swing_threshold, q_scope)
        summary.to_csv(out_dir / "e2_summary.csv", index=False)
        monthly = monthly_table(rows, args.swing_threshold)
        monthly.to_csv(out_dir / "e2_monthly.csv", index=False)
        set_alt_text(out_dir, FIG_NAME + ".png", make_figure(rows, summary, monthly, out_dir / FIG_NAME, args.swing_threshold))
        print(f"replotted {out_dir / (FIG_NAME + '.png')}")
        return
    df = load_hourly()
    nights = list_nights(df)
    seeds, shots, maxiter, max_n = args.seeds, args.shots, args.maxiter, args.max_nights
    if args.quick:
        seeds, shots, maxiter, max_n = [0], 512, 40, 3
        args.reps = 1
    if max_n:
        # spread the sample over the available nights instead of taking the first few
        idx = np.unique(np.linspace(0, len(nights) - 1, max_n).round().astype(int))
        nights = [nights[i] for i in idx]
    q_from, q_to = date.fromisoformat(args.qaoa_from), date.fromisoformat(args.qaoa_to)
    q_nights = {n for n in nights if q_from <= n <= q_to}
    if args.quick:  # tiny smoke run: QAOA on every selected night, whatever the range
        q_nights = set(nights)
        q_scope = "quick_subset"
    reuse = None
    csv_path = out_dir / "e2_all_nights.csv"
    if csv_path.exists() and not args.no_reuse and not args.quick:
        reuse = ensure_n_restarts(pd.read_csv(csv_path))
        reuse = reuse[reuse.family.isin(["p0", "qaoa"])]
    old_classical = None
    if csv_path.exists() and not args.quick:
        old_classical = pd.read_csv(csv_path)
        old_classical = old_classical[old_classical.family == "classical"]

    t_start = time.perf_counter()
    all_rows: list[dict] = []
    n_reused = 0
    for k, night in enumerate(nights, 1):
        t0 = time.perf_counter()
        r, reused = solve_night(df, night, seeds, args.reps, shots, maxiter, args.swing_threshold,
                                do_qaoa=night in q_nights, reuse=reuse)
        all_rows += r
        n_reused += int(reused)
        if k % 25 == 0 or k == len(nights) or night in q_nights:
            print(f"[{k}/{len(nights)}] {night} done in {time.perf_counter() - t0:.1f}s"
                  f"{' (QAOA re-used)' if reused else ''}", flush=True)
    rows = ensure_n_restarts(pd.DataFrame(all_rows))
    rows["nights_scope"] = np.where(rows.night.isin([str(n) for n in q_nights]), q_scope, "full_year_only")
    if reuse is not None:
        print(f"QAOA / p0 rows re-used for {n_reused} of {len(q_nights)} subset nights; "
              f"the remaining subset nights were run.")
    if old_classical is not None and len(old_classical):  # consistency of the recomputed classical rows
        new_c = rows[rows.family == "classical"]
        mg = new_c.merge(old_classical, on=["night", "method"], suffixes=("", "_old"))
        d = max(float((mg.co2_kg - mg.co2_kg_old).abs().max()), float((mg.cost_eur - mg.cost_eur_old).abs().max())) if len(mg) else float("nan")
        print(f"classical rows recomputed; {mg.night.nunique()} nights overlap the previous CSV, "
              f"max |difference| in kg CO2 / EUR = {d:.2e}")
    rows.to_csv(csv_path, index=False)
    summary = build_summary(rows, args.swing_threshold, q_scope)
    summary.to_csv(out_dir / "e2_summary.csv", index=False)
    monthly = monthly_table(rows, args.swing_threshold)
    monthly.to_csv(out_dir / "e2_monthly.csv", index=False)
    alt = make_figure(rows, summary, monthly, out_dir / FIG_NAME, args.swing_threshold)
    if not args.quick:
        set_alt_text(out_dir, FIG_NAME + ".png", alt)

    # ---- console summary
    pn = per_night_table(rows[rows.family == "classical"])
    print(f"\nnights: {len(pn)}  windy (swing >= {args.swing_threshold:g}): {int(pn.windy.sum())}  "
          f"calm: {int((~pn.windy).sum())}  QAOA subset: {len(q_nights)} nights   "
          f"total {time.perf_counter() - t_start:.0f}s")
    show = summary[summary.quantity.isin(["saving_co2_kg", "saving_eur", "frac_nights_opt_emits_more_co2",
                                          "n_nights_opt_emits_more_co2"])
                   & summary.comparison.isin(["milp_vs_on_arrival", "milp_vs_median_feasible"])]
    print(show[["nights_scope", "group", "n_nights", "comparison", "quantity", "median", "q25", "q75", "min", "max"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    qs = summary[summary.quantity.isin(["approx_ratio_expected", "p_within_1pct", "feasible_prob",
                                        "gain_approx_ratio_expected", "gain_p_within_1pct", "gain_feasible_prob"])
                 & summary.comparison.str.contains("qaoa_|p0_")]
    print(qs[["nights_scope", "group", "comparison", "quantity", "median", "q25", "q75", "n_positive"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(monthly[["month", "n_nights", "n_windy_nights", "median_saving_co2_kg", "median_saving_co2_pct",
                   "median_saving_eur", "n_nights_opt_emits_more_co2"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"wrote {csv_path}, e2_summary.csv, e2_monthly.csv, {FIG_NAME}.png/.svg")


if __name__ == "__main__":
    main()
