"""E3: penalty QUBO + X mixer vs constraint-preserving XY mixer (ideal and noisy), reps 1-3.

Grid on one night (default the main night 2026-10-08, i.e. Oct 8 -> 9; 16 qubits):

  ideal : mixer x  (penalty margin 1.5; margins 3 and 6 only at reps = --margin-reps, default 2)
          x reps {1, 2, 3} x seeds 0..4;  mixer xy (conflict-only, margin 1.5) x reps {1, 2, 3} x seeds 0..4.
          Training uses carbon_ev.qaoa.TRAIN_CONFIG (CVaR 0.25, ramp init, maxiter 1000; recorded in the
          CSV). Training is deterministic, so each config is trained once and the seeds are replicates
          of the 4096-shot sampling only (not of the optimization).
  noisy : the ideally trained angles (margin 1.5, seeds 0..2) re-sampled with the
          approximate IonQ-Forte-class depolarizing model (carbon_ev.noise); no re-training
  p = 0 : sampling the start state only (uniform for X, Dicke for XY), seeds 0..4,
          from qaoa.random_feasible_baseline - the reference QAOA must beat

``--nights main`` (default) uses ``--night``; ``--nights all`` or a comma list of dates
repeats the grid on those nights (slow: use with --skip-noisy or --seeds).

Outputs (results/, or results/scratch/ with --quick):
  e3_penalty_vs_xy.csv   one row per run          e3_summary.csv   median / IQR per configuration
  e3_penalty_vs_xy.png / .svg   (figure for the first night processed)

    .venv/Scripts/python experiments/e3_penalty_vs_xy.py
    .venv/Scripts/python experiments/e3_penalty_vs_xy.py --quick
    .venv/Scripts/python experiments/e3_penalty_vs_xy.py --skip-noisy
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
from matplotlib.lines import Line2D  # noqa: E402

import _style as S  # noqa: E402

from carbon_ev.data import list_nights, load_hourly  # noqa: E402
from carbon_ev.problem import build_instance  # noqa: E402
from carbon_ev.qaoa import (TRAIN_CONFIG, evaluate_params, exact_optimum,  # noqa: E402
                            random_feasible_baseline, run_qaoa)
from carbon_ev.qubo import default_penalties  # noqa: E402

MAIN_NIGHT = "2026-10-08"
FIG_NAME = "e3_penalty_vs_xy"
METRICS = ["feasible_prob", "demand_prob", "approx_ratio", "approx_ratio_expected", "p_opt", "p_within_1pct",
           "depth", "n_2q", "n_evals", "wall_s", "train_wall_s", "best_co2_kg", "best_cost_eur"]


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


def row_from(res, night, condition, mixer, reps, seed, margin, lam, n_evals=None, **extra) -> dict:
    """Tidy row from a QAOAResult (or a sample_metrics dict for p = 0)."""
    g = (lambda k: getattr(res, k)) if not isinstance(res, dict) else (lambda k: res.get(k, np.nan))
    best_eval = g("best_eval")
    return dict(
        night=night, condition=condition, mixer=mixer, reps=reps, seed=seed, margin=margin, lam=lam,
        feasible_prob=g("feasible_prob"), demand_prob=g("demand_prob"), approx_ratio=g("approx_ratio"),
        approx_ratio_expected=g("approx_ratio_expected"), p_opt=g("p_opt"), p_within_1pct=g("p_within_1pct"),
        depth=g("depth") if not isinstance(res, dict) else 0,
        n_2q=g("n_2q") if not isinstance(res, dict) else 0,
        n_evals=(g("n_evals") if n_evals is None else n_evals) if not isinstance(res, dict) else 0,
        wall_s=g("wall_s") if not isinstance(res, dict) else 0.0,
        best_co2_kg=best_eval["co2_kg"] if best_eval else np.nan,
        best_cost_eur=best_eval["cost_eur"] if best_eval else np.nan,
        **extra)


def run_night(df, night: date, a) -> list[dict]:
    from carbon_ev.noise import aer_available, noisy_sampler

    inst = build_instance(df, night)
    opt_obj = exact_optimum(inst)[1]
    rows: list[dict] = []
    nstr = str(night)

    # ---- p = 0 references
    for seed in a.seeds:
        b = random_feasible_baseline(inst, shots=a.shots, seed=seed, opt_obj=opt_obj)
        rows.append(row_from(b["uniform"], nstr, "p0", "x", 0, seed, np.nan, np.nan, label="uniform start"))
        rows.append(row_from(b["demand_satisfying"], nstr, "p0", "xy", 0, seed, np.nan, np.nan,
                             label="Dicke start"))

    # ---- ideal grid. Training with TRAIN_CONFIG is deterministic (``seed`` only changes shot
    # sampling), so every config is trained ONCE and re-sampled with each seed via evaluate_params.
    # Seeds are therefore replicates of the 4096-shot sampling, not of the optimization.
    trained: dict[tuple, np.ndarray] = {}
    for mixer in ("x", "xy"):
        for margin in (a.margins if mixer == "x" else [a.margins[0]]):
            lam_d, lam_c = default_penalties(inst, margin)
            # extra penalty margins only at reps = a.margin_reps (keeps the grid small)
            reps_list = a.reps if (mixer == "xy" or margin == a.margins[0]) else [a.margin_reps]
            for reps in reps_list:
                cfg = dict(TRAIN_CONFIG)
                if a.quick:
                    cfg["maxiter"] = a.maxiter
                tr = run_qaoa(inst, mixer, reps, shots=a.shots, seed=a.seeds[0], opt_obj=opt_obj,
                              lam_demand=lam_d, lam_conflict=lam_c, **cfg)
                trained[(mixer, margin, reps)] = tr.params
                for seed in a.seeds:
                    r = tr if seed == a.seeds[0] else evaluate_params(
                        inst, mixer, reps, tr.params, shots=a.shots, seed=seed, opt_obj=opt_obj,
                        lam_demand=lam_d, lam_conflict=lam_c)
                    rows.append(row_from(r, nstr, "ideal", mixer, reps, seed, margin, lam_c, label="",
                                         n_evals=tr.n_evals, train_wall_s=tr.wall_s,
                                         train_objective=cfg["objective"], train_cvar_alpha=cfg["cvar_alpha"],
                                         train_init=cfg["init"], train_maxiter=cfg["maxiter"]))
                print(f"  ideal {nstr} {mixer} margin={margin:g} p={reps}: train {tr.wall_s:.0f}s "
                      f"evals={tr.n_evals} P(feas)={tr.feasible_prob:.3f}", flush=True)

    # ---- noisy re-evaluation of the ideal angles (margin = first margin, i.e. 1.5)
    if not a.skip_noisy:
        if not aer_available():
            print("qiskit-aer not installed: skipping the noisy part (pip install qiskit-aer)")
        else:
            m0 = a.margins[0]
            lam_d, lam_c = default_penalties(inst, m0)
            for mixer in ("x", "xy"):
                for reps in a.reps:
                    for seed in a.noisy_seeds:
                        t0 = time.perf_counter()
                        r = evaluate_params(inst, mixer, reps, trained[(mixer, m0, reps)],
                                            sampler=noisy_sampler(seed=seed), shots=a.noisy_shots, seed=seed,
                                            opt_obj=opt_obj, lam_demand=lam_d, lam_conflict=lam_c)
                        rows.append(row_from(r, nstr, "noisy", mixer, reps, seed, m0, lam_c, label=""))
                        print(f"  noisy {nstr} {mixer} p={reps} seed={seed}: {time.perf_counter() - t0:.0f}s "
                              f"P(feas)={r.feasible_prob:.3f}", flush=True)
    return rows


def ensure_n_restarts(rows: pd.DataFrame) -> pd.DataFrame:
    """Add the ``n_restarts`` column (training restarts of the QAOA rows; empty for non-QAOA rows).

    Every QAOA run in this script uses ``run_qaoa(..., n_restarts=1)`` (restart 0 only, deterministic), so
    the seeds are sampling replicates of one training run. Old CSVs written before this column existed
    get it here (value 1) without any recomputation.
    """
    if "n_restarts" not in rows.columns:
        rows = rows.copy()
        rows["n_restarts"] = np.where(rows["condition"].isin(["ideal", "noisy"]), 1.0, np.nan)
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["night", "condition", "mixer", "margin", "reps"]
    out = []
    for k, g in rows.groupby(keys, dropna=False, sort=False):
        d = dict(zip(keys, k))
        d["n_runs"] = len(g)
        d["n_no_feasible_sample"] = int(g["approx_ratio"].isna().sum())
        for m in METRICS:
            v = g[m].to_numpy(float)
            d[f"{m}_median"] = float(np.nanmedian(v)) if np.any(~np.isnan(v)) else np.nan
            d[f"{m}_q25"] = float(np.nanpercentile(v, 25)) if np.any(~np.isnan(v)) else np.nan
            d[f"{m}_q75"] = float(np.nanpercentile(v, 75)) if np.any(~np.isnan(v)) else np.nan
        out.append(d)
    return pd.DataFrame(out)


def make_figure(rows: pd.DataFrame, out_base: Path, night: str, a) -> str:
    """P(feasible) and approximation ratio per QAOA variant, ideal and noisy, with the p = 0 start-state references."""
    S.apply()
    r = rows[rows.night == night]
    margins = a.margins
    groups = ([("x", margins[0], p) for p in a.reps] + [("x", m, a.margin_reps) for m in margins[1:]]
              + [("xy", margins[0], p) for p in a.reps])
    has_noisy = (r.condition == "noisy").any()

    def sel(cond, mixer, margin, reps):
        s = r[(r.condition == cond) & (r.mixer == mixer) & (r.reps == reps)]
        if mixer == "x":
            s = s[np.isclose(s.margin, margin)]
        return s

    fig, axes = plt.subplots(2, 1, figsize=(S.FIG_WIDTH, 8.6), sharex=True)
    ref = {mx: r[(r.condition == "p0") & (r.mixer == mx)] for mx in ("x", "xy")}
    xs = np.arange(len(groups))
    for ax, metric, ylabel in ((axes[0], "feasible_prob", "P(feasible)"),
                               (axes[1], "approx_ratio_expected", "Expected-cost ratio (1 = optimal)")):
        for gi, (mixer, margin, reps) in enumerate(groups):
            col = S.X if mixer == "x" else S.XY
            for cond, dx, mk, filled in (("ideal", -0.13, "o", True), ("noisy", 0.13, "s", False)):
                s = sel(cond, mixer, margin, reps)
                v = s[metric].dropna().to_numpy(float)
                if len(v) == 0:
                    continue
                med, q1, q3 = np.median(v), np.percentile(v, 25), np.percentile(v, 75)
                jit = np.linspace(-0.05, 0.05, len(v)) if len(v) > 1 else np.zeros(1)
                ax.scatter(gi + dx + jit, v, s=16, color=col, alpha=0.35, marker=mk, zorder=2)
                ax.errorbar([gi + dx], [med], yerr=[[med - q1], [q3 - med]], fmt=mk, color=col, ms=9,
                            mfc=col if filled else "white", mew=1.8, capsize=4, lw=1.6, zorder=3)
        for mixer, ls in (("x", "--"), ("xy", ":")):
            v = ref[mixer][metric].dropna()
            if len(v):
                idx = [i for i, g in enumerate(groups) if g[0] == mixer]
                ax.hlines(v.median(), min(idx) - 0.45, max(idx) + 0.45, colors="black", linestyles=ls, lw=1.6, zorder=1)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(-0.02, 1.02)
    lo = np.nanmin(r["approx_ratio_expected"].to_numpy(float))
    axes[1].set_ylim(max(0.0, min(0.5, lo - 0.05)), 1.02)

    labels = []
    for mixer, margin, reps in groups:
        n2q = sel("ideal", mixer, margin, reps)["n_2q"]
        cx = int(n2q.median()) if len(n2q) else "?"
        first = f"p = {reps}" if (mixer == "xy" or np.isclose(margin, margins[0])) else f"{margin:g}x"
        labels.append(f"{first}\n{cx} CX")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    n_main = len(a.reps)
    spans = ((0, n_main - 1, f"X mixer, penalty {margins[0]:g}x"),
             (n_main, n_main + len(margins) - 2, f"X mixer, penalty {', '.join(f'{m:g}x' for m in margins[1:])} (p = {a.margin_reps})"),
             (n_main + len(margins) - 1, len(groups) - 1, "XY mixer, Dicke start"))
    for lo_i, hi_i, name in spans:
        if hi_i >= lo_i:
            axes[1].text((lo_i + hi_i) / 2, -0.24, name, transform=axes[1].get_xaxis_transform(), ha="center", va="top",
                         fontsize=11, fontweight="bold")

    handles = [Line2D([], [], marker="o", color=S.X, ls="", ms=9, label="X mixer, ideal"),
               Line2D([], [], marker="o", color=S.XY, ls="", ms=9, label="XY mixer, ideal"),
               Line2D([], [], marker="s", color=S.X, mfc="white", mew=1.8, ls="", ms=9, label="X mixer, noisy"),
               Line2D([], [], marker="s", color=S.XY, mfc="white", mew=1.8, ls="", ms=9, label="XY mixer, noisy"),
               Line2D([], [], color="black", ls="--", label="p = 0 uniform start"),
               Line2D([], [], color="black", ls=":", label="p = 0 Dicke start")]
    axes[0].legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=11)

    def med(cond, mixer, margin, reps, metric):
        v = sel(cond, mixer, margin, reps)[metric].dropna()
        return float(v.median()) if len(v) else float("nan")

    p1 = a.reps[-1]
    fx, fxy = med("ideal", "x", margins[0], p1, "feasible_prob"), med("ideal", "xy", margins[0], p1, "feasible_prob")
    fxn = med("noisy", "x", margins[0], p1, "feasible_prob") if has_noisy else float("nan")
    fxyn = med("noisy", "xy", margins[0], p1, "feasible_prob") if has_noisy else float("nan")
    title = f"The XY mixer samples feasible schedules {fxy * 100:.0f} % of the time, vs {fx * 100:.0f} % for the penalty X mixer"
    caption = (f"Main night {night}, 16 qubits, p = {p1}, ideal simulation, 4096 shots; median and IQR over {len(a.seeds)} sampling seeds. "
               "CX = two-qubit gates after transpiling to rz, sx, x, cx.")
    if has_noisy:
        caption += (f" Hollow squares: ideal angles re-sampled with an approximate IonQ-like noise model "
                    f"({len(a.noisy_seeds)} seeds); feasibility {fxyn * 100:.0f} % (XY) and {fxn * 100:.0f} % (X) at p = {p1}.")
    top = S.heading(fig, title, caption)
    fig.subplots_adjust(top=top - 0.07, bottom=0.2, left=0.075, right=0.985, hspace=0.12)
    S.save(fig, out_base)
    return (f"{title}. Two stacked panels over the QAOA variants (X mixer at p = 1 to 3 and other penalties, XY mixer at "
            f"p = 1 to 3): top, probability of a feasible sample; bottom, expected-cost ratio. Filled circles are ideal "
            f"simulation, hollow squares an approximate noisy simulation; the X mixer is blue and the XY mixer vermilion. "
            f"Dashed and dotted lines mark the untrained start states. Two-qubit gate counts are printed under each variant.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny settings; writes to results/scratch/")
    ap.add_argument("--night", default=MAIN_NIGHT, help="night D (window D 20:00 -> D+1 08:00), default main night")
    ap.add_argument("--nights", default="main", help="'main' (= --night), 'all', or comma-separated dates")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--noisy-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--margin-reps", type=int, default=2, help="reps at which the extra X penalty margins are run")
    ap.add_argument("--margins", type=float, nargs="+", default=[1.5, 3.0, 6.0],
                    help="penalty margins for the X mixer (first one is used for XY and the noisy runs)")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--noisy-shots", type=int, default=4096)
    ap.add_argument("--maxiter", type=int, default=None, help="quick mode only; default training uses TRAIN_CONFIG")
    ap.add_argument("--skip-noisy", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw the figure from the existing CSV (no runs, CSVs untouched)")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.noisy_seeds, a.reps, a.margins, a.margin_reps = [0, 1], [0], [1], [1.5, 3.0], 1
        a.shots, a.noisy_shots, a.maxiter = 512, 256, 40

    out_dir = a.out_dir or (ROOT / "results" / ("scratch" if a.quick else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_hourly()
    avail = list_nights(df)
    if a.nights == "main":
        nights = [date.fromisoformat(a.night)]
    elif a.nights == "all":
        nights = avail
    else:
        nights = [date.fromisoformat(s) for s in a.nights.split(",")]
    for n in nights:
        if n not in avail:
            sys.exit(f"night {n} has no complete 20:00-08:00 coverage in the data")
    tag = "" if a.nights == "main" else "_nights"

    if a.replot:
        csv = out_dir / f"e3_penalty_vs_xy{tag}.csv"
        rows = ensure_n_restarts(pd.read_csv(csv))
        set_alt_text(out_dir, f"{FIG_NAME}{tag}.png", make_figure(rows, out_dir / (FIG_NAME + tag), str(nights[0]), a))
        print(f"replotted from {csv}")
        return
    t_start = time.perf_counter()
    rows_all: list[dict] = []
    for n in nights:
        print(f"night {n}", flush=True)
        rows_all += run_night(df, n, a)
    rows = ensure_n_restarts(pd.DataFrame(rows_all))
    rows.to_csv(out_dir / f"e3_penalty_vs_xy{tag}.csv", index=False)
    summ = summarize(rows)
    summ.to_csv(out_dir / f"e3_summary{tag}.csv", index=False)
    alt = make_figure(rows, out_dir / (FIG_NAME + tag), str(nights[0]), a)
    if not a.quick:
        set_alt_text(out_dir, f"{FIG_NAME}{tag}.png", alt)

    cols = ["night", "condition", "mixer", "margin", "reps", "n_runs", "feasible_prob_median", "feasible_prob_q25",
            "feasible_prob_q75", "approx_ratio_expected_median", "approx_ratio_median", "p_opt_median",
            "p_within_1pct_median", "n_2q_median", "depth_median", "n_evals_median", "wall_s_median"]
    print(f"\ntotal {time.perf_counter() - t_start:.0f}s")
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out_dir / ('e3_penalty_vs_xy' + tag + '.csv')}, e3_summary{tag}.csv, {FIG_NAME}{tag}.png/.svg")


if __name__ == "__main__":
    main()
