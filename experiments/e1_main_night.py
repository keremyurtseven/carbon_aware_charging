"""E1: every method on the main night (Oct 8 20:00 -> Oct 9 08:00).

Classical: charge-on-arrival, median feasible schedule, greedy, simulated annealing
(5 seeds), exact MILP. Sampling baselines (QAOA at p = 0): uniform random bitstrings
and the Dicke start state, 5 seeds. QAOA: mixer {x, xy} x reps {1, 2, 3} x seeds 0-4.

Writes results/e1_main_night.csv (one row per run), results/e1_summary.csv
(median and IQR per method) and results/e1_main_night.png/.svg.

    python experiments/e1_main_night.py            # full run
    python experiments/e1_main_night.py --quick    # 2 seeds, reps 1 only, small maxiter, writes to results/scratch
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
import main as demo  # noqa: E402  (shared row helpers; main() is not executed on import)
from carbon_ev.classical import greedy, milp, on_arrival, simulated_annealing  # noqa: E402
from carbon_ev.data import load_hourly  # noqa: E402
from carbon_ev.problem import build_instance, evaluate  # noqa: E402
from carbon_ev.qaoa import TRAIN_CONFIG, random_feasible_baseline, run_qaoa  # noqa: E402
from carbon_ev.qubo import default_penalties, to_qubo  # noqa: E402

RESULTS = ROOT / "results"
ALT = RESULTS / "figures_alt_text.md"
NUMERIC = ["co2_kg", "cost_eur", "objective", "saved_co2_pct", "saved_eur_pct", "saved_co2_vs_median_pct",
           "p_feasible", "approx_ratio", "approx_ratio_expected", "p_opt", "p_within_1pct",
           "depth", "n_2q", "n_evals", "wall_s"]


def run(args) -> pd.DataFrame:
    df = load_hourly()
    inst = build_instance(df, date.fromisoformat(args.night))
    seeds = list(range(args.seeds))
    reps_list = [1] if args.quick else [1, 2, 3]
    train = dict(TRAIN_CONFIG)
    if args.quick:
        train['maxiter'] = 30
    n_restarts = 1 if args.quick else args.restarts

    x_arr = on_arrival(inst)
    ev_arr = evaluate(inst, x_arr)
    base = (ev_arr["co2_kg"], ev_arr["cost_eur"])
    x_med, _med_obj, _ = demo.median_feasible(inst)
    ev_med = evaluate(inst, x_med)
    med = (ev_med["co2_kg"], ev_med["cost_eur"])
    t = time.perf_counter()
    x_opt = milp(inst)
    t_milp = time.perf_counter() - t
    opt_obj = evaluate(inst, x_opt)["objective"]

    rows = []

    def add(row, method, family, mixer="", reps=0, seed=-1):
        row.update(method=method, family=family, mixer=mixer, reps=reps, seed=seed, night=args.night)
        rows.append(row)

    add(demo.classical_row("", inst, x_arr, np.nan, base, med), "on-arrival", "classical")
    add(demo.classical_row("", inst, x_med, np.nan, base, med), "median feasible schedule", "reference")
    t = time.perf_counter()
    xg = greedy(inst)
    add(demo.classical_row("", inst, xg, time.perf_counter() - t, base, med), "greedy", "classical")
    add(demo.classical_row("", inst, x_opt, t_milp, base, med), "exact (MILP)", "classical")
    dd, dc = default_penalties(inst)
    Q, off = to_qubo(inst, dd, dc)
    for s in seeds:
        t = time.perf_counter()
        xs = simulated_annealing(Q, off, seed=s)
        add(demo.classical_row("", inst, xs, time.perf_counter() - t, base, med), "SA (QUBO)", "classical", seed=s)
    for s in seeds:
        rb = random_feasible_baseline(inst, args.shots, s, opt_obj=opt_obj)
        add(demo.metrics_row("", inst, rb["uniform"], base, med), "p=0 uniform random (X start)", "p0", "x", 0, s)
        add(demo.metrics_row("", inst, rb["demand_satisfying"], base, med), "p=0 Dicke start (XY start)", "p0",
            "xy", 0, s)
    for mixer in ("x", "xy"):
        for reps in reps_list:
            for s in seeds:
                res = run_qaoa(inst, mixer=mixer, reps=reps, shots=args.shots, seed=s, opt_obj=opt_obj,
                               n_restarts=n_restarts, **train)
                add(demo.metrics_row("", inst, demo.result_metrics(res), base, med, depth=res.depth,
                                     n_2q=res.n_2q, n_evals=res.n_evals, wall_s=res.wall_s),
                    f"QAOA {mixer.upper()} p={reps}", "qaoa", mixer, reps, s)
                print(f"  QAOA {mixer} p={reps} seed={s}: P(feas)={res.feasible_prob:.3f} "
                      f"P(<1%)={res.p_within_1pct:.4f} apx_exp={res.approx_ratio_expected:.3f} "
                      f"({res.wall_s:.1f}s)", flush=True)
    out = pd.DataFrame(rows)
    out["shots"] = args.shots
    is_q = out.family == "qaoa"
    # training configuration of the QAOA rows (from carbon_ev.qaoa.TRAIN_CONFIG); empty for non-QAOA rows
    out["train_objective"] = np.where(is_q, train["objective"], "")
    out["cvar_alpha"] = np.where(is_q, train["cvar_alpha"], np.nan)
    out["train_init"] = np.where(is_q, train["init"], "")
    out["train_maxiter"] = np.where(is_q, train["maxiter"], np.nan)
    out["n_restarts"] = np.where(is_q, n_restarts, np.nan)
    cols = ["night", "method", "family", "mixer", "reps", "seed", "shots", "schedule"] + NUMERIC + ["feasible", "train_objective", "cvar_alpha", "train_init",
                                                                                   "train_maxiter", "n_restarts"]
    return out[cols]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Median and interquartile range (q25, q75) of every metric per method."""
    recs = []
    for method, g in runs.groupby("method", sort=False):
        rec = {"method": method, "family": g.family.iloc[0], "n_runs": len(g)}
        for c in NUMERIC:
            v = g[c].dropna()
            rec[f"{c}_median"] = float(v.median()) if len(v) else np.nan
            rec[f"{c}_q25"] = float(v.quantile(0.25)) if len(v) else np.nan
            rec[f"{c}_q75"] = float(v.quantile(0.75)) if len(v) else np.nan
        rec["all_feasible"] = bool(g.feasible.all())
        recs.append(rec)
    return pd.DataFrame(recs)


def print_summary(s: pd.DataFrame) -> None:
    print(f"{'method':<30}{'n':>3}{'kg CO2':>8}{'EUR':>7}{'P(feas) med [IQR]':>26}{'apx.exp med [IQR]':>26}"
          f"{'apx.best':>9}{'P(<1%)':>8}{'depth':>6}{'2q':>5}{'evals':>6}{'time s':>8}")

    def f(v, fmt):
        w = int("".join(ch for ch in fmt.split(".")[0] if ch.isdigit()))
        return "-".rjust(w) if pd.isna(v) else format(v, fmt)

    def rng(r, c):
        if pd.isna(r[f"{c}_median"]):
            return "-".rjust(26)
        return f"{r[f'{c}_median']:.3f} [{r[f'{c}_q25']:.3f}, {r[f'{c}_q75']:.3f}]".rjust(26)

    for _, r in s.iterrows():
        print(f"{r.method:<30}{r.n_runs:>3}{f(r.co2_kg_median, '8.2f')}{f(r.cost_eur_median, '7.2f')}"
              f"{rng(r, 'p_feasible')}{rng(r, 'approx_ratio_expected')}{f(r.approx_ratio_median, '9.4f')}"
              f"{f(r.p_within_1pct_median, '8.4f')}{f(r.depth_median, '6.0f')}{f(r.n_2q_median, '5.0f')}"
              f"{f(r.n_evals_median, '6.0f')}{f(r.wall_s_median, '8.2f')}")


def make_figure(runs: pd.DataFrame, out_png: Path) -> str:
    """Dot plots of P(feasible), P(within 1 % of optimum) and expected-cost ratio per configuration.

    Returns the alt text.
    """
    from matplotlib.lines import Line2D
    S.apply()
    order = list(runs[runs.family.isin(["p0", "qaoa"])].method.unique())
    style = {"x": (S.X, "o"), "xy": (S.XY, "s")}
    short = {"p=0 uniform random (X start)": "random\nstart", "p=0 Dicke start (XY start)": "Dicke\nstart"}
    labels = [short.get(m, m.split()[-1]) for m in order]          # "QAOA X p=1" -> "p=1"
    fam = ["no QAOA (p=0)" if m.startswith("p=0") else ("X mixer" if m.startswith("QAOA X ") else
           "XY mixer") for m in order]
    # x positions with a gap between the three groups
    pos, k, prev = [], 0.0, None
    for f in fam:
        if prev is not None and f != prev:
            k += 0.9
        pos.append(k)
        k += 1.5 if f.startswith("no QAOA") else 1.0
        prev = f
    fig = plt.figure(figsize=(S.FIG_WIDTH, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.34)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
    key = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)
    panels = ((axes[0], "p_feasible", "(a) Feasible samples", "P(feasible)"),
              (axes[1], "p_within_1pct", "(b) Near-optimal samples", "P(within 1 % of optimum)"),
              (axes[2], "approx_ratio_expected", "(c) Expected-cost ratio", "Optimum / mean cost of\nfeasible samples (1 = optimal)"))
    for ax, col, ttl, ylab in panels:
        for x0, m in zip(pos, order):
            g = runs[runs.method == m]
            colr, mk = style[g.mixer.iloc[0]]
            hollow = g.family.iloc[0] == "p0"
            v = g[col].to_numpy()
            ax.scatter(x0 + rng.uniform(-0.12, 0.12, len(v)), v, s=40, marker=mk, alpha=0.6,
                       facecolors="none" if hollow else colr, edgecolors=colr, linewidths=1.5, zorder=3)
            q1, med, q3 = np.nanpercentile(v, [25, 50, 75])
            ax.errorbar(x0 + 0.3, med, yerr=[[med - q1], [q3 - med]], fmt=mk, color="black", ms=8,
                        capsize=5, mfc="white" if hollow else "black", zorder=4)
        for i in range(1, len(pos)):
            if fam[i] != fam[i - 1]:
                ax.axvline((pos[i] + pos[i - 1]) / 2, color="#999999", ls="--", lw=1)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.9)
        S.panel_title(ax, ttl)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1.05 if col != "p_within_1pct" else None)
        for f in dict.fromkeys(fam):                                  # group names under the tick labels
            xs = [x for x, ff in zip(pos, fam) if ff == f]
            ax.text(np.mean(xs), -0.23, f, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11,
                    fontweight="bold")
    axes[2].axhline(1, color=S.GREY, ls=":", lw=1)
    key.axis("off")
    key.legend(handles=[
        Line2D([], [], marker="o", color=S.X, ls="", ms=8, label="X mixer, penalty QUBO"),
        Line2D([], [], marker="s", color=S.XY, ls="", ms=8, label="XY mixer, Dicke start"),
        Line2D([], [], marker="o", mfc="none", mec=S.GREY, ls="", ms=8, label="hollow: start state only (p = 0)"),
        Line2D([], [], marker="o", color="black", ls="", ms=8, label="black: median over seeds, bar = IQR"),
        Line2D([], [], marker="o", color="black", alpha=0.5, ls="", ms=6, label="light: one seed")],
        loc="center left", fontsize=12, labelspacing=0.9)

    def med(m, c):
        return float(runs[runs.method == m][c].median())

    q = runs[runs.family == "qaoa"].groupby("method")["p_within_1pct"].median()
    best = q.idxmax()
    base_m = "p=0 Dicke start (XY start)" if best.startswith("QAOA XY") else "p=0 uniform random (X start)"
    b0 = med(base_m, "p_within_1pct")
    n_seeds = int(runs[runs.family == "qaoa"].groupby("method").size().max())
    qaoa_rows = runs[runs.family == "qaoa"]
    restarts = int(qaoa_rows.n_restarts.max()) if "n_restarts" in runs else 1
    alpha = float(qaoa_rows.cvar_alpha.max()) if "cvar_alpha" in runs else float("nan")
    mixer_name = "XY-mixer" if best.startswith("QAOA XY") else "X-mixer"
    p_best = best.split()[-1].replace("p=", "p = ")
    title = f"{mixer_name} QAOA ({p_best}) finds near-optimal schedules {q.max() / b0:.0f}x more often than its start state"
    caption = (f"Main night Oct 8 to 9, 16 qubits, 4096 shots, n = {n_seeds} seeds per QAOA configuration "
               f"(CVaR training, alpha = {alpha:g}, {restarts} restarts). Near-optimal: {q.max():.3f} for {best} vs "
               f"{b0:.4f} at p = 0; expected-cost ratio {med(best, 'approx_ratio_expected'):.2f} vs "
               f"{med(base_m, 'approx_ratio_expected'):.2f}.")
    top = S.heading(fig, title, caption)
    fig.subplots_adjust(top=top - 0.02, bottom=0.08, left=0.075, right=0.985)
    S.save(fig, out_png)
    return (f"Dot plots of P(feasible), P(within 1 % of optimum) and expected-cost ratio for the start states (p = 0) and "
            f"QAOA with the X and XY mixer at p = 1 to 3 on the main night, one dot per seed with median and interquartile range. "
            f"{title}. Expected-cost ratio {med(best, 'approx_ratio_expected'):.2f} vs {med(base_m, 'approx_ratio_expected'):.2f}.")


def append_alt(png: Path, text: str) -> None:
    lines = ALT.read_text(encoding="utf-8").splitlines(keepends=True) if ALT.exists() else ["# Figure alt text\n\n"]
    lines = [ln for ln in lines if f"`{png.name}`" not in ln]
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.append(f"- `{png.name}`: {text}\n")
    ALT.write_text("".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    global RESULTS, ALT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--night", default="2026-10-08")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--restarts", type=int, default=3,
                    help="QAOA training restarts (restart 0 deterministic, others random from the seed)")
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds, reps=1 only, maxiter 30, 1 restart; outputs go to results/scratch")
    ap.add_argument("--out-dir", default=None, help="output directory (default results/)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw the figure from results/e1_main_night.csv (no runs)")
    args = ap.parse_args(argv)
    if args.quick:
        args.seeds = min(args.seeds, 2)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    RESULTS = Path(args.out_dir) if args.out_dir else (ROOT / "results" / "scratch" if args.quick else ROOT / "results")
    ALT = RESULTS / "figures_alt_text.md"
    RESULTS.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.replot:
        runs = pd.read_csv(RESULTS / "e1_main_night.csv")
        runs["mixer"] = runs["mixer"].fillna("").astype(str)
        alt = make_figure(runs, RESULTS / "e1_main_night.png")
        append_alt(RESULTS / "e1_main_night.png", alt)
        print(f"replotted {RESULTS / 'e1_main_night.png'}")
        return 0
    runs = run(args)
    summ = summarize(runs)
    runs.to_csv(RESULTS / "e1_main_night.csv", index=False, float_format="%.6g")
    summ.to_csv(RESULTS / "e1_summary.csv", index=False, float_format="%.6g")
    print()
    print_summary(summ)
    print("\napx.exp = optimum / mean objective of feasible samples; apx.best = optimum / best feasible sample "
          "(near 1 for any sampler with 4096 shots over only 500 feasible schedules).")
    if not args.no_plot:
        alt = make_figure(runs, RESULTS / "e1_main_night.png")
        append_alt(RESULTS / "e1_main_night.png", alt)
        print(f"Figure: {RESULTS / 'e1_main_night.png'}")
    print(f"Total runtime: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
