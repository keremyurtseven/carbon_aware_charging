"""e8: QAOA as the subproblem solver inside the real 100-bus depot (hybrid LNS).

Question: what does QAOA deliver when it is used where it fits, as the solver of small
subproblems inside a large-neighbourhood search (LNS) over the whole depot?

Setup (all numbers are written to results/e8_*.csv, none are typed here):
  * fleet: ``depot_fleet()`` (modelled on public figures), carbon-first w = 1.0
  * nights: 2026-10-08 (windy, main night) and 2026-10-03 (calm, the e4 training night)
  * connection: the first-come-first-served cap ``min_cap_on_arrival_kw``; start schedule x0 =
    ``on_arrival_capacity`` at that cap (the FCFS start)
  * reference: global optimum of ``milp_depot`` on the same connection (proven gap recorded)
  * LNS (``carbon_ev.hybrid.lns_depot``): groups of 2 buses (at most 16 qubits), max_passes = 3,
    subsolvers "exact" (enumeration), "qaoa_xy" (Dicke start + ring XY mixer) and "qaoa_x"
    (penalty QUBO, X mixer); a candidate is accepted only if it is feasible and strictly cheaper

Notes (also recorded in the CSV columns ``qaoa_simulated_classically`` and ``note``)
  * QAOA here is SIMULATED CLASSICALLY. Angles are trained on the numpy ``ExactSimulator``
    (tested against Qiskit's Statevector); the 4096 shots are drawn from its exact output
    probabilities (``sample_with="exact"``). A cross-check re-runs one pass on the main night with the
    real Qiskit circuit through ``StatevectorSampler`` (``sample_with="sampler"``) to show that both
    paths agree statistically.
  * The "exact" subsolver is the reference: enumeration solves every 2-bus subproblem
    optimally in milliseconds, so it shows the best result any subsolver can reach.
  * Training inside LNS uses COBYLA maxiter = 200 per subproblem (TRAIN_CONFIG uses 1000 elsewhere),
    CVaR 0.25, ramp start, p = 2 layers.
  * The QAOA runs are executed in parallel worker processes, so the per-subproblem times are
    contended wall-clock seconds on one laptop: read orders of magnitude, not exact values.

Usage:
    python experiments/e8_hybrid_depot.py            # full run (about 20-30 min with 6 workers)
    python experiments/e8_hybrid_depot.py --quick    # tiny fleet, 1 seed, 1 pass -> results/scratch
    python experiments/e8_hybrid_depot.py --replot   # redraw the figure from the CSVs
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _style as S  # noqa: E402
from carbon_ev.data import load_hourly  # noqa: E402
from carbon_ev.depot import depot_fleet, milp_depot, min_cap_on_arrival_kw, on_arrival_capacity  # noqa: E402
from carbon_ev.hybrid import lns_depot  # noqa: E402
from carbon_ev.problem import build_instance, evaluate, objective_coefficients  # noqa: E402

FIG = "e8_hybrid_depot"
NIGHTS = {"2026-10-08": "windy main night", "2026-10-03": "calm night (e4 training night)"}
W = 1.0
NOTE = ("QAOA simulated classically (numpy ExactSimulator training, shots drawn from exact probabilities unless "
        "sample_with=sampler); exact enumeration is the reference subsolver")
COL = {"exact": S.EXACT, "qaoa_xy": S.XY, "qaoa_x": S.X}
MK = {"exact": "o", "qaoa_xy": "s", "qaoa_x": "^"}
LS = {"exact": "-", "qaoa_xy": "--", "qaoa_x": ":"}
LAB = {"exact": "exact enumeration", "qaoa_xy": "QAOA XY (simulated)", "qaoa_x": "QAOA X (simulated)"}


# ----------------------------------------------------------------------------- one run
def _instance(df, night: str, cap: float, n_buses: int):
    fleet = depot_fleet(n_buses=n_buses)
    return build_instance(df, date.fromisoformat(night), fleet=fleet, n_blocks=12, p_max_kw=cap, w=W)


def run_one(task: dict) -> tuple[dict, list[dict]]:
    """Run one LNS configuration. Returns (per-run row, per-subproblem trace rows)."""
    df = load_hourly()
    inst = _instance(df, task["night"], task["cap_kw"], task["n_buses"])
    x0 = on_arrival_capacity(inst)
    res = lns_depot(inst, x0, subsolver=task["subsolver"], group_size=2, max_passes=task["max_passes"],
                    seed=task["seed"], sample_with=task["sample_with"], shots=4096)
    ev = evaluate(inst, res.x)
    ev0 = evaluate(inst, x0)
    obj_ref = task["milp_objective"]
    start, final = res.objective_initial, res.objective_final
    row = dict(
        night=task["night"], night_kind=NIGHTS.get(task["night"], "quick"), purpose=task["purpose"],
        subsolver=task["subsolver"], seed=task["seed"], sample_with=task["sample_with"],
        w=W, n_buses=task["n_buses"], cap_kw=task["cap_kw"], group_size=2, max_passes=task["max_passes"],
        train_maxiter=200, shots=4096, workers=task["workers"],
        objective_fcfs_start=start, objective_final=final, objective_milp=obj_ref,
        milp_proven_gap=task["milp_gap"],
        gap_to_milp_pct=100 * (final - obj_ref) / obj_ref,
        gap_closed_pct=100 * (start - final) / (start - obj_ref),
        co2_kg=ev["co2_kg"], cost_eur=ev["cost_eur"], co2_kg_fcfs=ev0["co2_kg"], cost_eur_fcfs=ev0["cost_eur"],
        co2_kg_milp=task["milp_co2_kg"], cost_eur_milp=task["milp_cost_eur"],
        co2_saved_vs_fcfs_kg=ev0["co2_kg"] - ev["co2_kg"], co2_excess_vs_milp_kg=ev["co2_kg"] - task["milp_co2_kg"],
        n_subproblems=res.n_subproblems, n_improved=res.n_improved, passes_run=res.passes_run,
        improvements_per_pass=";".join(str(v) for v in res.improvements_per_pass),
        n_trivial=res.calls["trivial"], wall_s=res.wall_s,
        qaoa_simulated_classically=task["subsolver"] != "exact", note=NOTE,
    )
    for k in ("n_calls", "n_feasible_improvement", "feasible_improvement_rate", "n_fallback", "n_exact_improvable",
              "mean_feasible_prob", "mean_p_within_1pct", "mean_p_opt", "mean_baseline_p_within_1pct",
              "mean_gap_to_sub_opt"):
        row[f"qaoa_{k}"] = res.qaoa.get(k, np.nan)
    # trace: objective after each subproblem; seconds per subproblem from the log
    trace = [dict(night=task["night"], purpose=task["purpose"], subsolver=task["subsolver"], seed=task["seed"],
                  sample_with=task["sample_with"], index=0, pass_index=-1, kind="start", objective=res.trace[0],
                  seconds=np.nan, improved=False)]
    for k, rec in enumerate(res.log, start=1):
        trace.append(dict(night=task["night"], purpose=task["purpose"], subsolver=task["subsolver"],
                          seed=task["seed"], sample_with=task["sample_with"], index=k, pass_index=rec["pass_"],
                          kind=rec["kind"], objective=res.trace[k], seconds=rec["seconds"],
                          improved=bool(rec["improved"])))
    return row, trace


# ----------------------------------------------------------------------------- summary / plot
MIN_GAP_REL = 0.01   # "gap closed" is only meaningful if the FCFS start is >= 1 % above the MILP reference


def add_validity(runs: pd.DataFrame) -> pd.DataFrame:
    """Flag per run whether gap_closed_pct is meaningful (start-to-reference gap >= 1 % of the reference)."""
    runs = runs.copy()
    runs["gap_metric_valid"] = ((runs.objective_fcfs_start - runs.objective_milp) / runs.objective_milp) >= MIN_GAP_REL
    return runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    runs = add_validity(runs)
    metrics = ["objective_final", "gap_to_milp_pct", "gap_closed_pct", "co2_kg", "cost_eur", "co2_saved_vs_fcfs_kg",
               "co2_excess_vs_milp_kg", "n_subproblems", "n_improved", "wall_s", "qaoa_feasible_improvement_rate",
               "qaoa_mean_feasible_prob", "qaoa_mean_p_within_1pct", "qaoa_mean_baseline_p_within_1pct",
               "qaoa_n_fallback", "qaoa_mean_gap_to_sub_opt"]
    out = []
    for (purpose, night, sub, sw), g in runs.groupby(["purpose", "night", "subsolver", "sample_with"], sort=False):
        rec = dict(purpose=purpose, night=night, subsolver=sub, sample_with=sw, n_runs=len(g),
                   seeds=",".join(str(s) for s in sorted(g.seed)), max_passes=int(g.max_passes.iloc[0]),
                   objective_fcfs_start=g.objective_fcfs_start.iloc[0], objective_milp=g.objective_milp.iloc[0],
                   milp_proven_gap=g.milp_proven_gap.iloc[0], qaoa_simulated_classically=bool(sub != "exact"))
        valid = bool(g.gap_metric_valid.all())
        rec["gap_metric_valid"] = valid
        rec["gap_metric_note"] = ("" if valid else "FCFS start is within 1 % of the MILP reference: gap_closed_pct is not "
                                  "meaningful (set to NaN); see co2_saved_vs_fcfs_kg and gap_to_milp_pct instead")
        by_seed = g.sort_values("seed")
        rec["by_seed_co2_kg"] = ";".join(f"{int(r.seed)}:{r.co2_kg:.1f}" for r in by_seed.itertuples())
        rec["by_seed_gap_closed_pct"] = (";".join(f"{int(r.seed)}:{r.gap_closed_pct:.2f}" for r in by_seed.itertuples())
                                         if valid else "")
        ex = runs[(runs.purpose == purpose) & (runs.night == night) & (runs.subsolver == "exact")]
        same = ex[ex.seed.isin(g.seed)].sort_values("seed")
        rec["exact_same_seed_co2_kg"] = ";".join(f"{int(r.seed)}:{r.co2_kg:.1f}" for r in same.itertuples()) if sub != "exact" else ""
        for m in metrics:
            v = g[m].dropna()
            if m == "gap_closed_pct" and not valid:
                v = v.iloc[0:0]
            rec[f"{m}_median"] = float(v.median()) if len(v) else np.nan
            rec[f"{m}_q25"] = float(v.quantile(0.25)) if len(v) else np.nan
            rec[f"{m}_q75"] = float(v.quantile(0.75)) if len(v) else np.nan
        rec["sec_per_subproblem_mean"] = float((g.wall_s / g.n_subproblems).mean())   # wall_s / n_subproblems, mean over seeds
        rec["note"] = NOTE
        out.append(rec)
    return pd.DataFrame(out)


def make_figure(runs: pd.DataFrame, trace: pd.DataFrame, out_base: Path) -> str:
    """CO2 trajectory of the LNS per subsolver on two nights, and the time per subproblem."""
    S.apply()
    from matplotlib.lines import Line2D
    main_runs = runs[runs.purpose == "main"]
    nights = [n for n in NIGHTS if n in set(main_runs.night)] or sorted(main_runs.night.unique())
    windy_night = nights[0]
    fig = plt.figure(figsize=(S.FIG_WIDTH, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.27)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])][:len(nights)]
    for ax, night, letter in zip(axes, nights, "ab"):
        r = main_runs[main_runs.night == night]
        k_kg = float(r.co2_kg_fcfs.iloc[0] / r.objective_fcfs_start.iloc[0])      # kg CO2 per objective unit (w = 1)
        for sub in ("exact", "qaoa_xy", "qaoa_x"):
            ts = trace[(trace.purpose == "main") & (trace.night == night) & (trace.subsolver == sub)]
            for j, (seed, g) in enumerate(ts.groupby("seed")):
                g = g.sort_values("index")
                ax.plot(g["index"], g.objective * k_kg, color=COL[sub], ls=LS[sub], lw=1.8 if j == 0 else 1.0,
                        alpha=1.0 if j == 0 else 0.45, marker=MK[sub], markevery=max(len(g) // 8, 1), ms=6,
                        mfc=COL[sub] if sub != "qaoa_x" else "white")
        ax.axhline(r.co2_kg_fcfs.iloc[0], color="black", ls="-.", lw=1.4)
        ax.axhline(r.co2_kg_milp.iloc[0], color=S.GREY, ls=(0, (1, 1)), lw=2.0)
        if night != windy_night:
            saved = float(r[r.subsolver == "exact"].co2_saved_vs_fcfs_kg.median())
            ax.text(0.98, 0.9, f"Only ~{saved:.0f} kg CO₂ to gain on this calm night;" + chr(10)
                    + "the LNS ends within the MILP tolerance",
                    transform=ax.transAxes, ha="right", va="top", fontsize=11,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.set_xlabel("LNS subproblem index (2-bus groups, up to 3 passes)")
        ax.set_ylabel("Depot CO₂ of the night (kg)")
        S.panel_title(ax, f"({letter}) {night}: {NIGHTS.get(night, '').split(' (')[0]}")
    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    subs = [s_ for s_ in ("exact", "qaoa_xy", "qaoa_x") if s_ in set(main_runs.subsolver)]
    # seconds per subproblem = wall_s / n_subproblems of a run (includes the subproblem construction and,
    # for QAOA, the diagnostic enumeration); summaries use the mean over seeds per night
    main_runs = main_runs.assign(sec_per_sub=(main_runs.wall_s / main_runs.n_subproblems).to_numpy())
    for k, sub in enumerate(subs):
        v = main_runs[main_runs.subsolver == sub].sec_per_sub.to_numpy()
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=50, alpha=0.65, color=COL[sub], marker=MK[sub],
                   facecolors=COL[sub] if sub != "qaoa_x" else "none", edgecolors=COL[sub], linewidths=1.5)
        ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color="black", lw=2.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(subs)), [LAB[s_].replace(" (simulated)", "\n(simulated)").replace("exact enumeration", "exact\nenumeration")
                                      for s_ in subs])
    ax.set_xlim(-0.6, len(subs) - 0.4)
    ax.set_ylabel("Seconds per subproblem (log scale)")
    S.panel_title(ax, "(c) Time per subproblem")
    ax.grid(axis="x", visible=False)
    key = fig.add_subplot(gs[1, 1])
    key.axis("off")
    key.legend(handles=[
        Line2D([], [], color=COL["exact"], ls=LS["exact"], marker=MK["exact"], label=LAB["exact"]),
        Line2D([], [], color=COL["qaoa_xy"], ls=LS["qaoa_xy"], marker=MK["qaoa_xy"], label=LAB["qaoa_xy"]),
        Line2D([], [], color=COL["qaoa_x"], ls=LS["qaoa_x"], marker=MK["qaoa_x"], mfc="white", label=LAB["qaoa_x"]),
        Line2D([], [], color="black", ls="-.", label="FCFS start (5.4 MW cap)"),
        Line2D([], [], color=S.GREY, ls=(0, (1, 1)), lw=2.0, label="MILP reference (within 0.2 %)"),
        Line2D([], [], color="black", alpha=0.45, lw=1.0, label="thin lines: other seeds")],
        loc="center left", fontsize=12, labelspacing=0.9)

    def closed(sub):  # windy night only: on the calm night the FCFS-to-optimum gap is ~0 and the ratio is not meaningful
        v = main_runs[(main_runs.subsolver == sub) & (main_runs.night == windy_night)].gap_closed_pct
        return float(v.median()) if len(v) else float("nan")

    med_t = {s_: float(main_runs[(main_runs.subsolver == s_) & (main_runs.night == windy_night)].sec_per_sub.mean())
             for s_ in subs}
    lo = min(closed(q) for q in ("exact", "qaoa_xy", "qaoa_x"))
    hi = max(closed(q) for q in ("exact", "qaoa_xy", "qaoa_x"))
    title = f"QAOA subsolvers inside the 100-bus depot close {lo:.0f}-{hi:.0f} % of the CO₂ gap, like exact enumeration"
    caption = (f"Carbon-first (w = 1), 100 buses, 2-bus subproblems of at most 16 qubits. Windy night, mean seconds per subproblem: "
               f"exact {med_t.get('exact', float('nan')) * 1000:.1f} ms, QAOA-XY {med_t.get('qaoa_xy', float('nan')):.1f} s, "
               f"QAOA-X {med_t.get('qaoa_x', float('nan')):.1f} s (QAOA simulated on a statevector simulator, parallel workers).")
    top = S.heading(fig, title, caption)
    fig.subplots_adjust(top=top - 0.06, bottom=0.08, left=0.085, right=0.985)
    S.save(fig, out_base)
    return (f"{title}. Top row: depot CO₂ in kg against the index of the LNS subproblem for the windy main night and a calm night, "
            f"one line per subsolver (exact enumeration solid green with circles, QAOA XY dashed vermilion with squares, QAOA X "
            f"dotted blue with hollow triangles; thin lines other seeds), a dash-dotted line for the first-come-first-served start "
            f"and a dotted grey line for the MILP reference. Bottom left: seconds per subproblem on a log axis; exact enumeration "
            f"takes milliseconds, simulated QAOA seconds.")


def set_alt_text(path: Path, name: str, text: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else ["# Figure alt text\n\n"]
    lines = [ln for ln in lines if f"`{name}`" not in ln]
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.append(f"- `{name}`: {text}\n")
    path.write_text("".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="12-bus fleet, main night, seed 0, 1 pass; writes to results/scratch")
    ap.add_argument("--qaoa-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--exact-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--x-seeds", type=int, nargs="+", default=None, help="seeds for qaoa_x (default: same as --qaoa-seeds)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    out = a.out_dir or (ROOT / "results" / ("scratch" if a.quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    if a.replot:
        runs = pd.read_csv(out / "e8_hybrid_depot.csv")
        trace = pd.read_csv(out / "e8_trace.csv")
        runs = add_validity(runs)
        runs.to_csv(out / "e8_hybrid_depot.csv", index=False)
        summarize(runs).to_csv(out / "e8_summary.csv", index=False)
        alt = make_figure(runs, trace, out / FIG)
        set_alt_text(out / "figures_alt_text.md", FIG + ".png", alt)
        print(f"replotted {out / (FIG + '.png')}")
        return

    t0 = time.time()
    df = load_hourly()
    n_buses = 12 if a.quick else 100
    nights = ["2026-10-08"] if a.quick else list(NIGHTS)
    max_passes = 1 if a.quick else 3
    exact_seeds = [0] if a.quick else a.exact_seeds
    q_seeds = [0] if a.quick else a.qaoa_seeds
    x_seeds = q_seeds if a.x_seeds is None else a.x_seeds
    tasks = []
    for night in nights:
        free = _instance(df, night, 1e12, n_buses)
        cap = min_cap_on_arrival_kw(free)
        inst = _instance(df, night, cap, n_buses)
        x_opt, gap = milp_depot(inst)
        ev_opt = evaluate(inst, x_opt)
        obj_opt = float(objective_coefficients(inst) @ x_opt)
        base = dict(night=night, cap_kw=cap, n_buses=n_buses, milp_objective=obj_opt, milp_gap=gap,
                    milp_co2_kg=ev_opt["co2_kg"], milp_cost_eur=ev_opt["cost_eur"], workers=a.workers)
        print(f"{night}: cap {cap / 1000:.2f} MW, MILP optimum {ev_opt['co2_kg']:.0f} kg CO2 (proven gap {gap:.4f}), "
              f"FCFS start {evaluate(inst, on_arrival_capacity(inst))['co2_kg']:.0f} kg", flush=True)
        for sub, seeds in (("qaoa_x", x_seeds), ("qaoa_xy", q_seeds), ("exact", exact_seeds)):   # slow ones first
            for s in seeds:
                tasks.append(dict(base, subsolver=sub, seed=s, sample_with="exact", max_passes=max_passes,
                                  purpose="main"))
    main_night = nights[0]
    for s in ([0] if a.quick else q_seeds[:2]):            # cross-check: real circuit vs ExactSimulator, one pass
        for sw in ("sampler", "exact"):
            b = next(t for t in tasks if t["night"] == main_night)
            tasks.insert(0, dict(b, subsolver="qaoa_xy", seed=s, sample_with=sw, max_passes=1, purpose="cross_check"))
    print(f"{len(tasks)} runs on {a.workers} workers", flush=True)
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (row, tr) in enumerate(ex.map(run_one, tasks), 1):
            rows.append(row)
            traces += tr
            print(f"[{k}/{len(tasks)}] {row['purpose']} {row['night']} {row['subsolver']} seed {row['seed']} "
                  f"({row['sample_with']}): gap closed {row['gap_closed_pct']:.1f} %, {row['n_subproblems']} subproblems, "
                  f"{row['wall_s']:.0f} s", flush=True)
    runs = add_validity(pd.DataFrame(rows))
    trace = pd.DataFrame(traces)
    runs.to_csv(out / "e8_hybrid_depot.csv", index=False)
    trace.to_csv(out / "e8_trace.csv", index=False)
    summ = summarize(runs)
    summ.to_csv(out / "e8_summary.csv", index=False)
    alt = make_figure(runs, trace, out / FIG)
    if not a.quick:
        set_alt_text(out / "figures_alt_text.md", FIG + ".png", alt)
    show = summ[["purpose", "night", "subsolver", "sample_with", "n_runs", "gap_closed_pct_median", "gap_to_milp_pct_median",
                 "co2_kg_median", "co2_saved_vs_fcfs_kg_median", "n_improved_median", "wall_s_median",
                 "qaoa_feasible_improvement_rate_median", "qaoa_mean_p_within_1pct_median",
                 "qaoa_mean_baseline_p_within_1pct_median", "qaoa_n_fallback_median"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    (out / "e8_runtime.txt").write_text(f"total_runtime_s={time.time() - t0:.1f}\nworkers={a.workers}\nruns={len(tasks)}\n",
                                        encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
