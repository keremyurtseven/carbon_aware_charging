"""E6: scaling study - penalty (X mixer) vs XY encoding as the fleet grows (11 -> 16 -> 20 qubits).

Modelled fleets (assumptions, not data; only carbon intensity and price are measured),
all on the main night (default 2026-10-08, w = 0.7):

  2 EVs  A (11 kW, 2 blocks, blocks 0-5), B (9.2 kW, 1 block, blocks 0-4)               P_max = 19 kW  (11 qubits)
  3 EVs  + C (7.4 kW, 2 blocks, blocks 1-5)  = problem.DEFAULT_FLEET                   P_max = 19 kW  (16 qubits)
  4 EVs  + D (7.4 kW, 1 block, blocks 2-5)                                              P_max = 16 kW  (20 qubits)

P_max = 16 kW for the 4-EV fleet because with 19 kW the triple A+C+D (25.8 kW) is over the limit although no
pair is, so the site limit would not be a pairwise conflict (the script REQUIRES
``site_limit_is_pairwise_exact`` and stops otherwise). At 16 kW the pairs A+B, A+C, A+D, B+C conflict.

Per instance: the exact optimum, on-arrival and the gap to the second best are printed and stored; then for
mixer in {x, xy} and reps in {1, 2, 3} (seed 0, ``TRAIN_CONFIG``, n_restarts = 1; training is deterministic)
the QAOA metrics, plus the p = 0 baselines. Time budget: after a run, the next (larger) reps is skipped
if 1.5 x that run's wall time exceeds ``--max-run-s`` (default 180 s); skipped rows are written with
``status = skipped_time_budget`` and empty metrics.

Outputs (results/, or results/scratch/ with --quick):
  e6_scaling.csv   one row per (instance, mixer, reps)      e6_summary.csv   same plus gain over p = 0
  e6_scaling.png / .svg

    .venv/Scripts/python experiments/e6_scaling.py
    .venv/Scripts/python experiments/e6_scaling.py --quick
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
from carbon_ev.classical import brute_force, on_arrival  # noqa: E402
from carbon_ev.data import load_hourly  # noqa: E402
from carbon_ev.problem import (DEFAULT_FLEET, EV, build_instance, evaluate,  # noqa: E402
                               site_limit_is_pairwise_exact, var_index)
from carbon_ev.qaoa import TRAIN_CONFIG, random_feasible_baseline, run_qaoa  # noqa: E402

MAIN_NIGHT = "2026-10-08"
FIG_NAME = "e6_scaling"
FLOAT_FMT = "%.6g"

_A, _B, _C = DEFAULT_FLEET
_D = EV("D", 7.4, 1, (2, 3, 4, 5))
FLEETS = {  # name -> (fleet, P_max kW)
    "2EV": ([_A, _B], 19.0),
    "3EV": ([_A, _B, _C], 19.0),
    "4EV": ([_A, _B, _C, _D], 16.0),
}


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


def describe(fleet) -> str:
    return "; ".join(f"{e.name} {e.power_kw:g} kW x{e.k_blocks} blocks in {min(e.available)}-{max(e.available)}"
                     for e in fleet)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="2-EV and 3-EV fleets, reps 1, maxiter 40; writes to results/scratch/")
    ap.add_argument("--night", default=MAIN_NIGHT)
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--max-run-s", type=float, default=180.0,
                    help="skip the next reps if 1.5 x the last run's wall time exceeds this (0 = no limit)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--replot", action="store_true", help="redraw figure and summary from results/e6_scaling.csv (no runs)")
    a = ap.parse_args()
    reps_list, shots, fleets, maxiter = a.reps, a.shots, dict(FLEETS), None
    if a.quick:
        reps_list, shots, maxiter = [1], 512, 40
        fleets = {k: v for k, v in FLEETS.items() if k != "4EV"}

    out_dir = a.out_dir or (ROOT / "results" / ("scratch" if a.quick else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    night = date.fromisoformat(a.night)
    df = load_hourly()
    cfg = dict(TRAIN_CONFIG, **({"maxiter": maxiter} if maxiter else {}))

    t_start = time.perf_counter()
    if a.replot:
        res = pd.read_csv(out_dir / "e6_scaling.csv")
    else:
        rows: list[dict] = []
        for name, (fleet, p_max) in fleets.items():
            if not site_limit_is_pairwise_exact(fleet, p_max):
                sys.exit(f"{name}: site limit {p_max} kW is not pairwise exact for this fleet; choose another P_max")
            inst = build_instance(df, night, fleet=fleet, p_max_kw=p_max)
            n_q = len(var_index(inst))
            _, feas = brute_force(inst)
            objs = np.array([o for _, o in feas])
            opt = float(objs[0])
            oa = evaluate(inst, on_arrival(inst))["objective"]
            gap2 = float((objs[1] - objs[0]) / objs[0] * 100) if len(objs) > 1 else float("nan")
            inst_info = dict(instance=name, n_evs=len(fleet), n_qubits=n_q, p_max_kw=p_max, night=str(night), w=inst.w,
                             hypothetical_fleet=True, fleet=describe(fleet), n_feasible=len(feas),
                             opt_objective=opt, on_arrival_objective=oa, on_arrival_vs_opt_pct=(oa / opt - 1) * 100,
                             gap_to_second_best_pct=gap2, median_feasible_objective=float(np.median(objs)),
                             pairwise_exact=True)
            print(f"\n{name}: {n_q} qubits, {len(feas)} feasible, optimum {opt:.3f}, on-arrival {oa:.3f} "
                  f"(+{(oa / opt - 1) * 100:.1f} %), gap to 2nd best {gap2:.3f} %, P_max {p_max:g} kW", flush=True)

            base = random_feasible_baseline(inst, shots=shots, seed=0, opt_obj=opt)
            for mixer, key in (("x", "uniform"), ("xy", "demand_satisfying")):
                m0 = base[key]
                rows.append(dict(inst_info, mixer=mixer, reps=0, status="p0_start_state", feasible_prob=m0["feasible_prob"],
                                 approx_ratio=m0["approx_ratio"], approx_ratio_expected=m0["approx_ratio_expected"],
                                 p_opt=m0["p_opt"], p_within_1pct=m0["p_within_1pct"], depth=0, n_2q=0, n_evals=0, wall_s=0.0))
            for mixer in ("x", "xy"):
                last_wall = 0.0
                for reps in reps_list:
                    if a.max_run_s and 1.5 * last_wall > a.max_run_s and last_wall > 0:
                        rows.append(dict(inst_info, mixer=mixer, reps=reps, status="skipped_time_budget"))
                        print(f"  {mixer} p={reps}: skipped (previous run {last_wall:.0f}s, budget {a.max_run_s:.0f}s)", flush=True)
                        continue
                    r = run_qaoa(inst, mixer, reps, shots=shots, seed=0, opt_obj=opt, n_restarts=1, **cfg)
                    last_wall = r.wall_s
                    rows.append(dict(inst_info, mixer=mixer, reps=reps, status="ok", feasible_prob=r.feasible_prob,
                                     approx_ratio=r.approx_ratio, approx_ratio_expected=r.approx_ratio_expected,
                                     p_opt=r.p_opt, p_within_1pct=r.p_within_1pct, depth=r.depth, n_2q=r.n_2q,
                                     n_evals=r.n_evals, wall_s=r.wall_s, train_objective=cfg["objective"],
                                     train_cvar_alpha=cfg["cvar_alpha"], train_init=cfg["init"],
                                     train_maxiter=cfg["maxiter"], n_restarts=1, seed=0))
                    print(f"  {mixer} p={reps}: {r.wall_s:.0f}s evals={r.n_evals} P(feas)={r.feasible_prob:.3f} "
                          f"ratio_exp={r.approx_ratio_expected:.3f} P(<=1%)={r.p_within_1pct:.4f} 2q={r.n_2q}", flush=True)
        res = pd.DataFrame(rows)
        res.to_csv(out_dir / "e6_scaling.csv", index=False, float_format=FLOAT_FMT)

    # ---- summary: gain over the p = 0 start state of the same encoding and instance
    summ = res.copy()
    p0 = res[res.reps == 0].set_index(["instance", "mixer"])
    for m in ("feasible_prob", "approx_ratio_expected", "p_within_1pct"):
        summ[f"p0_{m}"] = [p0.loc[(i, mx), m] for i, mx in zip(summ.instance, summ.mixer)]
        summ[f"gain_{m}"] = summ[m] - summ[f"p0_{m}"]
    keep = ["instance", "n_evs", "n_qubits", "n_feasible", "on_arrival_vs_opt_pct", "gap_to_second_best_pct", "mixer",
            "reps", "status", "feasible_prob", "p_within_1pct", "approx_ratio_expected", "gain_feasible_prob",
            "gain_p_within_1pct", "gain_approx_ratio_expected", "depth", "n_2q", "n_evals", "wall_s"]
    summ[keep].to_csv(out_dir / "e6_summary.csv", index=False, float_format=FLOAT_FMT)

    # ---- figure
    S.apply()
    fig, axes = plt.subplots(2, 2, figsize=(S.FIG_WIDTH, 9.0))
    style = {1: ("o", "-"), 2: ("s", "--"), 3: ("^", ":")}
    mcol = {"x": S.X, "xy": S.XY}
    panels = (("feasible_prob", "P(feasible)", "(a) Feasible samples", False),
              ("p_within_1pct", "P(within 1 % of optimum)", "(b) Near-optimal samples", True),
              ("approx_ratio_expected", "Expected-cost ratio (1 = optimal)", "(c) Expected-cost ratio", False),
              ("n_2q", "Two-qubit gates (CX)", "(d) Circuit size", False))
    ok = res[res.status == "ok"]
    p0r = res[res.reps == 0]
    for ax, (metric, ylabel, ttl, logy) in zip(axes.ravel(), panels):
        for mixer in ("x", "xy"):
            for reps in sorted(ok.reps.unique()):
                s = ok[(ok.mixer == mixer) & (ok.reps == reps)].sort_values("n_qubits")
                mk, ls = style.get(reps, ("D", "-."))
                ax.plot(s.n_qubits, s[metric], color=mcol[mixer], marker=mk, ls=ls, lw=1.8, ms=8,
                        mfc=mcol[mixer] if mixer == "xy" else "white", mew=1.6)
            if metric != "n_2q":
                s0 = p0r[p0r.mixer == mixer].sort_values("n_qubits")
                ax.plot(s0.n_qubits, s0[metric], color="black", marker="x", ls="-" if mixer == "xy" else "--", lw=1.0,
                        ms=7, alpha=0.7)
        ax.set_xticks(sorted(res.n_qubits.unique()))
        ax.set_xlabel("Number of qubits (fleet of 2, 3, 4 EVs)")
        ax.set_ylabel(ylabel)
        S.panel_title(ax, ttl)
        if logy:
            ax.set_yscale("log")
    handles = [Line2D([], [], color=S.X, marker="o", mfc="white", mew=1.6, ls="", ms=8, label="X mixer (penalty QUBO)"),
               Line2D([], [], color=S.XY, marker="o", ls="", ms=8, label="XY mixer (Dicke start)")]
    handles += [Line2D([], [], color="0.3", marker=style[r][0], ls=style[r][1], label=f"p = {r}") for r in sorted(ok.reps.unique())]
    handles += [Line2D([], [], color="black", marker="x", ls="--", label="p = 0, uniform start (X)"),
                Line2D([], [], color="black", marker="x", ls="-", label="p = 0, Dicke start (XY)")]

    def val(mixer, metric, n):
        r = ok[(ok.mixer == mixer) & (ok.n_qubits == n)]
        r = r[r.reps == r.reps.max()]
        return (float(r[metric].iloc[0]), int(r.reps.iloc[0])) if len(r) else (float("nan"), 0)

    # headline: the instance where the XY mixer's feasibility advantage at its largest p is greatest
    sizes = sorted(res.n_qubits.unique())
    n_best = max(sizes, key=lambda n: val("xy", "feasible_prob", n)[0] - val("x", "feasible_prob", n)[0])
    n_big = sizes[-1]
    fx, px = val("x", "feasible_prob", n_best)
    fxy, pxy = val("xy", "feasible_prob", n_best)
    bx, bpx = val("x", "feasible_prob", n_big)
    bxy, bpxy = val("xy", "feasible_prob", n_big)
    title = f"At {n_best} qubits the XY mixer samples feasible schedules {fxy * 100:.0f} % of the time, vs {fx * 100:.0f} % for the X mixer"
    caption = (f"Modelled fleets of 2, 3 and 4 EVs on the main night, one deterministic training run (seed 0) per point; "
               f"headline at p = {pxy}. At {n_big} qubits (p = {bpxy}) the XY mixer samples {bxy * 100:.0f} % feasible schedules "
               f"and the X mixer {bx * 100:.0f} %.")
    top = S.heading(fig, title, caption)
    fig.subplots_adjust(top=top - 0.05, bottom=0.19, left=0.08, right=0.985, hspace=0.38, wspace=0.27)
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=11)
    S.save(fig, out_dir / FIG_NAME)
    if not a.quick:
        set_alt_text(out_dir, FIG_NAME + ".png",
                     f"{title}. Four line panels against the number of qubits (11, 16 and 20 for fleets of 2, 3 and 4 EVs): "
                     "probability of a feasible sample, probability of a sample within 1 percent of the optimum (log axis), "
                     "expected-cost ratio, and two-qubit gate count. Blue hollow markers are the penalty X mixer and vermilion "
                     "filled markers the XY mixer; marker shape and line style give the depth p = 1, 2, 3, and black crosses "
                     "show the untrained start states.")

    print(f"\ntotal {time.perf_counter() - t_start:.0f}s")
    show = summ[keep]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out_dir / 'e6_scaling.csv'}, e6_summary.csv, {FIG_NAME}.png/.svg")


if __name__ == "__main__":
    main()
