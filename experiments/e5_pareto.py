"""E5: w-sweep Pareto study (kg CO2 vs EUR), exact on every night + light QAOA on the main night.

For w = 0, 0.1, ..., 1 (CLI ``--w-step``) and every night from ``list_nights``:
the MILP optimum of the weighted objective, its kg CO2 and EUR, and the savings vs
charge-on-arrival and vs the median feasible schedule (median by objective at that w,
same convention as e2). Windy / calm uses the same carbon-swing threshold as e2 (1.3).

Scope: the classical sweep runs on EVERY night in the data (a full year); the QAOA overlay is main night only.
Summary rows carry ``nights_scope`` (``full_year_N`` for the classical rows, ``main_night_<date>`` for QAOA).
QAOA rows already present in an existing results/e5_pareto.csv are carried over unchanged unless ``--qaoa``
is given (then they are re-run); the QAOA overlay is never re-run by default.

Key question: at which w do calm nights stop emitting more CO2 than on-arrival?
(reported per night and per w in e5_summary.csv, quantity ``smallest_w_no_extra_co2``).

``--qaoa``: main night only, XY mixer, reps ``--qaoa-reps`` (default 3), w in {0, 0.5, 0.7, 1.0},
seeds 0..2 (shot-sampling replicates: training with ``TRAIN_CONFIG`` is deterministic). Adds rows with method ``qaoa_xy``: the best feasible
sample, and the mean CO2/EUR of the feasible samples ("expected feasible").

Outputs (results/, or results/scratch/ with --quick):
  e5_pareto.csv        classical rows: one per (night, w, method)
  e5_pareto_qaoa.csv   QAOA overlay rows (main night): one per (w, method, seed)
  e5_summary.csv  per (group, w) statistics across nights + crossover rows
  e5_pareto.png / .svg

    .venv/Scripts/python experiments/e5_pareto.py
    .venv/Scripts/python experiments/e5_pareto.py --qaoa
    .venv/Scripts/python experiments/e5_pareto.py --quick --qaoa
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
from carbon_ev.classical import brute_force, milp, on_arrival  # noqa: E402
from carbon_ev.data import list_nights, load_hourly  # noqa: E402
from carbon_ev.problem import build_instance, evaluate, var_index  # noqa: E402
from carbon_ev.qaoa import (TRAIN_CONFIG, counts_to_matrix, evaluate_params, run_qaoa,  # noqa: E402
                            schedule_metrics)

MAIN_NIGHT = "2026-10-08"
FIG_NAME = "e5_pareto"
W_MAIN = 0.7
QAOA_W = [0.0, 0.5, 0.7, 1.0]


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


def stats(a) -> tuple[float, float, float, float, float]:
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return (np.nan,) * 5
    return (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)),
            float(a.min()), float(a.max()))


def physical_coeffs(inst) -> tuple[np.ndarray, np.ndarray]:
    """Per-variable kg CO2 and EUR if the variable is 1 (var_index order)."""
    vi = var_index(inst)
    kwh = np.array([inst.evs[i].power_kw * inst.block_hours for i, _ in vi])
    t = np.array([t for _, t in vi])
    return kwh * inst.co2[t] / 1000.0, kwh * inst.price[t] / 1000.0


def sweep_night(df, night: date, ws: list[float], threshold: float) -> list[dict]:
    rows = []
    for w in ws:
        inst = build_instance(df, night, w=w)
        swing = float(inst.co2.max() / inst.co2.min())
        _, feas = brute_force(inst)
        med_x = feas[len(feas) // 2][0]
        ref = {"on_arrival": on_arrival(inst), "median_feasible": med_x, "milp": milp(inst)}
        ev = {k: evaluate(inst, x) for k, x in ref.items()}
        for method, e in ev.items():
            rows.append(dict(
                night=str(night), w=round(w, 4), method=method, seed=-1, windy=bool(swing >= threshold),
                carbon_swing=swing, co2_kg=e["co2_kg"], cost_eur=e["cost_eur"], objective=e["objective"],
                feasible=e["feasible"],
                saving_co2_vs_on_arrival=ev["on_arrival"]["co2_kg"] - e["co2_kg"],
                saving_eur_vs_on_arrival=ev["on_arrival"]["cost_eur"] - e["cost_eur"],
                saving_co2_vs_median=ev["median_feasible"]["co2_kg"] - e["co2_kg"],
                saving_eur_vs_median=ev["median_feasible"]["cost_eur"] - e["cost_eur"],
                nights_scope="classical_all_nights",
            ))
    return rows


def ensure_n_restarts(rows: pd.DataFrame) -> pd.DataFrame:
    """Add the ``n_restarts`` column (training restarts of the QAOA rows; empty for non-QAOA rows).

    Every QAOA run in this script uses ``run_qaoa(..., n_restarts=1)`` (restart 0 only, deterministic), so
    the seeds are sampling replicates of one training run. Old CSVs written before this column existed
    get it here (value 1) without any recomputation.
    """
    if "n_restarts" not in rows.columns:
        rows = rows.copy()
        rows["n_restarts"] = np.where(rows["method"].astype(str).str.startswith("qaoa"), 1.0, np.nan)
    return rows


QAOA_FILE = "e5_pareto_qaoa.csv"
FLOAT_FMT = "%.6g"  # 6 significant digits keeps every results/ file well under Qollab's 2 MB limit


def read_rows(out_dir: Path) -> pd.DataFrame:
    """Classical rows (e5_pareto.csv) + QAOA overlay rows (e5_pareto_qaoa.csv) as one table.

    A legacy combined e5_pareto.csv (QAOA rows inside) is still understood when no QAOA file exists.
    """
    main = pd.read_csv(out_dir / "e5_pareto.csv")
    q_path = out_dir / QAOA_FILE
    if q_path.exists():
        main = main[~main.method.astype(str).str.startswith("qaoa")]
        main = pd.concat([main, pd.read_csv(q_path)], ignore_index=True)
    return ensure_n_restarts(main)


def write_rows(rows: pd.DataFrame, out_dir: Path) -> None:
    """Write the classical rows to e5_pareto.csv and the QAOA rows to e5_pareto_qaoa.csv.

    All-empty and constant bookkeeping columns are dropped per file (``nights_scope`` lives in
    e5_summary.csv); floats are written with 6 significant digits.
    """
    is_q = rows.method.astype(str).str.startswith("qaoa")
    for part, name in ((rows[~is_q], "e5_pareto.csv"), (rows[is_q], QAOA_FILE)):
        part = part.dropna(axis=1, how="all").drop(columns=["nights_scope"], errors="ignore")
        if name == "e5_pareto.csv":  # constant in the classical rows (seed = -1, feasible = True)
            part = part.drop(columns=[c for c in ("seed", "feasible") if c in part.columns and part[c].nunique() == 1])
        if len(part) or name == "e5_pareto.csv":
            part.to_csv(out_dir / name, index=False, float_format=FLOAT_FMT)
        elif (out_dir / name).exists():
            (out_dir / name).unlink()


def qaoa_rows(df, night: date, seeds: list[int], shots: int, threshold: float, reps: int, maxiter=None) -> list[dict]:
    rows = []
    for w in QAOA_W:
        inst = build_instance(df, night, w=w)
        swing = float(inst.co2.max() / inst.co2.min())
        co2c, eurc = physical_coeffs(inst)
        base = evaluate(inst, on_arrival(inst))
        cfg = dict(TRAIN_CONFIG, **({"maxiter": maxiter} if maxiter else {}))
        tr = run_qaoa(inst, "xy", reps, shots=shots, seed=seeds[0], **cfg)  # training is deterministic
        for seed in seeds:
            r = tr if seed == seeds[0] else evaluate_params(inst, "xy", reps, tr.params, shots=shots, seed=seed)
            X, wt = counts_to_matrix(r.counts)
            _, _, feas = schedule_metrics(inst, X)
            common = dict(night=str(night), w=round(w, 4), seed=seed, windy=bool(swing >= threshold),
                          nights_scope=f"main_night_{night}",
                          carbon_swing=swing, feasible_prob=r.feasible_prob,
                          approx_ratio=r.approx_ratio, approx_ratio_expected=r.approx_ratio_expected,
                          p_opt=r.p_opt, p_within_1pct=r.p_within_1pct, n_2q=r.n_2q, depth=r.depth, reps=reps,
                          train_objective=cfg["objective"], train_cvar_alpha=cfg["cvar_alpha"],
                          train_init=cfg["init"], train_maxiter=cfg["maxiter"], n_evals=tr.n_evals)
            if r.best_eval is not None:
                rows.append(dict(common, method="qaoa_xy", co2_kg=r.best_eval["co2_kg"],
                                 cost_eur=r.best_eval["cost_eur"], objective=r.best_eval["objective"],
                                 feasible=True,
                                 saving_co2_vs_on_arrival=base["co2_kg"] - r.best_eval["co2_kg"],
                                 saving_eur_vs_on_arrival=base["cost_eur"] - r.best_eval["cost_eur"]))
            if feas.any():
                wf = wt[feas] / wt[feas].sum()
                co2m, eurm = float(wf @ (X[feas] @ co2c)), float(wf @ (X[feas] @ eurc))
                rows.append(dict(common, method="qaoa_xy_expected_feasible", co2_kg=co2m, cost_eur=eurm,
                                 objective=np.nan, feasible=True,
                                 saving_co2_vs_on_arrival=base["co2_kg"] - co2m,
                                 saving_eur_vs_on_arrival=base["cost_eur"] - eurm))
        print(f"  qaoa w={w:g} done", flush=True)
    return rows


def build_summary(rows: pd.DataFrame, threshold: float, main_night: str = MAIN_NIGHT) -> pd.DataFrame:
    opt = rows[rows.method == "milp"]
    scope = f"full_year_{opt.night.nunique()}"
    out = []
    groups = {"all": opt, "windy": opt[opt.windy], "calm": opt[~opt.windy]}
    for g, sub in groups.items():
        for w, s in sub.groupby("w"):
            n = s.night.nunique()
            for q in ("saving_co2_vs_on_arrival", "saving_eur_vs_on_arrival", "saving_co2_vs_median",
                      "saving_eur_vs_median", "co2_kg", "cost_eur"):
                med, q25, q75, lo, hi = stats(s[q])
                out.append(dict(group=g, w=w, quantity=q, n_nights=n, median=med, q25=q25, q75=q75, min=lo, max=hi))
            frac = float((s.saving_co2_vs_on_arrival < -1e-9).mean())
            out.append(dict(group=g, w=w, quantity="frac_nights_opt_emits_more_co2_than_on_arrival", n_nights=n,
                            median=frac, q25=np.nan, q75=np.nan, min=np.nan, max=np.nan))
            out.append(dict(group=g, w=w, quantity="n_nights_opt_emits_more_co2_than_on_arrival", n_nights=n,
                            median=float((s.saving_co2_vs_on_arrival < -1e-9).sum()), q25=np.nan, q75=np.nan,
                            min=np.nan, max=np.nan))
        # per night: smallest grid w at which the optimum no longer emits more CO2 than on-arrival
        smallest = []
        for night, s in sub.groupby("night"):
            ok = s[s.saving_co2_vs_on_arrival >= -1e-9].w
            smallest.append(float(ok.min()) if len(ok) else np.nan)
        med, q25, q75, lo, hi = stats(smallest)
        out.append(dict(group=g, w=np.nan, quantity="smallest_w_no_extra_co2", n_nights=len(smallest),
                        median=med, q25=q25, q75=q75, min=lo, max=hi))
        # smallest w at which NO night of the group (and all larger w) emits extra CO2
        fr = sub.groupby("w").saving_co2_vs_on_arrival.apply(lambda v: bool((v >= -1e-9).all()))
        ws = fr.index.to_numpy()
        good = [w for k, w in enumerate(ws) if fr.to_numpy()[k:].all()]
        out.append(dict(group=g, w=np.nan, quantity="smallest_w_all_nights_no_extra_co2", n_nights=len(smallest),
                        median=float(min(good)) if good else np.nan, q25=np.nan, q75=np.nan, min=np.nan, max=np.nan))
        # smallest w at which the group MEDIAN CO2 saving is > 0 and stays > 0
        med_s = sub.groupby("w").saving_co2_vs_on_arrival.median()
        goodm = [w for k, w in enumerate(med_s.index) if (med_s.to_numpy()[k:] > 0).all()]
        out.append(dict(group=g, w=np.nan, quantity="smallest_w_median_co2_saving_positive", n_nights=len(smallest),
                        median=float(min(goodm)) if goodm else np.nan, q25=np.nan, q75=np.nan, min=np.nan, max=np.nan))
    out = pd.DataFrame(out)
    out["nights_scope"] = scope
    q = rows[rows.method.str.startswith("qaoa")]
    if len(q):
        extra = []
        for (method, w), s in q.groupby(["method", "w"]):
            for qty in ("co2_kg", "cost_eur", "saving_co2_vs_on_arrival", "saving_eur_vs_on_arrival",
                        "feasible_prob", "approx_ratio", "approx_ratio_expected", "p_opt"):
                med, q25, q75, lo, hi = stats(s[qty])
                extra.append(dict(group=method, w=w, quantity=qty, n_nights=1, median=med, q25=q25, q75=q75,
                                  min=lo, max=hi))
        extra = pd.DataFrame(extra)
        extra["nights_scope"] = f"main_night_{main_night}"
        out = pd.concat([out, extra], ignore_index=True)
    out["swing_threshold"] = threshold
    return out


def make_figure(df, rows: pd.DataFrame, summary: pd.DataFrame, night: date, out_base: Path,
                threshold: float) -> str:
    """(a) cost vs CO2 trade-off on the main night; (b)-(d) effect of the carbon weight w over all nights."""
    S.apply()
    opt = rows[rows.method == "milp"]
    n_w = opt[opt.windy].night.nunique()
    n_c = opt[~opt.windy].night.nunique()
    fig, axes = plt.subplots(2, 2, figsize=(S.FIG_WIDTH, 9.6))

    # ---- (a) main-night front
    ax = axes[0, 0]
    inst = build_instance(df, night, w=W_MAIN)
    _, feas = brute_force(inst)
    co2c, eurc = physical_coeffs(inst)
    Xf = np.array([x for x, _ in feas])
    cc, ee = Xf @ co2c, Xf @ eurc
    ax.scatter(cc, ee, s=14, color=S.LIGHT_GREY, alpha=0.6, label=f"all {len(feas)} feasible schedules", zorder=1)
    order = np.argsort(cc, kind="stable")
    front, best_e = [], np.inf
    for a in order:  # non-dominated: lower CO2 and lower EUR
        if ee[a] < best_e - 1e-12:
            front.append(a)
            best_e = ee[a]
    ax.step(cc[front], ee[front], where="post", color=S.GREY, lw=1.2, ls="-", zorder=2,
            label="Pareto-optimal schedules")
    n_sel = rows[(rows.night == str(night)) & (rows.method == "milp")].sort_values("w")
    ax.plot(n_sel.co2_kg, n_sel.cost_eur, color=S.EXACT, lw=1.0, zorder=3)
    ax.scatter(n_sel.co2_kg, n_sel.cost_eur, s=60, marker="o", color=S.EXACT, edgecolor="black", zorder=4,
               label="exact optimum, w = 0 to 1")
    for (c_, e_), g in n_sel.assign(c=n_sel.co2_kg.round(6), e=n_sel.cost_eur.round(6)).groupby(["c", "e"]):
        lab = f"w = {g.w.min():g}" if len(g) == 1 else f"w = {g.w.min():g} to {g.w.max():g}"
        ax.annotate(lab, (g.co2_kg.iloc[0], g.cost_eur.iloc[0]), textcoords="offset points",
                    xytext=(14, -4) if g.w.min() == 0 else (6, 12), fontsize=11)
    w7 = n_sel[np.isclose(n_sel.w, W_MAIN)]
    if len(w7):
        ax.scatter(w7.co2_kg, w7.cost_eur, s=220, marker="o", facecolors="none", edgecolors=S.BLACK, linewidths=2,
                   zorder=5, label="w = 0.7")
    arr = rows[(rows.night == str(night)) & (rows.method == "on_arrival")].iloc[0]
    med = rows[(rows.night == str(night)) & (rows.method == "median_feasible") & np.isclose(rows.w, W_MAIN)].iloc[0]
    ax.scatter([arr.co2_kg], [arr.cost_eur], s=110, marker="X", color=S.BLACK, edgecolor="white", zorder=5,
               label="charge-on-arrival")
    ax.scatter([med.co2_kg], [med.cost_eur], s=100, marker="D", color=S.GREY, edgecolor="black", zorder=5,
               label="median feasible schedule")
    q = rows[(rows.night == str(night)) & rows.method.str.startswith("qaoa")]
    if len(q):
        qb = q[q.method == "qaoa_xy"]
        qe = q[q.method == "qaoa_xy_expected_feasible"]
        ax.scatter(qb.co2_kg, qb.cost_eur, s=70, marker="^", color=S.XY, edgecolor="black", zorder=6, alpha=0.9,
                   label=f"QAOA-XY (p = {int(q.reps.iloc[0])}), best sample")
        ax.scatter(qe.co2_kg, qe.cost_eur, s=70, marker="v", facecolors="white", edgecolors=S.XY, linewidths=1.8,
                   zorder=6, label="QAOA-XY, mean of feasible samples")
    ax.set_xlabel("CO₂ of the night's charging (kg)")
    ax.set_ylabel("Electricity cost (€)")
    ax.set_ylim(bottom=ax.get_ylim()[0] - 0.4)
    S.panel_title(ax, f"(a) Main night {night}")
    h_a, l_a = ax.get_legend_handles_labels()

    # ---- (b), (c) all-nights savings vs w
    for ax, qty, ylabel, ttl in ((axes[0, 1], "saving_co2_vs_on_arrival", "CO₂ saved (kg per night)", "(b) CO₂ saved vs w"),
                                 (axes[1, 0], "saving_eur_vs_on_arrival", "Cost saved (€ per night)", "(c) Cost saved vs w")):
        ax.axhline(0, color="black", lw=1)
        for grp, col, mk, ls, n in (("windy", S.WINDY, "o", "-", n_w), ("calm", S.CALM, "s", "--", n_c)):
            s = summary[(summary.group == grp) & (summary.quantity == qty)].sort_values("w")
            ax.fill_between(s.w, s.q25, s.q75, color=col, alpha=0.2, lw=0)
            ax.plot(s.w, s["median"], color=col, marker=mk, ls=ls, lw=1.8, ms=7, mfc=col if grp == "windy" else "white",
                    mec="black" if grp == "windy" else col, mew=1.4, label=f"{grp} nights (n = {n})")
        ax.set_xlabel("Carbon weight w (0 = cost only, 1 = CO₂ only)")
        ax.set_ylabel(ylabel)
        S.panel_title(ax, ttl)
    axes[0, 1].legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=11)

    # ---- (d) share of nights where the optimum emits more CO2 than on-arrival
    ax = axes[1, 1]
    for grp, col, mk, ls, n in (("windy", S.WINDY, "o", "-", n_w), ("calm", S.CALM, "s", "--", n_c)):
        sd = summary[(summary.group == grp) & (summary.quantity == "frac_nights_opt_emits_more_co2_than_on_arrival")
                     & (summary.nights_scope.str.startswith("full_year"))].sort_values("w")
        ax.plot(sd.w, 100 * sd["median"], color=col, marker=mk, ls=ls, lw=1.8, ms=7,
                mfc=col if grp == "windy" else "white", mec="black" if grp == "windy" else col, mew=1.4,
                label=f"{grp} nights (n = {n})")
    ax.set_xlabel("Carbon weight w (0 = cost only, 1 = CO₂ only)")
    ax.set_ylabel("Nights with more CO₂ than\ncharge-on-arrival (%)")
    S.panel_title(ax, "(d) Share of nights with extra CO₂")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left")

    def cross(group):
        r = summary[(summary.group == group) & (summary.quantity == "smallest_w_all_nights_no_extra_co2")
                    & summary.nights_scope.str.startswith("full_year")]
        return float(r["median"].iloc[0]) if len(r) else float("nan")

    w1 = n_sel[np.isclose(n_sel.w, 1.0)].iloc[0]
    less = 100 * (1 - w1.co2_kg / arr.co2_kg)
    c_all = cross("all")
    when = f"at w = {c_all:g}" if c_all >= 1 else f"from w = {c_all:g}"
    title = f"One weight w trades cost for CO₂: {when} no night emits more than charge-on-arrival"
    caption = (f"(a) Main night {night}: at w = 1 the optimum emits {w1.co2_kg:.1f} kg CO₂ against {arr.co2_kg:.1f} kg for "
               f"charge-on-arrival ({less:.0f} % less). (b) to (d) Classical exact optimum on all {n_w + n_c} nights "
               f"({n_c} calm, {n_w} windy); line = median, band = IQR.")
    top = S.heading(fig, title, caption)
    fig.subplots_adjust(top=top - 0.05, bottom=0.17, left=0.075, right=0.985, hspace=0.38, wspace=0.27)
    fig.legend(h_a, l_a, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=11)
    S.save(fig, out_base)
    return (f"{title}. Four panels. (a) Electricity cost against CO₂ for all feasible schedules of the main night as grey points "
            f"with the Pareto-optimal staircase, exact optima for w from 0 to 1 as green circles, charge-on-arrival as a black cross, "
            f"the median schedule as a grey diamond and XY-QAOA samples as vermilion triangles. (b) and (c) CO₂ and cost saved versus "
            f"charge-on-arrival as a function of w, median and IQR band over {n_w + n_c} nights: windy nights orange filled circles with "
            f"a solid line, calm nights sky-blue hollow squares with a dashed line. (d) Share of nights where the optimum emits more CO₂ "
            f"than charge-on-arrival, against w, same markers; it reaches zero {when}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3 nights, w in {0, 0.5, 1}; writes to results/scratch/")
    ap.add_argument("--night", default=MAIN_NIGHT, help="main night for panel (a) and QAOA")
    ap.add_argument("--w-step", type=float, default=0.1)
    ap.add_argument("--swing-threshold", type=float, default=1.3, help="windy if max/min block CO2 >= this (as e2)")
    ap.add_argument("--qaoa", action="store_true", help="also run XY QAOA on the main night (depth --qaoa-reps, default 3), w in {0,.5,.7,1}")
    ap.add_argument("--qaoa-reps", type=int, default=3, help="QAOA depth for --qaoa (default 3, TRAIN_CONFIG)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="QAOA seeds")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--replot", action="store_true", help="redraw the figure from results/e5_pareto.csv and e5_summary.csv (no runs)")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()

    out_dir = a.out_dir or (ROOT / "results" / ("scratch" if a.quick else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_hourly()
    nights = list_nights(df)
    main_night = date.fromisoformat(a.night)
    if main_night not in nights:
        sys.exit(f"night {main_night} has no complete coverage in the data")
    if a.replot:
        rows = read_rows(out_dir)           # the CSVs are left untouched: replot only redraws the figure
        summary = build_summary(rows, a.swing_threshold, a.night)
        alt = make_figure(df, rows, summary, main_night, out_dir / FIG_NAME, a.swing_threshold)
        set_alt_text(out_dir, FIG_NAME + ".png", alt)
        print(f"replotted {out_dir / (FIG_NAME + '.png')}")
        return
    ws = [round(float(x), 4) for x in np.arange(0.0, 1.0 + 1e-9, a.w_step)]
    seeds, shots = a.seeds, a.shots
    if a.quick:
        ws = [0.0, 0.5, 1.0]
        idx = np.unique(np.linspace(0, len(nights) - 1, 3).round().astype(int))
        nights = [nights[i] for i in idx]
        if main_night not in nights:
            nights.append(main_night)
        seeds, shots = [0], 512

    if not any(abs(w - W_MAIN) < 1e-9 for w in ws):  # the default w is always shown
        ws = sorted(ws + [W_MAIN])
    t0 = time.perf_counter()
    all_rows: list[dict] = []
    for k, n in enumerate(nights, 1):
        all_rows += sweep_night(df, n, ws, a.swing_threshold)
        print(f"[{k}/{len(nights)}] {n} done", flush=True)
    old_path = out_dir / "e5_pareto.csv"
    if a.qaoa:
        all_rows += qaoa_rows(df, main_night, seeds, shots, a.swing_threshold, a.qaoa_reps, 40 if a.quick else None)
    elif old_path.exists() and not a.quick:  # carry over the existing main-night QAOA rows (never re-run by default)
        old = read_rows(out_dir)
        keep = old[old.method.astype(str).str.startswith("qaoa")]
        if len(keep):
            print(f"carrying over {len(keep)} existing QAOA rows (main night) from {old_path.name}")
            all_rows += keep.to_dict("records")
    rows = ensure_n_restarts(pd.DataFrame(all_rows))
    write_rows(rows, out_dir)
    summary = build_summary(rows, a.swing_threshold, a.night)
    summary.to_csv(out_dir / "e5_summary.csv", index=False)
    alt = make_figure(df, rows, summary, main_night, out_dir / FIG_NAME, a.swing_threshold)
    if not a.quick:
        set_alt_text(out_dir, FIG_NAME + ".png", alt)

    # ---- console summary
    mil = rows[rows.method == "milp"]
    n_w = mil[mil.windy].night.nunique()
    print(f"\nnights {mil.night.nunique()} (windy {n_w}, calm {mil.night.nunique() - n_w}; threshold "
          f"{a.swing_threshold:g}), w grid {ws}, total {time.perf_counter() - t0:.0f}s")
    show = summary[summary.quantity.isin(["saving_co2_vs_on_arrival", "saving_eur_vs_on_arrival"])
                   & summary.group.isin(["windy", "calm"])]
    print(show.pivot_table(index="w", columns=["group", "quantity"], values="median").round(2).to_string())
    fr = summary[summary.quantity == "frac_nights_opt_emits_more_co2_than_on_arrival"]
    print(fr.pivot_table(index="w", columns="group", values="median").round(2).to_string())
    print(summary[summary.quantity.str.startswith("smallest_w")][["group", "quantity", "median", "q25", "q75", "min", "max"]]
          .to_string(index=False))
    print(f"wrote {out_dir / 'e5_pareto.csv'}, {QAOA_FILE}, e5_summary.csv, {FIG_NAME}.png/.svg")


if __name__ == "__main__":
    main()
