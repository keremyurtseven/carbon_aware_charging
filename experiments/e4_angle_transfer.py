"""E4: "train once, schedule every night" - transfer of QAOA angles between nights.

Angles are trained ONCE (X and XY mixer, reps = --reps, ``carbon_ev.qaoa.TRAIN_CONFIG``) on
  * the main night (2026-10-08), and
  * a calm night: the calm night (carbon swing < --swing-threshold, as e2) with the MEDIAN
    carbon swing among the calm nights present in the e2 table (documented in the CSV),
and then applied UNCHANGED (``evaluate_params``, ideal sampler, 4096 shots, seeds 0..2) on every
other night. Each transferred result is compared with
  * per-night training (taken from results/e2_all_nights.csv, same config, same reps), and
  * p = 0 (untrained start state, also from the e2 table).

Needs results/e2_all_nights.csv produced with the same --reps (run e2 first).
Quality lost by transfer = per-night value - transferred value (positive = transfer is worse),
summarized as median / IQR over nights, split windy / calm (threshold stored in the e2 table).

Outputs (results/, or results/scratch/ with --quick):
  e4_angle_transfer.csv  one row per (train set, mixer, target night, seed, condition)
  e4_summary.csv         per (train set, mixer, group, metric)
  e4_angle_transfer.png / .svg

    .venv/Scripts/python experiments/e4_angle_transfer.py
    .venv/Scripts/python experiments/e4_angle_transfer.py --quick
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

from carbon_ev.data import load_hourly  # noqa: E402
from carbon_ev.problem import build_instance  # noqa: E402
from carbon_ev.qaoa import TRAIN_CONFIG, evaluate_params, exact_optimum, run_qaoa  # noqa: E402

MAIN_NIGHT = "2026-10-08"
FIG_NAME = "e4_angle_transfer"
METRICS = ["approx_ratio_expected", "p_within_1pct", "feasible_prob", "p_opt", "approx_ratio"]


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


def ensure_n_restarts(rows: pd.DataFrame) -> pd.DataFrame:
    """Add the ``n_restarts`` column (training restarts of the QAOA rows; empty for non-QAOA rows).

    Every QAOA run in this script uses ``run_qaoa(..., n_restarts=1)`` (restart 0 only, deterministic), so
    the seeds are sampling replicates of one training run. Old CSVs written before this column existed
    get it here (value 1) without any recomputation.
    """
    if "n_restarts" not in rows.columns:
        rows = rows.copy()
        rows["n_restarts"] = np.where(rows["condition"].isin(["transfer", "per_night"]), 1.0, np.nan)
    return rows


def summarize_transfer(all_df: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-night table ``wide`` (median over sampling seeds) and the summary table, both from the per-run CSV."""
    train_sets = all_df.drop_duplicates("train_set").set_index("train_set")["train_night"].to_dict()
    key = ["train_set", "mixer", "night"]
    t = all_df[all_df.condition == "transfer"].groupby(key)[METRICS].median().add_prefix("transfer_")
    pnv = all_df[all_df.condition == "per_night"].set_index(key)[METRICS].add_prefix("pernight_")
    p0v = all_df[all_df.condition == "p0"].set_index(key)[METRICS].add_prefix("p0_")
    meta = all_df.groupby(key)[["windy", "carbon_swing"]].first()
    wide = meta.join(t).join(pnv).join(p0v)
    summ = []
    for (tname, mixer), g in wide.groupby(level=["train_set", "mixer"]):
        for grp, sub in (("all", g), ("windy", g[g.windy]), ("calm", g[~g.windy])):
            for m in METRICS:
                loss = sub[f"pernight_{m}"] - sub[f"transfer_{m}"]
                gain = sub[f"transfer_{m}"] - sub[f"p0_{m}"]
                full = sub[f"pernight_{m}"] - sub[f"p0_{m}"]
                d = dict(train_set=tname, train_night=train_sets[tname], mixer=mixer, group=grp, metric=m,
                         n_nights=len(sub), reps=reps)
                for lab, v in (("transfer", sub[f"transfer_{m}"]), ("per_night", sub[f"pernight_{m}"]),
                               ("p0", sub[f"p0_{m}"]), ("loss_vs_per_night", loss),
                               ("transfer_gain_over_p0", gain), ("per_night_gain_over_p0", full)):
                    med, q25, q75, lo, hi = stats(v)
                    d.update({f"{lab}_median": med, f"{lab}_q25": q25, f"{lab}_q75": q75})
                d["n_nights_transfer_gain_positive"] = int((gain > 1e-12).sum())
                d["n_nights_loss_positive"] = int((loss > 1e-12).sum())
                summ.append(d)
    summ = pd.DataFrame(summ)
    return wide, summ


def make_figure(e2, q2, wide, summ, main_night, threshold, reps, calm_night, p0name, out_dir, quick):
    """Draw the e4 figure (rows = mixer, columns = metric) and write its alt text."""
    S.apply()
    fig, axes = plt.subplots(2, 2, figsize=(S.FIG_WIDTH, 8.8), sharex=True)
    series = [("p0", "untrained start state (p = 0)", S.GREY, "D", False),
              ("pernight", "trained on that night", S.BLACK, "o", True),
              ("transfer:main", f"angles from main night {main_night}", S.WINDY, "^", True),
              ("transfer:calm_median", f"angles from calm night {calm_night}", S.CALM, "s", False)]
    for ri, mixer in enumerate(("x", "xy")):
        for ci, (m, mlab) in enumerate((("approx_ratio_expected", "Expected-cost ratio (1 = optimal)"),
                                        ("p_within_1pct", "P(within 1 % of optimum)"))):
            ax = axes[ri, ci]
            for tag, lab, col, mk, filled in series:
                if tag == "p0":  # reference rows come straight from e2 so that every night is shown
                    g = e2[e2.method == p0name[mixer]]
                    y = g[m]; x = g.carbon_swing
                elif tag == "pernight":
                    g = q2[q2.mixer == mixer]
                    y = g[m]; x = g.carbon_swing
                else:
                    g = wide.xs((tag.split(":")[1], mixer), level=("train_set", "mixer"))
                    y = g[f"transfer_{m}"]; x = g.carbon_swing
                ax.scatter(x, y, s=46, marker=mk, facecolors=col if filled else "white", edgecolors=col,
                           linewidths=1.5, alpha=0.9, label=lab, zorder=3)
            ax.axvline(threshold, color=S.GREY, ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xticks([1.0, 1.2, 1.5, 2.0, 3.0])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.minorticks_off()
            ax.set_ylabel(mlab)
            if ri == 0:
                S.panel_title(ax, f"{'(a)' if ci == 0 else '(b)'} {mlab.split(' (')[0]}")
            if ci == 0:
                ax.text(0.02, 0.04, "X mixer (penalty)" if mixer == "x" else "XY mixer (Dicke start)",
                        transform=ax.transAxes, fontsize=12, fontweight="bold",
                        color=S.X if mixer == "x" else S.XY, va="bottom")
            else:
                ax.text(0.98, 0.96, "X mixer (penalty)" if mixer == "x" else "XY mixer (Dicke start)",
                        transform=ax.transAxes, fontsize=12, fontweight="bold",
                        color=S.X if mixer == "x" else S.XY, va="top", ha="right")
    for ax in axes[1]:
        ax.set_xlabel("Carbon swing of the target night (max / min 2-h block CO₂)")

    def med_of(tset, mixer, metric, col):
        r = summ[(summ.train_set == tset) & (summ.mixer == mixer) & (summ.metric == metric) & (summ.group == "all")]
        return float(r[col].iloc[0])

    n_t = int(summ[(summ.train_set == "main") & (summ.group == "all")].n_nights.iloc[0])
    tm, tc = med_of("main", "xy", "p_within_1pct", "transfer_median"), med_of("calm_median", "xy", "p_within_1pct", "transfer_median")
    pn_, p0_ = med_of("main", "xy", "p_within_1pct", "per_night_median"), med_of("main", "xy", "p_within_1pct", "p0_median")
    title = f"Train once, schedule every night: reused XY angles give {tm / p0_:.1f}x the untrained near-optimal rate"
    caption = (f"Median P(within 1 % of optimum) of the XY mixer over {n_t} nights (p = {reps}, ideal simulation): "
               f"{tm:.3f} with angles from the main night, {tc:.3f} with angles from the calm night {calm_night}, "
               f"{pn_:.3f} trained on each night, {p0_:.3f} untrained. Dashed line: windy/calm threshold. "
               f"Reused-angle points are medians over shot-sampling seeds of one training run.")
    top = S.heading(fig, title, caption)
    fig.subplots_adjust(top=top - 0.06, bottom=0.18, left=0.085, right=0.985, hspace=0.12, wspace=0.27)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=11)
    S.save(fig, out_dir / FIG_NAME)
    alt = (f"{title}. Four scatter panels, one point per night against the carbon swing (log axis); top row X mixer, bottom row XY "
           f"mixer at p = {reps}; left expected-cost ratio, right probability of a sample within 1 % of the optimum. Grey hollow "
           f"diamonds are the untrained start state, black circles QAOA trained on that night, orange triangles angles trained once "
           f"on the main night and reused, sky-blue hollow squares angles trained once on a calm night and reused.")
    if not quick:
        set_alt_text(out_dir, FIG_NAME + ".png", alt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="reps 1, maxiter 40, 512 shots; reads/writes results/scratch/")
    ap.add_argument("--reps", type=int, default=3, help="must equal the --reps used for the e2 table")
    ap.add_argument("--main-night", default=MAIN_NIGHT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="shot-sampling seeds for the transfer evaluation")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--swing-threshold", type=float, default=1.3)
    ap.add_argument("--replot", action="store_true", help="redraw the figure from the existing e4/e2 CSVs (no runs)")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    reps, shots, maxiter, seeds = a.reps, a.shots, None, a.seeds
    if a.quick:
        reps, shots, maxiter, seeds = 1, 512, 40, [0]

    out_dir = a.out_dir or (ROOT / "results" / ("scratch" if a.quick else ""))
    e2_csv = out_dir / "e2_all_nights.csv"
    if not e2_csv.exists():
        sys.exit(f"{e2_csv} not found: run experiments/e2_all_nights.py first (same --reps)")
    e2 = pd.read_csv(e2_csv)
    # e2 holds classical results for every night but QAOA rows only for a subset of nights:
    # e4 works on the nights that have QAOA rows (also for choosing the calm training night).
    e2 = e2[e2.night.isin(e2[e2.family == "qaoa"].night.unique())]
    q2 = e2[e2.family == "qaoa"]
    if not (q2.reps == reps).all():
        sys.exit(f"e2 table was made with reps={sorted(q2.reps.unique())}, but e4 uses reps={reps}")
    info = e2.groupby("night")[["windy", "carbon_swing"]].first()
    nights = [date.fromisoformat(n) for n in info.index]
    if a.main_night not in info.index:
        sys.exit(f"main night {a.main_night} is not in the e2 table")
    calm = info[~info.windy & (info.index != a.main_night)].sort_values("carbon_swing")
    calm_night = calm.index[len(calm) // 2]  # median carbon swing among calm nights
    print(f"calm training night: {calm_night} (swing {calm.loc[calm_night, 'carbon_swing']:.3f}; "
          f"median of {len(calm)} calm nights)")
    train_sets = {"main": a.main_night, "calm_median": calm_night}

    if a.replot:
        all_df = ensure_n_restarts(pd.read_csv(out_dir / "e4_angle_transfer.csv"))
        all_df.to_csv(out_dir / "e4_angle_transfer.csv", index=False)
        wide, summ = summarize_transfer(all_df, reps)
        summ.to_csv(out_dir / "e4_summary.csv", index=False)
        make_figure(e2, q2, wide, summ, a.main_night, a.swing_threshold, reps, calm_night,
                    {"x": "p0_uniform", "xy": "p0_dicke"}, out_dir, a.quick)
        print(f"replotted {out_dir / (FIG_NAME + '.png')}")
        return

    df = load_hourly()
    cfg = dict(TRAIN_CONFIG, **({"maxiter": maxiter} if maxiter else {}))
    t0 = time.perf_counter()
    rows: list[dict] = []
    for tname, tnight in train_sets.items():
        tinst = build_instance(df, date.fromisoformat(tnight))
        for mixer in ("x", "xy"):
            tr = run_qaoa(tinst, mixer, reps, shots=shots, seed=0, **cfg)
            print(f"trained {mixer} p={reps} on {tname} ({tnight}): {tr.wall_s:.0f}s evals={tr.n_evals} "
                  f"params={np.round(tr.params, 3).tolist()}", flush=True)
            for night in nights:
                if str(night) == tnight:
                    continue
                inst = build_instance(df, night)
                opt = exact_optimum(inst)[1]
                meta = dict(train_set=tname, train_night=tnight, train_swing=float(info.loc[tnight, "carbon_swing"]),
                            mixer=mixer, reps=reps, night=str(night), windy=bool(info.loc[str(night), "windy"]),
                            carbon_swing=float(info.loc[str(night), "carbon_swing"]),
                            train_objective=cfg["objective"], train_cvar_alpha=cfg["cvar_alpha"],
                            train_init=cfg["init"], train_maxiter=cfg["maxiter"])
                for seed in seeds:
                    r = evaluate_params(inst, mixer, reps, tr.params, shots=shots, seed=seed, opt_obj=opt)
                    rows.append(dict(meta, condition="transfer", seed=seed,
                                     **{m: getattr(r, m) for m in METRICS}))
    tr_df = pd.DataFrame(rows)
    # reference conditions from e2 (same nights, same mixers)
    ref_rows = []
    p0name = {"x": "p0_uniform", "xy": "p0_dicke"}
    for (tname, tnight, mixer), g in tr_df.groupby(["train_set", "train_night", "mixer"]):
        for night in g.night.unique():
            base = dict(train_set=tname, train_night=tnight, mixer=mixer, reps=reps, night=night,
                        windy=bool(info.loc[night, "windy"]), carbon_swing=float(info.loc[night, "carbon_swing"]),
                        seed=-1)
            pn = q2[(q2.night == night) & (q2.mixer == mixer)].iloc[0]
            ref_rows.append(dict(base, condition="per_night", **{m: pn[m] for m in METRICS}))
            p0 = e2[(e2.night == night) & (e2.method == p0name[mixer])].iloc[0]
            ref_rows.append(dict(base, condition="p0", **{m: p0[m] for m in METRICS}))
    all_df = pd.concat([tr_df, pd.DataFrame(ref_rows)], ignore_index=True)
    all_df = ensure_n_restarts(all_df)
    all_df.to_csv(out_dir / "e4_angle_transfer.csv", index=False)
    wide, summ = summarize_transfer(all_df, reps)
    summ.to_csv(out_dir / "e4_summary.csv", index=False)

    make_figure(e2, q2, wide, summ, a.main_night, a.swing_threshold, reps, calm_night, p0name, out_dir, a.quick)

    print(f"\ntotal {time.perf_counter() - t0:.0f}s")
    show = summ[(summ.group.isin(["all", "windy", "calm"])) & summ.metric.isin(["approx_ratio_expected", "p_within_1pct"])]
    cols = ["train_set", "mixer", "group", "metric", "n_nights", "transfer_median", "per_night_median", "p0_median",
            "loss_vs_per_night_median", "loss_vs_per_night_q25", "loss_vs_per_night_q75", "n_nights_transfer_gain_positive"]
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out_dir / 'e4_angle_transfer.csv'}, e4_summary.csv, {FIG_NAME}.png/.svg")


if __name__ == "__main__":
    main()
