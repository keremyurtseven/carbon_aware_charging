"""Carbon-aware charging for an electric bus depot, with QAOA in Qiskit: end-to-end demo (Qollab entry point).

Main night (Oct 8 20:00 to Oct 9 08:00, 3 EVs, 16 qubits): charge-on-arrival, greedy,
simulated annealing, exact MILP and two QAOA variants (penalty QUBO with X mixer, and
XY mixer with Dicke start). Then three depot-scale sections: 100 e-buses on one night,
30 nights, and QAOA as the subsolver inside a depot optimizer.

Prints the comparison tables and writes CSVs and one figure to results/
(qollab_output/ in the browser). Needs numpy, scipy, pandas, matplotlib and qiskit;
no API key, no network.

    python main.py                       # main night, QAOA p=2
    python main.py --night 2026-10-01    # a calm control night
    python main.py --no-plot --no-depot

The fleets are modelling assumptions; only carbon intensity and day-ahead prices are
measured data (Energy-Charts.info, CC BY 4.0).
"""

import argparse
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

# Qollab (Pyodide) may run this file without __file__; the root is then the working directory.
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))
# On Qollab the code runs in the browser (Pyodide, sys.platform "emscripten"): figures are shown
# on the page and outputs go to a fresh folder, because unloaded project files cannot be overwritten.
IN_BROWSER = sys.platform == "emscripten"
if IN_BROWSER:
    import warnings

    # the browser's matplotlib backend passes float pixel sizes; harmless, so keep the console clean
    warnings.filterwarnings("ignore", message="The (width|height) parameter as float")

import matplotlib  # noqa: E402

if not IN_BROWSER:
    matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from carbon_ev.classical import greedy, milp, on_arrival, simulated_annealing  # noqa: E402
from carbon_ev.data import list_nights, load_hourly  # noqa: E402
from carbon_ev.depot import (  # noqa: E402
    block_load_kw, depot_fleet, milp_depot, min_cap_on_arrival_kw, min_peak_kw, on_arrival_capacity)
from carbon_ev.hybrid import lns_depot  # noqa: E402
from carbon_ev.problem import build_instance, evaluate, var_index  # noqa: E402
from carbon_ev.qaoa import TRAIN_CONFIG, feasible_schedules, random_feasible_baseline, run_qaoa  # noqa: E402
from carbon_ev.qubo import default_penalties, to_qubo  # noqa: E402

RESULTS = ROOT / ("qollab_output" if IN_BROWSER else "results")
# Okabe-Ito palette
BLUE, ORANGE, VERMILION, GREEN, SKY, GREY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#56B4E9", "#555555"
ALT_TEXT_FILE = RESULTS / "figures_alt_text.md"


WIDTH = 92
STEPS = 5


def section(title: str) -> None:
    """Print a section banner."""
    print(f"\n{'─' * WIDTH}\n{title}\n{'─' * WIDTH}")


def step(n: int, title: str, what: str) -> None:
    """Print a numbered step banner with one line saying what the step shows."""
    print(f"\n{'━' * WIDTH}\n[{n}/{STEPS}] {title}\n{'━' * WIDTH}")
    print(f"{what}\n")


def bar(value: float, vmax: float, width: int = 30) -> str:
    """Horizontal text bar of ``value`` on a 0..vmax scale."""
    n = int(round(width * max(value, 0) / vmax)) if vmax > 0 else 0
    return "█" * n + "·" * (width - n)


def sparkline(values, width: int = 40) -> str:
    """Compact text plot of a sequence (lower is better), resampled to ``width`` characters."""
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return ""
    vals = np.minimum.accumulate(vals)          # best value so far, so the line shows training progress
    idx = np.linspace(0, len(vals) - 1, min(width, len(vals))).round().astype(int)
    v = vals[idx]
    lo, hi = v.min(), v.max()
    ticks = "▁▂▃▄▅▆▇█"                        # tall bar = high cost; the line falls as training improves
    if hi == lo:
        return ticks[-1] * len(v)
    return "".join(ticks[int((x - lo) / (hi - lo) * (len(ticks) - 1) + 0.5)] for x in v)


def circuit_sketch(inst, reps: int) -> str:
    """Text drawing of the XY-encoded QAOA circuit structure for this instance (one box per block of gates)."""
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import Gate

    from carbon_ev.qaoa import registers_from_instance

    regs, ks = registers_from_instance(inst)
    qregs = [QuantumRegister(len(r), ev.name) for r, ev in zip(regs, inst.evs)]
    qc = QuantumCircuit(*qregs)
    for qr, k in zip(qregs, ks):
        qc.append(Gate("dicke", len(qr), [], label=f"Dicke({len(qr)},{k})"), qr)
    for layer in range(1, reps + 1):
        qc.append(Gate("cost", qc.num_qubits, [], label=f"Cost γ{layer}"), qc.qubits)
        for qr in qregs:
            qc.append(Gate("xy", len(qr), [], label=f"XY β{layer}"), qr)
    return str(qc.draw(output="text", fold=-1))


def try_write(what: str, write) -> None:
    """Run a file write; if the file system refuses it, print a note and carry on."""
    try:
        RESULTS.mkdir(exist_ok=True)
        write()
    except OSError as exc:
        print(f"(note: could not write {what}: {exc})")


def schedule_str(inst, x) -> str:
    """Blocks per EV, e.g. 'A:4,5 B:3 C:3,4'."""
    vi = var_index(inst)
    parts = []
    for i, ev in enumerate(inst.evs):
        blocks = [t for a, (ii, t) in enumerate(vi) if ii == i and x[a] > 0.5]
        parts.append(f"{ev.name}:{','.join(map(str, blocks)) or '-'}")
    return " ".join(parts)


def make_row(name, inst, ev, sched, base, med, **extra):
    """One table row.

    ``ev`` is an ``evaluate`` dict (None if no feasible sample); ``base`` and ``med`` are
    (co2_kg, eur) of the on-arrival baseline and of the median feasible schedule.
    """
    row = {"method": name, "co2_kg": np.nan, "cost_eur": np.nan, "objective": np.nan,
           "schedule": sched, "feasible": False, "saved_co2_pct": np.nan, "saved_eur_pct": np.nan,
           "saved_co2_vs_median_pct": np.nan, "p_feasible": np.nan, "approx_ratio": np.nan,
           "approx_ratio_expected": np.nan, "p_opt": np.nan, "p_within_1pct": np.nan,
           "depth": np.nan, "n_2q": np.nan, "n_evals": np.nan, "wall_s": np.nan}
    if ev is not None:
        row.update(co2_kg=ev["co2_kg"], cost_eur=ev["cost_eur"], objective=ev["objective"],
                   feasible=bool(ev["feasible"]),
                   saved_co2_pct=100 * (1 - ev["co2_kg"] / base[0]),
                   saved_eur_pct=100 * (1 - ev["cost_eur"] / base[1]),
                   saved_co2_vs_median_pct=100 * (1 - ev["co2_kg"] / med[0]))
    row.update(extra)
    return row


def classical_row(name, inst, x, t_s, base, med):
    """Table row for a classical schedule ``x`` that took ``t_s`` seconds."""
    return make_row(name, inst, evaluate(inst, x), schedule_str(inst, x), base, med, wall_s=t_s)


def metrics_row(name, inst, m, base, med, **extra):
    """Table row from a metrics dict (``sample_metrics`` output or ``result_metrics(QAOAResult)``)."""
    sched = "none feasible" if m["best_x"] is None else schedule_str(inst, m["best_x"])
    return make_row(name, inst, m["best_eval"], sched, base, med,
                    p_feasible=m["feasible_prob"], approx_ratio=m["approx_ratio"],
                    approx_ratio_expected=m["approx_ratio_expected"], p_opt=m["p_opt"],
                    p_within_1pct=m["p_within_1pct"], **extra)


def result_metrics(res) -> dict:
    """QAOAResult -> the dict layout used by ``metrics_row``."""
    return dict(best_x=res.best_x, best_eval=res.best_eval, feasible_prob=res.feasible_prob,
                approx_ratio=res.approx_ratio, approx_ratio_expected=res.approx_ratio_expected,
                p_opt=res.p_opt, p_within_1pct=res.p_within_1pct)


def median_feasible(inst):
    """Median feasible schedule: (x, objective, all feasible objectives). Lower median if N is even."""
    X, obj = feasible_schedules(inst)
    order = np.argsort(obj, kind="stable")
    a = int(order[(len(order) - 1) // 2])
    return X[a].astype(int), float(obj[a]), obj


def print_classical(table: pd.DataFrame) -> None:
    """Classical methods: the schedule each one picks and the CO2 it saves against charging on arrival."""
    print(f"  {'method':<28}{'kg CO2':>8}{'EUR':>8}  {'schedule (blocks per EV)':<26}{'CO2 saved':>10}")
    labels = {"on-arrival": "charge on arrival", "median feasible schedule": "a typical valid schedule",
              "greedy": "greedy", "SA (QUBO)": "simulated annealing", "exact (MILP)": "exact optimum (MILP)"}
    for r in table.itertuples():
        if r.method in labels:
            saved = "" if r.method == "on-arrival" else f"{r.saved_co2_pct:9.1f}%"
            print(f"  {labels[r.method]:<28}{r.co2_kg:8.2f}{r.cost_eur:8.2f}  {r.schedule:<26}{saved:>10}")


def print_qaoa(table: pd.DataFrame, shots: int) -> None:
    """QAOA vs its untrained start states: how often one sample is valid, near-optimal, and how good on average."""
    rows = [("random guessing", "random sampling"), ("XY start state only (untrained)", "Dicke start only"),
            ("penalty QAOA, X mixer", "QAOA penalty-X"), ("XY QAOA (Dicke start + XY mixer)", "QAOA XY")]
    print(f"  Sampling {shots} shots from each circuit on the statevector simulator:\n")
    print(f"  {'':<34}{'valid':>11}{'near-optimal':>15}{'avg quality':>15}{'2-qubit':>10}")
    print(f"  {'':<34}{'schedules':>11}{'(within 1 %)':>15}{'(1 = optimal)':>15}{'gates':>10}")
    for label, key in rows:
        r = table[table.method.str.startswith(key)].iloc[0]
        gates = "" if pd.isna(r.n_2q) else f"{r.n_2q:.0f}"
        print(f"  {label:<34}{100 * r.p_feasible:10.1f}%{100 * r.p_within_1pct:14.2f}%{r.approx_ratio_expected:15.2f}{gates:>10}")


def make_figure(inst, x_arr, x_opt, saved_pct, out_png: Path) -> None:
    """Summary figure (carbon and price per block, on-arrival vs optimal grids) plus its alt text."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})
    T = inst.n_blocks
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(9, 8.5), gridspec_kw={"height_ratios": [3, 1.2, 1.2]}, sharex=True)

    ax0.bar(range(T), inst.co2, width=0.8, color=SKY, edgecolor="black", hatch="//",
            label="Carbon intensity (bars, left)")
    ax0.set_ylabel("Carbon intensity (g CO$_2$/kWh)")
    ax0.set_ylim(0, inst.co2.max() * 1.15)
    axp = ax0.twinx()
    axp.plot(range(T), inst.price, color=VERMILION, marker="o", ms=8, lw=2, ls="--",
             label="Day-ahead price (line, right)")
    axp.set_ylabel("Day-ahead price (EUR/MWh)")
    axp.set_ylim(min(0, inst.price.min() * 1.1), inst.price.max() * 1.25)
    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = axp.get_legend_handles_labels()
    ax0.legend(h0 + h1, l0 + l1, loc="upper right", framealpha=0.95)
    if saved_pct >= 5:
        head = f"Shifting charging into the clean hours saves {saved_pct:.0f}% CO$_2$ vs charge-on-arrival"
    else:
        head = f"Little CO$_2$ to gain from shifting on this night ({saved_pct:+.0f}% vs charge-on-arrival)"
    ax0.set_title(head + "\n"
                  f"night {inst.night} 20:00 to next day 08:00, 3 EVs, w = {inst.w}", loc="left")

    vi = var_index(inst)
    marks = ["o", "s", "^", "D"]
    cols = [BLUE, ORANGE, GREEN, VERMILION]

    def grid(ax, x, label):
        for i, ev in enumerate(inst.evs):
            for t in range(T):
                avail = t in ev.available
                ax.add_patch(plt.Rectangle((t - 0.4, i - 0.4), 0.8, 0.8, fill=not avail,
                                           facecolor="#dddddd", edgecolor="#888888", lw=0.8))
            for a, (ii, t) in enumerate(vi):
                if ii == i and x[a] > 0.5:
                    ax.add_patch(plt.Rectangle((t - 0.4, i - 0.4), 0.8, 0.8, facecolor=cols[i % 4],
                                               edgecolor="black", hatch=["", "xx", ".."][i % 3], lw=1.2))
                    ax.plot(t, i, marker=marks[i % 4], color="white", mec="black", ms=10)
        ax.set_yticks(range(len(inst.evs)))
        ax.set_yticklabels([f"EV {e.name}\n{e.power_kw:g} kW" for e in inst.evs])
        ax.set_ylim(len(inst.evs) - 0.5, -0.5)
        ax.set_ylabel(label)

    grid(ax1, x_arr, "Charge on\narrival")
    grid(ax2, x_opt, "Optimal\n(exact)")
    ax2.set_xticks(range(T))
    ax2.set_xticklabels(inst.block_labels)
    ax2.set_xlim(-0.5, T - 0.5)
    ax2.set_xlabel("Time block (local hour of day, 2 h blocks)")
    keys = [Patch(facecolor="white", edgecolor="#888888", label="available, idle"),
            Patch(facecolor="#dddddd", edgecolor="#888888", label="not plugged in")]
    for i, ev in enumerate(inst.evs):
        keys.append(Patch(facecolor=cols[i % 4], edgecolor="black", hatch=["", "xx", ".."][i % 3],
                          label=f"EV {ev.name} charging ({['circle', 'square', 'triangle', 'diamond'][i % 4]})"))
    ax2.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3, frameon=False)
    fig.tight_layout()
    try_write(f"figure {out_png.name}", lambda: (fig.savefig(out_png, dpi=200),
                                                 fig.savefig(out_png.with_suffix(".svg"))))
    if IN_BROWSER:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.show()
    plt.close(fig)

    if saved_pct >= 5:
        finding = "the optimal schedule moves charging to the low-carbon blocks near the end of the night."
    else:
        finding = (f"on this night the optimal schedule saves little or no CO2 against charge-on-arrival "
                   f"({saved_pct:+.1f} percent).")
    alt = (f"- `{out_png.name}`: top, bar chart of carbon intensity per 2-hour block with the day-ahead price as a "
           f"dashed line with markers; below, two EV-by-block grids (charge-on-arrival versus the optimal "
           f"schedule) in which each EV has its own fill, hatch and marker (A blue circle, B orange crosshatched "
           f"square, C green dotted triangle), white cells are available but idle and grey cells are not plugged "
           f"in; {finding}\n")
    if IN_BROWSER:
        return
    existing = ALT_TEXT_FILE.read_text(encoding="utf-8") if ALT_TEXT_FILE.exists() else ""
    lines = [ln for ln in existing.splitlines(keepends=True) if f"`{out_png.name}`" not in ln]
    if not lines:
        lines = ["# Figure alt text\n\n"]
    lines.append(alt)
    try_write(ALT_TEXT_FILE.name, lambda: ALT_TEXT_FILE.write_text("".join(lines), encoding="utf-8"))


def depot_section(df, night, out_csv: Path) -> float:
    """Depot-scale demo: 100 e-buses on one night, charged three ways. Returns seconds taken.

    Connection sizes: unmanaged peak, smallest cap that first-come-first-served charging
    survives (``min_cap_on_arrival_kw``) and smallest connection a smart schedule needs
    (``min_peak_kw``).
    """
    t0 = time.perf_counter()
    fleet = depot_fleet()
    energy_mwh = sum(ev.k_blocks * ev.power_kw for ev in fleet) / 1000.0
    free = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=1e12)
    cap_unmanaged = float(block_load_kw(free, on_arrival_capacity(free)).max())
    cap_fcfs = min_cap_on_arrival_kw(free)
    cap_smart = min_peak_kw(free)
    step(4, f"The real-scale problem: a {len(fleet)}-bus depot (classical optimizer)",
         f"{len(fleet)} e-buses (modelled on public figures) need {energy_mwh:.1f} MWh tonight, in 1-hour blocks: "
         f"{len(var_index(free))} yes/no decisions.\nToo large for today's qubits, so a MILP solver plans it.")
    print(f"  Smallest grid connection that still charges every bus in time:")
    for label, v in (("unmanaged (charge on arrival)", cap_unmanaged), ("first-come-first-served cap", cap_fcfs),
                     ("smart schedule", cap_smart)):
        print(f"    {label:<32}{v / 1000:5.2f} MW  {bar(v, cap_unmanaged, 34)}")
    print()
    rows = [dict(kind="connection", method=m, connection_kw=v)
            for m, v in (("unmanaged peak", cap_unmanaged), ("FCFS cap", cap_fcfs), ("smart minimum", cap_smart))]
    # carbon-first (w = 1) is the headline; cost-balanced (w = 0.7) is shown as a second row
    inst_u = replace(free, p_max_kw=cap_unmanaged, w=1.0)
    base = evaluate(inst_u, on_arrival_capacity(inst_u))
    inst_f = replace(free, p_max_kw=cap_fcfs, w=1.0)
    fcfs = evaluate(inst_f, on_arrival_capacity(inst_f))
    x_s, gap = milp_depot(inst_u)
    smart = evaluate(inst_u, x_s)
    peak_smart = float(block_load_kw(inst_u, x_s).max())
    inst_b = replace(inst_u, w=0.7)
    x_b, gap_b = milp_depot(inst_b)
    balanced = evaluate(inst_b, x_b)
    print(f"  Tonight ({night}):")
    print(f"    {'':<40}{'kg CO2':>9}{'EUR':>9}{'CO2 saved':>11}{'EUR saved':>11}")
    mw = f"{cap_unmanaged / 1000:.1f} MW"
    for name, ev, conn, g, w_, is_smart in (
            ("unmanaged (charge on arrival)", base, cap_unmanaged, np.nan, np.nan, False),
            ("first-come-first-served cap", fcfs, cap_fcfs, np.nan, np.nan, False),
            (f"smart, carbon-first ({mw})", smart, cap_unmanaged, gap, 1.0, True),
            (f"smart, cost-balanced ({mw})", balanced, cap_unmanaged, gap_b, 0.7, True)):
        sc = 100 * (1 - ev["co2_kg"] / base["co2_kg"])
        se = 100 * (1 - ev["cost_eur"] / base["cost_eur"])
        print(f"    {name:<40}{ev['co2_kg']:>9,.0f}{ev['cost_eur']:>9,.0f}{sc:>10.1f}%{se:>10.1f}%")
        rows.append(dict(kind="night", method=name, night=str(night), w=w_, n_buses=len(fleet),
                         energy_mwh=energy_mwh, connection_kw=conn, co2_kg=ev["co2_kg"], cost_eur=ev["cost_eur"],
                         co2_saved_pct=sc, eur_saved_pct=se, mip_gap=g,
                         peak_kw=float(block_load_kw(inst_u, x_s if w_ == 1.0 else x_b).max()) if is_smart else np.nan))
    try_write(out_csv.name, lambda: pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6g"))
    elapsed = time.perf_counter() - t0
    print(f"    (EUR = wholesale day-ahead energy; solved in {elapsed:.1f} s, within {100 * max(gap, gap_b):.1f} % "
          f"of the optimum)")
    return elapsed


def depot_month_demo(df, night, out_csv: Path, n_nights: int = 30) -> float:
    """Short version of experiments/e7_depot_year.py: the 100-bus depot over the last ``n_nights`` nights.

    Carbon-first (w = 1) smart charging vs unmanaged charging on the unmanaged connection.
    The full year (362 nights, two baselines, two weights) is in results/e7_summary.csv.
    Returns seconds taken.
    """
    t0 = time.perf_counter()
    fleet = depot_fleet()
    nights = [n for n in list_nights(df) if n <= night][-n_nights:]
    free = build_instance(df, nights[0], fleet=fleet, n_blocks=12, p_max_kw=1e12)
    cap = float(block_load_kw(free, on_arrival_capacity(free)).max())
    rows = []
    for n in nights:
        inst = build_instance(df, n, fleet=fleet, n_blocks=12, p_max_kw=cap, w=1.0)
        un = evaluate(inst, on_arrival_capacity(inst))
        x, gap = milp_depot(inst)
        sm = evaluate(inst, x)
        rows.append(dict(night=str(n), unmanaged_co2_kg=un["co2_kg"], smart_co2_kg=sm["co2_kg"],
                         unmanaged_cost_eur=un["cost_eur"], smart_cost_eur=sm["cost_eur"], mip_gap=gap))
    res = pd.DataFrame(rows)
    co2_u, co2_s = res.unmanaged_co2_kg.sum() / 1000, res.smart_co2_kg.sum() / 1000
    eur_u, eur_s = res.unmanaged_cost_eur.sum(), res.smart_cost_eur.sum()
    print(f"\n  The last {len(nights)} nights ({nights[0]} to {nights[-1]}), carbon-first, same "
          f"{cap / 1000:.1f} MW connection:")
    print(f"    CO2   {co2_u:8.1f} t unmanaged  ->  {co2_s:8.1f} t smart   ({100 * (1 - co2_s / co2_u):.1f} % less)")
    print(f"    EUR   {eur_u:8,.0f}   unmanaged  ->  {eur_s:8,.0f}   smart   ({100 * (1 - eur_s / eur_u):.1f} % less)")
    print("    Full year (362 nights): 252 t CO2 and EUR 124,000 less (experiments/e7_depot_year.py).")
    try_write(out_csv.name, lambda: res.to_csv(out_csv, index=False, float_format="%.6g"))
    elapsed = time.perf_counter() - t0
    print(f"    ({len(nights)} nights solved in {elapsed:.1f} s)")
    return elapsed


def hybrid_demo(df, night, out_csv: Path, n_buses: int = 16) -> float:
    """Short version of experiments/e8_hybrid_depot.py: QAOA as the subsolver inside a depot optimizer.

    A 16-bus depot on ``night`` (w = 1, first-come-first-served connection). Large-neighbourhood
    search re-optimizes 2-bus groups (at most 16 qubits each) for one pass, once with exact
    enumeration and once with simulated XY-mixer QAOA, and is compared with the global MILP optimum.
    The full experiment (100 buses, two nights, three passes, several seeds) is in
    results/e8_summary.csv. Returns seconds taken.
    """
    t0 = time.perf_counter()
    fleet = depot_fleet(n_buses=n_buses)
    free = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=1e12)
    cap = min_cap_on_arrival_kw(free)
    inst = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=cap, w=1.0)
    x0 = on_arrival_capacity(inst)
    x_opt, _ = milp_depot(inst, mip_rel_gap=0.0)
    co2_start, co2_opt = evaluate(inst, x0)["co2_kg"], evaluate(inst, x_opt)["co2_kg"]
    step(5, "QAOA inside the depot optimizer (hybrid quantum-classical search)",
         f"A {n_buses}-bus depot on {cap / 1000:.2f} MW. The search frees 2 buses at a time; QAOA (XY mixer, "
         f"simulated) re-plans their\nhours as one small QUBO and the new plan is kept if it lowers the depot's CO2.")
    print(f"  Start: first-come-first-served, {co2_start:,.0f} kg CO2.   Classical optimum (the yardstick): "
          f"{co2_opt:,.0f} kg CO2.\n")
    rows = []
    for name, sub in (("exact enumeration", "exact"), ("QAOA, XY mixer", "qaoa_xy")):
        if sub == "qaoa_xy":
            print("  Running the search with QAOA as the subsolver ...")
        r = lns_depot(inst, x0, subsolver=sub, max_passes=1, seed=0)
        co2 = evaluate(inst, r.x)["co2_kg"]
        closed = 100 * (co2_start - co2) / (co2_start - co2_opt) if co2_start > co2_opt else float("nan")
        if sub == "qaoa_xy":
            kg = inst.co2.mean() / 1000.0          # w = 1: objective x mean(co2) / 1000 = kg CO2
            for k, rec in enumerate(r.log, start=1):
                buses = " + ".join(f"bus {int(inst.evs[i].name[3:6])}" for i in rec["group"])
                if rec["kind"] == "trivial":
                    what = "nothing to choose"
                else:
                    what = "improved" if rec["improved"] else "kept the current plan"
                print(f"    {k:>2}/{len(r.log)}  {buses:<16}{rec['n_vars']:>3} qubits   "
                      f"depot CO2 {r.trace[k] * kg:7,.0f} kg   {what}")
            print()
        rows.append(dict(night=str(night), n_buses=n_buses, connection_kw=cap, subsolver=sub, passes=1,
                         co2_start_kg=co2_start, co2_final_kg=co2, co2_milp_kg=co2_opt, gap_closed_pct=closed,
                         n_subproblems=r.n_subproblems, wall_s=r.wall_s))
    q, e = rows[1], rows[0]
    print(f"  Result: QAOA as subsolver  {q['co2_final_kg']:,.0f} kg CO2, {q['gap_closed_pct']:.1f} % of the way "
          f"to the optimum")
    print(f"          exact enumeration  {e['co2_final_kg']:,.0f} kg CO2, {e['gap_closed_pct']:.1f} % (the same "
          f"subproblems solved classically)")
    print("  Full experiment, 100 buses and 3 passes: experiments/e8_hybrid_depot.py.")
    try_write(out_csv.name, lambda: pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6g"))
    return time.perf_counter() - t0


def backend_check(selected, inst, params, reps, opt_obj, shots, seed, out_csv: Path) -> None:
    """Run the trained XY circuit once on the backend chosen in Qollab's Select QPU dialog.

    The angles come from the simulator training; only the final sampling runs on the selected
    backend (one job). The same angles are also sampled on the ideal simulator with the same
    number of shots for comparison.
    """
    from qiskit import transpile

    from carbon_ev.qaoa import build_qaoa_circuit, evaluate_params, param_order, sample_metrics

    name = str(getattr(selected, "name", type(selected).__name__))
    qc = build_qaoa_circuit(inst, "xy", reps)
    bound = qc.assign_parameters(np.asarray(params, dtype=float)[param_order(qc)])
    section(f"Selected backend: {name}")
    print(f"The trained XY circuit (p = {reps}, {qc.num_qubits} qubits) runs there once with {shots} shots; "
          f"the angles come from the simulator training above.")
    t0 = time.perf_counter()
    try:
        tqc = transpile(bound, backend=selected, optimization_level=1)
        counts = selected.run(tqc, shots=shots).result().get_counts()
    except Exception as exc:  # a remote backend can be offline or refuse the job; the rest of the demo stays valid
        print(f"(the run on {name} did not complete: {type(exc).__name__}: {exc})")
        return
    counts = {k.replace(" ", ""): v for k, v in counts.items()}
    elapsed = time.perf_counter() - t0
    on_backend = sample_metrics(inst, counts, opt_obj)
    ideal = evaluate_params(inst, "xy", reps, params, shots=shots, seed=seed, opt_obj=opt_obj)
    rows = []
    print(f"{'':<22}{'P(feasible)':>12}{'P(<1%)':>10}{'apx.exp':>10}")
    for label, m in (("ideal simulator", dict(feasible_prob=ideal.feasible_prob, p_within_1pct=ideal.p_within_1pct,
                                                approx_ratio_expected=ideal.approx_ratio_expected)),
                     (name, on_backend)):
        print(f"{label[:22]:<22}{m['feasible_prob']:>12.3f}{m['p_within_1pct']:>10.4f}"
              f"{m['approx_ratio_expected']:>10.4f}")
        rows.append(dict(backend=label, mixer="xy", reps=reps, shots=shots, feasible_prob=m["feasible_prob"],
                         p_within_1pct=m["p_within_1pct"], approx_ratio_expected=m["approx_ratio_expected"]))
    print(f"({elapsed:.1f} s on {name}; transpiled two-qubit gates: "
          f"{sum(1 for i in tqc.data if i.operation.num_qubits == 2)})")
    try_write(out_csv.name, lambda: pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6g"))


def main(argv=None) -> int:
    """Run the demo; returns 0 on success, 1 if the requested night has no complete data."""
    ap = argparse.ArgumentParser(prog="main.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--night", default="2026-10-08", help="night start date YYYY-MM-DD (20:00 -> 08:00)")
    ap.add_argument("--reps", type=int, default=2, help="QAOA layers p for both mixers (default 2)")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-depot", action="store_true",
                    help="skip all depot sections (one night, 30-night month, QAOA inside the depot)")
    ap.add_argument("--no-month", action="store_true", help="skip the 30-night depot demo (short e7)")
    ap.add_argument("--no-hybrid", action="store_true", help="skip the QAOA-inside-the-depot demo (short e8)")
    ap.add_argument("--backend-shots", type=int, default=1000,
                    help="shots for the selected-backend check (only on Qollab with a non-built-in backend)")
    ap.add_argument("--out-prefix", default=None,
                    help="output file stem in results/ (default: main_demo for the main night, "
                         "main_demo_<night> otherwise)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.perf_counter()
    night = date.fromisoformat(args.night)
    df = load_hourly(ROOT / "data/nl_grid_hourly.csv")  # literal path: Qollab loads only data files named in a string
    if night not in list_nights(df):
        print(f"Night {night} has no complete 20:00-08:00 coverage. Available: "
              f"{', '.join(str(d) for d in list_nights(df))}")
        return 1
    inst = build_instance(df, night)
    n_qubits = len(var_index(inst))
    print(f"{'═' * WIDTH}\n Carbon-aware charging for an electric bus depot\n Classical optimizer + QAOA in Qiskit, "
          f"on real Dutch grid data  ·  Team Bountum\n{'═' * WIDTH}")
    print(" This demo runs five steps (about a minute):\n"
          "   1  the grid data for one night\n"
          f"   2  classical baselines on a {n_qubits}-qubit scheduling problem\n"
          "   3  QAOA in Qiskit on the same problem: circuit, training, sampling\n"
          "   4  the 100-bus depot, planned by the classical optimizer\n"
          "   5  QAOA as the subsolver inside the depot optimizer")

    step(1, f"Grid data: night of {night}, 20:00 to 08:00 (Energy-Charts, Netherlands)",
         "Carbon intensity and day-ahead price for each 2-hour block. Charging when the bars are short emits less.")
    for lbl, c, p in zip(inst.block_labels, inst.co2, inst.price):
        print(f"  {lbl}   {c:4.0f} g CO2/kWh  {bar(c, inst.co2.max(), 36)}   {p:6.1f} EUR/MWh")

    def timed(fn, *a, **k):
        s = time.perf_counter()
        out = fn(*a, **k)
        return out, time.perf_counter() - s

    rows = []
    x_arr, t_arr = timed(on_arrival, inst)
    base_ev = evaluate(inst, x_arr)
    x_med, med_obj, all_obj = median_feasible(inst)
    med_ev = evaluate(inst, x_med)
    base = (base_ev["co2_kg"], base_ev["cost_eur"])
    med = (med_ev["co2_kg"], med_ev["cost_eur"])
    rows.append(classical_row("on-arrival", inst, x_arr, t_arr, base, med))
    rows.append(classical_row("median feasible schedule", inst, x_med, np.nan, base, med))
    x_gr, t = timed(greedy, inst)
    rows.append(classical_row("greedy", inst, x_gr, t, base, med))
    dd, dc = default_penalties(inst)
    Q, off = to_qubo(inst, dd, dc)
    x_sa, t = timed(simulated_annealing, Q, off, seed=args.seed)
    rows.append(classical_row("SA (QUBO)", inst, x_sa, t, base, med))
    x_opt, t = timed(milp, inst)
    rows.append(classical_row("exact (MILP)", inst, x_opt, t, base, med))
    opt_obj = evaluate(inst, x_opt)["objective"]

    n_feas = len(all_obj)
    step(2, f"Classical baselines: {len(inst.evs)} EVs, 6 blocks, {n_qubits} yes/no decisions ({n_qubits} qubits)",
         f"Each EV needs a set number of 2-hour blocks before it leaves, and the site limit forbids some pairs "
         f"charging together.\nThere are {n_feas} valid schedules; the exact optimum is the yardstick for QAOA.")
    print_classical(pd.DataFrame(rows))

    rb = random_feasible_baseline(inst, args.shots, args.seed, opt_obj=opt_obj)
    rows.append(metrics_row(f"random sampling, {args.shots} shots", inst, rb["uniform"], base, med))
    rows.append(metrics_row(f"Dicke start only (p=0), {args.shots} shots", inst, rb["demand_satisfying"], base, med))

    step(3, f"QAOA in Qiskit on the same {n_qubits}-qubit problem",
         "Two encodings: a penalty QUBO with the standard X mixer, and an XY mixer that starts each EV in a Dicke "
         "state,\nso every sample already gives each EV its charging blocks. The XY circuit "
         f"({args.reps} layer{'s' if args.reps > 1 else ''}):")
    print(circuit_sketch(inst, args.reps))
    print(f"\n  Training the angles (CVaR objective, alpha = {TRAIN_CONFIG['cvar_alpha']}; "
          "the line shows the training cost falling):")
    trained = {}
    for name, mixer, label in (("QAOA penalty-X", "x", "penalty QAOA, X mixer"), ("QAOA XY", "xy", "XY QAOA")):
        res = run_qaoa(inst, mixer=mixer, reps=args.reps, shots=args.shots, seed=args.seed, opt_obj=opt_obj, **TRAIN_CONFIG)
        trained[mixer] = res.params
        print(f"    {label:<22}{sparkline(res.history, 40)}  {res.n_evals:>4} evaluations, {res.wall_s:5.1f} s")
        rows.append(metrics_row(f"{name} (p={args.reps})", inst, result_metrics(res), base, med,
                                depth=res.depth, n_2q=res.n_2q, n_evals=res.n_evals, wall_s=res.wall_s))

    table = pd.DataFrame(rows)
    table.insert(0, "night", str(night))
    table.insert(1, "reps", args.reps)
    table.insert(2, "seed", args.seed)
    stem = args.out_prefix or ("main_demo" if args.night == "2026-10-08" else f"main_demo_{args.night}")
    try_write(f"{stem}.csv", lambda: table.to_csv(RESULTS / f"{stem}.csv", index=False, float_format="%.6g"))
    print()
    print_qaoa(table, args.shots)
    xy = table[table.method.str.startswith("QAOA XY")].iloc[0]
    start = table[table.method.str.startswith("Dicke start only")].iloc[0]
    opt_sched = table[table.method == "exact (MILP)"].iloc[0].schedule
    if start.p_within_1pct > 0:
        print(f"\n  XY QAOA samples near-optimal schedules {xy.p_within_1pct / start.p_within_1pct:.1f}x as often as its "
              f"untrained start, and the average quality\n  of its valid samples rises from "
              f"{start.approx_ratio_expected:.2f} to {xy.approx_ratio_expected:.2f}.")
    print(f"  Its best sample is the exact optimum ({opt_sched}). With 3 layers the near-optimal rate is 39x the "
          f"start\n  (experiment e1), and the trained circuit also ran on IonQ Forte 1 hardware (see the README).")

    if not args.no_plot:
        opt_row = table[table.method == "exact (MILP)"].iloc[0]
        make_figure(inst, x_arr, x_opt, opt_row.saved_co2_pct, RESULTS / f"{stem}.png")

    depot_s = month_s = hybrid_s = float("nan")
    if not args.no_depot:
        depot_s = depot_section(df, night, RESULTS / f"{stem}_depot.csv")
        if not args.no_month:
            month_s = depot_month_demo(df, night, RESULTS / f"{stem}_e7_month.csv")
        if not args.no_hybrid:
            hybrid_s = hybrid_demo(df, night, RESULTS / f"{stem}_e8_mini.csv")

    selected = globals().get("backend")  # Qollab pre-creates `backend` from the Select QPU dialog
    if selected is not None and "basic" not in str(getattr(selected, "name", "")).lower():
        backend_check(selected, inst, trained["xy"], args.reps, opt_obj, args.backend_shots, args.seed,
                      RESULTS / f"{stem}_backend.csv")

    total_s = time.perf_counter() - t0
    print(f"\n{'═' * WIDTH}\n Done in {total_s:.0f} s. Tables and the figure are in {RESULTS.name}/.\n"
          " Full experiments: experiments/e7_depot_year.py (the depot over a year) and "
          "experiments/e8_hybrid_depot.py\n (QAOA inside the 100-bus depot); all results and figures are in the README.\n"
          f"{'═' * WIDTH}")
    # timings are saved so that the documentation can cite them (per-method wall_s is in the main table CSV)
    runtime = pd.DataFrame([dict(night=str(night), reps=args.reps, shots=args.shots, seed=args.seed,
                       total_runtime_s=total_s, depot_section_s=depot_s, month_demo_s=month_s, hybrid_demo_s=hybrid_s,
                       qaoa_x_wall_s=float(table.loc[table.method.str.startswith("QAOA penalty-X"), "wall_s"].iloc[0]),
                       qaoa_xy_wall_s=float(table.loc[table.method.str.startswith("QAOA XY"), "wall_s"].iloc[0]),
                       note="wall-clock seconds on the machine that ran it; total includes plotting and the depot section")])
    try_write(f"{stem}_runtime.csv",
              lambda: runtime.to_csv(RESULTS / f"{stem}_runtime.csv", index=False, float_format="%.4g"))
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:  # no SystemExit on success: browser runtimes may report it as an error
        sys.exit(exit_code)
