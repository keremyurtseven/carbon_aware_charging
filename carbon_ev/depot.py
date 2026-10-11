"""Electric bus depot at realistic scale (classical; the QAOA benchmark stays small).

A Dutch bus depot charges its e-buses overnight under a fixed grid connection.
This module builds such a depot as an ``Instance`` with hourly resolution
(20:00 -> 08:00, 12 one-hour blocks) so the same objective, ``evaluate`` and
exact ``classical.milp`` apply, and adds what a depot needs on top:

* ``depot_fleet``: a synthetic fleet whose parameters are modelled on public
  figures (see ``DEPOT_SOURCES``). It is a modelling assumption, not data.
* ``on_arrival_capacity``: the do-nothing baseline under a kW limit (each bus
  starts charging when it arrives, if the connection has room).
* ``min_cap_on_arrival_kw``: the smallest connection simple first-come-first-
  served load capping can run on (no look-ahead).
* ``min_peak_kw``: the smallest grid connection that still charges every bus
  in time (MILP, 0.5 % proven gap).
* ``milp_depot``: the cost/CO2 optimum at depot scale (MILP, 0.2 % proven gap).

Only the carbon intensity and day-ahead prices are measured data.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp as _scipy_milp

from carbon_ev.problem import EV, Instance, objective_coefficients, var_index

# Public figures the synthetic fleet is modelled on (checked Oct 10, 2026).
DEPOT_SOURCES = {
    "fleet_size": "100 e-buses on the Amstelland-Meerlanden concession, 13 MW of charging at the "
                  "Amstelveen/Schiphol depots: https://www.electrive.com/2018/03/28/"
                  "extra-large-13-mw-charge-depot-for-100-e-buses-live-in-amsterdam",
    "consumption": "Dutch e-buses: 1.15 kWh/km (12 m), 1.63 kWh/km (18 m), ViriCiti data via "
                   "https://sustainable-bus.com/wp-json/wp/v2/posts/10542",
    "daily_km": "About 265 km per bus per day at an overnight-charged Dutch depot (Groningen): "
                "https://www.LowCVP.org.uk/assets/presentations/"
                "6._EVBus_Rollout_Experiences_from_NL-BIbiFabius_EVConsult.pdf",
}

KWH_PER_KM = {"12m": 1.15, "18m": 1.63}


def depot_fleet(
    n_buses: int = 100,
    share_18m: float = 0.3,
    km_mean: float = 265.0,
    km_spread: float = 40.0,
    charger_kw: float = 100.0,
    seed: int = 0,
) -> list[EV]:
    """Synthetic depot fleet on the hourly 20:00-08:00 grid (blocks 0..11).

    Each bus drives ``km_mean`` +/- ``km_spread`` km a day (uniform) and uses
    the measured kWh/km of its length, so it needs E kWh. On a charger rated
    ``charger_kw`` that takes k = ceil(E / charger_kw) one-hour blocks; the bus
    then draws the constant power E / k kW (<= charger_kw) in each of them, so
    the delivered energy equals the need exactly (no rounding surplus).
    Modelling assumptions (ours, not data): charging can pause and resume
    between blocks, power is constant within a block (no battery taper), buses
    return between 20:00 and 00:00 and leave between 05:00 and 07:00 (uniform
    whole hours), 100 kW chargers. Fixed ``seed`` -> same fleet.
    """
    rng = np.random.default_rng(seed)
    fleet = []
    for b in range(n_buses):
        length = "18m" if b < round(share_18m * n_buses) else "12m"
        km = rng.uniform(km_mean - km_spread, km_mean + km_spread)
        need_kwh = km * KWH_PER_KM[length]
        arrive = int(rng.integers(0, 5))       # 20:00 .. 00:00 -> blocks 0..4
        leave = int(rng.integers(9, 12))       # 05:00 .. 07:00 -> blocks 9..11 (exclusive end)
        available = tuple(range(arrive, leave))
        k = min(math.ceil(need_kwh / charger_kw), len(available))
        power = round(need_kwh / k, 1)         # kW drawn while charging (1-h blocks)
        fleet.append(EV(f"bus{b:03d}_{length}", power, k, available))
    return fleet


def block_load_kw(inst: Instance, x: np.ndarray) -> np.ndarray:
    """Site load per block (kW) of schedule ``x``."""
    load = np.zeros(inst.n_blocks)
    for (i, t), xv in zip(var_index(inst), np.asarray(x)):
        if xv:
            load[t] += inst.evs[i].power_kw
    return load


def is_feasible_kw(inst: Instance, x: np.ndarray, p_max_kw: float | None = None) -> bool:
    """Every bus gets its k blocks and the site load never exceeds the limit."""
    x = np.asarray(x)
    limit = inst.p_max_kw if p_max_kw is None else p_max_kw
    vi = var_index(inst)
    for i, ev in enumerate(inst.evs):
        if sum(int(xv) for (j, _), xv in zip(vi, x) if j == i) != ev.k_blocks:
            return False
    return bool(np.all(block_load_kw(inst, x) <= limit + 1e-6))


def on_arrival_capacity(inst: Instance) -> np.ndarray:
    """Charge-on-arrival under the kW limit (the do-nothing baseline at depot scale).

    Buses are served in arrival order (ties: fleet order). Each takes its
    earliest available blocks that still have room on the connection. Raises
    ValueError if some bus cannot be fully charged that way.
    """
    vi = var_index(inst)
    pos = {v: a for a, v in enumerate(vi)}
    x = np.zeros(len(vi), dtype=int)
    load = np.zeros(inst.n_blocks)
    order = sorted(range(len(inst.evs)), key=lambda i: (min(inst.evs[i].available), i))
    for i in order:
        ev = inst.evs[i]
        got = 0
        for t in sorted(ev.available):
            if got == ev.k_blocks:
                break
            if load[t] + ev.power_kw <= inst.p_max_kw + 1e-6:
                x[pos[(i, t)]] = 1
                load[t] += ev.power_kw
                got += 1
        if got < ev.k_blocks:
            raise ValueError(f"{ev.name} cannot be charged on arrival under {inst.p_max_kw} kW")
    return x


def min_cap_on_arrival_kw(inst: Instance, step_kw: float = 100.0) -> float:
    """Smallest site limit (multiple of ``step_kw``) under which first-come-first-served
    charge-on-arrival (``on_arrival_capacity``) still charges every bus.

    This is what simple load capping without any look-ahead can achieve, a fairer
    reference than the unmanaged peak when talking about grid connection size.
    """
    free = replace(inst, p_max_kw=1e12)
    hi = float(block_load_kw(free, on_arrival_capacity(free)).max())
    lo = 0.0
    hi = math.ceil(hi / step_kw) * step_kw
    while hi - lo > step_kw:                   # feasibility is monotone in the cap
        mid = math.ceil((lo + hi) / 2 / step_kw) * step_kw
        try:
            on_arrival_capacity(replace(inst, p_max_kw=mid))
            hi = mid
        except ValueError:
            lo = mid
    return hi


def _rows(inst: Instance, peak_column: bool):
    """Demand rows (== k_i) and per-block power rows (<= P_max, or <= peak variable)."""
    vi = var_index(inst)
    n = len(vi)
    extra = [0.0] if peak_column else []
    rows, lo, hi = [], [], []
    for i, ev in enumerate(inst.evs):
        rows.append([1.0 if vi[a][0] == i else 0.0 for a in range(n)] + extra)
        lo.append(ev.k_blocks)
        hi.append(ev.k_blocks)
    for t in range(inst.n_blocks):
        row = [inst.evs[vi[a][0]].power_kw if vi[a][1] == t else 0.0 for a in range(n)]
        if any(row):
            rows.append(row + ([-1.0] if peak_column else []))
            lo.append(-np.inf)
            hi.append(0.0 if peak_column else inst.p_max_kw)
    return LinearConstraint(np.array(rows), np.array(lo), np.array(hi)), n


def _gap(res) -> float:
    bound = getattr(res, "mip_dual_bound", None)
    if bound is None or res.fun == 0:
        return 0.0
    return float(abs(res.fun - bound) / abs(res.fun))


def milp_depot(inst: Instance, mip_rel_gap: float = 2e-3, time_limit: float = 60.0) -> tuple[np.ndarray, float]:
    """Depot-scale optimum of the same objective as ``classical.milp``.

    Unequal bus powers make the per-block power limit a packing problem, so the
    solver stops at a proven relative optimality gap (default 0.2 %) instead of
    proving exact optimality. Returns (x, proven relative gap).
    """
    cons, n = _rows(inst, peak_column=False)
    res = _scipy_milp(c=objective_coefficients(inst), constraints=cons, integrality=np.ones(n),
                      bounds=Bounds(0, 1), options={"mip_rel_gap": mip_rel_gap, "time_limit": time_limit})
    if res.x is None:
        raise ValueError(f"depot MILP found no feasible schedule: {res.message}")
    return np.rint(res.x).astype(int), _gap(res)


def min_peak_kw(inst: Instance, mip_rel_gap: float = 5e-3, time_limit: float = 60.0) -> float:
    """Smallest site limit (kW) under which every bus can still be fully charged.

    MILP with a peak variable; stops at a proven relative gap (default 0.5 %),
    so the value is within that tolerance of the true minimum (and always
    achievable: it comes with a feasible schedule).
    """
    cons, n = _rows(inst, peak_column=True)
    c = np.zeros(n + 1)
    c[-1] = 1.0
    res = _scipy_milp(c=c, constraints=cons, integrality=np.r_[np.ones(n), 0],
                      bounds=Bounds(np.zeros(n + 1), np.r_[np.ones(n), np.inf]),
                      options={"mip_rel_gap": mip_rel_gap, "time_limit": time_limit})
    if res.x is None:
        raise ValueError(f"min-peak MILP failed: {res.message}")
    return float(res.x[-1])
