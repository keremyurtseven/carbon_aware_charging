"""Problem definition: night instance, fleet, constraints and evaluation.

Variables
---------
One binary x[i, t] per EV i and block t in ``EV.available``. ``var_index(inst)``
lists the (i, t) pairs in a fixed order (EV-major, then block ascending); the
position in that list is the variable number a. A solution ``x`` is a 0/1
numpy array of length ``len(var_index(inst))`` in that order. Qubit a of the
quantum circuit carries x[a] (see ``qubo.py`` for the bit-order convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd

from carbon_ev.data import NIGHT_HOURS, night_frame


@dataclass(frozen=True)
class EV:
    """An EV: charger power, number of blocks it must charge, blocks it is plugged in."""

    name: str
    power_kw: float
    k_blocks: int
    available: tuple[int, ...]


# Modelling assumption, not data: chargers of 11, 9.2 and 7.4 kW (9.2 kW breaks ties between equal chargers).
DEFAULT_FLEET: list[EV] = [
    EV("A", 11.0, 2, (0, 1, 2, 3, 4, 5)),
    EV("B", 9.2, 1, (0, 1, 2, 3, 4)),
    EV("C", 7.4, 2, (1, 2, 3, 4, 5)),
]
DEFAULT_P_MAX_KW = 19.0


def block_cost(co2: np.ndarray, price: np.ndarray, w: float) -> np.ndarray:
    """Dimensionless block cost c_t(w) = w*co2_t/mean(co2) + (1-w)*price_t/mean(price)."""
    return w * co2 / co2.mean() + (1.0 - w) * price / price.mean()


@dataclass
class Instance:
    """One night, one fleet. ``co2``/``price`` are per-block means (g/kWh, EUR/MWh).

    ``cost_weights`` is c_t(w), derived in ``__post_init__`` (do not pass it).
    """

    night: date
    block_hours: float
    block_labels: list[str]
    co2: np.ndarray
    price: np.ndarray
    evs: list[EV]
    p_max_kw: float
    w: float
    cost_weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.co2 = np.asarray(self.co2, dtype=float)
        self.price = np.asarray(self.price, dtype=float)
        self.cost_weights = block_cost(self.co2, self.price, self.w)

    @property
    def n_blocks(self) -> int:
        return len(self.co2)


def build_instance(
    df: pd.DataFrame,
    night: date,
    fleet: list[EV] | None = None,
    n_blocks: int = 6,
    w: float = 0.7,
    p_max_kw: float = DEFAULT_P_MAX_KW,
) -> Instance:
    """Build the instance for the night D 20:00 -> D+1 08:00.

    The 12 hourly values are averaged into ``n_blocks`` equal blocks (n_blocks
    must divide 12; default 6 blocks of 2 h). The fleet's ``available`` blocks
    refer to these block numbers, so a fleet written for 6 blocks must be
    adapted if n_blocks differs.
    """
    fleet = DEFAULT_FLEET if fleet is None else fleet
    if NIGHT_HOURS % n_blocks != 0:
        raise ValueError(f"n_blocks must divide {NIGHT_HOURS}, got {n_blocks}")
    sub = night_frame(df, night)
    if sub is None:
        raise ValueError(f"night {night} does not have complete 20:00-08:00 coverage")
    per = NIGHT_HOURS // n_blocks
    co2 = sub["co2_g_per_kwh"].to_numpy().reshape(n_blocks, per).mean(axis=1)
    price = sub["price_eur_per_mwh"].to_numpy().reshape(n_blocks, per).mean(axis=1)
    labels = [f"{(20 + b * per) % 24:02d}-{(20 + (b + 1) * per) % 24:02d}" for b in range(n_blocks)]
    for ev in fleet:
        if any(t < 0 or t >= n_blocks for t in ev.available):
            raise ValueError(f"EV {ev.name}: available blocks outside 0..{n_blocks - 1}")
        if ev.k_blocks > len(ev.available):
            raise ValueError(f"EV {ev.name}: needs more blocks than available")
    return Instance(night, float(per), labels, co2, price, list(fleet), float(p_max_kw), float(w))


def var_index(inst: Instance) -> list[tuple[int, int]]:
    """(ev index i, block t) for every variable/qubit, EV-major then block ascending."""
    return [(i, t) for i, ev in enumerate(inst.evs) for t in sorted(ev.available)]


def conflicts(inst: Instance) -> list[tuple[int, int]]:
    """EV pairs (i < j) whose combined power exceeds the site limit."""
    return [
        (i, j)
        for i, j in combinations(range(len(inst.evs)), 2)
        if inst.evs[i].power_kw + inst.evs[j].power_kw > inst.p_max_kw + 1e-9
    ]


def site_limit_is_pairwise_exact(evs: list[EV], p_max_kw: float) -> bool:
    """Brute-force check that "total power <= p_max" equals "no conflicting pair".

    For every subset S of EVs charging in the same block, the site limit holds
    iff sum(P_i, i in S) <= p_max. The pairwise encoding says it holds iff no
    pair in S has P_i + P_j > p_max. Returns True iff the two agree for all
    subsets, i.e. the QUBO needs no higher-order terms or slack bits.
    """
    n = len(evs)
    for r in range(n + 1):
        for s in combinations(range(n), r):
            limit_ok = sum(evs[i].power_kw for i in s) <= p_max_kw + 1e-9
            pair_ok = all(
                evs[i].power_kw + evs[j].power_kw <= p_max_kw + 1e-9 for i, j in combinations(s, 2)
            )
            if limit_ok != pair_ok:
                return False
    return True


def objective_coefficients(inst: Instance) -> np.ndarray:
    """Linear objective coefficient P_i * h * c_t(w) of every variable (var_index order)."""
    return np.array(
        [inst.evs[i].power_kw * inst.block_hours * inst.cost_weights[t] for i, t in var_index(inst)]
    )


def evaluate(inst: Instance, x: np.ndarray) -> dict:
    """Evaluate a 0/1 vector ``x`` (var_index order).

    Returns dict with
      objective : sum x*P_i*h*c_t(w)   (dimensionless weighted cost)
      co2_kg    : sum x*P_i*h*co2_t/1000
      cost_eur  : sum x*P_i*h*price_t/1000
      violations: {"demand": sum_i (sum_t x_it - k_i)^2,
                   "conflict": number of (conflicting pair, block) both charging}
                  These are exactly the quantities multiplied by lam_demand and
                  lam_conflict in the QUBO.
      feasible  : True iff both violation counts are 0.
    """
    x = np.asarray(x)
    vi = var_index(inst)
    if x.shape != (len(vi),):
        raise ValueError(f"x must have length {len(vi)}, got shape {x.shape}")
    xi = x.astype(int)
    kwh = xi * np.array([inst.evs[i].power_kw * inst.block_hours for i, _ in vi])
    t_idx = np.array([t for _, t in vi])
    objective = float(np.sum(kwh * inst.cost_weights[t_idx]))
    co2_kg = float(np.sum(kwh * inst.co2[t_idx]) / 1000.0)
    cost_eur = float(np.sum(kwh * inst.price[t_idx]) / 1000.0)

    charging = {v for v, xv in zip(vi, xi) if xv}
    demand = 0
    for i, ev in enumerate(inst.evs):
        n_charged = sum(1 for (j, _) in charging if j == i)
        demand += (n_charged - ev.k_blocks) ** 2
    conflict = sum(
        1
        for (i, j) in conflicts(inst)
        for t in range(inst.n_blocks)
        if (i, t) in charging and (j, t) in charging
    )
    return {
        "objective": objective,
        "co2_kg": co2_kg,
        "cost_eur": cost_eur,
        "feasible": bool(demand == 0 and conflict == 0),
        "violations": {"demand": int(demand), "conflict": int(conflict)},
    }
