"""Classical baselines on the same instance and objective.

Every function returns ``x`` as a 0/1 numpy array in ``problem.var_index`` order
(``brute_force`` returns it together with the list of all feasible schedules).

* ``on_arrival``  - no smart charging: what happens by default.
* ``greedy``      - cheapest (EV, block) first, never closing off feasibility.
* ``simulated_annealing`` - single-bit-flip annealing on the penalty QUBO.
* ``brute_force`` - exact, enumerates feasible schedules only.
* ``milp``        - exact, ``scipy.optimize.milp`` with the true site power limit.
"""
from __future__ import annotations

from itertools import combinations, product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp as _scipy_milp

from carbon_ev.problem import Instance, conflicts, objective_coefficients, var_index


def _to_x(inst: Instance, schedule: dict[int, set[int]]) -> np.ndarray:
    """{ev index: set of blocks} -> 0/1 vector in var_index order."""
    return np.array([1 if t in schedule.get(i, ()) else 0 for i, t in var_index(inst)], dtype=int)


def _overlaps(conf: set[tuple[int, int]], i: int, j: int) -> bool:
    return (min(i, j), max(i, j)) in conf


def on_arrival(inst: Instance) -> np.ndarray:
    """Charge-on-arrival baseline.

    EVs are served in arrival order (arrival = first available block; ties are
    broken by the order of ``inst.evs``). Each EV takes its earliest available
    blocks, skipping any block where a conflicting EV that was served earlier
    is already charging (so it shifts later). Raises ValueError if an EV cannot
    get its k blocks at all.
    """
    conf = set(conflicts(inst))
    order = sorted(range(len(inst.evs)), key=lambda i: (min(inst.evs[i].available), i))
    sched: dict[int, set[int]] = {}
    for i in order:
        ev = inst.evs[i]
        taken: list[int] = []
        for t in sorted(ev.available):
            if len(taken) == ev.k_blocks:
                break
            if any(t in sched[j] and _overlaps(conf, i, j) for j in sched):
                continue
            taken.append(t)
        if len(taken) < ev.k_blocks:
            raise ValueError(f"EV {ev.name} cannot be served on arrival")
        sched[i] = set(taken)
    return _to_x(inst, sched)


def _completable(inst: Instance, conf: set[tuple[int, int]], sched: dict[int, set[int]]) -> bool:
    """Can the partial schedule be extended to a feasible one? (DFS over EVs, small instances.)"""
    n = len(inst.evs)

    def ok_add(i: int, blocks: tuple[int, ...], cur: dict[int, set[int]]) -> bool:
        return not any(t in cur[j] and _overlaps(conf, i, j) for t in blocks for j in cur if j != i)

    # the partial assignment itself must be conflict-free and not exceed k
    for i in sched:
        if len(sched[i]) > inst.evs[i].k_blocks:
            return False
        if not ok_add(i, tuple(sched[i]), sched):
            return False

    def rec(i: int, cur: dict[int, set[int]]) -> bool:
        if i == n:
            return True
        ev = inst.evs[i]
        have = cur.get(i, set())
        need = ev.k_blocks - len(have)
        free = [t for t in sorted(ev.available) if t not in have]
        for extra in combinations(free, need):
            trial = dict(cur)
            trial[i] = have | set(extra)
            if ok_add(i, extra, trial) and rec(i + 1, trial):
                return True
        return False

    return rec(0, {i: set(s) for i, s in sched.items()})


def greedy(inst: Instance) -> np.ndarray:
    """Greedy: repeatedly add the cheapest (EV, block) (lowest P*h*c_t) whose addition
    still allows a feasible completion. Ties broken by var_index order."""
    conf = set(conflicts(inst))
    vi = var_index(inst)
    coef = objective_coefficients(inst)
    order = np.argsort(coef, kind="stable")
    sched: dict[int, set[int]] = {i: set() for i in range(len(inst.evs))}
    for a in order:
        i, t = vi[a]
        if len(sched[i]) >= inst.evs[i].k_blocks:
            continue
        sched[i].add(t)
        if not _completable(inst, conf, sched):
            sched[i].discard(t)
    return _to_x(inst, sched)


def simulated_annealing(
    Q: np.ndarray,
    offset: float,
    seed: int = 0,
    n_sweeps: int = 3000,
    t_start: float | None = None,
    t_end: float | None = None,
) -> np.ndarray:
    """Single-bit-flip simulated annealing on E(x) = x^T Q x + offset.

    Geometric cooling from t_start to t_end over ``n_sweeps`` sweeps (each sweep
    proposes every bit once, in random order). Default temperatures scale with
    the largest |Q| entry. Returns the lowest-energy x seen (a QUBO minimiser,
    not necessarily feasible if the penalties are weak). ``offset`` does not
    affect the search; it is accepted for interface symmetry.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    S = Q + Q.T
    np.fill_diagonal(S, 0.0)
    diag = np.diag(Q).copy()
    scale = float(np.max(np.abs(Q))) or 1.0
    t_start = scale if t_start is None else t_start
    t_end = 0.01 * scale if t_end is None else t_end
    cool = (t_end / t_start) ** (1.0 / max(n_sweeps - 1, 1))

    x = rng.integers(0, 2, size=n).astype(float)
    field = S @ x  # sum_b S_ab x_b
    energy = float(x @ Q @ x)
    best_x, best_e = x.copy(), energy
    temp = t_start
    for _ in range(n_sweeps):
        for a in rng.permutation(n):
            delta = (1.0 - 2.0 * x[a]) * (diag[a] + field[a])
            if delta <= 0 or rng.random() < np.exp(-delta / temp):
                field += S[:, a] * (1.0 - 2.0 * x[a])
                x[a] = 1.0 - x[a]
                energy += delta
                if energy < best_e - 1e-12:
                    best_e, best_x = energy, x.copy()
        temp *= cool
    return best_x.astype(int)


def brute_force(inst: Instance) -> tuple[np.ndarray, list[tuple[np.ndarray, float]]]:
    """Exact optimum by enumerating feasible schedules only.

    Enumerates every choice of k_i blocks for each EV (product of combinations),
    drops those with a conflict, and returns (best_x, [(x, objective), ...]) with
    the list sorted by objective ascending (best_x is its first element).
    """
    conf = list(conflicts(inst))
    vi = var_index(inst)
    pos = {v: a for a, v in enumerate(vi)}
    coef = objective_coefficients(inst)
    choices = [list(combinations(sorted(ev.available), ev.k_blocks)) for ev in inst.evs]
    out: list[tuple[np.ndarray, float]] = []
    for combo in product(*choices):
        sets = [set(c) for c in combo]
        if any(sets[i] & sets[j] for i, j in conf):
            continue
        x = np.zeros(len(vi), dtype=int)
        for i, blocks in enumerate(combo):
            for t in blocks:
                x[pos[(i, t)]] = 1
        out.append((x, float(coef @ x)))
    if not out:
        raise ValueError("no feasible schedule")
    out.sort(key=lambda r: r[1])
    return out[0][0], out


def milp(inst: Instance) -> np.ndarray:
    """Exact optimum with scipy.optimize.milp.

    min coef^T x  s.t.  sum_t x_it = k_i (each EV)  and, for each block t,
    sum_i P_i x_it <= P_max. The power constraint is used directly (not the
    pairwise-conflict reduction), so it is exact for any fleet and any number
    of blocks, e.g. an hourly-resolution instance.
    """
    vi = var_index(inst)
    n = len(vi)
    coef = objective_coefficients(inst)
    rows, lo, hi = [], [], []
    for i, ev in enumerate(inst.evs):
        rows.append([1.0 if vi[a][0] == i else 0.0 for a in range(n)])
        lo.append(ev.k_blocks)
        hi.append(ev.k_blocks)
    for t in range(inst.n_blocks):
        row = [inst.evs[vi[a][0]].power_kw if vi[a][1] == t else 0.0 for a in range(n)]
        if any(row):
            rows.append(row)
            lo.append(-np.inf)
            hi.append(inst.p_max_kw)
    res = _scipy_milp(
        c=coef,
        constraints=LinearConstraint(np.array(rows), np.array(lo), np.array(hi)),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        raise ValueError(f"MILP failed: {res.message}")
    return np.rint(res.x).astype(int)
