"""Hybrid quantum-classical large-neighbourhood search (LNS) for the e-bus depot.

The depot problem (``depot.py``, ~100 buses, ~900 binary variables) is far too
big for QAOA on a laptop simulator. LNS uses QAOA where it fits:

1. Start from a feasible schedule ``x0`` (e.g. ``depot.on_arrival_capacity``).
2. Pick a small group of buses (default 2, at most ``max_vars`` = 16 qubits).
   Free only their variables; all other buses stay fixed at their current hours.
3. The fixed buses leave a residual capacity ``R_t = cap - load_t`` per block.
   The freed buses must (a) get exactly ``k_i`` blocks each and (b) respect
   ``sum_i P_i x_it <= R_t``. Variables with ``P_i > R_t`` are removed. For two
   buses (a pair) this is exactly a QUBO with one conflict term
   ``x_it * x_jt`` wherever ``P_i + P_j > R_t`` (no slack bits needed).
4. A subsolver reoptimises the group: ``"exact"`` (enumeration), ``"qaoa_xy"``
   (Dicke start + ring-XY mixer, only conflicts penalised) or ``"qaoa_x"``
   (|+> start, X mixer, demand and conflicts penalised).
5. The new assignment is accepted only if it is feasible and strictly cheaper;
   otherwise the current one is kept. The schedule can never get worse or
   infeasible, whatever the subsolver returns.

Notes
-----
* Groups are limited to ``group_size`` 1 or 2. Beyond pairs the capacity
  constraint is no longer pairwise-exact (it would need slack bits).
* For QAOA runs the exact enumeration of the subproblem is ALSO computed, but
  only for diagnostics (QAOA gap to the subproblem optimum, the Dicke-start
  baseline); the accept/reject decision uses the QAOA sample only. One
  exception: subproblems with at most one feasible assignment (nothing to
  choose) are detected by that enumeration and skipped before QAOA runs.
* Angles are trained with ``qaoa.ExactSimulator`` (a numpy copy of the circuit,
  tested against ``Statevector``) and, by default, the 4096 shots are drawn from
  its exact probabilities (``sample_with="exact"``). ``sample_with="sampler"``
  runs the real circuit through ``StatevectorSampler`` instead (about 1-2 s
  slower per subproblem, same distribution).
* The subsolver sees a tiny piece of a big problem. At this size the exact
  enumeration is faster and always optimal.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from carbon_ev.depot import is_feasible_kw
from carbon_ev.problem import Instance, objective_coefficients, var_index
from carbon_ev.qaoa import (
    TRAIN_CONFIG,
    ExactSimulator,
    build_qaoa_circuit_from_qubo,
    counts_to_matrix,
    default_init_params,
    ramp_init_params,
    sample_counts,
)

TOL = 1e-6          # kW tolerance, same as depot.is_feasible_kw
GAIN_TOL = 1e-9     # minimum objective improvement that counts


# ---------------------------------------------------------------------------
# Subproblem construction
# ---------------------------------------------------------------------------
@dataclass
class _Context:
    """Per-run constants of the full depot problem."""

    vi: list[tuple[int, int]]
    coef: np.ndarray                  # objective coefficient of every variable
    power: np.ndarray                 # bus power of every variable
    block: np.ndarray                 # block of every variable
    bus_cols: list[list[int]]         # global columns of each bus
    n_blocks: int


def _context(inst: Instance) -> _Context:
    """Precompute the per-variable arrays of the full problem."""
    vi = var_index(inst)
    bus_cols: list[list[int]] = [[] for _ in inst.evs]
    for a, (i, _t) in enumerate(vi):
        bus_cols[i].append(a)
    return _Context(
        vi=vi,
        coef=objective_coefficients(inst),
        power=np.array([inst.evs[i].power_kw for i, _ in vi]),
        block=np.array([t for _, t in vi]),
        bus_cols=bus_cols,
        n_blocks=inst.n_blocks,
    )


@dataclass
class _Sub:
    """The LNS subproblem of one group of buses."""

    group: tuple[int, ...]
    var_list: list[tuple[int, int]]   # (bus, block) per local variable / qubit
    cols: np.ndarray                  # global column per local variable
    coef: np.ndarray
    power: np.ndarray
    blocks: np.ndarray
    registers: list[list[int]]        # local variable indices per bus of the group
    ks: list[int]
    residual: np.ndarray              # R_t per block (kW)
    x_local: np.ndarray               # current assignment restricted to the variables

    @property
    def n(self) -> int:
        return len(self.var_list)

    @property
    def current_cost(self) -> float:
        return float(self.coef @ self.x_local)

    def block_matrix(self) -> np.ndarray:
        """(n_vars, n_blocks) matrix M with M[a, t] = P_a if variable a sits in block t."""
        M = np.zeros((self.n, len(self.residual)))
        M[np.arange(self.n), self.blocks] = self.power
        return M


def _build_sub(inst: Instance, ctx: _Context, x: np.ndarray, group: tuple[int, ...]) -> _Sub:
    """Free the buses in ``group``; the other buses stay fixed and set the residual capacity."""
    x = np.asarray(x, dtype=int)
    load = np.bincount(ctx.block, weights=ctx.power * x, minlength=ctx.n_blocks)
    group_cols = np.array([c for i in group for c in ctx.bus_cols[i]], dtype=int)
    load = load - np.bincount(ctx.block[group_cols], weights=(ctx.power * x)[group_cols],
                              minlength=ctx.n_blocks)
    residual = inst.p_max_kw - load
    cols, registers, ks = [], [], []
    for i in group:
        reg = []
        for c in ctx.bus_cols[i]:
            if ctx.power[c] <= residual[ctx.block[c]] + TOL:     # drop variables that can never fit
                reg.append(len(cols))
                cols.append(c)
        registers.append(reg)
        ks.append(inst.evs[i].k_blocks)
    cols_a = np.array(cols, dtype=int)
    sub = _Sub(
        group=tuple(group), var_list=[ctx.vi[c] for c in cols], cols=cols_a, coef=ctx.coef[cols_a],
        power=ctx.power[cols_a], blocks=ctx.block[cols_a], registers=registers, ks=ks,
        residual=residual, x_local=x[cols_a],
    )
    for reg, k in zip(registers, ks):
        if len(reg) < k:
            raise ValueError("current schedule is infeasible for the group (too few usable blocks)")
    return sub


def _qubo_from_sub(sub: _Sub, include_demand: bool, lam_margin: float) -> tuple[np.ndarray, float]:
    """E(x) = x^T Q x + offset; Q upper-triangular (same convention as qubo.to_qubo).

    Objective + lam * (conflict pairs) [+ lam * sum_i (sum_t x_it - k_i)^2 if include_demand],
    lam = lam_margin * max coefficient (the rule of ``qubo.default_penalties``).
    """
    n = sub.n
    Q = np.zeros((n, n))
    offset = 0.0
    if n == 0:
        return Q, offset
    Q[np.arange(n), np.arange(n)] += sub.coef
    lam = lam_margin * float(sub.coef.max())
    if include_demand:
        for reg, k in zip(sub.registers, sub.ks):
            for a in reg:
                Q[a, a] += lam * (1 - 2 * k)
            for p in range(len(reg)):
                for q in range(p + 1, len(reg)):
                    Q[reg[p], reg[q]] += 2 * lam
            offset += lam * k * k
    owner = np.empty(n, dtype=int)
    for r, reg in enumerate(sub.registers):
        owner[reg] = r
    for a in range(n):
        for b in range(a + 1, n):
            if (owner[a] != owner[b] and sub.blocks[a] == sub.blocks[b]
                    and sub.power[a] + sub.power[b] > sub.residual[sub.blocks[a]] + TOL):
                Q[a, b] += lam
    return Q, float(offset)


def subproblem_qubo(
    inst: Instance,
    x: np.ndarray,
    group: tuple[int, ...] | list[int],
    include_demand: bool = True,
    lam_margin: float = 1.5,
) -> tuple[np.ndarray, float, list[tuple[int, int]], list[list[int]], list[int]]:
    """QUBO of the LNS subproblem that frees the buses in ``group`` of schedule ``x``.

    Returns ``(Q, offset, var_list, registers, ks)``: qubit a carries the
    variable ``var_list[a] = (bus, block)``; ``registers[g]`` lists the qubits
    of the g-th group bus and ``ks[g]`` is its required number of blocks.
    ``include_demand=True`` adds the exact-k penalty (X mixer); with ``False``
    only the capacity conflicts are penalised (XY mixer + Dicke start keeps
    demand). Valid for groups of at most two buses (pairwise-exact).
    """
    if len(group) > 2:
        raise ValueError("groups of more than two buses are not pairwise-exact")
    sub = _build_sub(inst, _context(inst), x, tuple(group))
    Q, offset = _qubo_from_sub(sub, include_demand, lam_margin)
    return Q, offset, sub.var_list, sub.registers, sub.ks


# ---------------------------------------------------------------------------
# Exact subsolver (enumeration) - also the diagnostic reference for QAOA
# ---------------------------------------------------------------------------
@dataclass
class _Enum:
    best_x: np.ndarray        # optimal local assignment
    best_cost: float
    n_feasible: int           # feasible combinations
    n_demand: int             # combinations meeting demand (the Dicke-start support)
    n_within_1pct: int        # feasible with cost <= 1.01 * best


def _bus_combos(sub: _Sub, g: int):
    """All k-subsets of one bus's variables: (0/1 matrix, cost, block loads)."""
    reg, k = sub.registers[g], sub.ks[g]
    idx = np.array(list(combinations(range(len(reg)), k)), dtype=int).reshape(-1, k)
    C = np.zeros((len(idx), len(reg)))
    np.put_along_axis(C, idx, 1.0, axis=1)
    M = sub.block_matrix()[reg]
    return C, C @ sub.coef[reg], C @ M


def _enumerate(sub: _Sub) -> _Enum | None:
    """Best feasible assignment of a 1- or 2-bus group by full enumeration."""
    parts = [_bus_combos(sub, g) for g in range(len(sub.registers))]
    if len(parts) == 1:
        C, cost, load = parts[0]
        feas = (load <= sub.residual + TOL).all(axis=1)
        n_demand = len(cost)
        if not feas.any():
            return None
        masked = np.where(feas, cost, np.inf)
        a = int(np.argmin(masked))
        x = np.zeros(sub.n)
        x[sub.registers[0]] = C[a]
        best = float(cost[a])
        return _Enum(x, best, int(feas.sum()), n_demand,
                     int((feas & (cost <= 1.01 * best + 1e-12)).sum()))
    (C0, c0, L0), (C1, c1, L1) = parts
    n_demand = len(c0) * len(c1)
    if n_demand > 4_000_000:
        raise ValueError("subproblem too large to enumerate; lower max_vars")
    feas = ((L0[:, None, :] + L1[None, :, :]) <= sub.residual + TOL).all(axis=2)
    if not feas.any():
        return None
    cost = c0[:, None] + c1[None, :]
    a, b = np.unravel_index(int(np.argmin(np.where(feas, cost, np.inf))), cost.shape)
    x = np.zeros(sub.n)
    x[sub.registers[0]] = C0[a]
    x[sub.registers[1]] = C1[b]
    best = float(cost[a, b])
    return _Enum(x, best, int(feas.sum()), n_demand,
                 int((feas & (cost <= 1.01 * best + 1e-12)).sum()))


# ---------------------------------------------------------------------------
# QAOA subsolver
# ---------------------------------------------------------------------------
def _sub_feasible(sub: _Sub, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(demand_ok, feasible) per row of X over the subproblem's own constraints."""
    X = np.asarray(X, dtype=float)
    demand = np.ones(len(X), dtype=bool)
    for reg, k in zip(sub.registers, sub.ks):
        demand &= X[:, reg].sum(axis=1) == k
    cap_ok = ((X @ sub.block_matrix()) <= sub.residual + TOL).all(axis=1)
    return demand, demand & cap_ok


def _train_qaoa(sim: ExactSimulator, Q: np.ndarray, reps: int, cfg: dict, maxiter: int) -> np.ndarray:
    """COBYLA on [gamma*max|Q|, beta] with the CVaR (or mean) objective, exact probabilities."""
    scale = max(float(np.abs(Q).max()), 1e-12)
    x0 = ramp_init_params(Q, reps) if cfg["init"] == "ramp" else default_init_params(Q, reps)
    scalev = np.concatenate([np.full(reps, scale), np.ones(reps)])
    use_cvar = cfg["objective"] == "cvar"
    alpha = cfg["cvar_alpha"]

    def f(u):
        return sim.cvar(u / scalev, alpha) if use_cvar else sim.expectation(u / scalev)

    res = minimize(f, x0 * scalev, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
    return res.x / scalev


def _solve_qaoa(sub: _Sub, mixer: str, reps: int, shots: int, seed: int, cfg: dict,
                maxiter: int, lam_margin: float, sample_with: str):
    """Train, sample and return (best feasible local x or None, stats dict)."""
    t0 = time.perf_counter()
    Q, offset = _qubo_from_sub(sub, include_demand=(mixer == "x"), lam_margin=lam_margin)
    regs = sub.registers if mixer == "xy" else None
    ks = sub.ks if mixer == "xy" else None
    sim = ExactSimulator(Q, offset, reps, mixer, sub.registers, sub.ks)
    params = _train_qaoa(sim, Q, reps, cfg, maxiter)
    rng = np.random.default_rng(seed)
    if sample_with == "sampler":
        from qiskit.primitives import StatevectorSampler

        circuit = build_qaoa_circuit_from_qubo(Q, reps, mixer, regs, ks)
        counts = sample_counts(circuit, params, shots, StatevectorSampler(seed=seed))
        X, w = counts_to_matrix(counts)
    else:
        prob = sim.probabilities(params)
        prob = prob / prob.sum()
        draw = rng.multinomial(shots, prob)
        idx = np.nonzero(draw)[0]
        X = ((idx[:, None] >> np.arange(sub.n)) & 1).astype(np.int8)
        w = draw[idx].astype(float)
    demand, feas = _sub_feasible(sub, X)
    cost = X @ sub.coef
    out = dict(
        qaoa_feasible_prob=float(w[feas].sum() / shots),
        qaoa_demand_prob=float(w[demand].sum() / shots),
        qaoa_train_s=time.perf_counter() - t0,
        n_qubits=sub.n,
    )
    if not feas.any():
        return None, out
    a = int(np.argmin(np.where(feas, cost, np.inf)))
    out["qaoa_best_cost"] = float(cost[a])
    return X[a].astype(float), out | {"_X": X, "_w": w, "_cost": cost, "_feas": feas}


# ---------------------------------------------------------------------------
# LNS driver
# ---------------------------------------------------------------------------
@dataclass
class LNSResult:
    """Outcome of :func:`lns_depot`.

    ``trace`` holds the objective before the first subproblem and after each
    one (length ``n_subproblems + 1``). ``qaoa`` is empty for ``"exact"`` and
    otherwise has:
      n_calls, n_feasible_improvement (QAOA returned a feasible strictly better
      assignment), feasible_improvement_rate, n_fallback (kept current),
      mean_feasible_prob, mean_p_within_1pct / mean_p_opt (QAOA samples vs the
      subproblem optimum), mean_baseline_p_within_1pct (same quantity for the
      p=0 Dicke start, XY only), n_exact_improvable (subproblems where an
      improvement existed), mean_gap_to_sub_opt (relative cost gap of the best
      QAOA sample, among subproblems with a feasible sample).
    """

    x: np.ndarray
    objective_initial: float
    objective_final: float
    trace: list[float]
    n_subproblems: int
    n_improved: int
    passes_run: int
    improvements_per_pass: list[int]
    calls: dict[str, int]
    qaoa: dict[str, float]
    wall_s: float
    subsolver: str
    log: list[dict] = field(default_factory=list)


def make_groups(inst: Instance, group_size: int, seed: int, pass_index: int,
                max_vars: int = 16) -> list[tuple[int, ...]]:
    """Fixed-order partition of the buses into groups, deterministic in (seed, pass_index).

    Buses are shuffled; each takes the first unplaced partner whose available
    hours sum to at most ``max_vars`` (so the subproblem fits ``max_vars``
    qubits); buses without a partner become singleton groups.
    """
    if group_size not in (1, 2):
        raise ValueError("group_size must be 1 or 2")
    rng = np.random.default_rng([seed, pass_index])
    order = [int(i) for i in rng.permutation(len(inst.evs))]
    if group_size == 1:
        return [(i,) for i in order]
    width = {i: len(inst.evs[i].available) for i in order}
    free = list(order)
    groups: list[tuple[int, ...]] = []
    while free:
        a = free.pop(0)
        partner = next((b for b in free if width[a] + width[b] <= max_vars), None)
        if partner is None:
            groups.append((a,))
        else:
            free.remove(partner)
            groups.append((a, partner))
    return groups


def lns_depot(
    inst: Instance,
    x0: np.ndarray,
    subsolver: str = "exact",
    group_size: int = 2,
    max_passes: int = 3,
    seed: int = 0,
    reps: int = 2,
    shots: int = 4096,
    max_vars: int = 16,
    train_maxiter: int = 200,
    lam_margin: float = 1.5,
    sample_with: str = "exact",
    time_limit_s: float | None = None,
    train_config: dict | None = None,
    keep_log: bool = True,
    patience: int = 1,
) -> LNSResult:
    """Large-neighbourhood search over the depot with a pluggable subsolver.

    ``inst.p_max_kw`` is the connection limit; ``x0`` must satisfy it (e.g.
    ``depot.on_arrival_capacity(inst)``). Each pass visits every group once
    (see :func:`make_groups`). The search stops after ``max_passes``, after
    ``patience`` consecutive passes without improvement (default 1), or when
    ``time_limit_s`` is exceeded.
    ``train_config`` defaults to ``qaoa.TRAIN_CONFIG`` (CVaR 0.25, ramp init)
    with ``train_maxiter`` (200 instead of 1000) COBYLA evaluations to stay
    around a few seconds per subproblem.
    """
    if subsolver not in ("exact", "qaoa_xy", "qaoa_x"):
        raise ValueError("subsolver must be 'exact', 'qaoa_xy' or 'qaoa_x'")
    if sample_with not in ("exact", "sampler"):
        raise ValueError("sample_with must be 'exact' or 'sampler'")
    x = np.asarray(x0, dtype=int).copy()
    if not is_feasible_kw(inst, x):
        raise ValueError("x0 must be a feasible depot schedule")
    cfg = dict(TRAIN_CONFIG if train_config is None else train_config)
    mixer = subsolver.split("_")[1] if subsolver.startswith("qaoa") else None
    ctx = _context(inst)
    t0 = time.perf_counter()
    obj = float(ctx.coef @ x)
    trace = [obj]
    obj_initial = obj
    calls = {"exact": 0, "qaoa": 0, "trivial": 0}
    stats = dict(n_calls=0, n_feasible_improvement=0, n_fallback=0, n_exact_improvable=0,
                 _fp=[], _pw=[], _po=[], _bw=[], _gap=[])
    improvements_per_pass: list[int] = []
    idle_passes = 0
    log: list[dict] = []
    n_sub = n_improved = 0
    out_of_time = False

    for ps in range(max_passes):
        improved_here = 0
        for group in make_groups(inst, group_size, seed, ps, max_vars):
            if time_limit_s is not None and time.perf_counter() - t0 > time_limit_s:
                out_of_time = True
                break
            sub = _build_sub(inst, ctx, x, group)
            n_sub += 1
            enum = _enumerate(sub)
            cur = sub.current_cost
            rec = dict(pass_=ps, group=group, n_vars=sub.n, current_cost=cur)
            t_sub = time.perf_counter()
            new_local = None
            if enum is None or enum.n_feasible < 2:
                calls["trivial"] += 1                     # nothing to choose between
                rec["kind"] = "trivial"
            elif mixer is None:
                calls["exact"] += 1
                rec["kind"] = "exact"
                if enum.best_cost < cur - GAIN_TOL:
                    new_local = enum.best_x
            else:
                calls["qaoa"] += 1
                rec["kind"] = subsolver
                seed_sub = int(np.random.SeedSequence([seed, n_sub]).generate_state(1)[0])
                best_local, info = _solve_qaoa(sub, mixer, reps, shots, seed_sub, cfg,
                                               train_maxiter, lam_margin, sample_with)
                stats["n_calls"] += 1
                improvable = enum.best_cost < cur - GAIN_TOL
                stats["n_exact_improvable"] += int(improvable)
                stats["_fp"].append(info["qaoa_feasible_prob"])
                if best_local is not None:
                    q_cost = info["qaoa_best_cost"]
                    X, w, cost, feas = info["_X"], info["_w"], info["_cost"], info["_feas"]
                    stats["_pw"].append(float(w[feas & (cost <= 1.01 * enum.best_cost + 1e-12)].sum() / shots))
                    stats["_po"].append(float(w[feas & (cost <= enum.best_cost + GAIN_TOL)].sum() / shots))
                    stats["_gap"].append((q_cost - enum.best_cost) / max(abs(enum.best_cost), 1e-12))
                    if q_cost < cur - GAIN_TOL:
                        new_local = best_local
                        stats["n_feasible_improvement"] += 1
                    else:
                        stats["n_fallback"] += 1
                else:
                    stats["_pw"].append(0.0)
                    stats["_po"].append(0.0)
                    stats["n_fallback"] += 1
                if mixer == "xy":
                    stats["_bw"].append(enum.n_within_1pct / enum.n_demand)
                rec.update({k: v for k, v in info.items() if not k.startswith("_")})
                rec["exact_best_cost"] = enum.best_cost
            if new_local is not None:
                new_cost = float(sub.coef @ new_local)
                x[sub.cols] = np.rint(new_local).astype(int)
                obj += new_cost - cur
                improved_here += 1
                n_improved += 1
                rec["new_cost"] = new_cost
            rec["improved"] = new_local is not None
            rec["seconds"] = time.perf_counter() - t_sub
            trace.append(obj)
            if keep_log:
                log.append(rec)
        improvements_per_pass.append(improved_here)
        idle_passes = idle_passes + 1 if improved_here == 0 else 0
        if out_of_time or idle_passes >= patience:
            break

    assert is_feasible_kw(inst, x), "LNS produced an infeasible schedule"  # only feasible, strictly better moves are accepted
    q: dict[str, float] = {}
    if mixer is not None and stats["n_calls"]:
        nc = stats["n_calls"]
        mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
        q = dict(
            n_calls=nc, n_feasible_improvement=stats["n_feasible_improvement"],
            feasible_improvement_rate=stats["n_feasible_improvement"] / nc,
            n_fallback=stats["n_fallback"], n_exact_improvable=stats["n_exact_improvable"],
            mean_feasible_prob=mean(stats["_fp"]), mean_p_within_1pct=mean(stats["_pw"]),
            mean_p_opt=mean(stats["_po"]), mean_baseline_p_within_1pct=mean(stats["_bw"]),
            mean_gap_to_sub_opt=mean(stats["_gap"]),
        )
    return LNSResult(
        x=x, objective_initial=obj_initial, objective_final=float(ctx.coef @ x), trace=trace,
        n_subproblems=n_sub, n_improved=n_improved, passes_run=len(improvements_per_pass),
        improvements_per_pass=improvements_per_pass, calls=calls, qaoa=q,
        wall_s=time.perf_counter() - t0, subsolver=subsolver, log=log,
    )
