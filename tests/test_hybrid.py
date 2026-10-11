"""Tests for the hybrid LNS depot search (hybrid.py)."""
from datetime import date

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from carbon_ev.data import load_hourly
from carbon_ev.depot import (
    block_load_kw, depot_fleet, is_feasible_kw, milp_depot, min_cap_on_arrival_kw, on_arrival_capacity,
)
from carbon_ev.hybrid import (
    _build_sub, _context, _enumerate, _qubo_from_sub, _sub_feasible, lns_depot, make_groups,
    subproblem_qubo,
)
from carbon_ev.problem import EV, build_instance, objective_coefficients
from carbon_ev.qaoa import ExactSimulator, build_qaoa_circuit_from_qubo, param_order
from carbon_ev.qubo import all_bitstrings, qubo_energy

NIGHT = date(2026, 10, 8)


@pytest.fixture(scope="module")
def df():
    return load_hourly()


def _random_small_fleet(rng, n_buses=5):
    fleet = []
    for b in range(n_buses):
        w = int(rng.integers(3, 7))
        a = int(rng.integers(0, 12 - w + 1))
        k = int(rng.integers(1, w))
        fleet.append(EV(f"b{b}", float(rng.integers(20, 100)), k, tuple(range(a, a + w))))
    return fleet


def _tight_instance(df, fleet):
    free = build_instance(df, NIGHT, fleet=fleet, n_blocks=12, p_max_kw=1e9)
    cap = min_cap_on_arrival_kw(free, step_kw=10.0)
    return build_instance(df, NIGHT, fleet=fleet, n_blocks=12, p_max_kw=cap)


def _tiny_depot(df, n=6, seed=1):
    free = build_instance(df, NIGHT, fleet=depot_fleet(n, seed=seed), n_blocks=12, p_max_kw=1e9)
    cap = min_cap_on_arrival_kw(free, step_kw=10.0)
    return build_instance(df, NIGHT, fleet=depot_fleet(n, seed=seed), n_blocks=12, p_max_kw=cap)


@pytest.mark.parametrize("seed", range(6))
def test_subproblem_qubo_argmin_equals_exact_enumeration(df, seed):
    rng = np.random.default_rng(seed)
    inst = _tight_instance(df, _random_small_fleet(rng))
    x0 = on_arrival_capacity(inst)
    ctx = _context(inst)
    checked = 0
    for group in [(0, 1), (2, 3), (1, 4), (0, 3), (4,)]:
        sub = _build_sub(inst, ctx, x0, group)
        enum = _enumerate(sub)
        assert enum is not None
        Xall = all_bitstrings(sub.n)
        demand, feas = _sub_feasible(sub, Xall)
        # penalty QUBO (X mixer): unconstrained argmin is the constrained optimum
        Q, off, var_list, regs, ks = subproblem_qubo(inst, x0, group, include_demand=True)
        assert len(var_list) == sub.n
        E = qubo_energy(Q, off, Xall)
        a = int(np.argmin(E))
        assert feas[a]
        assert float(Xall[a] @ sub.coef) == pytest.approx(enum.best_cost)
        assert E[a] == pytest.approx(enum.best_cost)          # penalties vanish when feasible
        # conflict-only QUBO (XY mixer): argmin over the demand sector
        Qc, offc, *_ = subproblem_qubo(inst, x0, group, include_demand=False)
        Ec = np.where(demand, qubo_energy(Qc, offc, Xall), np.inf)
        b = int(np.argmin(Ec))
        assert feas[b] and Ec[b] == pytest.approx(enum.best_cost)
        # feasible set sizes agree between enumeration and brute force
        assert enum.n_feasible == int(feas.sum())
        assert enum.n_demand == int(demand.sum())
        checked += 1
    assert checked == 5


def test_subproblem_variables_fit_16_qubits_and_current_x_is_feasible(df):
    inst = _tiny_depot(df, n=20, seed=3)
    x0 = on_arrival_capacity(inst)
    ctx = _context(inst)
    for group in make_groups(inst, 2, seed=0, pass_index=0):
        sub = _build_sub(inst, ctx, x0, group)
        assert sub.n <= 16
        _, feas = _sub_feasible(sub, sub.x_local[None, :])
        assert feas[0]


def test_make_groups_partition_and_determinism(df):
    inst = build_instance(df, NIGHT, fleet=depot_fleet(40, seed=0), n_blocks=12, p_max_kw=1e9)
    g1 = make_groups(inst, 2, seed=5, pass_index=0)
    assert g1 == make_groups(inst, 2, seed=5, pass_index=0)
    assert g1 != make_groups(inst, 2, seed=5, pass_index=1)
    flat = sorted(i for g in g1 for i in g)
    assert flat == list(range(40))
    for g in g1:
        assert sum(len(inst.evs[i].available) for i in g) <= 16
    assert sum(1 for g in g1 if len(g) == 2) >= 15       # mostly real pairs


def test_xy_and_x_circuits_match_exact_simulator_on_a_subproblem(df):
    rng = np.random.default_rng(0)
    inst = _tight_instance(df, _random_small_fleet(rng, 4))
    x0 = on_arrival_capacity(inst)
    sub = _build_sub(inst, _context(inst), x0, (0, 1))
    for mixer in ("x", "xy"):
        Q, off = _qubo_from_sub(sub, include_demand=(mixer == "x"), lam_margin=1.5)
        regs, ks = (sub.registers, sub.ks) if mixer == "xy" else (None, None)
        qc = build_qaoa_circuit_from_qubo(Q, 2, mixer, regs, ks, measure=False)
        params = np.array([0.3, 0.5, 0.4, 0.2])
        vals = params[param_order(qc)]
        ref = Statevector(qc.assign_parameters(dict(zip(qc.parameters, vals)))).data
        sim = ExactSimulator(Q, off, 2, mixer, sub.registers, sub.ks)
        assert abs(np.vdot(ref, sim.state(params))) == pytest.approx(1.0, abs=1e-9)


def test_lns_exact_never_worse_and_feasible(df):
    inst = _tiny_depot(df, n=24, seed=2)
    x0 = on_arrival_capacity(inst)
    res = lns_depot(inst, x0, "exact", max_passes=3, seed=0)
    assert is_feasible_kw(inst, res.x)
    assert np.all(np.diff(res.trace) <= 1e-9)
    assert res.objective_final <= res.objective_initial + 1e-9
    assert res.objective_final == pytest.approx(float(objective_coefficients(inst) @ res.x))
    assert res.trace[-1] == pytest.approx(res.objective_final)
    assert res.n_subproblems == len(res.trace) - 1
    assert res.objective_final < res.objective_initial        # tight cap leaves room to improve
    assert block_load_kw(inst, res.x).max() <= inst.p_max_kw + 1e-6


def test_lns_exact_reaches_or_nears_global_milp_on_tiny_fleet(df):
    for seed in (1, 4):
        inst = _tiny_depot(df, n=6, seed=seed)
        x0 = on_arrival_capacity(inst)
        res = lns_depot(inst, x0, "exact", max_passes=40, seed=0, patience=8)
        xm, gap = milp_depot(inst, mip_rel_gap=1e-6)
        opt = float(objective_coefficients(inst) @ xm)
        assert res.objective_final >= opt - 1e-6              # cannot beat the optimum
        assert res.objective_final <= opt * 1.02              # near the global optimum
        assert is_feasible_kw(inst, res.x)


@pytest.mark.parametrize("subsolver", ["qaoa_xy", "qaoa_x"])
def test_lns_qaoa_stays_feasible_and_never_worse(df, subsolver):
    inst = _tiny_depot(df, n=6, seed=1)
    x0 = on_arrival_capacity(inst)
    res = lns_depot(inst, x0, subsolver, max_passes=1, seed=0, reps=1, shots=512, train_maxiter=40)
    assert is_feasible_kw(inst, res.x)
    assert np.all(np.diff(res.trace) <= 1e-9)
    q = res.qaoa
    assert q["n_calls"] == res.calls["qaoa"] >= 1
    assert 0.0 <= q["feasible_improvement_rate"] <= 1.0
    assert q["n_feasible_improvement"] + q["n_fallback"] == q["n_calls"]
    assert 0.0 <= q["mean_feasible_prob"] <= 1.0
    assert 0.0 <= q["mean_p_within_1pct"] <= 1.0
    if subsolver == "qaoa_xy":
        assert q["mean_baseline_p_within_1pct"] >= 0.0


def test_lns_rejects_bad_input(df):
    inst = _tiny_depot(df, n=4, seed=1)
    x0 = on_arrival_capacity(inst)
    with pytest.raises(ValueError):
        lns_depot(inst, x0, "bogus")
    with pytest.raises(ValueError):
        lns_depot(inst, np.zeros_like(x0), "exact")             # infeasible start
    with pytest.raises(ValueError):
        subproblem_qubo(inst, x0, (0, 1, 2))                    # beyond pairwise-exact
