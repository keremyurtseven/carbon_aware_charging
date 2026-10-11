"""Tests for the constraint-preserving QAOA building blocks."""
from math import comb

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from scipy.linalg import expm

from carbon_ev.qaoa import dicke_state, ring_edges, xy_ring_mixer

BASIS = ["rz", "sx", "x", "cx"]


def _weights(n: int) -> np.ndarray:
    return np.array([bin(i).count("1") for i in range(2**n)])


@pytest.mark.parametrize("n,k", [(5, 1), (5, 2), (6, 2), (6, 1), (3, 0), (4, 4), (4, 2), (6, 3), (2, 1), (1, 1)])
def test_dicke_state_amplitudes(n, k):
    sv = Statevector(dicke_state(n, k)).data
    w = _weights(n)
    amp = sv[w == k]
    assert np.allclose(sv[w != k], 0, atol=1e-10)
    # uniform magnitude 1/sqrt(C(n,k)) and a single global phase
    assert np.allclose(np.abs(amp), 1 / np.sqrt(comb(n, k)), atol=1e-10)
    assert np.allclose(amp / amp[0], 1, atol=1e-10)


def test_dicke_invalid():
    with pytest.raises(ValueError):
        dicke_state(3, 4)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_ring_edges_cover_ring_once(n):
    layers = ring_edges(n)
    edges = [frozenset(e) for layer in layers for e in layer]
    expected = {frozenset((i, (i + 1) % n)) for i in range(n)}
    assert len(edges) == len(set(edges))
    assert set(edges) == expected
    for layer in layers:  # edges inside a layer are disjoint
        qs = [q for e in layer for q in e]
        assert len(qs) == len(set(qs))


def test_ring_edges_tiny():
    assert ring_edges(1) == []
    assert ring_edges(2) == [[(0, 1)]]


def test_xy_two_qubit_matches_exact_exponential():
    beta = 0.37
    h = SparsePauliOp(["XX", "YY"], [0.5, 0.5]).to_matrix()
    assert Operator(xy_ring_mixer([[0, 1]], beta)).equiv(Operator(expm(-1j * beta * h)))


def _random_weight_k_state(layout, ks, rng):
    """Random superposition of strings with weight ks[r] in every register."""
    n = sum(layout)
    offsets = np.cumsum([0] + list(layout))
    idx = np.arange(2**n)
    ok = np.ones(2**n, dtype=bool)
    for r, (size, k) in enumerate(zip(layout, ks)):
        bits = (idx >> np.arange(offsets[r], offsets[r] + size)[:, None]) & 1
        ok &= bits.sum(axis=0) == k
    amp = np.zeros(2**n, dtype=complex)
    amp[ok] = rng.normal(size=ok.sum()) + 1j * rng.normal(size=ok.sum())
    return amp / np.linalg.norm(amp), ok


@pytest.mark.parametrize("layout,ks", [
    ([6, 5, 5], [2, 1, 2]),     # main-night layout, 16 qubits
    ([3, 4], [1, 2]),
    ([5, 1, 2], [2, 1, 1]),     # size-1 register: nothing to mix
    ([4, 3], [0, 3]),           # k = 0 and k = n: trivial registers
])
def test_xy_mixer_preserves_weight_per_register(layout, ks):
    rng = np.random.default_rng(1)
    offsets = np.cumsum([0] + layout)
    regs = [list(range(offsets[r], offsets[r + 1])) for r in range(len(layout))]
    n = sum(layout)
    for _ in range(3):
        psi, ok = _random_weight_k_state(layout, ks, rng)
        beta = rng.uniform(0.1, 3.0)
        out = Statevector(psi).evolve(xy_ring_mixer(regs, beta)).data
        assert np.abs(out[~ok]).max() < 1e-9
        assert np.isclose(np.linalg.norm(out), 1.0)


def test_xy_mixer_actually_mixes():
    regs = [list(range(0, 6)), list(range(6, 11)), list(range(11, 16))]
    prep = QuantumCircuit(16)
    for reg, k in zip(regs, [2, 1, 2]):
        prep.compose(dicke_state(len(reg), k), reg, inplace=True)
    psi = Statevector(prep)
    out = psi.evolve(xy_ring_mixer(regs, 0.4))
    assert abs(psi.inner(out)) < 0.99
    # Dicke start is stationary for the *full-ring* XY Hamiltonian but a single
    # Trotter step with a non-trivial beta still stays in the feasible space.
    w = _weights(16)
    assert np.isclose((np.abs(out.data[w == 5]) ** 2).sum(), 1.0)  # total weight 5
    # a non-Dicke feasible start (all ones packed at the end) must change
    basis = QuantumCircuit(16)
    for reg, k in zip(regs, [2, 1, 2]):
        basis.x(reg[len(reg) - k:])
    b = Statevector(basis)
    assert abs(b.inner(b.evolve(xy_ring_mixer(regs, 0.4)))) < 0.99


def test_parametrized_mixer_binds():
    beta = Parameter("beta")
    qc = xy_ring_mixer([[0, 1, 2], [3, 4]], beta)
    assert set(qc.parameters) == {beta}
    bound = qc.assign_parameters({beta: 0.5})
    assert not bound.parameters


def test_primitive_gate_counts_report(capsys):
    """Informational: depth / 2q count after transpiling to the fixed basis."""
    rows = []
    for n, k in [(6, 2), (5, 1), (5, 2)]:
        t = transpile(dicke_state(n, k), basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
        rows.append((f"Dicke({n},{k})", t.depth(), t.count_ops().get("cx", 0)))
        assert t.count_ops().get("cx", 0) > 0
    regs = [list(range(0, 6)), list(range(6, 11)), list(range(11, 16))]
    t = transpile(xy_ring_mixer(regs, 0.3), basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
    rows.append(("XYring 6,5,5", t.depth(), t.count_ops().get("cx", 0)))
    with capsys.disabled():
        print()
        for r in rows:
            print("  %-14s depth=%-4d cx=%d" % r)


# ---------------------------------------------------------------------------
# QUBO-level QAOA
# ---------------------------------------------------------------------------
from carbon_ev.qaoa import (  # noqa: E402
    _run_qaoa_qubo, bitstring_to_x, build_qaoa_circuit_from_qubo, cost_layer,
    param_order, sample_counts, x_to_bitstring,
)


def _energy(Q, off, x):
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + off)


def _all_energies(Q, n):
    return np.array([_energy(Q, 0.0, [(i >> j) & 1 for j in range(n)]) for i in range(2**n)])


def test_bitstring_to_x():
    assert list(bitstring_to_x("0011")) == [1, 1, 0, 0]   # key is q3 q2 q1 q0
    assert list(bitstring_to_x("10 01")) == [1, 0, 0, 1]
    assert x_to_bitstring(bitstring_to_x("01101")) == "01101"


@pytest.mark.parametrize("symmetric", [False, True])
def test_cost_layer_phases_match_qubo_energy(symmetric):
    rng = np.random.default_rng(3)
    n = 5
    Q = np.triu(rng.normal(size=(n, n)))
    if symmetric:
        Q = (Q + Q.T) / 2
    gamma = 0.7
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.compose(cost_layer(Q, gamma, n), inplace=True)
    sv = Statevector(qc).data * np.sqrt(2**n)
    ratio = sv / np.exp(-1j * gamma * _all_energies(Q, n))
    assert np.allclose(ratio, ratio[0], atol=1e-9)         # constant => global phase only


def test_full_circuit_p1_x_mixer_beta_zero_matches_cost_phases():
    rng = np.random.default_rng(4)
    n = 4
    Q = np.triu(rng.normal(size=(n, n)))
    qc = build_qaoa_circuit_from_qubo(Q, reps=1, mixer="x", measure=False)
    vals = {p.name: 0.0 for p in qc.parameters}
    vals["gamma[0]"] = 0.9
    bound = qc.assign_parameters({p: vals[p.name] for p in qc.parameters})
    sv = Statevector(bound).data * np.sqrt(2**n)
    ratio = sv / np.exp(-1j * 0.9 * _all_energies(Q, n))
    assert np.allclose(ratio, ratio[0], atol=1e-9)


def test_param_order():
    qc = build_qaoa_circuit_from_qubo(np.eye(3), reps=2, mixer="x")
    names = [qc.parameters[i].name for i in np.argsort(param_order(qc))]
    assert names == ["gamma[0]", "gamma[1]", "beta[0]", "beta[1]"]


def test_xy_circuit_samples_only_right_weights():
    from qiskit.primitives import StatevectorSampler
    rng = np.random.default_rng(5)
    regs = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
    ks = [1, 1, 2]
    Q = np.triu(rng.normal(size=(9, 9)))
    qc = build_qaoa_circuit_from_qubo(Q, reps=2, mixer="xy", registers=regs, ks=ks)
    counts = sample_counts(qc, np.array([0.3, 0.5, 0.4, 0.8]), 2000, StatevectorSampler(seed=1))
    assert len(counts) > 5
    for key in counts:
        x = bitstring_to_x(key)
        assert [int(x[r].sum()) for r in regs] == ks


def test_qaoa_beats_uniform_on_pick_one_of_four():
    c = np.array([3.0, 1.0, 2.0, 4.0])          # best: index 1
    lam = 6.0
    Q = np.diag(c - lam) + np.triu(np.full((4, 4), 2 * lam), 1)
    off = lam
    qc = build_qaoa_circuit_from_qubo(Q, reps=1, mixer="x")
    res = _run_qaoa_qubo(Q, off, qc, shots=4096, seed=0, maxiter=80)
    p_opt = res.counts.get(x_to_bitstring([0, 1, 0, 0]), 0) / 4096
    assert p_opt > 2 / 16                         # uniform would give 1/16
    uniform_mean = np.mean(_all_energies(Q, 4)) + off
    assert res.mean_energy < uniform_mean
    assert res.n_evals > 5 and res.depth > 0 and res.n_2q > 0 and res.wall_s > 0


def test_run_qaoa_qubo_xy_deterministic_seed():
    regs, ks = [[0, 1, 2], [3, 4, 5]], [1, 1]
    Q = np.triu(np.arange(36, dtype=float).reshape(6, 6) % 5 - 2)
    qc = build_qaoa_circuit_from_qubo(Q, reps=1, mixer="xy", registers=regs, ks=ks)
    a = _run_qaoa_qubo(Q, 0.0, qc, shots=512, seed=7, maxiter=20)
    b = _run_qaoa_qubo(Q, 0.0, qc, shots=512, seed=7, maxiter=20)
    assert a.counts == b.counts and np.allclose(a.params, b.params)


# ---------------------------------------------------------------------------
# Fast exact simulator, schedule metrics, instance-level runner
# ---------------------------------------------------------------------------
from datetime import date  # noqa: E402

from carbon_ev.data import load_hourly  # noqa: E402
from carbon_ev.problem import build_instance, evaluate  # noqa: E402
from carbon_ev.qaoa import (  # noqa: E402
    ExactSimulator, exact_optimum, feasible_schedules, registers_from_instance,
    run_qaoa, schedule_metrics,
)


@pytest.fixture(scope="module")
def main_inst():
    return build_instance(load_hourly(), date(2026, 10, 8))


@pytest.mark.parametrize("mixer", ["x", "xy"])
def test_exact_simulator_matches_circuit_statevector(mixer):
    rng = np.random.default_rng(11)
    regs, ks = [[0, 1, 2, 3], [4, 5, 6], [7, 8]], [2, 1, 1]
    n = 9
    Q = np.triu(rng.normal(size=(n, n)))
    reps = 2
    qc = build_qaoa_circuit_from_qubo(Q, reps, mixer, regs if mixer == "xy" else None,
                                      ks if mixer == "xy" else None, measure=False)
    params = rng.uniform(0.1, 1.0, size=2 * reps)
    order = param_order(qc)
    bound = qc.assign_parameters({p: params[i] for p, i in zip(qc.parameters, order)})
    ref = Statevector(bound).data
    sim = ExactSimulator(Q, 0.0, reps, mixer, regs, ks)
    got = sim.state(params)
    assert abs(np.vdot(ref, got)) == pytest.approx(1.0, abs=1e-9)   # equal up to global phase


def test_schedule_metrics_matches_evaluate(main_inst):
    rng = np.random.default_rng(2)
    n = len(registers_from_instance(main_inst)[0][0]) + 10
    X = rng.integers(0, 2, size=(300, n))
    # include many feasible rows
    Xf, _ = feasible_schedules(main_inst)
    X = np.vstack([X, Xf[::10]])
    obj, dem, feas = schedule_metrics(main_inst, X)
    for x, o, d, f in zip(X, obj, dem, feas):
        ev = evaluate(main_inst, x)
        assert o == pytest.approx(ev["objective"])
        assert f == ev["feasible"]
        assert d == (ev["violations"]["demand"] == 0)


def test_exact_optimum_main_night(main_inst):
    X, obj = feasible_schedules(main_inst)
    assert len(X) == 500
    x, o = exact_optimum(main_inst)
    ev = evaluate(main_inst, x)
    assert ev["feasible"]
    assert ev["co2_kg"] == pytest.approx(15.69, abs=0.01)


@pytest.mark.parametrize("mixer", ["x", "xy"])
def test_run_qaoa_main_night(main_inst, mixer):
    res = run_qaoa(main_inst, mixer=mixer, reps=1, shots=4096, seed=0)
    assert res.best_x is not None
    assert evaluate(main_inst, res.best_x)["feasible"]
    assert 0.0 <= res.feasible_prob <= 1.0
    assert 0.0 < res.approx_ratio <= 1.0 + 1e-12
    assert 0.0 < res.approx_ratio_expected <= 1.0 + 1e-12
    assert 0.0 <= res.p_opt <= res.p_within_1pct <= res.feasible_prob <= res.demand_prob <= 1.0
    assert sum(res.counts.values()) == 4096
    assert res.depth > 0 and res.n_2q > 0 and res.n_evals > 5
    assert res.wall_s < 60
    if mixer == "xy":
        assert res.demand_prob == 1.0
        uniform_demand_fraction = 15 * 5 * 10 / 2**16
        assert res.feasible_prob >= uniform_demand_fraction


def test_run_qaoa_reproducible_and_warm_start(main_inst):
    a = run_qaoa(main_inst, "xy", 1, shots=1024, seed=3, maxiter=30)
    b = run_qaoa(main_inst, "xy", 1, shots=1024, seed=3, maxiter=30)
    assert a.counts == b.counts
    c = run_qaoa(main_inst, "xy", 1, shots=1024, seed=3, maxiter=10, init_params=a.params)
    assert c.history[0] <= a.history[0]          # warm start begins at the trained optimum
    assert c.history[0] == pytest.approx(min(a.history), rel=1e-6)


# ---------------------------------------------------------------------------
# Instance-level circuit, fixed-angle evaluation and sampling baselines
# ---------------------------------------------------------------------------
from carbon_ev.qaoa import build_qaoa_circuit, evaluate_params, random_feasible_baseline  # noqa: E402


def test_build_qaoa_circuit_wrapper(main_inst):
    for mixer in ("x", "xy"):
        qc = build_qaoa_circuit(main_inst, mixer=mixer, reps=2)
        assert qc.num_qubits == 16
        assert len(qc.parameters) == 4
        assert qc.num_clbits == 16


def test_evaluate_params_matches_run_qaoa_samples(main_inst):
    res = run_qaoa(main_inst, "xy", 1, shots=1024, seed=5, maxiter=15)
    ev = evaluate_params(main_inst, "xy", 1, res.params, shots=1024, seed=5)
    assert ev.demand_prob == 1.0
    assert ev.depth == res.depth and ev.n_2q == res.n_2q
    assert ev.opt_obj == pytest.approx(res.opt_obj)


def test_random_baselines(main_inst):
    base = random_feasible_baseline(main_inst, shots=20000, seed=0)
    u, d = base["uniform"], base["demand_satisfying"]
    assert d["demand_prob"] == 1.0
    assert u["demand_prob"] == pytest.approx(15 * 5 * 10 / 2**16, abs=0.005)
    assert d["feasible_prob"] == pytest.approx(500 / 750, abs=0.02)   # 500 feasible of 750 demand-ok
    assert u["feasible_prob"] < d["feasible_prob"]
    for m in (u, d):
        assert 0.0 <= m["p_opt"] <= m["p_within_1pct"] <= m["feasible_prob"] <= 1.0
    # trained XY must beat the Dicke-start sampling baseline on feasibility mass near the optimum
    res = run_qaoa(main_inst, "xy", 1, shots=20000, seed=0)
    assert res.p_within_1pct > d["p_within_1pct"]


# ---------------------------------------------------------------------------
# CVaR objective, ramp init, restarts
# ---------------------------------------------------------------------------
from carbon_ev.qaoa import cvar_from_distribution, ramp_init_params  # noqa: E402


def test_cvar_alpha_one_is_mean():
    E = np.array([3.0, 1.0, 2.0, 7.0])
    p = np.array([0.4, 0.1, 0.3, 0.2])
    assert cvar_from_distribution(E, p, 1.0) == pytest.approx(float(E @ p))


def test_cvar_by_hand_with_partial_last_bin():
    E = np.array([3.0, 1.0, 2.0])
    p = np.array([0.7, 0.1, 0.2])
    # sorted: E=1 (p .1), E=2 (p .2), E=3 (p .7). alpha=0.25 -> .1*1 + .15*2 = 0.4 -> /0.25
    assert cvar_from_distribution(E, p, 0.25) == pytest.approx(1.6)
    # alpha smaller than the first bin: just the lowest energy
    assert cvar_from_distribution(E, p, 0.05) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cvar_from_distribution(E, p, 0.0)


def test_exact_simulator_cvar_matches_distribution():
    rng = np.random.default_rng(8)
    Q = np.triu(rng.normal(size=(6, 6)))
    sim = ExactSimulator(Q, 0.5, 1, "x")
    params = np.array([0.4, 0.6])
    probs = sim.probabilities(params)
    assert sim.cvar(params, 1.0) == pytest.approx(sim.expectation(params))
    assert sim.cvar(params, 0.2) == pytest.approx(cvar_from_distribution(sim.energy, probs, 0.2))
    assert sim.cvar(params, 0.2) <= sim.expectation(params) + 1e-12


def test_ramp_init_shape_and_monotone():
    Q = np.triu(np.ones((4, 4)) * 3)
    r = ramp_init_params(Q, 3)
    assert r.shape == (6,)
    assert np.all(np.diff(r[:3]) > 0) and np.all(np.diff(r[3:]) < 0)


def test_run_qaoa_defaults_unchanged_by_new_kwargs(main_inst):
    a = run_qaoa(main_inst, "xy", 1, shots=512, seed=2, maxiter=25)
    b = run_qaoa(main_inst, "xy", 1, shots=512, seed=2, maxiter=25,
                 objective="mean", cvar_alpha=0.1, init="default", n_restarts=1)
    assert np.allclose(a.params, b.params) and a.counts == b.counts


def test_restarts_never_worse_and_count_evals(main_inst):
    from carbon_ev.qaoa import _run_qaoa_qubo
    Q, off, regs, ks = __import__("carbon_ev.qaoa", fromlist=["_problem_for"])._problem_for(main_inst, "xy")
    qc = build_qaoa_circuit_from_qubo(Q, 1, "xy", regs, ks)
    ex = ExactSimulator(Q, off, 1, "xy", regs, ks)
    kw = dict(shots=256, seed=4, maxiter=40, exact=ex, objective="cvar", cvar_alpha=0.25)
    one = _run_qaoa_qubo(Q, off, qc, n_restarts=1, **kw)
    three = _run_qaoa_qubo(Q, off, qc, n_restarts=3, **kw)
    assert three.train_objective <= one.train_objective + 1e-12
    assert three.n_evals > one.n_evals


def test_invalid_objective_and_init(main_inst):
    with pytest.raises(ValueError):
        run_qaoa(main_inst, "xy", 1, objective="median")
    with pytest.raises(ValueError):
        run_qaoa(main_inst, "xy", 1, init="random")
