from datetime import date

import numpy as np
import pytest

from carbon_ev.data import load_hourly
from carbon_ev.problem import build_instance, evaluate
from carbon_ev.qubo import (
    all_bitstrings,
    bitstring_to_x,
    default_penalties,
    qubo_energy,
    qubo_to_ising,
    to_qubo,
    x_to_bitstring,
)

NIGHTS = [date(2026, 10, 8), date(2026, 10, 7), date(2026, 10, 1)]


@pytest.fixture(scope="module")
def df():
    return load_hourly()


@pytest.fixture(scope="module")
def inst(df):
    return build_instance(df, NIGHTS[0])


@pytest.fixture(scope="module")
def X():
    return all_bitstrings(16)


def direct_energies(inst, X, lam_d, lam_c, include_demand=True):
    """objective + lam * violations computed straight from evaluate()."""
    out = np.empty(len(X))
    for r, x in enumerate(X):
        e = evaluate(inst, x)
        v = e["violations"]
        out[r] = e["objective"] + (lam_d * v["demand"] if include_demand else 0.0) + lam_c * v["conflict"]
    return out


@pytest.mark.parametrize("include_demand", [True, False])
def test_qubo_matches_direct_all_2_16(inst, X, include_demand):
    lam_d, lam_c = 1.7, 2.3  # arbitrary and different, to catch mixed-up penalties
    Q, off = to_qubo(inst, lam_d, lam_c, include_demand=include_demand)
    E = qubo_energy(Q, off, X)
    assert np.allclose(E, direct_energies(inst, X, lam_d, lam_c, include_demand))
    assert E[1234] == pytest.approx(qubo_energy(Q, off, X[1234]))


def test_qubo_upper_triangular(inst):
    Q, _ = to_qubo(inst, 1.0, 1.0)
    assert np.allclose(Q, np.triu(Q))


@pytest.mark.parametrize("margin", [1.5, 3.0])
@pytest.mark.parametrize("night", NIGHTS)
def test_qubo_argmin_is_feasible_optimum(df, X, night, margin):
    inst = build_instance(df, night)
    lam_d, lam_c = default_penalties(inst, margin)
    Q, off = to_qubo(inst, lam_d, lam_c)
    E = qubo_energy(Q, off, X)
    evals = [evaluate(inst, x) for x in X]
    feas = np.array([e["feasible"] for e in evals])
    obj = np.array([e["objective"] for e in evals])
    assert feas.sum() == 500
    assert np.allclose(E[feas], obj[feas])  # feasible x carry zero penalty
    assert feas[np.argmin(E)]
    assert E.min() == pytest.approx(obj[feas].min())
    assert np.sum(np.isclose(E, E.min())) == 1  # unique minimiser


def test_penalty_rule_conflict_only(inst, X):
    """Without the demand term, among demand-satisfying x the minimiser is still feasible."""
    lam_d, lam_c = default_penalties(inst)
    assert lam_d == lam_c > 0
    Q, off = to_qubo(inst, lam_d, lam_c, include_demand=False)
    ok = np.array([evaluate(inst, x)["violations"]["demand"] == 0 for x in X])
    Xok = X[ok]
    E = qubo_energy(Q, off, Xok)
    assert evaluate(inst, Xok[np.argmin(E)])["feasible"]


def test_ising_diagonal_matches_qubo_all_2_16(inst, X):
    lam_d, lam_c = default_penalties(inst)
    Q, off = to_qubo(inst, lam_d, lam_c)
    H, const = qubo_to_ising(Q, off)
    assert H.num_qubits == 16
    # basis index k <-> x[a] = (k >> a) & 1 <-> qubit a (little-endian)
    diag = np.asarray(H.to_matrix(sparse=True).diagonal()).real
    assert np.allclose(diag + const, qubo_energy(Q, off, X))
    assert all(set(p) <= {"I", "Z"} for p in H.paulis.to_labels())  # diagonal operator


def test_ising_bit_order_small():
    """Variable 0 <-> qubit 0 <-> rightmost bitstring character."""
    Q = np.zeros((3, 3))
    Q[0, 0] = 5.0
    Q[1, 2] = 7.0
    H, const = qubo_to_ising(Q, 0.5)
    d = np.asarray(H.to_matrix(sparse=True).diagonal()).real + const
    assert d[0b001] == pytest.approx(5.5)  # x0 = 1
    assert d[0b110] == pytest.approx(7.5)  # x1 = x2 = 1
    assert d[0] == pytest.approx(0.5)
    assert list(bitstring_to_x("001")) == [1, 0, 0]
    assert x_to_bitstring(np.array([1, 0, 0])) == "001"
    labels = dict(zip(H.paulis.to_labels(), H.coeffs.real))
    assert labels["IIZ"] == pytest.approx(-2.5)  # Z on qubit 0 is the rightmost character


def test_bitstring_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 2, size=16)
        assert np.array_equal(bitstring_to_x(x_to_bitstring(x)), x)
