"""QUBO and Ising construction.

Conventions (shared with qaoa.py / classical.py)
-----------------------------------------------
* Variable a = position in ``problem.var_index(inst)``; x is a 0/1 numpy array
  of length n in that order.
* Qubit a carries x[a]. Qiskit bitstrings are little-endian: in a counts key
  such as ``"0110"`` the RIGHTMOST character is qubit 0, so
  ``x[a] = int(bitstring[n - 1 - a])``. Use ``bitstring_to_x`` / ``x_to_bitstring``.
  Computational-basis index k (statevector position) has x[a] = (k >> a) & 1.
* QUBO energy is ``E(x) = x^T Q x + offset`` with Q upper-triangular:
  diagonal = linear coefficients (x^2 = x), Q[a, b] (a < b) = full coupling.
* Ising mapping: x_a = (1 - Z_a) / 2, so Z = +1 <-> x = 0 and Z = -1 <-> x = 1.
  ``qubo_to_ising`` returns (H, const) with E(x) = <x|H|x> + const; H has no
  identity term.

Penalty rule (``default_penalties``)
------------------------------------
g = max_a (P_i * h * c_t), the largest single-variable objective coefficient.
Dropping one charge saves at most g and breaking one conflict saves at most
g, so lam_demand = lam_conflict = margin * g with margin = 1.5 (default).
The rule is a heuristic, not a proof of exactness for every instance; the
tests verify that the QUBO minimum is the feasible optimum for margins
1.5 and 3. Larger lambda hurts QAOA conditioning, so keep it near g.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from carbon_ev.problem import Instance, conflicts, objective_coefficients, var_index


def default_penalties(inst: Instance, margin: float = 1.5) -> tuple[float, float]:
    """(lam_demand, lam_conflict) = margin * max single-variable objective coefficient."""
    g = float(np.max(objective_coefficients(inst)))
    return margin * g, margin * g


def to_qubo(
    inst: Instance,
    lam_demand: float,
    lam_conflict: float,
    include_demand: bool = True,
) -> tuple[np.ndarray, float]:
    """Build (Q, offset) with E(x) = x^T Q x + offset (Q upper-triangular).

    E(x) = sum_a coef_a x_a
           + lam_demand   * sum_i (sum_t x_it - k_i)^2      (if include_demand)
           + lam_conflict * sum_{(i,j) conflicting, t} x_it x_jt
    For feasible x the penalties vanish and E equals ``evaluate(...)['objective']``.
    With include_demand=False (XY mixer keeps demand satisfied) only the
    conflict penalty is added.
    """
    vi = var_index(inst)
    n = len(vi)
    pos = {v: a for a, v in enumerate(vi)}
    Q = np.zeros((n, n))
    offset = 0.0
    Q[np.arange(n), np.arange(n)] += objective_coefficients(inst)

    if include_demand:
        for i, ev in enumerate(inst.evs):
            idx = [pos[(i, t)] for t in sorted(ev.available)]
            k = ev.k_blocks
            for a in idx:
                Q[a, a] += lam_demand * (1 - 2 * k)
            for p in range(len(idx)):
                for q in range(p + 1, len(idx)):
                    Q[idx[p], idx[q]] += 2 * lam_demand
            offset += lam_demand * k * k

    for i, j in conflicts(inst):
        for t in range(inst.n_blocks):
            if (i, t) in pos and (j, t) in pos:
                a, b = sorted((pos[(i, t)], pos[(j, t)]))
                Q[a, b] += lam_conflict
    return Q, float(offset)


def qubo_energy(Q: np.ndarray, offset: float, x: np.ndarray):
    """E(x) = x^T Q x + offset. ``x`` may be (n,) -> float, or (m, n) -> array of m energies."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return float(x @ Q @ x + offset)
    return np.einsum("mi,ij,mj->m", x, Q, x) + offset


def qubo_to_ising(Q: np.ndarray, offset: float) -> tuple[SparsePauliOp, float]:
    """Convert the QUBO to (H, const) with x_a = (1 - Z_a)/2.

    E(x) = sum_a Q_aa x_a + sum_{a<b} J_ab x_a x_b + offset, J_ab = Q_ab + Q_ba.
    Using x_a = (1 - Z_a)/2:
        const = offset + sum_a Q_aa/2 + sum_{a<b} J_ab/4
        h_a   = -Q_aa/2 - sum_{b != a} J_ab/4         (coefficient of Z_a)
        c_ab  = J_ab/4                                (coefficient of Z_a Z_b)
    H = sum_a h_a Z_a + sum_{a<b} c_ab Z_a Z_b acts on qubit a = variable a
    (Qiskit little-endian: qubit 0 is the rightmost bitstring character).
    The diagonal of H in the computational basis at index k (x[a] = (k>>a)&1)
    plus const equals qubo_energy.
    """
    n = Q.shape[0]
    J = Q + Q.T
    np.fill_diagonal(J, 0.0)
    diag = np.diag(Q)
    const = float(offset + diag.sum() / 2 + np.triu(J, 1).sum() / 4)
    h = -diag / 2 - J.sum(axis=1) / 4
    terms: list[tuple[str, list[int], float]] = []
    for a in range(n):
        if h[a] != 0.0:
            terms.append(("Z", [a], float(h[a])))
    for a in range(n):
        for b in range(a + 1, n):
            if J[a, b] != 0.0:
                terms.append(("ZZ", [a, b], float(J[a, b] / 4)))
    if not terms:
        return SparsePauliOp("I" * n, coeffs=[0.0]), const
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n), const


def bitstring_to_x(bitstring: str) -> np.ndarray:
    """Qiskit little-endian bitstring (rightmost char = qubit 0) -> x array (x[a] = qubit a)."""
    return np.array([int(c) for c in reversed(bitstring.replace(" ", ""))], dtype=int)


def x_to_bitstring(x: np.ndarray) -> str:
    """Inverse of ``bitstring_to_x``."""
    return "".join(str(int(v)) for v in reversed(list(x)))


def all_bitstrings(n: int) -> np.ndarray:
    """All 2^n assignments as a (2^n, n) int8 array; row k has x[a] = (k >> a) & 1."""
    k = np.arange(2**n, dtype=np.int64)
    return ((k[:, None] >> np.arange(n)) & 1).astype(np.int8)
