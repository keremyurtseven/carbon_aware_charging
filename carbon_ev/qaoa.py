"""QAOA building blocks and runner for carbon-aware EV charging.

Contents:

* :func:`dicke_state`   - exact Dicke-state preparation (Baertschi-Eidenbenz 2019).
* :func:`xy_ring_mixer` - parametrized ring-XY mixer, one ring per register.
* :func:`cost_layer`, :func:`build_qaoa_circuit_from_qubo` - QAOA circuit for a raw QUBO.
* :class:`ExactSimulator` - fast numpy model of that circuit used for training.
* :func:`build_qaoa_circuit`, :func:`run_qaoa`, :func:`evaluate_params` - instance-level
  circuit, trainer and fixed-angle evaluator (:class:`QAOAResult`).
* :func:`random_feasible_baseline` - metrics of just sampling the start state (p = 0).

Simulation: angles are TRAINED with :class:`ExactSimulator`, a numpy copy of the
circuit (verified against ``qiskit.quantum_info.Statevector`` in the tests);
the reported SAMPLES come from ``qiskit.primitives.StatevectorSampler`` running
the real Qiskit circuit (or from a noisy sampler, see noise.py).

Qiskit conventions: qubit 0 is the least-significant bit of a bitstring.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.circuit.library import CRYGate, RYGate, XXPlusYYGate
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from carbon_ev.problem import Instance, conflicts, evaluate, objective_coefficients, var_index
from carbon_ev.qubo import (
    all_bitstrings, bitstring_to_x, default_penalties, qubo_energy, to_qubo, x_to_bitstring,
)


# ---------------------------------------------------------------------------
# Dicke state D(n, k) = uniform superposition of all weight-k n-bit strings
# ---------------------------------------------------------------------------
def _scs(qc: QuantumCircuit, qubits: Sequence[int], k: int) -> None:
    """Append the split-and-cyclic-shift block SCS_{m,k} on ``qubits`` (m = len).

    ``qubits[-1]`` is the "split" qubit. Using the recursion
    D(m,k) = sqrt((m-k)/m) |D(m-1,k)>|0> + sqrt(k/m) |D(m-1,k-1)>|1>,
    the block moves weight onto the last qubit with the right amplitude and
    leaves the remaining m-1 qubits in a state the next (smaller) block
    continues to process. Requires k < m.
    """
    m = len(qubits)
    q = list(qubits)
    # Two-qubit block on the last two qubits (amplitude sqrt(1/m)).
    qc.cx(q[m - 2], q[m - 1])
    qc.append(CRYGate(2 * np.arccos(np.sqrt(1.0 / m))), [q[m - 1], q[m - 2]])
    qc.cx(q[m - 2], q[m - 1])
    # Three-qubit blocks (amplitude sqrt(l/m)), l = 2..k.
    for l in range(2, k + 1):
        qc.cx(q[m - l - 1], q[m - 1])
        qc.append(
            RYGate(2 * np.arccos(np.sqrt(l / m))).control(2, annotated=False),
            [q[m - 1], q[m - l], q[m - l - 1]],
        )
        qc.cx(q[m - l - 1], q[m - 1])


def dicke_state(n: int, k: int) -> QuantumCircuit:
    """Return a circuit preparing the Dicke state D(n, k) from |0...0>.

    D(n, k) = C(n, k)^(-1/2) * sum of |x> over all x in {0,1}^n with |x| = k.
    Deterministic, ancilla-free construction of Baertschi & Eidenbenz
    (FCT 2019): flip the last k qubits to |1>, then apply SCS blocks on the
    first n, n-1, ..., 2 qubits (O(n k) gates, depth O(n)). The cascade
    shrinks toward qubit 0. Edge cases k = 0 and k = n give |0..0> and |1..1>
    (no superposition is possible, so only X gates are used).
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need n >= 1 and 0 <= k <= n, got n={n}, k={k}")
    qc = QuantumCircuit(n, name=f"Dicke({n},{k})")
    if k > 0:
        qc.x(range(n - k, n))  # |0^(n-k) 1^k>
    if k == 0 or k == n:
        return qc
    for m in range(n, 1, -1):
        _scs(qc, range(m), min(k, m - 1))
    return qc


# ---------------------------------------------------------------------------
# Ring XY mixer
# ---------------------------------------------------------------------------
def ring_edges(n: int) -> list[list[tuple[int, int]]]:
    """Return the edges of an n-cycle split into commuting layers.

    Positions are local (0..n-1). Layer A holds (0,1),(2,3),..., layer B holds
    (1,2),(3,4),... plus the closing edge (n-1,0) when n is even; for odd n the
    closing edge is a third layer C. n = 1 has no edges, n = 2 a single edge
    (the "ring" (0,1),(1,0) would double-count it).
    """
    if n < 2:
        return []
    a = [(i, i + 1) for i in range(0, n - 1, 2)]
    b = [(i, i + 1) for i in range(1, n - 1, 2)]
    layers = [a, b]
    if n > 2:
        if n % 2 == 0:
            b.append((n - 1, 0))
        else:
            layers.append([(n - 1, 0)])
    return [layer for layer in layers if layer]


def xy_ring_mixer(
    registers: Sequence[Sequence[int]],
    beta: float | ParameterExpression | None = None,
    num_qubits: int | None = None,
) -> QuantumCircuit:
    """Ring-XY mixer, one ring per register: prod exp(-i beta (XX+YY)/2).

    ``registers`` is a list of lists of qubit indices; each list is one EV's
    register and is mixed as a ring in the given order. Every term (XX+YY)/2
    conserves the Hamming weight, so the product conserves it within each
    register. The edges are applied in a Trotterized layer order (see
    :func:`ring_edges`); this is an approximation of exp(-i beta H_ring), which
    is fine for QAOA because only weight conservation matters.

    Registers of size 1 (or 0) have no edges and are left untouched: nothing
    can mix, and the single bit is fixed by its Dicke state (k = 0 or 1).
    Registers with k = 0 or k = n are also fine: the gates act trivially on
    |0..0> and |1..1>.

    ``beta`` may be a float or a Parameter (default: ``Parameter("beta")``).
    ``num_qubits`` defaults to 1 + the largest index used.
    """
    if beta is None:
        beta = Parameter("beta")
    used = [q for reg in registers for q in reg]
    if len(set(used)) != len(used):
        raise ValueError("registers must not overlap")
    nq = num_qubits if num_qubits is not None else (max(used) + 1 if used else 0)
    qc = QuantumCircuit(nq, name="XYring")
    # XXPlusYYGate(theta) = exp(-i theta/4 (XX+YY))  =>  theta = 2 beta.
    layer_lists = [ring_edges(len(reg)) for reg in registers]
    for depth in range(max((len(l) for l in layer_lists), default=0)):
        for reg, layers in zip(registers, layer_lists):
            if depth < len(layers):
                for i, j in layers[depth]:
                    qc.append(XXPlusYYGate(2 * beta), [reg[i], reg[j]])
    return qc


# ---------------------------------------------------------------------------
# QUBO-level QAOA
# ---------------------------------------------------------------------------
# Conventions (shared with qubo.py): E(x) = x^T Q x + offset, qubit a <-> x[a],
# Qiskit counts keys are little-endian (x = reversed key), x = (1 - Z)/2.
# Q may be upper-triangular or symmetric: only (Q + Q^T)/2 matters.
TRANSPILE_BASIS = ["rz", "sx", "x", "cx"]


def _sym(Q: np.ndarray) -> np.ndarray:
    """Symmetrized copy (Q + Q^T)/2 of a square matrix."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    return (Q + Q.T) / 2.0


def counts_to_matrix(counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """(X, w): one row x per distinct key of ``counts`` and its count."""
    keys = list(counts)
    X = np.array([bitstring_to_x(k) for k in keys], dtype=np.int8)
    return X, np.array([counts[k] for k in keys], dtype=float)


def cost_layer(
    Q: np.ndarray, gamma: float | ParameterExpression, num_qubits: int | None = None
) -> QuantumCircuit:
    """Phase separator exp(-i gamma H_C) up to a global phase.

    With x = (1 - Z)/2 and Qs = (Q + Q^T)/2:
        E(x) = const + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
        h_i  = -Qs_ii / 2 - (1/2) sum_{j != i} Qs_ij,     J_ij = Qs_ij / 2.
    exp(-i gamma h Z) = RZ(2 gamma h) and exp(-i gamma J ZZ) = RZZ(2 gamma J).
    """
    Qs = _sym(Q)
    n = Qs.shape[0] if num_qubits is None else num_qubits
    if Qs.shape[0] != n:
        raise ValueError("num_qubits must equal Q.shape[0]")
    qc = QuantumCircuit(n, name="cost")
    off = Qs - np.diag(np.diag(Qs))
    h = -0.5 * np.diag(Qs) - 0.5 * off.sum(axis=1)
    for i in range(n):
        if h[i] != 0.0:
            qc.rz(2 * gamma * float(h[i]), i)
    for i in range(n):
        for j in range(i + 1, n):
            if off[i, j] != 0.0:
                J = float(off[i, j]) / 2.0
                qc.rzz(2 * gamma * J, i, j)
    return qc


def build_qaoa_circuit_from_qubo(
    Q: np.ndarray,
    reps: int = 1,
    mixer: str = "x",
    registers: Sequence[Sequence[int]] | None = None,
    ks: Sequence[int] | None = None,
    measure: bool = True,
) -> QuantumCircuit:
    """Parametrized QAOA circuit for E(x) = x^T Q x.

    Parameters are ParameterVectors ``gamma[0..p-1]`` and ``beta[0..p-1]``.
    The logical parameter order used by this module is
    ``[gamma_1..gamma_p, beta_1..beta_p]`` (note: ``circuit.parameters`` is
    alphabetical, i.e. beta first, so use :func:`param_order`).

    mixer="x":  |+>^n start, RX(2 beta) mixer on every qubit.
    mixer="xy": each register in the Dicke state D(len(reg), k), ring-XY
                mixer exp(-i beta (XX+YY)/2) per register. Qubits outside
                every register (if any) stay in |0> and are not mixed.
    """
    n = np.asarray(Q).shape[0]
    if mixer not in ("x", "xy"):
        raise ValueError("mixer must be 'x' or 'xy'")
    gam = ParameterVector("gamma", reps)
    bet = ParameterVector("beta", reps)
    qc = QuantumCircuit(n, name=f"QAOA-{mixer}-p{reps}")
    if mixer == "x":
        qc.h(range(n))
    else:
        if registers is None or ks is None or len(registers) != len(ks):
            raise ValueError("mixer='xy' needs registers and ks of equal length")
        for reg, k in zip(registers, ks):
            qc.compose(dicke_state(len(reg), k), list(reg), inplace=True)
    for layer in range(reps):
        qc.compose(cost_layer(Q, gam[layer], n), inplace=True)
        if mixer == "x":
            for q in range(n):
                qc.rx(2 * bet[layer], q)
        else:
            qc.compose(xy_ring_mixer(registers, bet[layer], n), inplace=True)
    if measure:
        qc.measure_all()
    return qc


def param_order(circuit: QuantumCircuit) -> list[int]:
    """Index into [gamma_1..gamma_p, beta_1..beta_p] for each of circuit.parameters."""
    p = len(circuit.parameters) // 2
    out = []
    for par in circuit.parameters:
        m = re.fullmatch(r"(gamma|beta)\[(\d+)\]", par.name)
        out.append(int(m.group(2)) + (0 if m.group(1) == "gamma" else p))
    return out


def circuit_stats(circuit: QuantumCircuit) -> tuple[int, int]:
    """(depth, 2-qubit gate count) after transpiling to rz/sx/x/cx, level 1, seed 0."""
    t = transpile(circuit, basis_gates=TRANSPILE_BASIS, optimization_level=1, seed_transpiler=0)
    return t.depth(), int(t.count_ops().get("cx", 0))


def default_init_params(Q: np.ndarray, reps: int) -> np.ndarray:
    """Linear-ramp start [gamma_1..gamma_p, beta_1..beta_p].

    gamma_l = ((l + 1/2)/p) / max|Q|   (small phases: gamma * max|Q| <= 1)
    beta_l  = (pi/4) * (1 - (l + 1/2)/p)   (p = 1 gives the common pi/8 start)
    """
    scale = max(float(np.abs(Q).max()), 1e-12)
    frac = (np.arange(reps) + 0.5) / reps
    return np.concatenate([frac / scale, (np.pi / 4) * (1 - frac)])


def ramp_init_params(Q: np.ndarray, reps: int, dt: float = 0.75) -> np.ndarray:
    """Trotterized-quantum-annealing style linear ramp (opt-in via ``init="ramp"``).

    With s_k = k/(p+1), k = 1..p:  gamma_k = dt * s_k / max|Q|,  beta_k = dt * (1 - s_k).
    gamma rises and beta falls over the layers (dt = 0.75 is the usual TQA time step).
    For p = 1 this is gamma = 0.375/max|Q|, beta = 0.375.
    """
    scale = max(float(np.abs(Q).max()), 1e-12)
    s = (np.arange(reps) + 1.0) / (reps + 1.0)
    return np.concatenate([dt * s / scale, dt * (1.0 - s)])


def cvar_from_distribution(energies: np.ndarray, probs: np.ndarray, alpha: float) -> float:
    """CVaR_alpha: mean energy of the lowest-energy alpha fraction of the distribution.

    Exact tail (Barkoutsos et al., Quantum 4, 256 (2020)): outcomes are sorted by
    energy and the last included outcome is taken only partially so that exactly
    probability mass alpha is used. alpha = 1 gives the plain mean.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("cvar_alpha must be in (0, 1]")
    E = np.asarray(energies, dtype=float)
    p = np.asarray(probs, dtype=float)
    order = np.argsort(E, kind="stable")
    return _cvar_sorted(E[order], p[order], alpha)


def _cvar_sorted(E_sorted: np.ndarray, p_sorted: np.ndarray, alpha: float) -> float:
    """CVaR_alpha for outcomes already sorted by energy."""
    before = np.cumsum(p_sorted) - p_sorted
    w = np.clip(alpha - before, 0.0, p_sorted)
    return float((w * E_sorted).sum() / w.sum())


# ---------------------------------------------------------------------------
# Fast exact simulation (used for training; the circuit stays the reference)
# ---------------------------------------------------------------------------
class ExactSimulator:
    """Numpy statevector model of the QAOA circuit, for fast exact training.

    Qiskit's StatevectorSampler needs ~1 s per call for a 16-qubit circuit with
    a few hundred gates, which is too slow for ~100 optimizer evaluations. This
    class applies the same unitary directly: the cost layer is a diagonal phase
    exp(-i gamma (E(x) - offset)), the X mixer is a product of RX gates, and the
    XY mixer applies the same Trotterized ring edges as :func:`xy_ring_mixer`
    (each edge is a rotation in the {01, 10} subspace). Tests check it against
    ``Statevector`` of the real circuit.
    """

    def __init__(self, Q, offset, reps, mixer="x", registers=None, ks=None):
        self.n = np.asarray(Q).shape[0]
        self.reps, self.mixer, self.registers = reps, mixer, registers
        self.energy = qubo_energy(np.asarray(Q, dtype=float), offset, all_bitstrings(self.n))
        self.phase_energy = self.energy - offset
        self._order = np.argsort(self.energy, kind="stable")   # for CVaR
        self._energy_sorted = self.energy[self._order]
        if mixer == "x":
            self.psi0 = np.full(2**self.n, 2 ** (-self.n / 2), dtype=complex)
        else:
            prep = QuantumCircuit(self.n)
            for reg, k in zip(registers, ks):
                prep.compose(dicke_state(len(reg), k), list(reg), inplace=True)
            self.psi0 = Statevector(prep).data.astype(complex)
        self._edges = [
            [[(reg[i], reg[j]) for i, j in layer] for layer in ring_edges(len(reg))]
            for reg in (registers or [])
        ]

    def _rx_all(self, psi, beta):
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in range(self.n):
            v = psi.reshape(2 ** (self.n - 1 - q), 2, 2**q)
            a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :] = c * a + s * b
            v[:, 1, :] = s * a + c * b
        return psi

    def _xy_edge(self, psi, qa, qb, beta):
        lo, hi = sorted((qa, qb))
        v = psi.reshape(2 ** (self.n - 1 - hi), 2, 2 ** (hi - lo - 1), 2, 2**lo)
        a, b = v[:, 0, :, 1, :].copy(), v[:, 1, :, 0, :].copy()   # |hi=0,lo=1>, |hi=1,lo=0>
        c, s = np.cos(beta), -1j * np.sin(beta)
        v[:, 0, :, 1, :] = c * a + s * b
        v[:, 1, :, 0, :] = s * a + c * b

    def state(self, params: np.ndarray) -> np.ndarray:
        p = self.reps
        gam, bet = params[:p], params[p:]
        psi = self.psi0.copy()
        for layer in range(p):
            psi *= np.exp(-1j * gam[layer] * self.phase_energy)
            if self.mixer == "x":
                self._rx_all(psi, bet[layer])
            else:
                for depth in range(max((len(e) for e in self._edges), default=0)):
                    for layers in self._edges:
                        if depth < len(layers):
                            for qa, qb in layers[depth]:
                                self._xy_edge(psi, qa, qb, bet[layer])
        return psi

    def probabilities(self, params: np.ndarray) -> np.ndarray:
        return np.abs(self.state(params)) ** 2

    def expectation(self, params: np.ndarray) -> float:
        return float(self.probabilities(params) @ self.energy)

    def cvar(self, params: np.ndarray, alpha: float) -> float:
        """Exact CVaR_alpha of the QUBO energy distribution (see cvar_from_distribution)."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("cvar_alpha must be in (0, 1]")
        return _cvar_sorted(self._energy_sorted, self.probabilities(params)[self._order], alpha)


# ---------------------------------------------------------------------------
# Generic QUBO runner
# ---------------------------------------------------------------------------
@dataclass
class QuboRunResult:
    """Output of :func:`_run_qaoa_qubo`."""

    params: np.ndarray            # [gamma_1..gamma_p, beta_1..beta_p]
    counts: dict[str, int]        # final re-sampled counts (Qiskit keys)
    mean_energy: float            # mean QUBO energy of ``counts``
    n_evals: int
    wall_s: float
    depth: int
    n_2q: int
    history: list[float] = field(default_factory=list)
    train_objective: float = float("nan")   # final value of the training objective (best restart)


def sample_counts(circuit: QuantumCircuit, params: np.ndarray, shots: int, sampler) -> dict[str, int]:
    """Run ``circuit`` at ``params`` (logical order) and return counts."""
    vals = np.asarray(params, dtype=float)[param_order(circuit)]
    res = sampler.run([(circuit, vals)], shots=shots).result()[0]
    return res.data[circuit.cregs[0].name].get_counts()


def _run_qaoa_qubo(
    Q: np.ndarray,
    offset: float,
    circuit: QuantumCircuit,
    shots: int = 4096,
    seed: int = 0,
    init_params: np.ndarray | None = None,
    optimizer: str = "COBYLA",
    maxiter: int = 200,
    sampler=None,
    train_shots: int | None = None,
    exact: ExactSimulator | None = None,
    objective: str = "mean",
    cvar_alpha: float = 0.1,
    init: str = "default",
    n_restarts: int = 1,
) -> QuboRunResult:
    """Train angles by minimizing a QUBO-energy statistic, then re-sample with ``shots``.

    * Training objective: ``objective="mean"`` (default) is <E>; ``"cvar"`` is
      CVaR_alpha with alpha = ``cvar_alpha`` (mean of the best alpha fraction).
      With ``exact`` (an :class:`ExactSimulator`) and no ``sampler`` both are exact;
      otherwise they are computed from ``train_shots`` (default ``shots``) samples.
    * Start point: ``init_params`` if given, else ``init="default"`` (linear ramp of
      :func:`default_init_params`) or ``init="ramp"`` (:func:`ramp_init_params`).
    * ``n_restarts`` > 1: restart 0 uses that deterministic start, the others use
      random angles from ``np.random.default_rng(seed)`` (gamma * max|Q| ~ U(0, 1),
      beta ~ U(0, pi/2)). The restart with the lowest final training objective is
      kept; ``n_evals`` counts the evaluations of all restarts.
    * Final counts always come from ``sampler`` on ``circuit``. The default is
      ``StatevectorSampler(seed=seed)`` (ideal, local). Other V2 samplers (e.g.
      noisy Aer) receive the circuit as is; transpile first if the backend needs it.
    * The optimizer works in scaled variables u = [gamma * max|Q|, beta] so
      that COBYLA's step size (rhobeg = 0.3) suits both.
    """
    from qiskit.primitives import StatevectorSampler

    if objective not in ("mean", "cvar"):
        raise ValueError("objective must be 'mean' or 'cvar'")
    if init not in ("default", "ramp"):
        raise ValueError("init must be 'default' or 'ramp'")
    if not 0.0 < cvar_alpha <= 1.0:
        raise ValueError("cvar_alpha must be in (0, 1]")
    t0 = time.perf_counter()
    use_exact = exact is not None and sampler is None
    if sampler is None:
        sampler = StatevectorSampler(seed=seed)
    p = len(circuit.parameters) // 2
    scale = max(float(np.abs(Q).max()), 1e-12)
    if init_params is not None:
        x0 = np.asarray(init_params, dtype=float)
    elif init == "ramp":
        x0 = ramp_init_params(Q, p)
    else:
        x0 = default_init_params(Q, p)
    if x0.shape != (2 * p,):
        raise ValueError(f"init_params must have length {2 * p}")
    scalev = np.concatenate([np.full(p, scale), np.ones(p)])
    tshots = train_shots or shots
    history: list[float] = []
    cache: dict[str, float] = {}

    def train_objective(u: np.ndarray) -> float:
        if use_exact:
            if objective == "cvar":
                val = exact.cvar(u / scalev, cvar_alpha)
            else:
                val = exact.expectation(u / scalev)
        else:
            counts = sample_counts(circuit, u / scalev, tshots, sampler)
            new = [k for k in counts if k not in cache]
            if new:
                X, _ = counts_to_matrix({k: 1 for k in new})
                cache.update(zip(new, qubo_energy(Q, offset, X)))
            E = np.array([cache[k] for k in counts])
            w = np.array(list(counts.values()), dtype=float)
            if objective == "cvar":
                val = cvar_from_distribution(E, w / w.sum(), cvar_alpha)
            else:
                val = float(E @ w / w.sum())
        history.append(float(val))
        return float(val)

    options = {"maxiter": maxiter}
    if optimizer.upper() == "COBYLA":
        options["rhobeg"] = 0.3
    rng = np.random.default_rng(seed)
    best = None
    for r in range(max(1, int(n_restarts))):
        if r == 0:
            start = x0 * scalev
        else:
            start = np.concatenate([rng.uniform(0.0, 1.0, p), rng.uniform(0.0, np.pi / 2, p)])
        res = minimize(train_objective, start, method=optimizer, options=options)
        if best is None or res.fun < best.fun:
            best = res
    params = best.x / scalev
    counts = sample_counts(circuit, params, shots, sampler)
    X, w = counts_to_matrix(counts)
    depth, n2q = circuit_stats(circuit)
    return QuboRunResult(
        params=params, counts=counts,
        mean_energy=float(qubo_energy(Q, offset, X) @ w / w.sum()),
        n_evals=len(history), wall_s=time.perf_counter() - t0,
        depth=depth, n_2q=n2q, history=history, train_objective=float(best.fun),
    )


# ---------------------------------------------------------------------------
# Instance-level API
# ---------------------------------------------------------------------------
def schedule_metrics(inst: Instance, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized (objective, demand_ok, feasible) for the rows of X (var_index order).

    Matches ``problem.evaluate`` (checked in tests): demand_ok means every EV
    has exactly k_i charged blocks; feasible additionally needs no conflict.
    """
    X = np.asarray(X, dtype=np.int64)
    vi = var_index(inst)
    pos = {v: a for a, v in enumerate(vi)}
    obj = X @ objective_coefficients(inst)
    demand_ok = np.ones(len(X), dtype=bool)
    for i, ev in enumerate(inst.evs):
        cols = [pos[(i, t)] for t in ev.available]
        demand_ok &= X[:, cols].sum(axis=1) == ev.k_blocks
    feasible = demand_ok.copy()
    for i, j in conflicts(inst):
        for t in range(inst.n_blocks):
            if (i, t) in pos and (j, t) in pos:
                feasible &= (X[:, pos[(i, t)]] * X[:, pos[(j, t)]]) == 0
    return obj, demand_ok, feasible


def feasible_schedules(inst: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all feasible schedules as (X, objective). Product of C(n_i, k_i) rows."""
    vi = var_index(inst)
    pos = {v: a for a, v in enumerate(vi)}
    per_ev = []
    for i, ev in enumerate(inst.evs):
        cols = [pos[(i, t)] for t in sorted(ev.available)]
        per_ev.append([c for c in combinations(cols, ev.k_blocks)])
    rows = np.zeros((int(np.prod([len(c) for c in per_ev])), len(vi)), dtype=np.int8)
    for r, choice in enumerate(product(*per_ev)):
        for cols in choice:
            rows[r, list(cols)] = 1
    obj, _, feas = schedule_metrics(inst, rows)
    return rows[feas], obj[feas]


def exact_optimum(inst: Instance) -> tuple[np.ndarray, float]:
    """(x, objective) of the best feasible schedule, by enumeration (small fleets only)."""
    X, obj = feasible_schedules(inst)
    a = int(np.argmin(obj))
    return X[a], float(obj[a])


@dataclass
class QAOAResult:
    """Result of :func:`run_qaoa`.

    approx_ratio          = opt / objective(best feasible sample)      (in (0, 1], 1 = optimal)
    approx_ratio_expected = opt / mean objective of feasible samples   (same scale)
    feasible_prob         = P(demand met AND no conflict); demand_prob = P(demand met)
    p_opt / p_within_1pct = P(sample is feasible and has objective <= opt (+tol) / 1.01 * opt)
    Objectives are the dimensionless weighted costs of ``problem.evaluate``.
    If no sample is feasible: best_x = None, approx_ratio = nan.
    """

    params: np.ndarray
    best_x: np.ndarray | None
    counts: dict[str, int]
    feasible_prob: float
    approx_ratio: float
    p_opt: float
    n_evals: int
    wall_s: float
    depth: int
    n_2q: int
    p_within_1pct: float = float("nan")
    approx_ratio_expected: float = float("nan")
    demand_prob: float = float("nan")
    best_eval: dict | None = None      # problem.evaluate of best_x (co2_kg, cost_eur, ...)
    opt_obj: float = float("nan")
    mean_energy: float = float("nan")  # mean QUBO energy of the samples
    mixer: str = "x"
    reps: int = 1
    shots: int = 0
    history: list[float] = field(default_factory=list)


def registers_from_instance(inst: Instance) -> tuple[list[list[int]], list[int]]:
    """Per-EV qubit registers (var_index order) and Dicke weights k_i = k_blocks."""
    regs: list[list[int]] = [[] for _ in inst.evs]
    for a, (i, _t) in enumerate(var_index(inst)):
        regs[i].append(a)
    return regs, [ev.k_blocks for ev in inst.evs]


def _problem_for(inst: Instance, mixer: str, lam_demand=None, lam_conflict=None):
    """(Q, offset, registers, ks) for an instance and mixer."""
    if mixer not in ("x", "xy"):
        raise ValueError("mixer must be 'x' or 'xy'")
    dd, dc = default_penalties(inst)
    lam_demand = dd if lam_demand is None else lam_demand
    lam_conflict = dc if lam_conflict is None else lam_conflict
    Q, offset = to_qubo(inst, lam_demand, lam_conflict, include_demand=(mixer == "x"))
    regs, ks = registers_from_instance(inst)
    return Q, offset, regs, ks


def build_qaoa_circuit(
    inst: Instance,
    mixer: str = "x",
    reps: int = 1,
    lam_demand: float | None = None,
    lam_conflict: float | None = None,
    measure: bool = True,
) -> QuantumCircuit:
    """Parametrized QAOA circuit for an instance.

    mixer="x": penalty QUBO (demand + conflict), |+> start, RX mixer.
    mixer="xy": conflict-only QUBO, Dicke start per EV register, ring-XY mixer.
    Parameters: gamma[0..p-1], beta[0..p-1] (see :func:`build_qaoa_circuit_from_qubo`).
    """
    Q, _off, regs, ks = _problem_for(inst, mixer, lam_demand, lam_conflict)
    return build_qaoa_circuit_from_qubo(
        Q, reps, mixer, regs if mixer == "xy" else None, ks if mixer == "xy" else None, measure
    )


def sample_metrics(inst: Instance, counts: dict[str, int], opt_obj: float) -> dict:
    """Quality metrics of a set of sampled schedules (see :class:`QAOAResult`)."""
    X, w = counts_to_matrix(counts)
    obj, demand_ok, feas = schedule_metrics(inst, X)
    total = w.sum()
    tol = 1e-9 * max(1.0, abs(opt_obj))
    out = dict(
        feasible_prob=float(w[feas].sum() / total),
        demand_prob=float(w[demand_ok].sum() / total),
        approx_ratio=float("nan"), approx_ratio_expected=float("nan"),
        p_opt=float(w[feas & (obj <= opt_obj + tol)].sum() / total),
        p_within_1pct=float(w[feas & (obj <= 1.01 * opt_obj + tol)].sum() / total),
        best_x=None, best_eval=None,
    )
    if feas.any():
        a = int(np.argmin(np.where(feas, obj, np.inf)))
        out["best_x"] = X[a].astype(int)
        out["best_eval"] = evaluate(inst, out["best_x"])
        out["approx_ratio"] = float(opt_obj / obj[a])
        out["approx_ratio_expected"] = float(opt_obj / ((obj[feas] * w[feas]).sum() / w[feas].sum()))
    return out


def _result(inst, counts, params, opt_obj, mixer, reps, shots,
            depth, n_2q, n_evals, wall_s, mean_energy, history) -> QAOAResult:
    m = sample_metrics(inst, counts, opt_obj)
    return QAOAResult(
        params=np.asarray(params), best_x=m["best_x"], counts=counts,
        feasible_prob=m["feasible_prob"], approx_ratio=m["approx_ratio"], p_opt=m["p_opt"],
        n_evals=n_evals, wall_s=wall_s, depth=depth, n_2q=n_2q,
        p_within_1pct=m["p_within_1pct"], approx_ratio_expected=m["approx_ratio_expected"],
        demand_prob=m["demand_prob"], best_eval=m["best_eval"], opt_obj=float(opt_obj),
        mean_energy=mean_energy, mixer=mixer, reps=reps, shots=shots, history=history,
    )


# Training configuration used by every experiment (chosen from a scan on the
# main night, see docs/decisions.md). CVaR (Barkoutsos et al., Quantum 4, 256,
# 2020) trains on the best alpha-fraction of outcomes instead of the mean; the
# ramp start rescues the X mixer at p=3. Same settings for both mixers so the
# encoding comparison is fair. Use as run_qaoa(inst, mixer, reps, **TRAIN_CONFIG).
TRAIN_CONFIG = {"objective": "cvar", "cvar_alpha": 0.25, "init": "ramp", "maxiter": 1000}


def run_qaoa(
    inst: Instance,
    mixer: str = "x",
    reps: int = 1,
    shots: int = 4096,
    seed: int = 0,
    init_params: np.ndarray | None = None,
    optimizer: str = "COBYLA",
    maxiter: int = 200,
    sampler=None,
    opt_obj: float | None = None,
    lam_demand: float | None = None,
    lam_conflict: float | None = None,
    train_shots: int | None = None,
    objective: str = "mean",
    cvar_alpha: float = 0.1,
    init: str = "default",
    n_restarts: int = 1,
) -> QAOAResult:
    """Run QAOA on an instance with the X mixer (penalty) or ring-XY mixer (Dicke).

    mixer="x":  full penalty QUBO (demand + conflict), |+> start.
    mixer="xy": conflict-only QUBO; every EV register starts in D(n_i, k_i) and
                the ring-XY mixer keeps its weight (so demand always holds).
                Registers of size 1, or with k = 0 or k = n, cannot mix; the
                circuit handles them (they stay fixed), no special case needed.
    Penalties default to ``qubo.default_penalties(inst)`` (margin 1.5).
    Training uses the exact expectation (:class:`ExactSimulator`) unless a
    custom ``sampler`` is passed; reported metrics use ``shots`` sampled counts.
    ``opt_obj`` is the exact feasible optimum; if None it is found by enumeration.
    To evaluate already-trained angles on another sampler (e.g. noisy), use
    :func:`evaluate_params`.

    Training options (the defaults train on the plain mean from a linear-ramp start):
    ``objective="cvar"`` with ``cvar_alpha`` trains on CVaR_alpha of the QUBO energy,
    ``init="ramp"`` uses a TQA-style start, ``n_restarts`` > 1 adds random restarts
    (see :func:`_run_qaoa_qubo`). Training is deterministic; ``seed`` only affects
    shot sampling and the random restarts.
    """
    Q, offset, regs, ks = _problem_for(inst, mixer, lam_demand, lam_conflict)
    circuit = build_qaoa_circuit_from_qubo(
        Q, reps, mixer, regs if mixer == "xy" else None, ks if mixer == "xy" else None
    )
    exact = ExactSimulator(Q, offset, reps, mixer, regs, ks)
    run = _run_qaoa_qubo(Q, offset, circuit, shots, seed, init_params, optimizer, maxiter,
                         sampler, train_shots, exact, objective, cvar_alpha, init, n_restarts)
    if opt_obj is None:
        opt_obj = exact_optimum(inst)[1]
    return _result(inst, run.counts, run.params, opt_obj, mixer, reps, shots,
                   run.depth, run.n_2q, run.n_evals, run.wall_s, run.mean_energy, run.history)


def evaluate_params(
    inst: Instance,
    mixer: str,
    reps: int,
    params: np.ndarray,
    sampler=None,
    shots: int = 4096,
    seed: int = 0,
    opt_obj: float | None = None,
    lam_demand: float | None = None,
    lam_conflict: float | None = None,
) -> QAOAResult:
    """Sample the QAOA circuit at fixed ``params`` and report metrics (no training).

    Use for noisy evaluation of ideally trained angles (``sampler=noisy_sampler()``
    from noise.py) and for angle transfer between nights. Default sampler is the
    ideal ``StatevectorSampler(seed=seed)``.
    """
    from qiskit.primitives import StatevectorSampler

    t0 = time.perf_counter()
    Q, offset, regs, ks = _problem_for(inst, mixer, lam_demand, lam_conflict)
    circuit = build_qaoa_circuit_from_qubo(
        Q, reps, mixer, regs if mixer == "xy" else None, ks if mixer == "xy" else None
    )
    params = np.asarray(params, dtype=float)
    if params.shape != (2 * reps,):
        raise ValueError(f"params must have length {2 * reps}")
    counts = sample_counts(circuit, params, shots, sampler or StatevectorSampler(seed=seed))
    if opt_obj is None:
        opt_obj = exact_optimum(inst)[1]
    X, w = counts_to_matrix(counts)
    mean_e = float(qubo_energy(Q, offset, X) @ w / w.sum())
    depth, n2q = circuit_stats(circuit)
    return _result(inst, counts, params, opt_obj, mixer, reps, shots, depth, n2q, 0,
                   time.perf_counter() - t0, mean_e, [])


def random_feasible_baseline(
    inst: Instance, shots: int = 4096, seed: int = 0, opt_obj: float | None = None
) -> dict[str, dict]:
    """Metrics of QAOA at p = 0, i.e. just sampling the start state.

    Returns {"uniform": ..., "demand_satisfying": ...} where each value is the
    dict of :func:`sample_metrics` (feasible_prob, demand_prob, approx_ratio,
    approx_ratio_expected, p_opt, p_within_1pct, best_x, best_eval) for
    ``shots`` samples of
      uniform             - uniform random bitstrings (the |+>^n state of the X mixer),
      demand_satisfying   - uniform over bitstrings with exactly k_i ones in every EV
                            register (the Dicke start state of the XY mixer).
    This is the baseline QAOA has to beat; it is a sampling baseline, not a
    classical optimizer.
    """
    if opt_obj is None:
        opt_obj = exact_optimum(inst)[1]
    rng = np.random.default_rng(seed)
    n = len(var_index(inst))
    regs, ks = registers_from_instance(inst)
    Xu = rng.integers(0, 2, size=(shots, n))
    Xd = np.zeros((shots, n), dtype=np.int64)
    for reg, k in zip(regs, ks):
        # random k-subset per row: positions of the k smallest random keys
        keys = rng.random((shots, len(reg)))
        pick = np.argsort(keys, axis=1)[:, :k]
        cols = np.asarray(reg)[pick]
        np.put_along_axis(Xd, cols, 1, axis=1)

    def to_counts(X):
        counts: dict[str, int] = {}
        for row in X:
            key = x_to_bitstring(row)
            counts[key] = counts.get(key, 0) + 1
        return counts

    return {
        "uniform": sample_metrics(inst, to_counts(Xu), opt_obj),
        "demand_satisfying": sample_metrics(inst, to_counts(Xd), opt_obj),
    }
