"""Approximate trapped-ion (IonQ Forte-class) noise model for local simulation.

APPROXIMATION.  This is a simple depolarizing + readout model, NOT a
calibrated device model.  The numbers below are order-of-magnitude values for
IonQ Forte-class systems as published by IonQ (checked Oct 10 2026 at
https://ionq.com/quantum-systems/forte and
https://ionq.com/quantum-systems/forte-enterprise): single-qubit gate error
about 0.02 % (Clifford RB), two-qubit gate error about 0.4 % (fidelity
about 99.6 %), state preparation and measurement error about 0.5 %.  These
are vendor headline figures, not a calibration snapshot, and
real devices drift between calibrations.  Treat results as "what a machine of
roughly this quality might do", never as a prediction for a specific device.
Everything runs locally with qiskit-aer; no API key, no hardware.

Model details
-------------
* Gate set: the circuit is transpiled to ['rz', 'sx', 'x', 'cx'] (the fixed
  basis used for all gate counts in this project).  One CX is charged the error
  of one native two-qubit gate (IonQ's native MS gate is locally equivalent to
  CX), and all-to-all connectivity means no routing overhead is added.
* Depolarizing error for average gate fidelity F on d = 2**m levels:
  lambda = (1 - F) * d / (d - 1)   (qiskit-aer's depolarizing_error(lambda, m)).
  'rz' is virtual (error-free) as on trapped-ion hardware.
* Readout: symmetric bit-flip probability ``readout`` per measured qubit.
* No idling/decoherence, crosstalk or leakage is modelled.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile

# Assumed error rates (1 - fidelity); see module docstring: approximate, vendor headline figures.
P1_DEFAULT = 2e-4      # single-qubit gate error (fidelity ~99.98 %)
P2_DEFAULT = 4e-3      # two-qubit gate error    (fidelity ~99.6 %)
READOUT_DEFAULT = 5e-3  # SPAM / readout flip     (fidelity ~99.5 %)

NOISY_BASIS = ["rz", "sx", "x", "cx"]


def aer_available() -> bool:
    """True if qiskit-aer can be imported."""
    try:
        import qiskit_aer  # noqa: F401
    except ImportError:
        return False
    return True


def _require_aer():
    try:
        import qiskit_aer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "qiskit-aer is required for the noise study: pip install qiskit-aer "
            "(everything else in this project runs without it)."
        ) from exc
    return qiskit_aer


def build_noise_model(p1: float = P1_DEFAULT, p2: float = P2_DEFAULT, readout: float = READOUT_DEFAULT):
    """Approximate IonQ-Forte-class depolarizing noise model (see module docstring).

    ``p1`` / ``p2`` are average gate infidelities (1 - F) of 1- and 2-qubit gates,
    ``readout`` the symmetric readout flip probability.
    """
    _require_aer()
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    nm = NoiseModel(basis_gates=NOISY_BASIS)
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1 * 2.0, 1), ["sx", "x"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 * 4.0 / 3.0, 2), ["cx"])
    if readout > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return nm


class TranspilingSampler:
    """Wrap a V2 sampler so parametrized logical circuits can be run unchanged.

    The first time a circuit object is seen it is transpiled once to the fixed
    basis (optimization_level=1, seed_transpiler=0, parameters stay symbolic);
    later calls only bind new parameter values.  Exposes ``run(pubs, shots=)``
    like any V2 sampler, so it can be passed as ``sampler=`` to
    ``qaoa.run_qaoa`` / ``qaoa.evaluate_params``.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self._cache: dict[int, tuple[QuantumCircuit, QuantumCircuit]] = {}

    def _transpiled(self, circuit: QuantumCircuit) -> QuantumCircuit:
        hit = self._cache.get(id(circuit))
        if hit is None or hit[0] is not circuit:
            t = transpile(circuit, basis_gates=NOISY_BASIS, optimization_level=1, seed_transpiler=0)
            hit = (circuit, t)
            self._cache[id(circuit)] = hit
        return hit[1]

    def run(self, pubs, shots: int | None = None):
        new = []
        for pub in pubs:
            pub = tuple(pub)
            new.append((self._transpiled(pub[0]),) + pub[1:])
        return self.inner.run(new, shots=shots)


def noisy_sampler(
    seed: int = 0,
    p1: float = P1_DEFAULT,
    p2: float = P2_DEFAULT,
    readout: float = READOUT_DEFAULT,
    method: str = "statevector",
) -> TranspilingSampler:
    """Local noisy V2 sampler: aer SamplerV2 + approximate IonQ-Forte-class noise.

    APPROXIMATION (see module docstring for the assumed numbers).  Raises a clear
    ImportError if qiskit-aer is not installed.  Noise is applied by Monte-Carlo
    trajectories ('statevector' method, one noise realization per shot), which
    is the only option that fits 16 qubits in memory.

    Typical use (angles trained ideally, evaluated noisily)::

        res = run_qaoa(inst, "xy", 1)                       # ideal training
        noisy = evaluate_params(inst, "xy", 1, res.params, sampler=noisy_sampler(1))
    """
    _require_aer()
    from qiskit_aer.primitives import SamplerV2

    options = {
        "backend_options": {"noise_model": build_noise_model(p1, p2, readout), "method": method}
    }
    # qiskit-aer seeds noisy per-shot trajectories with roughly (seed + shot index),
    # so adjacent user seeds (0, 1, 2, ...) would reuse almost the same trajectories
    # and give near-identical counts. Spread them out with a SeedSequence.
    aer_seed = int(np.random.SeedSequence(seed).generate_state(1)[0])
    return TranspilingSampler(SamplerV2(seed=aer_seed, options=options))


def ionq_cloud_available() -> bool:
    """True if qiskit-ionq is installed and IONQ_API_KEY is set (optional, never required)."""
    import os

    try:
        import qiskit_ionq  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("IONQ_API_KEY"))
