"""Tests for the approximate IonQ-Forte-class noise model (needs qiskit-aer)."""
import sys
import time
from datetime import date

import pytest

pytest.importorskip("qiskit_aer")

from carbon_ev.data import load_hourly  # noqa: E402
from carbon_ev.noise import build_noise_model, noisy_sampler  # noqa: E402
from carbon_ev.problem import build_instance  # noqa: E402
from carbon_ev.qaoa import evaluate_params, run_qaoa  # noqa: E402


@pytest.fixture(scope="module")
def inst():
    return build_instance(load_hourly(), date(2026, 10, 8))


@pytest.fixture(scope="module")
def trained_xy(inst):
    return run_qaoa(inst, "xy", 1, seed=0)


def test_noise_model_has_expected_errors():
    nm = build_noise_model()
    assert set(nm.noise_instructions) >= {"sx", "x", "cx"}
    assert "rz" not in nm.noise_instructions        # virtual gate


def test_zero_noise_recovers_ideal_feasibility(inst, trained_xy):
    s = noisy_sampler(seed=1, p1=0.0, p2=0.0, readout=0.0)
    res = evaluate_params(inst, "xy", 1, trained_xy.params, sampler=s, shots=256)
    assert res.demand_prob == 1.0


def test_noise_lowers_xy_demand_probability_and_is_fast(inst, trained_xy):
    t0 = time.perf_counter()
    res = evaluate_params(inst, "xy", 1, trained_xy.params, sampler=noisy_sampler(seed=1), shots=1024)
    elapsed = time.perf_counter() - t0
    assert res.demand_prob < 1.0                     # noise leaks out of the Hamming-weight sectors
    assert res.feasible_prob < trained_xy.feasible_prob
    assert sum(res.counts.values()) == 1024
    assert elapsed < 60


def test_missing_aer_fails_with_clear_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "qiskit_aer", None)   # makes `import qiskit_aer` raise
    with pytest.raises(ImportError, match="qiskit-aer"):
        noisy_sampler()


def test_noisy_sampler_seeds_differ_and_reproduce():
    """Regression: adjacent seeds must not reuse aer's per-shot noise trajectories."""
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit

    from carbon_ev.noise import noisy_sampler

    qc = QuantumCircuit(3)
    qc.h(range(3))
    qc.measure_all()

    def counts(seed):
        return noisy_sampler(seed).run([(qc,)], shots=400).result()[0].data.meas.get_counts()

    assert counts(0) == counts(0)
    assert counts(0) != counts(1)
