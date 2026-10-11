"""Tests for the bus-depot extension (classical, realistic scale)."""
import numpy as np
import pytest

from carbon_ev.classical import brute_force, milp
from carbon_ev.data import list_nights, load_hourly
from carbon_ev.depot import (
    block_load_kw,
    depot_fleet,
    is_feasible_kw,
    min_peak_kw,
    on_arrival_capacity,
)
from carbon_ev.problem import EV, build_instance, evaluate


@pytest.fixture(scope="module")
def df():
    return load_hourly()


def test_fleet_is_deterministic_and_valid():
    a, b = depot_fleet(seed=3), depot_fleet(seed=3)
    assert a == b
    assert len(a) == 100
    for ev in a:
        assert 1 <= ev.k_blocks <= len(ev.available)
        assert all(0 <= t < 12 for t in ev.available)


def test_on_arrival_charges_earliest_and_respects_limit(df):
    night = list_nights(df)[0]
    fleet = depot_fleet(n_buses=20, seed=1)
    inst = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=1e9)
    x = on_arrival_capacity(inst)
    assert is_feasible_kw(inst, x)
    peak = block_load_kw(inst, x).max()
    capped = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=peak)
    assert is_feasible_kw(capped, on_arrival_capacity(capped))


def test_milp_beats_on_arrival_under_same_limit(df):
    night = list_nights(df)[-1]
    fleet = depot_fleet(n_buses=30, seed=2)
    free = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=1e9)
    peak = block_load_kw(free, on_arrival_capacity(free)).max()
    inst = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=peak)
    xa, xo = on_arrival_capacity(inst), milp(inst)
    assert is_feasible_kw(inst, xo)
    assert evaluate(inst, xo)["objective"] <= evaluate(inst, xa)["objective"] + 1e-9


def test_min_peak_matches_brute_force_on_tiny_fleet(df):
    night = list_nights(df)[0]
    fleet = [EV("a", 50.0, 2, (0, 1, 2)), EV("b", 30.0, 1, (1, 2)), EV("c", 20.0, 2, (0, 1, 2, 3))]
    inst = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=1e9)
    _, feas = brute_force(inst)
    best = min(block_load_kw(inst, x).max() for x, _ in feas)
    assert min_peak_kw(inst) == pytest.approx(best)
    tight = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=best)
    assert is_feasible_kw(tight, milp(tight))


def test_infeasible_limit_raises(df):
    night = list_nights(df)[0]
    fleet = [EV("a", 50.0, 2, (0, 1)), EV("b", 50.0, 2, (0, 1))]
    inst = build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=60.0)
    with pytest.raises(ValueError):
        on_arrival_capacity(inst)
    with pytest.raises(ValueError):
        milp(inst)
    assert np.isclose(min_peak_kw(build_instance(df, night, fleet=fleet, n_blocks=12, p_max_kw=1e9)), 100.0)


def test_fleet_energy_is_exact_not_rounded_up():
    fleet = depot_fleet(seed=0)
    for ev in fleet:
        assert ev.power_kw <= 100.0 + 1e-9
        assert ev.k_blocks == int(np.ceil(ev.power_kw * ev.k_blocks / 100.0 - 1e-6))


def test_min_cap_on_arrival_is_the_smallest_working_cap(df):
    from dataclasses import replace

    from carbon_ev.depot import min_cap_on_arrival_kw

    night = list_nights(df)[0]
    inst = build_instance(df, night, fleet=depot_fleet(n_buses=25, seed=4), n_blocks=12, p_max_kw=1e9)
    cap = min_cap_on_arrival_kw(inst, step_kw=50.0)
    on_arrival_capacity(replace(inst, p_max_kw=cap))           # works at the cap
    with pytest.raises(ValueError):
        on_arrival_capacity(replace(inst, p_max_kw=cap - 50.0))  # fails one step below
    assert min_peak_kw(inst) <= cap + 1e-6                     # smart never needs more


def test_milp_depot_matches_exact_milp_on_small_fleet(df):
    from carbon_ev.depot import milp_depot

    night = list_nights(df)[-1]
    inst = build_instance(df, night, fleet=depot_fleet(n_buses=8, seed=5), n_blocks=12, p_max_kw=400.0)
    x, gap = milp_depot(inst, mip_rel_gap=0.0)
    assert gap <= 1e-9
    assert evaluate(inst, x)["objective"] == pytest.approx(evaluate(inst, milp(inst))["objective"])
