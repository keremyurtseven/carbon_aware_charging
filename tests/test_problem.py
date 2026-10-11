import itertools
from datetime import date

import numpy as np
import pytest

from carbon_ev.data import load_hourly
from carbon_ev.problem import (
    DEFAULT_FLEET,
    EV,
    build_instance,
    conflicts,
    evaluate,
    site_limit_is_pairwise_exact,
    var_index,
)

MAIN = date(2026, 10, 8)


@pytest.fixture(scope="module")
def df():
    return load_hourly()


def feasible_schedules(inst):
    """Independent small brute force: all choices of k blocks per EV, filtered by evaluate."""
    vi = var_index(inst)
    pos = {v: a for a, v in enumerate(vi)}
    out = []
    choices = [itertools.combinations(sorted(e.available), e.k_blocks) for e in inst.evs]
    for combo in itertools.product(*choices):
        x = np.zeros(len(vi), dtype=int)
        for i, blocks in enumerate(combo):
            for t in blocks:
                x[pos[(i, t)]] = 1
        r = evaluate(inst, x)
        if r["feasible"]:
            out.append((r["objective"], r["co2_kg"], r["cost_eur"], combo))
    return sorted(out, key=lambda r: r[0])


def test_main_instance_structure(df):
    inst = build_instance(df, MAIN)
    assert inst.block_hours == 2.0 and inst.n_blocks == 6
    assert inst.block_labels == ["20-22", "22-00", "00-02", "02-04", "04-06", "06-08"]
    assert len(var_index(inst)) == 16
    assert var_index(inst)[:7] == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0)]
    assert conflicts(inst) == [(0, 1)]  # only A+B (20.2 kW > 19 kW)
    # block values are means of the hourly rows
    sub = df.loc["2026-10-08 20:00":"2026-10-09 07:00"]
    assert inst.co2[0] == pytest.approx(sub["co2_g_per_kwh"].iloc[:2].mean())
    assert inst.price[5] == pytest.approx(sub["price_eur_per_mwh"].iloc[10:].mean())
    # c_t(w) normalised by the mean: its weighted pieces average to 1
    assert inst.cost_weights.mean() == pytest.approx(1.0)


def test_site_limit_pairwise_exact_default_fleet():
    assert site_limit_is_pairwise_exact(DEFAULT_FLEET, 19.0)


def test_site_limit_not_pairwise_exact_detected():
    # three 7 kW chargers under a 15 kW limit: every pair fits (14), the triple does not (21)
    fleet = [EV(n, 7.0, 1, (0,)) for n in "XYZ"]
    assert not site_limit_is_pairwise_exact(fleet, 15.0)


def test_evaluate_known_schedule(df):
    inst = build_instance(df, MAIN)
    vi = var_index(inst)
    best = {(0, 4), (0, 5), (1, 3), (2, 3), (2, 4)}
    x = np.array([1 if v in best else 0 for v in vi])
    r = evaluate(inst, x)
    assert r["feasible"] and r["violations"] == {"demand": 0, "conflict": 0}
    kwh = {0: 22.0, 1: 18.4, 2: 14.8}  # P_i * 2 h
    co2 = sum(kwh[i] * inst.co2[t] for i, t in best) / 1000
    price = sum(kwh[i] * inst.price[t] for i, t in best) / 1000
    assert r["co2_kg"] == pytest.approx(co2)
    assert r["cost_eur"] == pytest.approx(price)


def test_evaluate_violations(df):
    inst = build_instance(df, MAIN)
    vi = var_index(inst)
    n = len(vi)
    r = evaluate(inst, np.zeros(n, dtype=int))
    assert not r["feasible"] and r["violations"]["demand"] == 4 + 1 + 4
    # A and B both charging in block 0 -> one conflict, demand satisfied
    x = np.zeros(n, dtype=int)
    for v in [(0, 0), (0, 1), (1, 0), (2, 1), (2, 2)]:
        x[vi.index(v)] = 1
    r = evaluate(inst, x)
    assert r["violations"] == {"demand": 0, "conflict": 1} and not r["feasible"]


@pytest.mark.parametrize(
    "night, kg, eur, blocks, gap_pct",
    [
        (date(2026, 10, 8), 15.69, 8.72, ((4, 5), (3,), (3, 4)), 0.20),
        (date(2026, 10, 7), 34.05, 11.55, ((4, 5), (3,), (3, 4)), 0.07),
        (date(2026, 10, 1), 51.49, 15.31, ((2, 3), (4,), (2, 3)), 0.03),
    ],
)
def test_section4_precheck(df, night, kg, eur, blocks, gap_pct):
    inst = build_instance(df, night, w=0.7)
    res = feasible_schedules(inst)
    assert len(res) == 500
    obj, co2, cost, combo = res[0]
    assert co2 == pytest.approx(kg, abs=0.01)
    assert cost == pytest.approx(eur, abs=0.01)
    assert combo == blocks
    gap = 100 * (res[1][0] - obj) / obj
    assert gap == pytest.approx(gap_pct, abs=0.01)


def test_main_night_non_degenerate(df):
    """Optimum is unique and the worst feasible schedule is far above it."""
    res = feasible_schedules(build_instance(df, MAIN))
    assert res[1][0] > res[0][0]
    assert res[-1][0] > 2 * res[0][0]


def test_build_instance_errors(df):
    with pytest.raises(ValueError):
        build_instance(df, date(2026, 10, 9))  # no D+1 data
    with pytest.raises(ValueError):
        build_instance(df, MAIN, n_blocks=5)
