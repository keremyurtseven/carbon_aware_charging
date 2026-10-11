from datetime import date

import numpy as np
import pytest

from carbon_ev.classical import brute_force, greedy, milp, on_arrival, simulated_annealing
from carbon_ev.data import load_hourly
from carbon_ev.problem import EV, build_instance, evaluate, var_index
from carbon_ev.qubo import default_penalties, to_qubo

NIGHTS = [date(2026, 10, 8), date(2026, 10, 7), date(2026, 10, 1)]
SEEDS = range(5)


@pytest.fixture(scope="module")
def df():
    return load_hourly()


def sa_x(inst, seed):
    lam_d, lam_c = default_penalties(inst)
    Q, off = to_qubo(inst, lam_d, lam_c)
    return simulated_annealing(Q, off, seed=seed)


def test_brute_force_main_night(df):
    inst = build_instance(df, NIGHTS[0])
    best, allf = brute_force(inst)
    assert len(allf) == 500
    assert all(evaluate(inst, x)["feasible"] for x, _ in allf)
    assert np.array_equal(best, allf[0][0])
    objs = [o for _, o in allf]
    assert objs == sorted(objs)
    assert evaluate(inst, best)["objective"] == pytest.approx(objs[0])
    assert var_index(inst)
    assert best.shape == (16,)


@pytest.mark.parametrize("night", NIGHTS)
def test_milp_equals_brute_force(df, night):
    inst = build_instance(df, night)
    bx, _ = brute_force(inst)
    mx = milp(inst)
    assert evaluate(inst, mx)["feasible"]
    assert evaluate(inst, mx)["objective"] == pytest.approx(evaluate(inst, bx)["objective"])
    assert np.array_equal(mx, bx)  # optimum is unique on these nights


@pytest.mark.parametrize("night", NIGHTS)
def test_baselines_feasible_and_not_better_than_optimum(df, night):
    inst = build_instance(df, night)
    opt = evaluate(inst, brute_force(inst)[0])["objective"]
    for name, x in [("on_arrival", on_arrival(inst)), ("greedy", greedy(inst))]:
        r = evaluate(inst, x)
        assert r["feasible"], name
        assert r["objective"] >= opt - 1e-9, name


@pytest.mark.parametrize("night", NIGHTS)
def test_simulated_annealing(df, night):
    inst = build_instance(df, night)
    opt = evaluate(inst, brute_force(inst)[0])["objective"]
    res = [evaluate(inst, sa_x(inst, s)) for s in SEEDS]
    assert all(r["feasible"] for r in res)
    objs = np.array([r["objective"] for r in res])
    assert np.all(objs >= opt - 1e-9)
    assert np.median(objs) <= 1.01 * opt


def test_sa_deterministic(df):
    inst = build_instance(df, NIGHTS[0])
    assert np.array_equal(sa_x(inst, 3), sa_x(inst, 3))


def test_on_arrival_main_night(df):
    inst = build_instance(df, NIGHTS[0])
    x = on_arrival(inst)
    vi = var_index(inst)
    sched = {(i, t) for (i, t), v in zip(vi, x) if v}
    # A (arrives first, listed first) takes 0,1; B must dodge A -> block 2; C arrives at 1 -> 1,2
    assert sched == {(0, 0), (0, 1), (1, 2), (2, 1), (2, 2)}
    r = evaluate(inst, x)
    assert r["feasible"]
    assert 36 <= r["co2_kg"] <= 40  # about 38 kg on the main night
    opt = evaluate(inst, brute_force(inst)[0])
    assert r["co2_kg"] > 2 * opt["co2_kg"]


def test_milp_nonpairwise_fleet_uses_true_power_limit(df):
    """Three 7 kW EVs, 15 kW limit: at most 2 charge together; MILP must respect it."""
    fleet = [EV(n, 7.0, 2, (0, 1, 2)) for n in "XYZ"]
    inst = build_instance(df, NIGHTS[0], fleet=fleet, p_max_kw=15.0)
    x = milp(inst)
    vi = var_index(inst)
    for t in range(inst.n_blocks):
        assert sum(7.0 for (i, tt), v in zip(vi, x) if v and tt == t) <= 15.0
    for i in range(3):
        assert sum(v for (j, _), v in zip(vi, x) if j == i) == 2
