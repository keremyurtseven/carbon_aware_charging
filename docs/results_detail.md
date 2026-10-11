# Detailed result tables

These tables were moved out of the README to keep it scannable. Every number is a value from the file named in the comment under each table; nothing is edited by hand. Back to the [README](../README.md).

## e2: savings of the exact optimum on the 3-EV slice, all 362 nights

Median per night over the 362 nights, w = 0.7, exact optimum against charge-on-arrival and against the median feasible schedule. The last column is the number of nights where the optimum emits more CO2 than charge-on-arrival.

| Night type | Nights | CO2 saved vs charge-on-arrival | Cost saved vs charge-on-arrival | CO2 saved vs median feasible schedule | Cost saved vs median feasible schedule | Nights with more CO2 than charge-on-arrival |
|---|---|---|---|---|---|---|
| Windy | 127 | 3.51 kg (14.7 %) | 1.43 EUR (16.5 %) | 3.50 kg | 0.92 EUR | 13 |
| Calm | 235 | 0.12 kg (0.3 %) | 1.43 EUR (11.7 %) | 0.84 kg | 0.93 EUR | 103 |
| All nights | 362 | 0.62 kg (1.7 %) | 1.43 EUR (13.0 %) | 1.46 kg | 0.93 EUR | 116 |

<!-- source: results/e2_summary.csv, nights_scope full_year_362, comparison milp_vs_on_arrival and milp_vs_median_feasible, median column; the last column is n_valid of frac_nights_opt_emits_more_co2 -->

## e2: QAOA gain over the untrained start state (29-night subset)

Median over nights of the per-night gain of QAOA (p = 3, one seed, CVaR training) over the matching untrained start state, with the number of nights where the gain is positive. Subset: the 29 nights Sep 10 to Oct 8, 2026.

| Quantity (median over nights) | XY QAOA gain over Dicke p = 0 (nights with a positive gain) | X QAOA gain over uniform p = 0 (nights with a positive gain) |
|---|---|---|
| All 29 nights: expected-cost ratio | +0.0058 (29/29) | +0.0012 (20/29) |
| All 29 nights: P(within 1 %) | +0.0366 (29/29) | +0.0024 (21/29) |
| All 29 nights: P(feasible) | +0.0945 (29/29) | +0.0305 (29/29) |
| Windy (6): expected-cost ratio | +0.0386 (6/6) | +0.0029 (6/6) |
| Windy (6): P(within 1 %) | +0.0140 (6/6) | +0.0042 (5/6) |
| Windy (6): P(feasible) | +0.0831 (6/6) | +0.6033 (6/6) |
| Calm (23): expected-cost ratio | +0.0055 (23/23) | +0.0012 (14/23) |
| Calm (23): P(within 1 %) | +0.0378 (23/23) | +0.0012 (16/23) |
| Calm (23): P(feasible) | +0.0945 (23/23) | +0.0122 (23/23) |

<!-- source: results/e2_summary.csv, comparison qaoa_*_minus_p0_*, quantity gain_*, median and n_positive columns -->

## e5: nights with more CO2 than charge-on-arrival, 3-EV slice

Medians over the 362 nights of the exact optimum against charge-on-arrival on the 3-EV slice, at w = 0.7 and w = 1.

| Nights | Count | Median CO2 saved at w = 0.7 | Median cost saved at w = 0.7 | Nights where the optimum emits **more** CO2 than charge-on-arrival, w = 0.7 | The same at w = 1 |
|---|---|---|---|---|---|
| Windy | 127 | 3.51 kg | 1.43 EUR | 13 (10.2 %) | 0 |
| Calm | 235 | 0.12 kg | 1.43 EUR | 103 (43.8 %) | 0 |
| All | 362 | 0.62 kg | 1.43 EUR | 116 (32.0 %) | 0 |

<!-- source: results/e5_summary.csv, nights_scope full_year_362, w = 0.7 and 1.0, quantities saving_co2_vs_on_arrival, saving_eur_vs_on_arrival, n_nights_opt_emits_more_co2_than_on_arrival, frac_nights_opt_emits_more_co2_than_on_arrival -->

## e6: scaling table, 11, 16 and 20 qubits

Rows with p = 0 are the untrained start states; the other rows are the largest completed p per mixer. All rows are in `results/e6_scaling.csv`.

| Fleet | Qubits | Feasible schedules | Mixer | p | P(feasible) | P(within 1 %) | Expected-cost ratio | Gain in expected-cost ratio over p = 0 | 2-qubit gates | Optimizer evaluations | Wall time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 EVs | 11 | 50 | X | 0 | 0.024 | 0.0002 | 0.615 | - | - | - | - |
| 2 EVs | 11 | 50 | X | 3 | 0.656 | 0.0271 | 0.645 | +0.030 | 180 | 1000 | 3 |
| 2 EVs | 11 | 50 | XY | 0 | 0.651 | 0.0239 | 0.630 | - | - | - | - |
| 2 EVs | 11 | 50 | XY | 3 | 0.669 | 0.1895 | 0.725 | +0.095 | 172 | 839 | 2 |
| 3 EVs | 16 | 500 | X | 0 | 0.007 | 0.0000 | 0.642 | - | - | - | - |
| 3 EVs | 16 | 500 | X | 3 | 0.624 | 0.0024 | 0.646 | +0.004 | 240 | 1000 | 32 |
| 3 EVs | 16 | 500 | XY | 0 | 0.670 | 0.0024 | 0.636 | - | - | - | - |
| 3 EVs | 16 | 500 | XY | 3 | 0.846 | 0.0811 | 0.770 | +0.134 | 248 | 1000 | 24 |
| 4 EVs | 20 | 228 | X | 0 | 0.000 | 0.0000 | 0.927 | - | - | - | - |
| 4 EVs | 20 | 228 | X | 2 | 0.105 | 0.0007 | 0.851 | -0.076 | 248 | 1000 | 1267 |
| 4 EVs | 20 | 228 | XY | 0 | 0.075 | 0.0002 | 0.836 | - | - | - | - |
| 4 EVs | 20 | 228 | XY | 2 | 0.107 | 0.0010 | 0.838 | +0.001 | 298 | 135 | 126 |

<!-- source: results/e6_summary.csv (rows with reps = 0 are the untrained start states; the other rows are the largest completed p per mixer; all rows are in results/e6_scaling.csv) -->

## e3: penalty versus XY mixer, ideal and noisy

Main night, 16 qubits. Medians; seeds are sampling replicates of one trained circuit (5 ideal, 3 noisy). `Penalty margin` is the multiple of the largest objective coefficient used as the penalty weight.

| Condition | Mixer | Penalty margin | p | P(feasible) | Expected-cost ratio | P(within 1 %) | 2-qubit gates | Depth |
|---|---|---|---|---|---|---|---|---|
| p = 0 (untrained) | X | - | 0 | 0.008 | 0.637 | 0.0000 | 0 | 0 |
| p = 0 (untrained) | XY | - | 0 | 0.670 | 0.636 | 0.0022 | 0 | 0 |
| Ideal | X | 1.5 | 1 | 0.164 | 0.639 | 0.0005 | 80 | 52 |
| Ideal | X | 1.5 | 2 | 0.507 | 0.645 | 0.0024 | 160 | 79 |
| Ideal | X | 1.5 | 3 | 0.625 | 0.649 | 0.0042 | 240 | 106 |
| Ideal | X | 3 | 2 | 0.476 | 0.642 | 0.0022 | 160 | 79 |
| Ideal | X | 6 | 2 | 0.379 | 0.639 | 0.0015 | 160 | 79 |
| Ideal | XY | 1.5 | 1 | 0.608 | 0.648 | 0.0068 | 164 | 215 |
| Ideal | XY | 1.5 | 2 | 0.707 | 0.703 | 0.0068 | 206 | 244 |
| Ideal | XY | 1.5 | 3 | 0.845 | 0.772 | 0.0862 | 248 | 267 |
| Noisy (approximate) | X | 1.5 | 1 | 0.136 | 0.644 | 0.0012 | 80 | 52 |
| Noisy (approximate) | X | 1.5 | 2 | 0.329 | 0.644 | 0.0017 | 160 | 79 |
| Noisy (approximate) | X | 1.5 | 3 | 0.325 | 0.646 | 0.0010 | 240 | 106 |
| Noisy (approximate) | XY | 1.5 | 1 | 0.332 | 0.647 | 0.0029 | 164 | 215 |
| Noisy (approximate) | XY | 1.5 | 2 | 0.332 | 0.688 | 0.0027 | 206 | 244 |
| Noisy (approximate) | XY | 1.5 | 3 | 0.330 | 0.755 | 0.0271 | 248 | 267 |

<!-- source: results/e3_summary.csv, *_median columns -->

## e4: angle transfer

Angles (p = 3) tuned on the main night or on a median calm night (Oct 3), reused on the other 28 nights; medians over those nights.

| Mixer | Angles tuned on | Metric | Reused | Re-tuned per night | Untrained (p = 0) | Nights where reuse beats p = 0 |
|---|---|---|---|---|---|---|
| XY | Main night (windy) | Expected-cost ratio | 0.9597 | 0.9655 | 0.9540 | 28/28 |
| XY | Main night (windy) | P(within 1 %) | 0.0391 | 0.0613 | 0.0194 | 27/28 |
| XY | Main night (windy) | P(feasible) | 0.882 | 0.761 | 0.670 | 28/28 |
| XY | Calm night | Expected-cost ratio | 0.9616 | 0.9641 | 0.9524 | 13/28 |
| XY | Calm night | P(within 1 %) | 0.0046 | 0.0613 | 0.0179 | 7/28 |
| XY | Calm night | P(feasible) | 0.697 | 0.767 | 0.670 | 25/28 |
| X | Main night (windy) | Expected-cost ratio | 0.9546 | 0.9557 | 0.9539 | 16/28 |
| X | Main night (windy) | P(within 1 %) | 0.0151 | 0.0027 | 0.0002 | 28/28 |
| X | Main night (windy) | P(feasible) | 0.478 | 0.031 | 0.007 | 28/28 |
| X | Calm night | Expected-cost ratio | 0.9588 | 0.9529 | 0.9525 | 14/28 |
| X | Calm night | P(within 1 %) | 0.0005 | 0.0026 | 0.0002 | 13/28 |
| X | Calm night | P(feasible) | 0.013 | 0.038 | 0.007 | 27/28 |

<!-- source: results/e4_summary.csv, group all, transfer_median, per_night_median, p0_median, n_nights_transfer_gain_positive -->
