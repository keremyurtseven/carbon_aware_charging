# Technical notes

The full write-up behind the project page: every experiment with its tables, the formulation, the training settings and the assumptions. The [project README](../README.md) is the short version. Every number cites the file in `results/` it comes from.

## 1. Results at a glance

"Per year" is the sum over 362 nights; scaled to 365 nights it is about 1 % more (`results/e7_summary.csv`, columns `*_per_365`). EUR figures are wholesale day-ahead energy.

| What we measured | Result | Source |
|---|---|---|
| **100-bus depot, one year** (362 nights, 12.6 GWh): carbon-first smart charging (w = 1) | Unmanaged charging: 5,019 t CO2 and €1.51 million. **Smart charging on the same 6.9 MW connection: 252 t CO2 less (5.0 %) and €124,000 less (8.2 %).** Against first-come-first-served charging capped at 5.4 MW: 200 t (4.0 %) and €90,000 (6.0 %). | `results/e7_summary.csv` |
| Windy and calm nights | 127 windy nights: 13.6 % CO2 saved (180 of the 252 t). 235 calm nights: 2.0 %. At w = 1 one night is above the unmanaged baseline, by 0.05 %, inside the solver's proven gap. | `results/e7_summary.csv` |
| Grid connection, three levels (the smallest on which every bus is still charged) | Unmanaged charging: **6.90 MW**. First-come-first-served charging, capped: **5.40 MW**. Smart scheduling: **3.31 MW** (MILP, within 0.5 %), 52 % below the unmanaged peak. | `results/e7_depot_year.csv` (columns `cap_unmanaged_kw`, `cap_fcfs_kw`, `cap_smart_min_kw`) |
| Cost-balanced variant (w = 0.7) | 217 t CO2 (4.3 %) and €194,000 (12.9 %) vs unmanaged; 171 t (3.4 %) and €155,000 (10.4 %) vs capped. Each night's price term is divided by that night's mean price, so the CO2/EUR balance varies by night (on 2025-10-24 the mean price is 12.4 EUR/MWh); 125 of 362 nights emit slightly more CO2 than unmanaged. | `results/e7_summary.csv`, `data/nl_grid_hourly.csv` |
| XY-mixer QAOA, p = 3, vs its own start state (16 qubits, median of 5 seeds) | P(within 1 % of optimum) 0.0862 vs 0.0022; expected-cost ratio 0.772 vs 0.636; P(feasible) 0.845 ideal and 0.330 under approximate IonQ-like noise. | `results/e1_summary.csv`, `results/e3_summary.csv` |
| Real IonQ Forte 1 hardware (16 qubits, XY p = 2, 1,000 shots) | P(feasible) 0.316 on hardware, 0.332 predicted by our noise model, 0.71 ideal. | `results/e9_forte1_hardware_check.csv` |
| QAOA subsolvers inside the depot (e8) | On the windy night all subsolvers close 97 to 99 % of the CO2 gap between the first-come-first-served start and the MILP reference. Exact enumeration takes about 0.4 ms per subproblem, XY QAOA about 2.8 s. | `results/e8_hybrid_depot.csv` |
| QAOA at 20 qubits (e6) | XY P(feasible) 0.107 at p = 2, against 0.846 at 16 qubits (p = 3). | `results/e6_summary.csv` |

**The question.** When should a bus depot charge its e-buses overnight, so that the same grid connection emits less CO2 and costs less? We answer it with optimization (MILP) on one full year of real Dutch grid data (Energy-Charts, 362 nights), and we run QAOA, a quantum algorithm in Qiskit, on small scheduling problems taken from the same depot.

![Three panels for a 100-bus Dutch depot modelled on public figures over 362 nights (Oct 2025 to Oct 2026), carbon-first scheduling (w = 1). (a) Bar chart of CO2 saved per month by smart charging versus unmanaged charge-on-arrival on the same 6.9 MW connection: savings in every month, largest in November 2025 (about 39 t), smallest in October 2026 (about 9 t, only 8 nights of data), and about 14 to 25 t in the other full months (lowest April 2026, highest July 2026). (b) Bar chart of wholesale energy cost saved per month, from about €5,000 (January 2026 and the partial October 2026) to about €24,000 (June 2026), with a second peak of about €18,000 in September 2026. (c) Smallest grid connection on which each strategy still charges every bus: 6.9 MW unmanaged, 5.4 MW with first-come-first-served capping, 3.3 MW with the smart schedule. Bars use distinct hatches as well as colours. Totals over the year at w = 1: 252 t CO2 (5.0 percent) and €124,000 (8.2 percent) saved (results/e7_summary.csv).](../results/e7_depot_year.png)

*Figure 1 (`results/e7_depot_year.png`): the depot over one year, carbon-first (w = 1) smart charging against unmanaged charging on the same 6.9 MW connection; yearly totals 252 t CO2 (5.0 %) and €124,000 (8.2 %) (`results/e7_summary.csv`). The first and last months are partial (21 and 8 nights).*

## 2. The problem

**The depot.** A bus depot parks its electric buses overnight and charges them all by the morning, under a limited grid connection. Charging each bus as soon as it returns piles the load into the evening and sets a high connection peak, while the Dutch grid's carbon intensity and day-ahead price change hour by hour. We decide **which bus charges in which hour** between 20:00 and 08:00, to minimize a weighted mix of CO2 and cost. Two rules apply:

1. **Demand:** every bus gets its required charging hours before it leaves.
2. **Site limit:** the grid connection caps how many buses can charge at once.

**Context in the Netherlands**, as reported by Sustainable Bus:

- About 2,748 zero-emission buses in the Netherlands in 2025, and only zero-emission bus purchases since 2025 ([Sustainable Bus, Netherlands zero-emission buses 2025](https://www.sustainable-bus.com/news/netherlands-zero-emission-buses-2025/)).
- Grid congestion is pushing some operators back to diesel: Arriva in Roosendaal, and EBS in Flevoland with 40 diesel buses ([Sustainable Bus, grid issues in the Dutch electric bus transition](https://sustainable-bus.com/news/netherlands-transition-electric-buses-grid-issues)).
- Operators already use smart charging: Transdev and Liander reduced the 10 MVA connection of the Amstelveen depot thanks to smarter charging ([Sustainable Bus, Transdev and Liander charging capacity](https://sustainable-bus.com/infrastructure/transdev-liander-charging-capacity-electric-buses)).

What we add is a **carbon-aware objective evaluated on a full year of measured carbon and price data**, a night-by-night account of where it helps, and a QAOA benchmark on the same problem.

**The modelled fleet**, anchored in public figures (parameters in `carbon_ev/depot.py`, `DEPOT_SOURCES`):

| Parameter | Value | Anchor or assumption |
|---|---|---|
| Fleet size | 100 e-buses | Anchor: 100 e-buses and 13 MW of charging at the Amstelveen/Schiphol depots ([electrive, 2018](https://www.electrive.com/2018/03/28/extra-large-13-mw-charge-depot-for-100-e-buses-live-in-amsterdam)) |
| Energy use | 1.15 kWh/km (12 m bus), 1.63 kWh/km (18 m bus) | Anchor: ViriCiti data via [Sustainable Bus](https://sustainable-bus.com/wp-json/wp/v2/posts/10542). Assumption: 30 % of buses are 18 m |
| Daily distance | about 265 km, uniform +/- 40 km | Anchor: about 265 km per bus per day at an overnight-charged Dutch depot ([EVConsult presentation](https://www.LowCVP.org.uk/assets/presentations/6._EVBus_Rollout_Experiences_from_NL-BIbiFabius_EVConsult.pdf)). Assumption: the spread |
| Charger rating and power draw | 100 kW depot chargers; each bus draws its energy need divided by its number of 1-hour blocks (75.7 to 99.8 kW), so the delivered energy equals the need | Assumption (charger rating); derived (power per bus, `carbon_ev/depot.py`) |
| Arrival and departure | return 20:00 to 00:00, leave 05:00 to 07:00 (uniform) | Assumption |
| Charging need | 3 to 5 hours per bus; 34.7 MWh per night for the fleet | Derived (`results/e7_depot_year.csv`, column `energy_mwh`) |

**Baselines and solver.** *Unmanaged* charging means every bus starts when it arrives, with no limit; its peak needs 6.90 MW. *Capped first-come-first-served* charging is the simple fix an operator can apply today: buses start in arrival order and wait if the connection has no room; the smallest connection it can run on (no look-ahead) is 5.40 MW. **Smart charging** is the MILP optimum of the weighted CO2 and cost objective under the same or a smaller connection. The cost/CO2 solves are stopped within 0.2 % of the optimum and the minimum-connection solve is within 0.5 % (largest recorded gap: `results/e7_summary.csv`, column `max_mip_gap`); "MILP" for the depot means these gaps throughout.

**The 16-qubit scheduling problem for the quantum benchmark.** The depot has 812 binary variables (bus and hour pairs). The quantum benchmark runs on a smaller problem built from the same objective: three vehicles over six 2-hour blocks, 16 binary variables, one qubit each. In e8, QAOA also solves 2-bus subproblems of the depot itself. The three vehicles: 11 kW and 7.4 kW are common European AC charger ratings, 9.2 kW for EV B breaks ties between identical chargers; the site limit P_max = 19 kW, the arrival and departure times, the energy needs and the weight w = 0.7 are our choices.

| EV | Charger power | Blocks needed | Energy | Available blocks (0 = 20:00-22:00 ... 5 = 06:00-08:00) |
|---|---|---|---|---|
| A (e-van) | 11 kW | 2 | 44 kWh | 0 to 5 (all) |
| B (car, leaves 06:00) | 9.2 kW | 1 | 18.4 kWh | 0 to 4 |
| C (car, arrives 22:00) | 7.4 kW | 2 | 29.6 kWh | 1 to 5 |

A and B together would draw 20.2 kW, above the 19 kW limit, so they cannot charge in the same block. A+C and B+C can.

The main night of this problem is Oct 8 to 9, 2026. Carbon intensity fell from about 470 g CO2/kWh at 23:00 to about 150 g/kWh at 03:00 to 04:00, roughly a threefold swing (hourly values in `data/nl_grid_hourly.csv`: 469.1 at 23:00, 155.3 at 03:00, 151.3 at 04:00). Price and carbon mostly move together but not always: at 23:00 the electricity is cheaper than at 19:00 to 22:00, but it is the dirtiest hour of the evening.

![top, bar chart of carbon intensity per 2-hour block with the day-ahead price as a dashed line with markers; below, two EV-by-block grids (charge-on-arrival versus the optimal schedule) in which each EV has its own fill, hatch and marker (A blue circle, B orange crosshatched square, C green dotted triangle), white cells are available but idle and grey cells are not plugged in; the optimal schedule moves charging to the low-carbon blocks near the end of the night.](../results/main_demo.png)

*Figure 2 (`results/main_demo.png`, made by `main.py`): the main night of the 3-EV problem. Top, carbon intensity and price per 2-hour block. Bottom, charge-on-arrival versus the optimal schedule. Fills, hatches and markers differ per EV.*

## 3. What smart charging saves

For a depot manager the questions are how much CO2 and money carbon-aware charging saves per year, and whether it needs a smaller connection. All numbers below are for the modelled 100-bus depot over 362 nights (Oct 10, 2025 to Oct 8, 2026; the two clock-change nights are excluded). "Per year" means the sum over these 362 nights.

**Connection needed (three levels)**

| Charging strategy | Smallest grid connection on which every bus is still charged |
|---|---|
| Unmanaged (uncontrolled) charging | 6.90 MW |
| Capped first-come-first-served charging (no look-ahead) | 5.40 MW |
| Smart scheduling (MILP, within 0.5 %) | 3.31 MW |

<!-- source: results/e7_depot_year.csv, columns cap_unmanaged_kw, cap_fcfs_kw, cap_smart_min_kw (identical every night because the fleet is the same every night) -->

Smart scheduling needs 52 % less connection than the unmanaged peak. The cost figures below contain no capacity charges, so this saving is not priced.

**Yearly savings, two baselines, two weights.** Smart charging runs on the same connection as its baseline: 6.9 MW against unmanaged charging, 5.4 MW against capped first-come-first-served charging. Unmanaged baseline over the 362 nights: 5,019 t CO2 and €1.51 million for 12.6 GWh. Capped baseline: 5,013 t and €1.50 million.

| Weight w | CO2 saved vs unmanaged (6.9 MW) | Cost saved vs unmanaged | Nights with more CO2 than unmanaged | CO2 saved vs capped (5.4 MW) | Cost saved vs capped | Nights with more CO2 than capped |
|---|---|---|---|---|---|---|
| **1.0, carbon-first (headline)** | 252 t (5.0 %) | €124,000 (8.2 %) | 1 of 362 | 200 t (4.0 %) | €90,000 (6.0 %) | 1 of 362 |
| 0.7, cost-balanced | 217 t (4.3 %) | €194,000 (12.9 %) | 125 of 362 | 171 t (3.4 %) | €155,000 (10.4 %) | 106 of 362 |

<!-- source: results/e7_summary.csv, group all, baselines unmanaged and fcfs, columns co2_saved_t_sum, co2_saved_pct, eur_saved_sum, eur_saved_pct, nights_more_co2_than_baseline. The one night above the baseline at w = 1 exceeds it by 0.05 % against a proven solver gap of at most 0.2 % (max_mip_gap). -->

**Why the headline uses w = 1.** The objective for each night is c_t(w) = w * co2_t / mean(co2) + (1 - w) * price_t / mean(price), where the means are taken over that night. A night with a low mean price therefore amplifies the price term, and the balance between CO2 and money varies from night to night. On 2025-10-24 the mean price of the night is 12.4 EUR/MWh (`data/nl_grid_hourly.csv`). At w = 1 the price term vanishes and the objective is pure carbon; smart charging then emits no more CO2 than unmanaged charging (one night is 0.05 % above, inside the solver gap). At w = 0.7 the savings in money are €70,000 per year larger, and 125 nights emit slightly more CO2. Both are shown.

Per-365-night scalings (`co2_saved_t_per_365`, `eur_saved_per_365`) are in the same file and are about 1 % higher.

**Windy nights carry most of the CO2 saving.** A night is windy if its carbon swing (highest divided by lowest 2-hour block CO2) is at least 1.3 (see [decisions.md](decisions.md)).

| Night type | Weight w | Nights | CO2 saved vs unmanaged | Cost saved vs unmanaged | Nights with more CO2 than unmanaged | CO2 saved vs capped | Nights with more CO2 than capped |
|---|---|---|---|---|---|---|---|
| Windy | 1.0 | 127 | 180 t (13.6 %) | €62,000 (15.1 %) | 1 | 142 t (10.8 %) | 0 |
| Calm | 1.0 | 235 | 72 t (2.0 %) | €62,000 (5.7 %) | 0 | 58 t (1.6 %) | 1 |
| Windy | 0.7 | 127 | 177 t (13.4 %) | €69,000 (16.8 %) | 19 | 139 t (10.6 %) | 8 |
| Calm | 0.7 | 235 | 41 t (1.1 %) | €125,000 (11.4 %) | 106 | 32 t (0.9 %) | 98 |

<!-- source: results/e7_summary.csv, groups windy and calm, w = 1.0 and 0.7 -->

**What the numbers say**

- CO2: 5.0 % at w = 1 against unmanaged charging, 13.6 % on the 127 windy nights and 2.0 % on the 235 calm nights, where carbon intensity is flat.
- Money: 8.2 % at w = 1 and 12.9 % at w = 0.7. Smart charging costs more than unmanaged charging on 50 of 362 nights at w = 1 and on 3 nights at w = 0.7 (`results/e7_depot_year.csv`, columns `unmanaged_cost_eur` and `smart_*_cost_eur`).
- Connection: smart scheduling needs 3.31 MW, capped first-come-first-served charging 5.40 MW, unmanaged charging 6.90 MW.
- Against the capped baseline the savings are 200 t CO2 and €90,000 at w = 1.

Calm nights and nights with more CO2 are listed next to the others in every table.

## 4. How the quantum benchmark works

The quantum benchmark uses the 3-EV problem from section 2. The depot uses the same objective, with hourly blocks (12 per night) and a MILP instead of QAOA.

**Terms, defined once**

- **QUBO** (Quadratic Unconstrained Binary Optimization): a cost function over 0/1 variables with terms up to degree two. Constraints are added as penalty terms.
- **QAOA** (Quantum Approximate Optimization Algorithm): a circuit with alternating "cost" and "mixer" layers. A classical optimizer tunes its angles to make good bit strings likely.
- **Mixer:** the part of the circuit that moves probability between bit strings. The standard one is the X mixer.
- **XY mixer:** a mixer that swaps neighbouring bits and keeps the number of ones fixed.
- **Dicke state D(n, k):** an equal superposition of all n-bit strings with exactly k ones. It is the natural starting state for an XY mixer.

**Formulation.** The night (20:00 to 08:00) is split into 6 blocks of 2 h. Block carbon and price are the means of their hourly values. The variable x(i,t) is 1 if EV i charges in block t. The objective is the energy charged in each block times a weighted, dimensionless cost:

c_t(w) = w * co2_t / mean(co2) + (1 - w) * price_t / mean(price), with w = 0.7 on this slice. The means are taken over the blocks of the same night. The depot headline uses w = 1 (see section 3); the QAOA slice keeps w = 0.7, the weight all its experiments were run with.

We also report physical outputs separately: kg CO2 and EUR.

**Two encodings (the core comparison)**

| | Penalty QAOA | XY QAOA |
|---|---|---|
| Demand constraint (each EV gets k blocks) | Penalty term in the QUBO | Enforced by a ring XY mixer per EV, with each EV register started in a Dicke state |
| Site limit (pairwise conflicts) | Penalty term | Penalty term |
| Mixer | Standard X | Ring XY (keeps the number of ones per EV) |
| Start state | Equal superposition of all strings | Product of Dicke states |

The pipeline, as text:

```
grid data (CSV) -> night instance (blocks, fleet) -> QUBO -> Ising operator
   -> QAOA circuit (X mixer, or XY mixer + Dicke start) -> sampled schedules
   -> feasibility check + metrics, compared with the classical baselines
```

**Classical baselines (same instance, same objective):** charge-on-arrival, greedy, simulated annealing on the same QUBO, and the exact optimum by brute force (and `scipy.optimize.milp` for larger cases).

**Metrics:** probability that a sample is feasible; approximation ratio (cost of the best feasible sample, or the expected feasible cost, relative to the optimum); probability of sampling the exact optimum and a schedule within 1 % of it; CO2 and EUR; circuit depth and 2-qubit gate count after transpiling to `rz, sx, x, cx`; optimizer iterations and wall time. All sampled results use 4096 shots and fixed seeds. What a "seed" varies depends on the experiment, so read the replication line of each section: e1 uses 5 seeds that also vary the training restarts; e2 uses 1 training run per night; e3 uses 5 (ideal) or 3 (noisy) sampling replicates of one training run; e4 and e5 use 3 sampling replicates.

Adjacent night blocks have similar carbon, so many schedules are near-ties and P(exact optimum) is small even for good samples. With 4096 shots the best feasible sample is the optimum for every method, including the untrained start state, so the headline QAOA metrics are the **expected feasible cost ratio** (`approx_ratio_expected`) and the **probability of landing within 1 % of the optimum** (`p_within_1pct`), shown next to the baselines below.

**Two scales of the expected-cost ratio.** The same metric (`approx_ratio_expected`, 1 is best) sits at about 0.64 to 0.77 on the main night (p = 0 start 0.636, XY p = 3 0.772, `results/e1_summary.csv`) but at about 0.95 to 0.96 when averaged over nights in e4 (`results/e4_summary.csv`). These are different nights, not different metrics: on a windy night the mean feasible schedule is far from the optimum, while calm nights have flat cost landscapes where every schedule is close to the optimum. Compare QAOA only with the p = 0 baseline on the same night.

**Baselines at p = 0 (circuit depth zero, no training).** Each QAOA result is compared with its own start state: (a) uniform random bit strings, the start state of the X mixer, and (b) the Dicke start state alone, the start state of the XY mixer. Every QAOA table shows these rows.

**How the angles are trained.**

- The QAOA angles are trained with our own numpy statevector simulator (`ExactSimulator`) for speed, because training needs thousands of evaluations. It was checked against Qiskit's `Statevector`. The **final samples** we report come from Qiskit's `StatevectorSampler`.
- The training settings are fixed in `TRAIN_CONFIG` in `carbon_ev/qaoa.py` and are the same for both mixers: a CVaR objective with alpha = 0.25 (it optimizes the best 25 % of sampled outcomes; Barkoutsos et al., [Quantum 4, 256 (2020)](https://doi.org/10.22331/q-2020-04-24-256)), a linear-ramp initial guess, and at most 1000 optimizer iterations.
- Why: with the plain mean objective the expected solution quality stayed at the p = 0 level. The settings come from a scan on the main night (scratch runs, not part of `results/`).

## 5. Experiments

Each experiment below states what it shows. Detail tables are in [docs/results_detail.md](results_detail.md).

### 5.1 XY QAOA puts 8.6 % of its samples within 1 % of the optimum, against 0.2 % for its start state

*Experiment e1: main night, all methods.*

Medians over 5 seeds for the sampled methods, 4096 shots, CVaR training with 3 random optimizer restarts (`n_restarts` = 3). Restarts depend on the seed, so training varies across seeds (per-seed table below). "Best-sample CO2" is the CO2 of the best feasible sample. "Expected-cost ratio" is `approx_ratio_expected`: the average cost of the feasible samples relative to the optimum (1 is best). Depth and 2-qubit gates are after transpiling to `rz, sx, x, cx`.

| Method | Best-sample CO2 (kg) | Cost (EUR) | P(feasible) | Expected-cost ratio | P(within 1 %) | 2-qubit gates | Depth |
|---|---|---|---|---|---|---|---|
| on-arrival | 38.14 | 15.01 | - | - | - | - | - |
| median feasible schedule | 26.39 | 11.79 | - | - | - | - | - |
| greedy | 15.74 | 8.71 | - | - | - | - | - |
| SA (QUBO) | 15.69 | 8.72 | - | - | - | - | - |
| exact (MILP) | 15.69 | 8.72 | - | - | - | - | - |
| p=0 uniform random (X start) | 18.98 | 9.39 | 0.008 | 0.637 | 0.0000 | - | - |
| p=0 Dicke start (XY start) | 15.69 | 8.72 | 0.670 | 0.636 | 0.0022 | - | - |
| QAOA X p=1 | 15.69 | 8.72 | 0.164 | 0.639 | 0.0005 | 80 | 52 |
| QAOA X p=2 | 15.69 | 8.72 | 0.508 | 0.643 | 0.0024 | 160 | 79 |
| QAOA X p=3 | 15.69 | 8.72 | 0.625 | 0.649 | 0.0042 | 240 | 106 |
| QAOA XY p=1 | 15.69 | 8.72 | 0.923 | 0.663 | 0.0059 | 164 | 215 |
| QAOA XY p=2 | 15.69 | 8.72 | 0.723 | 0.707 | 0.0083 | 206 | 244 |
| QAOA XY p=3 | 15.69 | 8.72 | 0.845 | 0.772 | 0.0862 | 248 | 267 |

<!-- source: results/e1_summary.csv, *_median columns -->

**Takeaways**

- The exact optimum, simulated annealing and the Dicke start state alone all reach the 15.69 kg schedule as the best of 4096 samples.
- The XY mixer at p = 3 moves the expected-cost ratio from 0.636 (start state) to 0.772 and P(within 1 %) from 0.0022 to 0.0862.
- The X mixer raises P(feasible) from 0.008 (random sampling) to 0.625 at p = 3; its expected-cost ratio moves from 0.637 to 0.649.
- XY circuits are deeper (depth 267 vs 106 at p = 3). Wall time per run (training and sampling on the simulator): 60 s for XY p = 3 and 194 s for X p = 3.
- **Training outcome at p = 1 and p = 2 depends on the seed.** Per-seed values from `results/e1_main_night.csv`: at p = 1, seeds 0, 2 and 4 reach P(feasible) of about 0.92 and seeds 1 and 3 about 0.60. At p = 2, seeds 0 and 1 reach P(within 1 %) of 0.053 and 0.076, while seeds 2, 3 and 4 stay near 0.006. 2 of 5 seeds end in a lower-quality optimum despite 3 random restarts. At p = 3 all 5 seeds agree.

| XY seed | p = 1 P(feasible) | p = 2 P(within 1 %) | p = 3 P(feasible) | p = 3 P(within 1 %) |
|---|---|---|---|---|
| 0 | 0.923 | 0.0530 | 0.846 | 0.0811 |
| 1 | 0.604 | 0.0762 | 0.840 | 0.0881 |
| 2 | 0.924 | 0.0083 | 0.845 | 0.0857 |
| 3 | 0.614 | 0.0054 | 0.847 | 0.0889 |
| 4 | 0.926 | 0.0071 | 0.842 | 0.0862 |

<!-- source: results/e1_main_night.csv, rows QAOA XY p=1, p=2, p=3, columns p_feasible and p_within_1pct per seed -->

![Three dot plots (a) P(feasible), (b) P(within 1 percent of optimum), (c) expected-cost ratio over eight configurations in three groups: p=0 baselines, X mixer at p=1 to 3, XY mixer at p=1 to 3; per-seed markers plus median and interquartile range. QAOA XY p=3 puts 0.086 probability on schedules within 1 % of optimum, against 0.0022 for its p=0 start state. Main night Oct 8 to 9, 16 qubits, 4096 shots. Median and IQR over n = 5 seeds per QAOA bar (CVaR training, alpha = 0.25, 3 restarts). Expected-cost ratio: XY p=3 0.77 vs 0.64 at p=0.](../results/e1_main_night.png)

*Figure 3 (`results/e1_main_night.png`): the main-night comparison of the p = 0 baselines and the two mixers.*

### 5.2 XY QAOA improves on its start state on all 29 nights tested

*Experiment e2: all nights.*

This is the 3-EV problem (16 qubits) on all 362 nights of the year. Classical results (exact optimum, greedy, simulated annealing) use all 362 nights. The QAOA rows use the 29 nights Sep 10 to Oct 8, 2026 (QAOA runs are slow); they are in the second table.

Savings of the exact optimum on the 3-EV problem against charge-on-arrival, w = 0.7, median per night over the 362 nights:

Full table of the savings on all 362 nights: [e2 savings table](results_detail.md#e2-savings-of-the-exact-optimum-on-the-3-ev-slice-all-362-nights).

On the main night charge-on-arrival is the worst feasible schedule, so the tables also give savings against the median feasible schedule. On 116 of 362 nights the optimum emits more CO2 than charge-on-arrival (largest excess 2.37 kg; `results/e2_summary.csv`). Windy and calm nights are shown side by side.

![Three panels over a full year of 362 nights (classical exact optimum, w = 0.7). (a) and (b) Scatter, one point per night, x axis carbon swing (ratio of the highest to the lowest 2-hour block CO2, log scale): kg CO2 and euros saved versus charge-on-arrival. Windy nights (swing >= 1.3, n = 127) are filled orange-red circles, calm nights (n = 235) hollow blue squares. Savings grow with the swing; on many calm nights the CO2 saving is about zero or negative while euros are saved. (c) Hatched bars: median CO2 saving per month with interquartile bars, and a line with markers for the number of windy nights per month.](../results/e2_savings_all_nights.png)

*Figure 4 (`results/e2_savings_all_nights.png`): per-night CO2 and euro savings of the 3-EV slice and the monthly median. Windy and calm nights differ by marker shape and fill, not only by colour. Monthly numbers are in `results/e2_monthly.csv`.*

QAOA runs (29-night subset) use p = 3, one seed (seed 0) and the CVaR training settings, so e2 has a single run per night. We report the median **over nights** of the per-night gain over the matching untrained start state (`results/e2_summary.csv`, rows `gain_*`).

Table of QAOA gains by night type: [e2 QAOA gain table](results_detail.md#e2-qaoa-gain-over-the-untrained-start-state-29-night-subset).

**Takeaways**

- XY QAOA improves on its start state on all 29 nights in expected-cost ratio (median gain +0.006): +0.039 on the 6 windy nights and +0.006 on the 23 calm nights, where cost landscapes are flat.
- The X mixer's median gain in expected-cost ratio is +0.0012 (positive on 20 of 29 nights).
- The Dicke start state alone reaches best-sample approximation ratio 1 on all 29 nights (`p0_dicke`, `approx_ratio` in the same file), so QAOA is compared on expected-cost ratio and P(within 1 %).

### 5.3 XY encoding keeps 85 % of samples valid in simulation; approximate noise and real IonQ hardware bring it to about a third

*Experiment e3: penalty versus XY mixer, ideal and noisy.*

Main night, 16 qubits, 1 optimizer restart, 5 seeds for ideal runs and 3 for noisy runs. The seeds are **sampling replicates** of one trained circuit. The noisy runs reuse the ideal angles and sample from an approximate depolarizing noise model with IonQ-like error rates (see [decisions.md](decisions.md)). The `margin` is the multiple of the largest objective coefficient used as the penalty weight.

**e1 and e3 differ at XY p = 1 because of restarts.** e1 uses 3 random restarts and e3 uses 1. At XY p = 1 the e1 median (P(feasible) 0.923) comes from the better of two training outcomes; e1 seeds 1 and 3 land near 0.60, where the single-restart e3 run (0.608) sits. The circuits are the same; the difference shows how much training outcomes vary at p = 1.

Full table: [e3 table](results_detail.md#e3-penalty-versus-xy-mixer-ideal-and-noisy).

**Takeaways**

- Ideal simulator: XY P(feasible) is 0.845 at p = 3, the X mixer's 0.625 (penalty margin 1.5). The XY expected-cost ratio rises with p (0.772 at p = 3); the X mixer's stays near its start.
- A larger penalty lowers the X mixer's P(feasible): at p = 2 it is 0.507 (margin 1.5), 0.476 (margin 3) and 0.379 (margin 6).
- **Noise.** Under the approximate noise model XY P(feasible) is 0.330 at p = 3 and about the same at p = 1 and 2, against 0.670 for the noiseless Dicke start state alone. The noisy X mixer reaches 0.325 at p = 3.
- The feasible samples that remain keep their cost quality: noisy XY expected-cost ratio 0.755 at p = 3 (ideal 0.772). Noise lowers feasibility; the cost of the feasible schedules is nearly unchanged.
- **Estimate (not a measured result):** the noisy plateau of about 0.33 is roughly (2-qubit gate fidelity) to the power of the 2-qubit gate count, times the ideal P(feasible). For XY p = 3: 0.996^248 is about 0.37, times 0.845 is about 0.31 (2-qubit error 4e-3 from the noise model, see [decisions.md](decisions.md); gate count and ideal P(feasible) from `results/e3_summary.csv`). The near-identical values at p = 1, 2 and 3 follow from that product: more gates are offset by higher ideal feasibility.
- Gate cost: XY uses 248 two-qubit gates at p = 3 and the X mixer 240, with much greater depth (267 vs 106).

**Checked on IonQ Forte 1 hardware.** The team ran `main.py` on Qollab with the IonQ Forte 1 quantum computer (36 qubits) selected. The XY-mixer circuit (p = 2, 16 qubits, main night) was sampled on the hardware once, 1,000 shots, with the angles trained on the simulator in the same run. IonQ's compilation used 183 two-qubit gates (206 in our `rz, sx, x, cx` basis). Console output: `results/qollab_ionq_forte1_hardware.txt`.

| Source | Backend | Shots | P(feasible) | Standard error | P(within 1 %) | Expected-cost ratio | 2-qubit gates |
|---|---|---|---|---|---|---|---|
| Ideal simulator, same Qollab run | Qiskit StatevectorSampler | 1,000 | 0.697 | 0.015 | 0.0080 | 0.704 | 206 |
| Ideal simulator, e3 (median of 5) | Qiskit StatevectorSampler | 4,096 | 0.707 | 0.007 | 0.0068 | 0.703 | 206 |
| Approximate IonQ-like noise model, e3 (median of 3) | qiskit-aer + carbon_ev/noise.py | 4,096 | 0.332 | 0.007 | 0.0027 | 0.688 | 206 |
| **IonQ Forte 1 quantum computer (real hardware, via Qollab)** | ionq_qpu.forte-1 | 1,000 | 0.316 | 0.015 | 0.0040 | 0.687 | 183 |

<!-- source: results/e9_forte1_hardware_check.csv; standard error is the binomial standard error of P(feasible) -->

- **Hardware matches the noise model.** P(feasible) is 0.316 on Forte 1 (standard error 0.015) against 0.332 from our approximate IonQ-like model, a difference of -1.06 standard errors, and 0.707 on the ideal simulator. Hardware noise roughly halves feasibility, as the model predicted.
- **Quality of the feasible schedules.** The expected-cost ratio is 0.687 on hardware, 0.688 in the model and 0.704 (same run) to 0.703 (e3) ideal.
- **P(within 1 %)** is 0.004 on hardware (4 of 1,000 samples), 0.0027 in the model; 4 samples are too few for a precise comparison.
- One circuit, one job of 1,000 shots. The job took 803.5 s, and the IonQ SDK reported an execution time of 415 s (`results/qollab_ionq_forte1_hardware.txt`).

![Two stacked panels comparing QAOA variants on the 2026-10-08 night instance (16 qubits). Top: probability that a sampled schedule is feasible; bottom: approximation ratio of the mean feasible sample. Groups on the x axis are the penalty X mixer at p = 1 to 3 (penalty 1.5x), the X mixer at penalty 3x and 6x (p = 2), and the XY mixer at p = 1 to 3, annotated with two-qubit gate counts. Filled circles with IQR bars are ideal simulation, hollow squares are an approximate noisy simulation using the ideal angles; dashed and dotted horizontal lines are the p = 0 start-state references. The XY mixer keeps feasibility far above the X mixer but uses many more gates.](../results/e3_penalty_vs_xy.png)

*Figure 5 (`results/e3_penalty_vs_xy.png`): penalty versus XY mixer, ideal and approximate noisy simulation.*

### 5.4 Angles tuned on the windy main night carry over to 28 other nights

*Experiment e4: train once, schedule every night.*

We tuned QAOA angles (p = 3) on one night and reused them unchanged on the other 28 nights, then compared with re-tuning on every night and with the untrained start state (3 sampling replicates per night, `results/e4_summary.csv`). Two training nights: the main night (Oct 8, windy) and a median calm night (Oct 3). Medians over the 28 nights.

Full table: [e4 table](results_detail.md#e4-angle-transfer).

**Takeaways**

- XY angles from the main night give expected-cost ratio 0.9597 when reused, against 0.9655 with per-night re-tuning and 0.9540 untrained, and beat the untrained start on 28 of 28 nights. P(within 1 %) is 0.0391 reused, 0.0613 re-tuned and 0.0194 untrained.
- Angles tuned on a calm night (Oct 3) give P(within 1 %) 0.0046 against 0.0179 untrained (better on 7 of 28 nights); the expected-cost ratio is about level with the start state (median per-night gain -0.0006, positive on 13 of 28 nights). One possible reason, not tested: a flat calm night gives the optimizer little signal.
- For the X mixer, per-night re-tuning gives P(feasible) 0.031 against 0.478 for the reused main-night angles (P(within 1 %) 0.0027 vs 0.0151). The re-tuned baseline uses 1 restart, which is under-trained for the X mixer, so the negative "loss" values for X (`loss_vs_per_night_median`) reflect that baseline.
- e4 uses 3 sampling replicates per night and one training run per training night.

![Four scatter panels, one point per night against carbon swing (log x axis). Rows: X mixer and XY mixer at p = 3; columns: approximation ratio of the mean feasible sample and probability of a sample within 1 percent of the optimum. Grey diamonds are the untrained start state, black filled circles QAOA trained on that same night, red-orange triangles angles trained once on the main night and reused, hollow blue squares angles trained once on a median calm night and reused.](../results/e4_angle_transfer.png)

*Figure 6 (`results/e4_angle_transfer.png`): reused versus re-tuned angles against the untrained start, one point per night. Marker shape and fill identify each series.*

### 5.5 Carbon-only weighting removes every night with extra CO2

*Experiment e5: cost versus carbon on the 3-EV problem.*

On the 3-EV problem we swept the weight w from 0 (cost only) to 1 (carbon only) and solved all 362 nights exactly (MILP). Medians over nights of the exact optimum versus charge-on-arrival, from `results/e5_summary.csv` (scope `full_year_362`; savings in kg CO2 and EUR; "windy" means a carbon swing of at least 1.3, see [decisions.md](decisions.md)).

Table by night type: [e5 table](results_detail.md#e5-nights-with-more-co2-than-charge-on-arrival-3-ev-slice).

**At w = 0.7**, about 44 % of calm nights emit more CO2 than charge-on-arrival while saving money: when the carbon signal is flat, the price term dominates and the cheapest schedule is not the cleanest. On windy nights it is 13 of 127.

**At w = 1** no night emits extra CO2 (0 nights), and the median cost saving on calm nights falls from 1.43 EUR (w = 0.7) to 0.28 EUR (w = 1.0). The weight is a policy choice between the two.

**Main night.** The exact front for Oct 8 to 9 has two points (`results/e5_pareto.csv`): 18.62 kg CO2 at 8.26 EUR for w up to 0.2, and 15.69 kg at 8.72 EUR for w from 0.3 up. Charge-on-arrival on that night is 38.14 kg and 15.01 EUR. It is the worst of the 500 feasible schedules on this night, so `results/e5_summary.csv` also gives savings against the median feasible schedule. The QAOA samples in panel (a) of Figure 7 are 3 sampling replicates of one training run per w; they are in `results/e5_pareto_qaoa.csv`, separate from the classical rows in `results/e5_pareto.csv` to keep each file under 2 MB.

![Four panels. (a) Scatter of kg CO2 against euros for all feasible schedules of the 2026-10-08 night (grey points) with the Pareto-optimal staircase, the exact optima for w from 0 to 1 (blue circles on a line), the default w = 0.7, charge-on-arrival (red cross), the median schedule (orange diamond) and XY-QAOA samples (green triangles, 3 sampling replicates of one training run per w). (b) and (c) Median and interquartile band, across all 362 nights of the year (127 windy, 235 calm; classical exact optimum), of CO2 and euro savings versus charge-on-arrival as a function of w: windy nights as filled circles with solid line, calm nights as hollow squares with dashed line. (d) Percentage of nights where the optimum emits more CO2 than charge-on-arrival, against w, same markers. Raising w increases CO2 savings and reduces euro savings; the share of nights with extra CO2 falls to zero only at high w.](../results/e5_pareto.png)

*Figure 7 (`results/e5_pareto.png`): cost versus carbon. Raising w increases CO2 savings and reduces euro savings; the share of nights with extra CO2 falls to zero only at w = 1.*

### 5.6 XY keeps its lead at 16 qubits; at 20 qubits both mixers reach about 11 % valid samples

*Experiment e6: scaling from 11 to 20 qubits.*

We ran fleets of 2, 3 and 4 EVs on the same main night (w = 0.7): 11, 16 and 20 qubits. The instances differ in more than size: the 4-EV fleet adds a fourth vehicle D (7.4 kW, 1 block, blocks 2 to 5) and needs P_max = 16 kW instead of 19 kW, because at 19 kW the triple A+C+D would break the site limit although no pair does, and the limit would no longer be exactly pairwise (the script refuses to run in that case). Charge-on-arrival is 118 %, 119 % and 28 % above the optimum in objective value, with 50, 500 and 228 feasible schedules.

QAOA settings are the same everywhere (`TRAIN_CONFIG`, seed 0, 1 restart, at most 1000 evaluations). A run is skipped if 1.5 times the previous run's wall time exceeds 180 s, so the 20-qubit instance has no p = 3 rows.

Full table: [e6 scaling table](results_detail.md#e6-scaling-table-11-16-and-20-qubits).

**Takeaways**

- At 11 qubits both mixers reach about the same P(feasible) at p = 3 (X 0.656, XY 0.669). At 16 qubits XY is clearly ahead (X 0.624, XY 0.846 at p = 3).
- **At 20 qubits** XY and X give the same P(feasible) with these settings: at p = 2 it is X 0.105 and XY 0.107. The Dicke start state alone has 0.075 P(feasible) there.
- The gain in expected-cost ratio over p = 0 at 20 qubits is -0.076 for X and +0.001 for XY at p = 2, and P(within 1 %) is about 0.0010.
- The X mixer at p = 2 used all 1000 allowed evaluations and took 1267 s; the XY runs stopped after 41 and 135 evaluations (we have not checked whether that is convergence or a stall).
- Each point is one seed and one restart.

![Four line panels against number of qubits (11, 16, 20; modelled fleets of 2, 3 and 4 EVs): probability of a feasible sample, probability of a sample within 1 percent of the optimum (log axis), approximation ratio of the mean feasible sample, and two-qubit gate count. Blue hollow markers are the penalty X mixer, filled vermilion markers the XY mixer; marker shape and line style give the depth p = 1, 2, 3; black crosses are the untrained start-state references. Runs skipped by the time budget are absent.](../results/e6_scaling.png)

*Figure 8 (`results/e6_scaling.png`): scaling of the penalty X mixer and the XY mixer with the number of qubits. Marker shape and line style identify p, not colour alone.*

### 5.7 QAOA subsolvers match exact enumeration inside the 100-bus depot

*Experiment e8: QAOA as the subproblem solver in a hybrid depot optimizer.*

**What we built.** A large-neighbourhood search (LNS) over the whole depot (carbon-first w = 1, the first-come-first-served 5.4 MW connection), starting from the first-come-first-served schedule. Each step re-optimizes a group of 2 buses (at most 16 qubits) and accepts the change only if it is feasible and strictly cheaper; the search runs for up to 3 passes. Subsolvers: exact enumeration (seeds 0 to 2), and QAOA with the XY mixer and with the X mixer (CVaR, at most 200 optimizer iterations per subproblem, seeds 0 and 1). The reference is the MILP optimum on the same connection. Setup details are in the docstring of `experiments/e8_hybrid_depot.py`.

The QAOA is simulated: angles are trained on our numpy `ExactSimulator` and the 4096 shots are drawn from its exact output probabilities. A one-pass cross-check with the Qiskit circuit through `StatevectorSampler` agrees. Exact enumeration also skips trivial subproblems (at most one feasible assignment) before QAOA runs; whether a QAOA proposal is accepted depends only on the QAOA sample.

| Night | Subsolver | Seeds | Gap to the MILP reference closed (per seed) | Final CO2 (median) | Seconds per subproblem (mean over seeds) |
|---|---|---|---|---|---|
| Windy, 2026-10-08 | Exact enumeration | 0, 1, 2 | 97.3, 99.2, 97.0 % | 7,627 kg | 0.4 ms |
| Windy, 2026-10-08 | QAOA, XY mixer | 0, 1 | 98.0, 99.2 % | 7,563 kg | 2.8 s |
| Windy, 2026-10-08 | QAOA, X mixer | 0, 1 | 98.2, 98.6 % | 7,573 kg | 4.8 s |
| Calm, 2026-10-03 | Exact enumeration | 0, 1, 2 | not defined (see below) | 20,360 kg | 0.7 ms |
| Calm, 2026-10-03 | QAOA, XY mixer | 0, 1 | not defined | 20,360 kg | 2.2 s |

<!-- source: results/e8_hybrid_depot.csv (columns gap_closed_pct, co2_kg, and wall_s divided by n_subproblems, averaged over seeds) and results/e8_summary.csv. Seconds per subproblem are wall-clock times with 6 parallel worker processes, including overhead: orders of magnitude, not exact values. -->

**Results**

- **Windy night (2026-10-08).** The first-come-first-served start emits 12,321 kg CO2 and the MILP reference 7,495 kg (proven gap 0.08 %). Every subsolver closes 97 to 99 % of that gap, and the results overlap seed by seed (exact seed 1: 99.2 %, XY seed 1: 99.2 %). QAOA subsolvers end within about 0.5 % of exact enumeration on each seed.
- **Time per subproblem** (wall time divided by the number of subproblems, mean over seeds, windy night): exact enumeration 0.4 ms, XY QAOA 2.8 s, X QAOA 4.8 s on the simulator, about 10^3 to 10^4 times the exact-enumeration time (6,875 and 11,632 times; measured with parallel workers). QAOA kept the current assignment in about 49 % of its calls.
- **Calm night (2026-10-03).** The first-come-first-served start is 20,373 kg and the MILP reference 20,366 kg, a difference of 7 kg. The LNS saves 12 to 13 kg and ends 0.02 to 0.03 % below the MILP reference, inside the MILP's proven gap (0.04 %). The share of the gap closed is not defined here because the start is within 1 % of the reference.
- **Cross-check.** After one pass on the windy night, the Qiskit `StatevectorSampler` path closes 93.8 % of the gap and the numpy path 93.6 %. The XY subproblem's probability of a sample within 1 % of its optimum is 0.08 (numpy path) and 0.07 (Qiskit sampler), against 0.024 to 0.026 for the Dicke start state alone (`results/e8_summary.csv`).
- **What it shows.** QAOA runs as a drop-in subsolver inside a depot-scale optimizer, with depot-level results equal to exact enumeration on these 16-qubit subproblems. The subsolver interface is a sampler call, so a hardware or cloud sampler can replace the simulator; we ran simulators only.

![Three panels. (a) and (b): depot CO2 in kg against the index of the LNS subproblem for the main windy night and a calm night, with one line per subsolver (exact enumeration solid with circles, QAOA XY dashed with squares, QAOA X dotted with hollow triangles; thin lines other seeds), a dash-dotted line for the first-come-first-served start and a dotted line for the MILP reference (within its 0.2 % tolerance). (c) Seconds per subproblem on a log axis for the three subsolvers: exact enumeration takes milliseconds, simulated QAOA seconds. In the 100-bus depot, simulated QAOA subsolvers match exact enumeration (all close 97-99 % of the CO2 gap, within seed noise) but take 1000 to 10000 times longer. w = 1, 100 buses, 2-bus subproblems. Windy night, mean seconds per subproblem: exact 0.4 ms, QAOA-XY 2.8 s, QAOA-X 4.8 s (parallel workers). QAOA simulated with a statevector simulator; exact enumeration is the reference. Thin lines: other seeds.](../results/e8_hybrid_depot.png)

*Figure 9 (`results/e8_hybrid_depot.png`): depot CO2 against the LNS subproblem index on a windy and a calm night, and seconds per subproblem (log axis). Subsolvers differ by marker and line style. Runtime of the whole experiment: 1,057 s with 6 workers (`results/e8_runtime.txt`).*

## 6. What is new

**Related work.** Carbon-aware and peak-aware e-bus charging is established (Rupp et al. 2020; Brinkel et al. 2023; Winschermann et al. 2025, on a Dutch site), quantum annealing has been applied to e-bus charging cost (Yu and Nabil 2021), and QAOA has been applied to EV charging (Dalyac et al., Kea et al., Ekstrøm et al., Sturm et al.). The figures below are taken from abstracts and not checked against the full papers.

- Dalyac et al., "Qualifying quantum approaches for hard industrial optimization problems. A case study in the field of smart-charging of electric vehicles", EPJ Quantum Technology 8, 12 (2021). EDF and Pasqal; a maximum-independent-set formulation; real EV charging-session data. It does not use a grid carbon signal.
- Kea, Huot, Han, IEEE Access 11, 109964-109973 (2023), [doi:10.1109/ACCESS.2023.3320800](https://doi.org/10.1109/ACCESS.2023.3320800). QAOA for EV charging with a knapsack-based formulation.
- Ekstrøm, Wang, Schmitt, qubit versus qudit QAOA encodings for EV charging on random instances, [arXiv 2605.10255](https://arxiv.org/abs/2605.10255) (May 2026).
- Sturm et al. (Fraunhofer IAO), [arXiv 2404.07171](https://arxiv.org/abs/2404.07171) (2024). QAOA and quantum annealing for EV (car) charging.
- Yu and Nabil (EDF R&D China), Frontiers in Physics 9, 730685 (2021), [open-access article](https://www.frontiersin.org/articles/10.3389/fphy.2021.730685/full). Quantum annealing (D-Wave 2000Q) for e-bus charging scheduling. Cost only, penalty QUBO, no carbon objective, no QAOA.
- Rupp, Rieke, Handschuh, Kuperjans, Transportation Research Part D 81, 102293 (2020). E-bus charging against variable prices and CO2-equivalent intensity: savings of up to 13.6 % in cost and 14.9 % in CO2-equivalent; the cheapest window is not the cleanest.
- Brinkel et al., Transportation Research Part E 172, 103085 (2023). Three Dutch Qbuzz depots: peak shaving cuts cost by 23 to 32 % against charge-on-arrival, and day-ahead optimization a further 6 to 11 %. No carbon objective.
- Winschermann, van der Bijl, Gerards, Hurink, [arXiv 2504.06078](https://arxiv.org/abs/2504.06078) (2025, preprint). Emissions and peak scheduling on a real Dutch bus charging site; peak up to 57 % lower.
- A student project, [Quantum Optimization of Electric Vehicle Charging Schedules Using QAOA (TJHSST QLab)](https://qlab.sites.tjhsst.edu/projects/detail/97/).

XY mixers and Dicke-state initialization are established tools (used for portfolios, vehicle routing, rostering and others).

**What this project adds.** We did not find prior work that applies QAOA to EV or bus charging against a measured grid carbon-intensity and price signal, or that pairs such a quantum benchmark with a full-year depot-scale evaluation including calm nights. Specifically:

1. **A measured year of grid carbon intensity and day-ahead prices at depot scale, with a QAOA benchmark.** Dalyac et al. use real charging-session data, arXiv 2605.10255 uses random instances, and the classical e-bus studies above do not run QAOA. We report how often and by how much carbon-aware charging helps or costs CO2 over 362 nights, calm nights included.
2. **A constraint-encoding benchmark on that problem:** penalty QUBO with the standard X mixer versus a per-EV ring XY mixer with Dicke-state start, with the p = 0 start states as baselines, on an ideal and an approximate noisy simulator. The noise model built from IonQ's published error rates was checked against a real IonQ Forte 1 run (section 5.3). arXiv 2605.10255 compares qubit and qudit encodings; we did not find a penalty-versus-XY comparison for EV charging.
3. **QAOA as a subsolver inside a depot-scale optimizer (e8)**, compared with exact enumeration on the same subproblems.
4. **"Train once, schedule every night":** tune QAOA angles on one night, reuse them on all others, and measure the quality loss against per-night re-optimization. We did not find prior work that asks this for EV charging.

**How our numbers compare** (published figures from the abstracts; the interpretation is ours). Our cost saving of 12.9 % (wholesale energy, w = 0.7, `results/e7_summary.csv`) is close to the 13.6 % of Rupp et al., although the setups differ. Our minimum connection is 52 % below the unmanaged peak (`results/e7_depot_year.csv`), against a peak reduction of up to 57 % reported by Winschermann et al. Our CO2 saving of 5.0 % (w = 1) is lower than the 14.9 % of Rupp et al., probably because we schedule overnight only (no midday solar hours) and use the average carbon intensity, which swings less at night than during the day. We have not tested this explanation.

## 7. Assumptions and limitations

**Model**

- **Fleets.** The 100-bus depot is modelled on public figures (section 2). The charger rating, the arrival and departure spreads and the 30 % share of 18 m buses are our assumptions, and so are the 3-EV problem's chargers, energy needs and site limit.
- **Idealised charging.** Charging can be interrupted and resumed, runs at constant power (each bus draws its energy need divided by its number of blocks) with no battery taper, and uses 1-hour blocks. Real chargers, batteries and operating rules (for example, a bus that must charge in one go) reduce the flexibility and therefore the savings.
- **Objective weights.** The CO2 and price terms are each divided by that night's mean, so at w < 1 a night with a low mean price (for example 2025-10-24) amplifies the price term. The depot headline uses w = 1; the 3-EV problem and its experiments use w = 0.7. At w = 0.7 on the 3-EV problem, about 44 % of calm nights emit more CO2 than charge-on-arrival (section 5.5).

**Data**

- **One year** (Oct 10, 2025 to Oct 9, 2026), 362 complete nights (the 2 clock-change nights with 11 and 13 local hours are excluded), in the Netherlands. Other years or grids can give different savings.
- **Average carbon intensity** of Dutch consumption, not the marginal emissions of an extra charger.
- **EUR figures are wholesale day-ahead energy.** They exclude grid fees, taxes and capacity charges, so they are not a depot's bill, and the freed connection capacity is not priced.

**Solvers and QAOA**

- **Depot MILP.** The cost/CO2 solves are within 0.2 % of the optimum and the minimum-connection solve within 0.5 %. The full year takes 167.9 s (`results/e7_runtime.txt`).
- **QAOA is mostly simulated.** Angles are trained on a numpy statevector simulator and the final samples come from Qiskit's `StatevectorSampler` (section 4). One circuit (XY mixer, p = 2, 16 qubits) was sampled once on IonQ Forte 1 hardware via Qollab, 1,000 shots, with simulator-trained angles (section 5.3); everything else is simulated.
- **Classical solvers are faster at this size.** On the 16-qubit problems exact solvers take milliseconds (`results/e1_summary.csv`); in e8 exact enumeration takes about 0.4 ms per subproblem and QAOA about 2.8 s (XY mixer).
- **Noise model.** A depolarizing model meant to resemble IonQ gate errors; it is an approximation.
- **Scaling.** At 20 qubits XY and X give the same P(feasible) (0.107 for XY at p = 2) and the p = 3 runs were skipped by the time budget (section 5.6).
- **Training settings** were chosen by a scan on the main night (scratch runs, not in `results/`), so the QAOA numbers on that night (e1, e3, e6) are tuned.
- **Statistics.** e1 has 5 seeds. e2 has one seed per night on 29 nights. In e3 and e4 the seeds are sampling replicates of one trained circuit, so they do not capture training variance. e6 has one seed and one restart per point.
- **Near-ties.** Many schedules are within a fraction of a percent of the optimum, so P(exact optimum) understates solution quality.
- **Related work** in section 6 is not an exhaustive literature search.

## 8. How to run

**Requirements:** Python 3.10 or newer. Everything runs on a local statevector simulator; no API key and no hardware are needed.

### On Qollab

Open the project and run `main.py`. Qollab runs Python in the browser (Pyodide): Qiskit (>= 2.5.2) is installed from `requirements.txt`, and NumPy, SciPy, pandas and Matplotlib come with Pyodide, so nothing else is needed. The run prints the main-night table, the depot sections (one night, 30 nights, QAOA inside the depot) and shows the figure; it writes its tables to a new `qollab_output/` folder (the published `results/` files stay unchanged). The full run took 63.1 s on Qollab, with the same results as the local run (console output: `results/qollab_run.txt`). If you pick a non-built-in backend in Qollab's Select QPU dialog (for example the free IonQ Forte 1 remote simulator, which applies IonQ's noise model), `main.py` also runs the trained XY circuit (p = 2, 16 qubits) there once, using the angles trained on the simulator, and prints its P(feasible), P(within 1 %) and expected-cost ratio next to the ideal values. Selecting the IonQ Forte 1 quantum computer instead runs a real hardware job: it uses Qollab credits and can take many minutes (803.5 s in our run, `results/qollab_ionq_forte1_hardware.txt`); the remote simulators are free. The longer experiments (`experiments/`) are meant for a local run.

### Locally

```bash
pip install -r requirements-local.txt
python main.py
```

`main.py` is the demo: the 16-qubit QAOA benchmark on the main night, the 100-bus depot on that night, a 30-night depot run (short version of e7), and QAOA as the subsolver inside a 16-bus depot (short version of e8). The full e7 and e8 results are the ones reported above. Target runtime: under 2 minutes. Runtime of the full `main.py`: 51.6 s in the last recorded run (`results/main_demo_runtime.csv`: 24.6 s for QAOA with the X mixer, 5.8 s with the XY mixer, 1.4 s for the depot night, 3.8 s for the 30-night run, 13.3 s for QAOA inside the depot; p = 2, wall-clock on the laptop that ran it).

### Reproduce every figure

```bash
python experiments/e7_depot_year.py      # the depot over the full year (classical, MILP within 0.2 %)
python experiments/e1_main_night.py
python experiments/e2_all_nights.py
python experiments/e3_penalty_vs_xy.py
python experiments/e4_angle_transfer.py
python experiments/e5_pareto.py --qaoa # writes e5_pareto.csv (classical) and e5_pareto_qaoa.csv (QAOA overlay)
python experiments/e6_scaling.py       # 11, 16 and 20 qubits; --quick for a reduced run
python experiments/e8_hybrid_depot.py  # QAOA subsolvers inside the depot (6 parallel workers; --quick for a tiny run)
```

`e7` is classical (MILP, proven within 0.2 %) and needs no Qiskit. A quick check on three nights, `python experiments/e7_depot_year.py --max-nights 3 --out-dir results/scratch`, took a few seconds when we tested it; the full year took 167.9 s (`results/e7_runtime.txt`). The 20-qubit rows of e6 are slow: the X mixer at p = 2 took 1267 s (`results/e6_summary.csv`). Each script uses fixed seeds, writes CSV files and figures to `results/`, and can be re-run from scratch. Options for each: `--help`. Wall time per QAOA run on the main night, from `results/e1_summary.csv`: 60 s (XY, p = 3) and 194 s (X, p = 3), median of 5 seeds. E7 took 167.9 s (`results/e7_runtime.txt`), e8 1,057 s (`results/e8_runtime.txt`, 6 workers) and e6 about 27 min. Optional extras: `qiskit-aer` for the noisy runs and `qiskit-ionq` for the IonQ cloud simulator (needs an API key; not required).

### Fetch more data

```bash
python data/fetch_data.py 2025-10-10 2026-10-09    # the full year used here
python data/build_dataset.py
```

The code works with whatever days are present in `data/nl_grid_hourly.csv`. Dutch clock changes are handled by `data/nl_time.py` (standard library only), see `data/DATA.md`.

### Tests

```bash
pytest -q
```

## 9. Extending the project

- Price the freed connection capacity (capacity charges, congestion fees), and add grid fees to the cost.
- Use a real depot's timetable and charger data instead of our modelled fleet; try other fleet sizes with `--buses` and `--charger-kw` in `e7_depot_year.py`.
- Larger LNS groups inside the depot (3+ buses per subproblem): needs slack bits for the capacity constraint, a test of how the qubit cost grows (e6 stops at 20 qubits, e8 at 2-bus groups).
- Finer time resolution (15-minute blocks; the raw data in `data/nl_grid_15min.csv` is already 15-minute) and battery charging curves instead of constant power.
- Marginal instead of average emissions, or other countries and bidding zones from Energy-Charts.
- Other constraint-preserving mixers, deeper circuits, warm starts, other optimizers.
- Run more circuits on IonQ hardware (one XY circuit was run, section 5.3) and compare with the noisy simulator here.
- Rolling-horizon scheduling with forecast errors.
- Run the e8 subsolver interface on a cloud or hardware sampler and larger subproblems (more buses per group).

