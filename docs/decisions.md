# Design decisions

The modelling choices behind the project, and why we made them.

## From a car park to a bus depot

We started with three vehicles in a car park, a problem small enough to put on 16 qubits. A depot operator thinks in tonnes of CO₂, euros and megawatts per year, so we moved the headline to a 100-bus depot over a full year, solved with a classical optimizer. The three-vehicle problem stayed as the quantum benchmark, and QAOA joined the depot itself as a subproblem solver (e8). We also considered trains. A railway timetable has no load that can be shifted by a few hours, so there is nothing to schedule against the carbon signal, which ruled them out. Overnight bus charging is flexible up to each bus's departure.

## One year of data and Dutch clock changes

The dataset covers 10 October 2025 to 9 October 2026: 8,760 hourly rows from the Energy-Charts API. We fetched it directly with `data/fetch_data.py`, after checking our first, smaller snapshot value by value against the API (960 of 960 identical). Dutch local time is computed in `data/nl_time.py` from the EU daylight-saving rule, with the standard library only. The two clock-change nights have 11 and 13 local hours and are left out, which gives 362 nights.

## The depot fleet

`carbon_ev/depot.py` builds 100 buses, 30 % of them 18 m long, each driving about 265 km a day (±40 km). The buses use the measured Dutch values of 1.15 and 1.63 kWh/km, return between 20:00 and 00:00, and leave between 05:00 and 07:00. The sources are listed in `DEPOT_SOURCES`. Chargers are rated at 100 kW. Each bus draws exactly its energy need spread over its charging hours (75.7 to 99.8 kW), so the depot needs 34.7 MWh per night. Charging happens in one-hour blocks at constant power and may pause between blocks.

## Baselines and connection sizes

We compare the smart schedule with two baselines:
- **Unmanaged charging:** every bus starts on arrival.
- **First-come-first-served charging:** buses start in arrival order and wait while the connection is full. This is what a simple cap does today.

For each strategy we compute the smallest connection on which every bus is still charged: 6.90, 5.40 and 3.31 MW. Savings are always measured on the same connection as the baseline, so a simple cap gets credit for whatever it already achieves.

## The objective and the carbon-first headline

Each night's cost per block is `w · co2 / mean(co2) + (1 − w) · price / mean(price)`. At w = 1 the schedule is purely carbon-driven, and smart charging never emits more CO₂ than unmanaged charging (one night sits 0.05 % above it, inside the solver tolerance). That makes w = 1 the clearest headline. With w < 1, each night's mean price sets how much the price term weighs, which varies from night to night. The cost-balanced setting w = 0.7 saves more money (€194,000 a year), so we report it as well.

## Solver precision

With unequal bus powers the depot problem is a packing problem, and a fully exact solve can take minutes per night. The MILP therefore stops at a proven gap of 0.2 % for the cost and CO₂ objective and 0.5 % for the minimum connection. Each night's gap is recorded in `results/e7_depot_year.csv`. The full year runs in 168 s (`results/e7_runtime.txt`).

## QUBO penalty weight

In the penalty encoding, λ is 1.5 times the largest objective coefficient, large enough that breaking a constraint never pays. A brute-force check over all 2^16 bit strings confirms that the QUBO minimum is the feasible optimum. e3 also tests λ at 3 and 6 times.

## Training QAOA

The angles are trained on a numpy copy of the circuit (`ExactSimulator`), which is checked against Qiskit's `Statevector`. Training needs thousands of evaluations, and this keeps it fast. The final samples come from Qiskit's `StatevectorSampler`. We train on CVaR with α = 0.25 (Barkoutsos et al., Quantum 4, 256, 2020) from a linear-ramp start, with the same settings for both mixers (`TRAIN_CONFIG` in `carbon_ev/qaoa.py`). CVaR focuses training on the best outcomes, and it is what moved the XY mixer well beyond its starting state.

## Metrics

With 4,096 shots, the best sample of almost any method hits the optimum, so we headline two quantities instead:
- the probability of sampling a schedule within 1 % of the optimum;
- the average quality of the valid samples.

Both are reported next to each mixer's untrained starting state.

## Windy and calm nights

A night counts as windy if its carbon intensity swings by at least 1.3× between its highest and lowest two-hour block. We fixed this rule before looking at the results.

## Noise model and the hardware check

The noise model uses depolarizing errors at IonQ Forte's published rates: 2·10⁻⁴ per one-qubit gate, 4·10⁻³ per two-qubit gate and 5·10⁻³ readout (`carbon_ev/noise.py`). Noisy seeds go through `numpy.random.SeedSequence`, so different seeds give independent noise.

We then ran one trained circuit on IonQ's Forte 1 quantum computer through Qollab: the XY mixer, p = 2, 16 qubits, 1,000 shots. Only the sampling ran on hardware, because training would need thousands of hardware evaluations. The hardware gave 0.316 valid samples (±0.015) against 0.332 from the model (`results/e9_forte1_hardware_check.csv`).

## QAOA inside the depot

The large-neighbourhood search frees two buses at a time. With two buses, the depot's connection limit becomes exact pairwise conflicts, so the subproblem is a clean QUBO of at most 16 qubits. A new assignment is accepted only if it is feasible and lowers the depot objective. Exact enumeration solves the same subproblems as the reference subsolver.

## Running on Qollab

Qollab runs Python in the browser (Pyodide). For that, `main.py`:
- names its data file as a literal path;
- imports the package by name (`carbon_ev/` sits at the project root);
- writes its outputs to `qollab_output/`.

Requirements list only Qiskit and `tzdata`; NumPy, SciPy, pandas and Matplotlib come with Pyodide. When a backend is chosen in Qollab's Select QPU dialog, `main.py` samples the trained circuit on it once.
