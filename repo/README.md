# Self-sufficiency conditions for demand-proportional safety floors

Code and results for:

> V. Patel and P. K. Mishra, "A Self-Sufficiency Condition for
> Demand-Proportional Safety Floors in DRL-Based 6G Network Slicing."
> Under review, *Journal of Network and Systems Management*.

A safety filter that clips a DRL agent's allocation so latency-critical slices
never fall below `demand × h` is usually assumed to be at worst conservative.
This repository contains the simulator, sweeps and analysis showing that below a
derivable threshold `h* = 1.3765` it is actively harmful: it censors the policy
gradient on exactly the states it exists to protect, and produces **10.5×** more
critical SLA violations during training than using no floor at all.

## Two things to read before the code

**The latency model is assumed, not measured.** `L(ρ) = L₀ + A·max(0, ρ − ρ_knee)^κ`
with `L₀ = 0.5 ms, A = 292.8, ρ_knee = 0.70, κ = 1.15` is a convex proxy. It is
not fitted to any testbed, and no measurements from real hardware exist for this
project. The phenomenon depends on convexity and on the presence of a knee; the
numerical value of `h*` does not transfer to a differently-shaped delay curve.

**The benchmark does not demonstrate a case for learning.** A static rule
granting `d × 1.5` achieves zero violations at a higher carried load (0.897)
than either learned policy (0.861 unshielded). The environment's optimum is
state-independent, so these results establish a property of the *filter*, not an
argument for the *agent*. `sweeps/baseline_eval.py` reproduces this and it is
discussed in the paper's limitations.

## Layout

```
slicing/        the simulator (environment, slices, traffic, allocators, rewards)
sweeps/         the four experiment drivers
figures/        scripts that regenerate the paper's figures from results/
results/        every CSV behind every number in the paper
data/           where the Milan dataset goes (not redistributed — see data/README.md)
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then follow `data/README.md` to obtain the traffic file. Nothing runs without it —
by design.

## Reproducing the paper

Every command is run from the repository root. Each sweep takes roughly 8 core-hours
per condition on commodity CPU; `--jobs` parallelises across seeds.

| Paper item | Command |
|---|---|
| Table 2, Fig. 2 | `python sweeps/plateau_sweep.py --jobs 8` |
| Table 3, Fig. 3 | `python sweeps/fix_sweep.py --jobs 8` |
| Table 4, Fig. 4 | `python sweeps/delay_sweep.py --jobs 8` |
| Table 5 | `python sweeps/baseline_eval.py` |

`baseline_eval.py` trains nothing and finishes in seconds — it is the cheapest
way to confirm the `h*` prediction independently of any agent. It should show
zero URLLC violations at `h = 1.5` and 432 of 432 at `h = 1.35`, bracketing
`h* = 1.3765`.

Figures regenerate from the committed CSVs without retraining:

```bash
python figures/chart_movie.py      # Fig. 2  fig_trap.pdf
python figures/chart_entropy.py    # Fig. 3  fig_entropy.pdf
python figures/chart_delay.py      # Fig. 4  fig_delay.pdf
```

## Provenance of the result files

This matters, so it is stated explicitly rather than left implicit.

An earlier round of this project ran on a synthetic diurnal stand-in without
anyone realising, because the downloaded dataset was named `milan.txt.txt` and
the loader looked only for `milan.txt`. Every number in the paper was re-run on
the real trace after that was found and fixed.

The raw sweep outputs on disk therefore contained both runs appended in
sequence. The files in `results/` are the **real-trace rows only**, extracted and
tagged with a `traffic` column so the ambiguity cannot recur:

- `*_sweep_real.csv` — per-run summaries
- `*_traj_real.csv` — per-checkpoint trajectories, `traffic=milan-real`
- `fix_traj_real.csv` additionally carries a `condition` column
  (`floor_only` / `floor_plus_override`)

The delay trajectory block assignment was verified against `delay_sweep_real.csv`
by matching `alloc_ratio`, `frac_plateau` and `frac_below_hstar` at the final
checkpoint; the retained block matches an order of magnitude more closely than
the discarded one.

## Known issues fixed here

Recorded because they affected published-looking numbers at some point:

1. **Synthetic-data fallback** — `MilanTraffic` now raises rather than silently
   substituting a stand-in.
2. **Seed mispairing in Fig. 4(b)** — `fix_traj` rows arrive in worker-completion
   order under `--jobs`, not seed order. Chunking by position and zipping against
   seed-sorted summary rows paired each run's occupancy with another run's
   violation count. The reported Spearman correlation was 0.95 as a result; it is
   0.93 correctly paired. Runs are now keyed by the seed they contain.

## Licence

Code: MIT (see `LICENSE`). The Telecom Italia dataset is under its own terms and
is not included here.
