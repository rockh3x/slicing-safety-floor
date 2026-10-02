# Self-sufficiency conditions for demand-proportional safety floors

Code and results for:

> V. Patel and P. K. Mishra, "A Self-Sufficiency Condition for
> Demand-Proportional Safety Floors in DRL-Based 6G Network Slicing."
> Manuscript prepared for the *Journal of Network and Systems Management*.

A safety filter that clips a DRL agent's allocation so latency-critical slices
never fall below `demand × h` is usually assumed to be at worst conservative.
This repository contains the simulator, sweeps and analysis showing that below a
derivable threshold `h* = 1.3765` it is actively harmful: it censors the policy
gradient on exactly the states it exists to protect, and produces **10.4×** more
critical SLA violations during training than using no floor at all. It also
contains the remedy the paper proposes, *reserve-then-allocate*
(`SlicingGymEnv(floor_mode="reserve")`): the same floor reserved first, with
the action dividing only the remaining capacity, which removes the invariant
region by construction.

## Two things to read before the code

**The latency model is assumed, not fitted.** `L(ρ) = L₀ + A·max(0, ρ − ρ_knee)^κ`
with `L₀ = 0.5 ms, A = 292.8, ρ_knee = 0.70, κ = 1.15` is a convex proxy chosen
for its shape. None of the four constants is fitted to hardware, and no result
in the paper is a hardware measurement. What the results require of a real
bearer is the scope condition `ρ_knee < 1/h` (the paper's knee bound), not these
constants; the numerical value of `h*` does not transfer to a differently-shaped
delay curve.

`measurement/` holds a separate harness that measures round-trip delay against
offered load on a traffic-shaped virtual Ethernet link, with the data from
three sweeps. That link queues only near saturation, so it does not meet the scope
condition. It is a check on the knee bound, not a calibration of the model. See
`measurement/README.md`.

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
measurement/    the delay-measurement harness and its three sweeps
data/           where the Milan dataset goes (not redistributed; see data/README.md)
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then follow `data/README.md` to obtain the traffic file. Nothing runs without it,
by design.

## Reproducing the paper

Run every command from the repository root; the scripts put the root on
`sys.path` themselves. A 10⁶-step run takes about 400 s on one CPU core, so an
8-seed condition is roughly one core-hour; `--jobs N` runs N seeds in parallel.

The paper's training configuration (learning rate 1e-4, KL cap 0.02, 10⁶ steps,
a checkpoint every 50k steps) is the default in `fix_sweep.py` and
`delay_sweep.py`. `plateau_sweep.py` keeps its original exploratory defaults, so
pass the flags shown.

| Paper item | Command |
|---|---|
| Table 5, Fig. 2: no floor and floor at h = 1.35 (Sec. 6.1–6.2) | `python sweeps/plateau_sweep.py --headrooms 0 1.35 --steps 1000000 --ckpt-every 50000 --lr 1e-4 --target-kl 0.02 --jobs 8` |
| Table 5: floor + override cost (Sec. 6.3) | `python sweeps/fix_sweep.py --overrides 0 2.0 --jobs 8` |
| Table 6, Fig. 3: entropy, floor only (Sec. 6.4) | `python sweeps/fix_sweep.py --overrides 0 --ent-coefs 0 0.001 0.003 0.01 0.03 --jobs 8` |
| Table 6, Fig. 3: entropy with override cost (Sec. 6.4) | `python sweeps/fix_sweep.py --overrides 2.0 --ent-coefs 0 0.001 0.003 0.01 0.03 --jobs 8` |
| Table 7, Fig. 4: deferred floor (Sec. 6.5) | `python sweeps/delay_sweep.py --jobs 8` |
| Table 8, Fig. 5: reservation (Sec. 6.6) | `python sweeps/plateau_sweep.py --floor-mode reserve --headrooms 1.0 1.2 1.35 1.38 --steps 1000000 --ckpt-every 50000 --lr 1e-4 --target-kl 0.02 --jobs 8 --out reserve_sweep.csv --traj reserve_traj.csv` |
| Table 8, Fig. 5: projection controls (Sec. 6.6) | `python sweeps/plateau_sweep.py --floor-mode clip --headrooms 1.2 1.38 --steps 1000000 --ckpt-every 50000 --lr 1e-4 --target-kl 0.02 --jobs 8 --out clip_sweep.csv --traj clip_traj.csv` |
| Table 9: fixed-rule baselines (Sec. 6.7) | `python sweeps/baseline_eval.py` |
| Learning-rate comparison (Sec. 6.2) | `python sweeps/plateau_sweep.py --headrooms 1.35 --steps 1000000 --ckpt-every 50000 --lr 3e-4 --jobs 8` |

Sweeps append to `*_sweep.csv` / `*_traj.csv` in the working directory. Each
training checkpoint is evaluated on one held-out 144-step episode (seed 100);
the final policy on three (seeds 100–102).

`baseline_eval.py` trains nothing and finishes in seconds. It is the cheapest
way to confirm the `h*` prediction independently of any agent: it should show
zero URLLC violations at `h = 1.5` and 432 of 432 at `h = 1.35`, bracketing
`h* = 1.3765`.

Figures regenerate from the committed CSVs without retraining, and are written
to the repository root:

```bash
python figures/chart_movie.py      # Fig. 2  fig_trap.pdf
python figures/chart_entropy.py    # Fig. 3  fig_entropy.pdf
python figures/chart_delay.py      # Fig. 4  fig_delay.pdf
python figures/chart_reserve.py    # Fig. 5  fig_reserve.pdf
python figures/reserve_analysis.py # Table 8 numbers and paired Wilcoxon tests
```

`plateau_sweep.py` refuses to append to an existing output file whose header
differs from the one it would write, so a changed script cannot silently
misalign old rows (known issue 3 below). Point `--out`/`--traj` at new files.

## What the violation count counts

`crit` and `total_viol` count SLA violations of **both** latency-critical slices,
`urllc` (5 ms) and `volte` (20 ms). `frac_below_hstar` is the fraction of steps
on which the **urllc** allocation ratio is below `h*`, which equals the URLLC
violation fraction exactly. With a floor of `h ≥ 1.35` the two agree, because
such a floor lies above VoLTE's own threshold (1.258) and VoLTE cannot violate.
Without a floor, or with one below 1.258, they do not: about half of the
unshielded runs' violations (1058 of 2148), and most of those under projection
at `h = 1.2` (2078 of 2816), are VoLTE violations.

## Provenance of the result files

This matters, so it is stated explicitly rather than left implicit.

An earlier round of this project ran on a synthetic diurnal stand-in without
anyone realising, because the downloaded dataset was named `milan.txt.txt` and
the loader looked only for `milan.txt`. Every number in the paper was re-run on
the real trace after that was found and fixed.

The raw sweep outputs on disk therefore contained both runs appended in
sequence. The files in `results/` are the **real-trace rows only**, extracted and
tagged with a `traffic` column so the ambiguity cannot recur:

- `*_sweep_real.csv`: per-run summaries
- `*_traj_real.csv`: per-checkpoint trajectories, `traffic=milan-real`
- `fix_traj_real.csv` additionally carries a `condition` column
  (`floor_only` / `floor_plus_override`)

The other files in `results/` were produced after the fix and hold real-trace
rows only:

- `ent_sweep.csv`, `ent_traj.csv`: entropy sweep, floor only
- `ent_fix.csv`, `ent_fix_traj.csv`: entropy sweep with the override cost
- `thr_plateau.csv`, `thr_plateau_traj.csv`, `thr_fix.csv`: re-runs of the
  principal conditions that also log carried load (`carried`, `served_*`), the
  source of the carried-load column in Table 5. The older `*_sweep_real.csv`
  summaries predate those columns.
- `lr3e4_1m.csv`, `lr3e4_1m_traj.csv`: the learning-rate comparison
- `baseline_eval.csv`: the fixed-rule allocators
- `reserve_sweep.csv`, `reserve_traj.csv`: reserve-then-allocate at
  `h = 1.0, 1.2, 1.35, 1.38` (column `floor_mode = reserve`)
- `clip_sweep.csv`, `clip_traj.csv`: projection controls at `h = 1.2, 1.38`

The delay trajectory block assignment was verified against `delay_sweep_real.csv`
by matching `alloc_ratio`, `frac_plateau` and `frac_below_hstar` at the final
checkpoint; the retained block matches an order of magnitude more closely than
the discarded one.

## Known issues fixed here

Recorded because they affected published-looking numbers at some point:

1. **Synthetic-data fallback.** `MilanTraffic` now raises rather than silently
   substituting a stand-in.
2. **Seed mispairing in Fig. 4(b).** `fix_traj` rows arrive in worker-completion
   order under `--jobs`, not seed order. Chunking by position and zipping against
   seed-sorted summary rows paired each run's occupancy with another run's
   violation count. The reported Spearman correlation was 0.95 as a result; it is
   0.93 correctly paired. Runs are now keyed by the seed they contain.
3. **Shifted columns in `fix_sweep_real.csv`.** Its header was written by an
   older version of `fix_sweep.py` that did not log `logstd_first` and
   `logstd_last`, so every column from `h_star` onward was read two places
   off, and the last two values were fused into one field. The header
   has been rebuilt and the field split; no value changed. Columns up to
   `viol_rate`, the only ones the figures read, were never affected.
4. **Comments describing measurements that were never made.** Earlier versions
   of `slices.py`, `environment.py` and `gym_env.py` described the latency
   constants as fitted to an HTB testbed and quoted latencies measured on it.
   That testbed was coded but never run. The comments have been removed; the
   model is assumed, as stated above.
5. **Sweeps not runnable as documented.** `python sweeps/<script>.py` from the
   repository root could not import `slicing`, because Python puts the script's
   own directory, not the working directory, on the path. Each sweep now adds
   the repository root itself.

## Licence

Code: MIT (see `LICENSE`). The Telecom Italia dataset is under its own terms and
is not included here.
