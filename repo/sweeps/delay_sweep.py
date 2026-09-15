"""
delay_sweep.py - does the pathology depend on WHEN the floor is switched on?
===========================================================================
THE QUESTION

  Every result so far applies the safety floor from step 0. The floor at
  h = 1.35 is below h* = 1.3765, so it cannot satisfy the SLA it enforces, and
  the policy sits in the action-invariant region for ~500k steps at 144/144
  violations.

  But the invariant region is only reachable if the floor BINDS. It binds when
  the policy requests less than d*h for the critical slices. At step 0 the
  policy is near-uniform and requests far too little, so the floor binds
  immediately and the gradient is censored from the outset.

  The naive version of the hypothesis is that a policy already above the floor
  never triggers the clip, so a late switch costs nothing. A 20k-step control
  run shows that version is WRONG, and it is worth writing down why, because it
  is the reason this experiment is interesting rather than decorative:

      switch_at = 0, seed 0     step  5,000   alloc 2.22   plateau 0.01
                                step 10,000   alloc 1.36   plateau 0.77
                                step 15,000   alloc 1.35   plateau 0.99

  The policy is FAR above the floor when the floor comes on - a near-uniform
  softmax hands the critical slice roughly 5x its demand - and it still ends up
  in the region. It gets there by descending: the reward penalises wasted
  capacity, so the policy trims over-provisioning, walks down onto the floor,
  and only then loses the gradient that would let it walk back up. Being clear
  of the floor at the switch is therefore not protective on its own.

  THE REAL QUESTION, then, is not whether a late floor binds immediately, but
  whether the policy still descends into the region afterwards - and if so,
  whether it does that from a converged policy as readily as from an untrained
  one. Three outcomes are distinguishable and all three are informative:

      AVOIDED   steps_to_trap_post = -1. The converged policy sits above the
                floor and stays there. Delaying the floor is a genuine remedy
                and the paper gains a second design rule.
      DELAYED   steps_to_trap_post large but finite. Delaying buys time and
                nothing more. Worth reporting as a negative result - it kills
                an obvious reviewer suggestion before they make it.
      NO EFFECT steps_to_trap_post ~ the same for every switch time. The
                pathology is a property of the floor's value alone.

  PREDICTION (registered before running): DELAYED. The efficiency pressure that
  drives the descent does not weaken with training, so the policy should still
  find the floor - but from 1.6-2.2 rather than from 5, so the descent should
  be shorter and the total damage smaller than the always-on condition.

WHY THIS IS WORTH A SUBSECTION

  If confirmed, it separates two things the paper currently conflates: an
  UNSOUND floor (h < h*) and a PREMATURELY IMPOSED one. The design implication
  changes - "derive h from the SLA" gains a companion, "or do not impose the
  floor until the policy is above it" - and it connects the result to
  curriculum and shielded-fine-tuning practice.

  It is also the honest test of the reverse claim, which is the one that would
  hurt: if a late floor still traps the policy, then delaying is not a remedy
  and the paper should say so before a reviewer says it for us.

BASELINES ARE NOT RE-RUN

  switch_at = 0        is already measured: fix_sweep.py --overrides 0
                       -> 1,886 total violations, 506,250 stagnation steps
  switch_at = never    is already measured: plateau_sweep.py --headrooms 0
                       -> 135 total violations, 0 stagnation steps
  This script runs only the intermediate switch times and reports against them.

REPORTED PER RUN
     alloc_at_switch     allocation ratio at the last checkpoint before the
                         switch - how far above the floor the policy was when
                         the floor arrived
     steps_to_trap_post  steps AFTER the switch before the first checkpoint at
                         total failure; -1 if it never happens. This is the
                         discriminating measurement.
     steps_trapped_post  stagnation steps counted only after the switch - the
                         quantity the always-on condition scores 506,250 on
     viol_pre/post       violations summed before / after the switch
     rebound             1 if the policy was clear of the floor at the switch
                         and fell in anyway (expected, per the note above)

USAGE
     python delay_sweep.py                          # 3 switch times x 8 seeds
     python delay_sweep.py --seeds 4                # first look, half the time
     python delay_sweep.py --jobs 8                 # parallel on the Ultra 9
     python delay_sweep.py --switch-at 100000 400000 --seeds 4

OUTPUT
     delay_sweep.csv, delay_traj.csv
"""

import argparse, csv, os, time
import numpy as np
from stable_baselines3 import PPO

try:
    import gymnasium as gym
except ImportError as exc:      # legacy `gym` is NOT a drop-in here: SB3 2.x
    raise ImportError(          # requires gymnasium, and a silent fallback to
        "gymnasium is required (pip install gymnasium). Legacy `gym` is not "
        "a substitute -- it is unmaintained, incompatible with NumPy 2, and "
        "falling back to it silently would change what the experiment "
        "measures.") from exc

from plateau_sweep import (self_sufficiency_threshold, make_env, evaluate,
                           window_metrics)
from stable_baselines3.common.callbacks import BaseCallback


class DelayedFloor(gym.Wrapper):
    """Train with the safety filter OFF, then switch it on at `switch_at`.

    SlicingGymEnv checks self.safe and self.headroom inside step(), so flipping
    the attributes mid-episode is enough - no re-construction, no reset, and
    the policy and optimiser state carry across untouched. That matters: the
    whole point is to change ONLY the floor, at one instant, and watch what the
    already-trained policy does with it.

    The counter is the wrapper's own step count. With a single environment that
    equals the PPO timestep count, which is what --switch-at is expressed in.
    """

    def __init__(self, env, switch_at, headroom):
        super().__init__(env)
        self.switch_at = int(switch_at)
        self.headroom = float(headroom)
        self.t = 0
        self.switched = False
        base = self.unwrapped
        base.safe = False                # start unshielded, whatever make_env did
        base.headroom = 1.0

    def step(self, action):
        self.t += 1
        if not self.switched and self.t >= self.switch_at:
            base = self.unwrapped
            base.safe = True
            base.headroom = self.headroom
            self.switched = True
        return self.env.step(action)


class SwitchingTrajectory(BaseCallback):
    """Checkpoint trace whose EVALUATION follows the floor state.

    Before the switch the deployed system has no floor, so the checkpoint is
    evaluated unshielded; after it, shielded. Evaluating post-switch
    checkpoints without the floor would measure a system that does not exist
    and would hide exactly the collapse this experiment is looking for.
    """

    def __init__(self, headroom, reward, seed, h_star, every, switch_at):
        super().__init__()
        self.hr_on, self.reward, self.seed = headroom, reward, seed
        self.h_star, self.every, self.switch_at = h_star, every, switch_at
        self._next = every
        self.history = []        # (timestep, crit)
        self.allocs = []         # (timestep, alloc_ratio)
        self.rows = []           # buffered trajectory rows; parent writes them

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        hr = self.hr_on if self.num_timesteps >= self.switch_at else 0.0
        m = evaluate(self.model, hr, self.reward, self.h_star, episodes=1)
        self.history.append((self.num_timesteps, m["crit"]))
        self.allocs.append((self.num_timesteps, m["alloc_ratio"]))
        self.rows.append([self.switch_at, self.seed, self.num_timesteps,
                          int(self.num_timesteps >= self.switch_at),
                          round(m["mean_req"], 3), round(m["alloc_ratio"], 4),
                          round(m["frac_plateau"], 4),
                          round(m["frac_below_hstar"], 4), m["crit"]])
        return True


def split_metrics(history, allocs, every, switch_at, headroom, full=140):
    """Trap metrics split at the switch, plus the predictor and the falsifier."""
    pre = [(t, c) for t, c in history if t < switch_at]
    post = [(t, c) for t, c in history if t >= switch_at]

    n100_post = sum(1 for _, c in post if c >= full)
    first_safe = -1
    for i, (t, _) in enumerate(history):
        if all(c == 0 for _, c in history[i:]):
            first_safe = t
            break

    # allocation ratio at the last checkpoint before the floor comes on
    pre_allocs = [a for t, a in allocs if t < switch_at]
    at_switch = pre_allocs[-1] if pre_allocs else float("nan")
    clear_at_switch = bool(np.isfinite(at_switch) and at_switch > headroom)

    # THE DISCRIMINATING QUANTITY. The policy descends towards the floor under
    # efficiency pressure, so being above it at the switch does not settle
    # anything - the question is whether it descends INTO the region afterwards.
    # -1 means it never did, which is the "avoided" outcome. A finite value is
    # the "merely delayed" outcome, and its size is the delay bought.
    to_trap = -1
    for t, c in post:
        if c >= full:
            to_trap = t - switch_at
            break

    return {
        "alloc_at_switch":    at_switch,
        "clear_at_switch":    int(clear_at_switch),
        "viol_pre":           int(sum(c for _, c in pre)),
        "viol_post":          int(sum(c for _, c in post)),
        "total_viol":         int(sum(c for _, c in history)),
        "ckpts_at_100_post":  n100_post,
        "steps_trapped_post": n100_post * every,
        "steps_to_trap_post": to_trap,
        "steps_to_safe":      first_safe,
        # falsifier for the naive version of the hypothesis: the policy was
        # clear of the floor when it came on, and fell in anyway.
        "rebound":            int(clear_at_switch and n100_post > 0),
    }


def run_point(args_tuple):
    """One (switch_at, seed) run. Returns a plain dict so it is picklable."""
    (switch_at, seed, headroom, reward, steps, h_star, hp, every, window) = args_tuple

    try:                              # keep parallel workers from oversubscribing
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    train_env = DelayedFloor(make_env(0.0, reward, seed), switch_at, headroom)
    m = PPO("MlpPolicy", train_env, verbose=0, seed=seed,
            n_steps=hp["n_steps"], batch_size=hp["batch_size"],
            learning_rate=hp["lr"], ent_coef=0.0,
            target_kl=hp["target_kl"], gamma=0.95, device="cpu")
    cb = SwitchingTrajectory(headroom, reward, seed, h_star, every, switch_at)
    t0 = time.time()
    m.learn(total_timesteps=steps, callback=cb)
    secs = time.time() - t0

    out = evaluate(m, headroom, reward, h_star)      # final policy, floor on
    out.update(window_metrics(cb.history, window))
    out.update(split_metrics(cb.history, cb.allocs, every, switch_at, headroom))
    out["switch_at"] = switch_at
    out["seed"] = seed
    out["train_s"] = secs
    out["rows"] = cb.rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom", type=float, default=1.35,
                    help="floor multiplier switched on at --switch-at")
    ap.add_argument("--switch-at", type=int, nargs="+",
                    default=[150_000, 300_000, 500_000])
    ap.add_argument("--reward", default="asymmetric_over0.5")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--ckpt-every", type=int, default=50_000)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel training runs; 1 keeps it sequential")
    ap.add_argument("--out", default="delay_sweep.csv")
    ap.add_argument("--traj", default="delay_traj.csv")
    a = ap.parse_args()
    hp = {"lr": a.lr, "target_kl": a.target_kl,
          "n_steps": a.n_steps, "batch_size": a.batch_size}

    h_star, _ = self_sufficiency_threshold()
    for s in a.switch_at:
        if s % a.ckpt_every:
            print(f"WARNING: switch-at {s:,} is not a multiple of the checkpoint "
                  f"interval {a.ckpt_every:,}; alloc_at_switch will be measured "
                  f"up to {a.ckpt_every:,} steps early.")

    jobs = [(s, seed, a.headroom, a.reward, a.steps, h_star, hp,
             a.ckpt_every, a.window)
            for s in a.switch_at for seed in range(a.seeds)]
    n = len(jobs)

    print(f"h* = {h_star:.4f}   floor = {a.headroom} "
          f"({'BELOW h* - not self-sufficient' if a.headroom < h_star else 'at/above h*'})")
    print(f"{n} runs: {len(a.switch_at)} switch times x {a.seeds} seeds "
          f"@ {a.steps:,} steps, {a.jobs} in parallel")
    print("already measured, not re-run:")
    print("   floor from step 0     1,886 violations, 506,250 stagnation steps")
    print("   floor never on          135 violations,       0 stagnation steps")
    print("registered prediction: DELAYED - the policy still descends into the")
    print("region after a late switch, but from a lower starting ratio, so")
    print("steps_to_trap_post is finite and total_viol < 1,886.")
    print("steps_to_trap_post = -1 on most seeds would falsify it.\n")

    new_s, new_t = not os.path.exists(a.out), not os.path.exists(a.traj)
    fs, ft = open(a.out, "a", newline=""), open(a.traj, "a", newline="")
    ws, wt = csv.writer(fs), csv.writer(ft)
    if new_s:
        ws.writerow(["headroom", "switch_at", "seed", "steps",
                     "carried", "served_urllc", "served_volte", "served_video", "utilisation",
                     "alloc_at_switch", "clear_at_switch",
                     "viol_pre", "viol_post", "total_viol",
                     "ckpts_at_100_post", "steps_trapped_post",
                     "steps_to_trap_post", "steps_to_safe",
                     "rebound", "mean_crit_win", "max_crit_win",
                     "frac_unsafe_win", "alloc_ratio", "frac_below_hstar",
                     "frac_plateau", "crit", "mean_req", "viol_rate",
                     "h_star", "lr", "target_kl", "train_s"])
    if new_t:
        wt.writerow(["switch_at", "seed", "timestep", "floor_on", "mean_req",
                     "alloc_ratio", "frac_plateau", "frac_below_hstar", "crit"])

    def emit(r):
        for row in r["rows"]:
            wt.writerow(row)
        ws.writerow([a.headroom, r["switch_at"], r["seed"], a.steps,
                     round(r["carried"], 4), round(r["served_urllc"], 4),
                     round(r["served_volte"], 4), round(r["served_video"], 4),
                     round(r["utilisation"], 4),
                     round(r["alloc_at_switch"], 4), r["clear_at_switch"],
                     r["viol_pre"], r["viol_post"], r["total_viol"],
                     r["ckpts_at_100_post"], r["steps_trapped_post"],
                     r["steps_to_trap_post"], r["steps_to_safe"], r["rebound"],
                     round(r["mean_crit_win"], 2), r["max_crit_win"],
                     round(r["frac_unsafe_win"], 4), round(r["alloc_ratio"], 4),
                     round(r["frac_below_hstar"], 4), round(r["frac_plateau"], 4),
                     r["crit"], round(r["mean_req"], 3), round(r["viol_rate"], 4),
                     round(h_star, 4), a.lr, a.target_kl, round(r["train_s"], 1)])
        fs.flush()
        ft.flush()
        ttt = (f"{r['steps_to_trap_post']:,}" if r["steps_to_trap_post"] >= 0
               else "never")
        flag = "  <-- REBOUND (fell in from above)" if r["rebound"] else ""
        print(f"switch={r['switch_at']:>7,} s{r['seed']}  "
              f"alloc@switch={r['alloc_at_switch']:.2f}  "
              f"fell_in_after={ttt:>9}  "
              f"trapped_post={r['steps_trapped_post']:>8,}  "
              f"viol pre/post={r['viol_pre']:>4}/{r['viol_post']:<5} "
              f"total={r['total_viol']:>5}{flag}")

    t0 = time.time()
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(a.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(run_point, jobs), 1):
                print(f"[{i:3d}/{n}] ", end="")
                emit(r)
    else:
        for i, j in enumerate(jobs, 1):
            print(f"[{i:3d}/{n}] ", end="", flush=True)
            emit(run_point(j))

    fs.close()
    ft.close()
    print(f"\nwritten: {a.out} and {a.traj}   ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
