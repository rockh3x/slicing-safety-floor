"""
fix_sweep.py - restoring the censored gradient
==============================================
WHAT THE STABILISED RUN ESTABLISHED (and why this is the last experiment)

  With lr=1e-4 and target_kl=0.02 the oscillation disappears: every run is a
  monotone learning curve. The shield effect survives, and in a much starker
  form.

  Real Milan trace, 8 seeds, 1M steps (mean total violations per run):

      condition        ckpts at 100% violation   total violations   worst
      no shield              0 / 160                    268          43%
      shield h=1.35        149 / 160                  2,802         100%

  Every shielded seed sits at exactly alloc/demand = 1.35, frac_plateau ~0.79,
  144/144 violations, for 700,000-1,000,000 steps. The unshielded agent never
  exceeds 62/144 violations at any checkpoint. The shield does not merely slow
  learning down - it holds the policy at total SLA failure for most of the
  budget, and no shielded seed reaches sustained safety within it.

WHY THE AGENT IS STUCK, AND WHAT ESCAPES IT

  Inside the plateau the granted allocation is independent of the action, so
  the reward is constant and the gradient is exactly zero. Deterministic
  evaluation shows the policy inside the region on ~79% of evaluated states.
  Escape is therefore NOT gradient-driven - it is driven by exploration noise
  occasionally sampling an action outside the plateau. That predicts escape
  time should scale inversely with learning rate, which is what happened when
  lr dropped 3e-4 -> 1e-4 and the mean trapped phase grew from 500k to 931k
  steps.

TWO REMEDIES, ONE OF WHICH IS THE LITERATURE'S

  A. OVERRIDE COST (--override).  Charge the policy for how far the filter had
     to move its action:  r <- r - c * override, where override is already
     logged by gym_env as the fraction of capacity the filter shifted. Inside
     the plateau this term DOES depend on the action - asking for less means a
     bigger correction - so it restores a gradient pointing out of the trap.
     This is the standard remedy, not our invention: Krasowski et al. (TMLR
     2023) report that "adding a reward penalty, every time the safety
     verification is engaged, improved training performance", and Odriozola-
     Olalde et al. (EAAI 2025) describe the same feedback reward for reducing
     shield intervention rate. We are testing whether it removes the trap in
     slicing, and at what cost in allocation efficiency.

  B. ENTROPY (--ent-coef).  If escape really is noise-driven, entropy alone
     should shorten the trapped phase without any reward change. This is the
     falsification test for the mechanism, and it is free.

REPORTED PER RUN
     ckpts_at_100     checkpoints with >=140/144 critical violations
     steps_trapped    those checkpoints x ckpt interval - the trapped phase
     steps_to_safe    first checkpoint after which violations stay at zero
     total_viol       violations summed over all checkpoints (training-time
                      safety, the metric that actually matters online)
     mean_alloc       final allocation ratio - the efficiency the fix costs

USAGE
     python fix_sweep.py                               # baseline vs 3 costs
     python fix_sweep.py --overrides 0 2.0 --seeds 4   # quick
     python fix_sweep.py --overrides 0 --ent-coefs 0 0.01   # mechanism test

OUTPUT
     fix_sweep.csv, fix_traj.csv
"""

import argparse, csv, os, sys, time

# Make the repository root importable when run as `python sweeps/fix_sweep.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                           window_metrics, Trajectory)


class OverrideCost(gym.Wrapper):
    """Charge the policy for the correction the safety filter had to apply.

    gym_env already computes info["override"] = the fraction of capacity the
    filter moved. Subtracting a multiple of it makes the reward depend on the
    requested action even when the granted allocation does not - which is
    precisely the gradient the shield was destroying.

    Applied ONLY during training. Evaluation is left untouched so that the
    safety numbers stay comparable to every earlier sweep.
    """

    def __init__(self, env, coef):
        super().__init__(env)
        self.coef = float(coef)

    def step(self, action):
        obs, r, te, tr, info = self.env.step(action)
        if self.coef:
            r = r - self.coef * info.get("override", 0.0)
        return obs, r, te, tr, info


class TrajectoryStd(Trajectory):
    """Trajectory, plus the policy's mean log-std at each checkpoint.

    WHY THIS EXISTS
      The entropy result needs a mechanism, not just an observation. SB3's
      Gaussian policy carries a STATE-INDEPENDENT log_std, so an entropy bonus
      has a gradient on the spread and none on the mean. The prediction is
      therefore specific: under entropy, log_std should climb steadily while
      mean_req stays frozen at the plateau value. If instead log_std barely
      moves, the explanation is wrong and the entropy result needs a different
      account before it goes in the paper.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.logstds = []

    def _on_step(self):
        before = len(self.history)
        ok = super()._on_step()
        if len(self.history) > before:
            try:
                ls = float(self.model.policy.log_std.mean().item())
            except Exception:
                ls = float("nan")
            self.logstds.append((self.num_timesteps, ls))
        return ok


def trap_metrics(history, every, full=140):
    """Trapped phase and time-to-safety, from the checkpoint trace."""
    crit = [c for _, c in history]
    n100 = sum(1 for c in crit if c >= full)
    first = None
    for i, (t, _) in enumerate(history):
        if all(c == 0 for _, c in history[i:]):
            first = t
            break
    return {
        "ckpts_at_100":  n100,
        "steps_trapped": n100 * every,
        "steps_to_safe": first if first is not None else -1,
        "total_viol":    int(sum(crit)),
    }


def run_point(args_tuple):
    """One (override, entropy, seed) run. Takes a single tuple and returns a
    plain dict so multiprocessing.Pool can dispatch it to a worker."""
    (headroom, reward, steps, seed, h_star, coef, ent, hp, every,
     window) = args_tuple
    try:                              # stop workers oversubscribing the cores
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    train_env = OverrideCost(make_env(headroom, reward, seed), coef)
    m = PPO("MlpPolicy", train_env, verbose=0, seed=seed,
            n_steps=hp["n_steps"], batch_size=hp["batch_size"],
            learning_rate=hp["lr"], ent_coef=ent,
            target_kl=hp["target_kl"], gamma=0.95, device="cpu")
    cb = TrajectoryStd(headroom, reward, seed, h_star, None, None, every)
    t0 = time.time()
    m.learn(total_timesteps=steps, callback=cb)
    secs = time.time() - t0

    out = evaluate(m, headroom, reward, h_star)     # unmodified reward
    out.update(window_metrics(cb.history, window))
    out.update(trap_metrics(cb.history, every))
    ls = [v for _, v in cb.logstds] or [float("nan")]
    out["logstd_first"], out["logstd_last"] = ls[0], ls[-1]
    out["train_s"] = secs
    out["override"], out["ent"], out["seed"] = coef, ent, seed
    out["rows"] = cb.rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom", type=float, default=1.35)
    ap.add_argument("--overrides", type=float, nargs="+",
                    default=[0.0, 0.5, 2.0, 5.0])
    ap.add_argument("--ent-coefs", type=float, nargs="+", default=[0.0])
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
    ap.add_argument("--out", default="fix_sweep.csv")
    ap.add_argument("--traj", default="fix_traj.csv")
    a = ap.parse_args()
    hp = {"lr": a.lr, "target_kl": a.target_kl,
          "n_steps": a.n_steps, "batch_size": a.batch_size}

    h_star, u = self_sufficiency_threshold()
    combos = [(c, e) for c in a.overrides for e in a.ent_coefs]
    n = len(combos) * a.seeds
    print(f"h* = {h_star:.4f}   headroom = {a.headroom} "
          f"({'BELOW h* - floor is not self-sufficient' if a.headroom < h_star else 'at/above h*'})")
    print(f"{n} runs: {len(combos)} (override, ent) combos x {a.seeds} seeds "
          f"@ {a.steps:,} steps, {a.jobs} in parallel")
    print("baseline for comparison - shield, no fix (paper config): "
          "trapped 931,250 steps on average, 2,802 total violations\n")

    new_s, new_t = not os.path.exists(a.out), not os.path.exists(a.traj)
    fs, ft = open(a.out, "a", newline=""), open(a.traj, "a", newline="")
    ws, wt = csv.writer(fs), csv.writer(ft)
    if new_s:
        ws.writerow(["headroom", "override", "ent_coef", "seed", "steps",
                     "carried", "served_urllc", "served_volte", "served_video", "utilisation",
                     "ckpts_at_100", "steps_trapped", "steps_to_safe",
                     "total_viol", "mean_crit_win", "max_crit_win",
                     "frac_unsafe_win", "alloc_ratio", "frac_below_hstar",
                     "frac_plateau", "crit", "mean_req", "viol_rate",
                     "logstd_first", "logstd_last",
                     "h_star", "lr", "target_kl", "train_s"])
    if new_t:
        wt.writerow(["headroom", "seed", "timestep", "mean_req", "alloc_ratio",
                     "frac_plateau", "frac_below_hstar", "crit"])

    jobs = [(a.headroom, a.reward, a.steps, seed, h_star, coef, ent, hp,
             a.ckpt_every, a.window)
            for coef, ent in combos for seed in range(a.seeds)]

    def emit(r):
        coef, ent, seed = r["override"], r["ent"], r["seed"]
        for row in r["rows"]:
            wt.writerow(row)
        ws.writerow([a.headroom, coef, ent, seed, a.steps,
                     round(r["carried"], 4), round(r["served_urllc"], 4),
                     round(r["served_volte"], 4), round(r["served_video"], 4),
                     round(r["utilisation"], 4),
                     r["ckpts_at_100"], r["steps_trapped"],
                         r["steps_to_safe"], r["total_viol"],
                         round(r["mean_crit_win"], 2), r["max_crit_win"],
                         round(r["frac_unsafe_win"], 4),
                         round(r["alloc_ratio"], 4),
                         round(r["frac_below_hstar"], 4),
                         round(r["frac_plateau"], 4), r["crit"],
                         round(r["mean_req"], 3), round(r["viol_rate"], 4),
                         round(r["logstd_first"], 4), round(r["logstd_last"], 4),
                     round(h_star, 4), a.lr, a.target_kl,
                     round(r["train_s"], 1)])
        fs.flush(); ft.flush()
        sts = f"{r['steps_to_safe']:,}" if r["steps_to_safe"] > 0 else "never"
        print(f"ovr={coef:<5} ent={ent:<6} s{seed}  "
              f"trapped={r['steps_trapped']:>8,}  safe_at={sts:>9}  "
              f"total_viol={r['total_viol']:>5}  alloc={r['alloc_ratio']:.2f}  "
              f"logstd {r['logstd_first']:+.2f}->{r['logstd_last']:+.2f}")

    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(a.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(run_point, jobs), 1):
                print(f"[{i:3d}/{n}] ", end=""); emit(r)
    else:
        for i, j in enumerate(jobs, 1):
            print(f"[{i:3d}/{n}] ", end="", flush=True); emit(run_point(j))
    fs.close()
    ft.close()
    print(f"\nwritten: {a.out} and {a.traj}")


if __name__ == "__main__":
    main()
