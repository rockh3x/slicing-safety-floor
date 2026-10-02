"""
plateau_sweep.py - the control the abdication sweep was missing, plus the
                   diagnostic that explains it
==========================================================================
WHY THIS RUN EXISTS

  1) THE MISSING CONTROL.  Every version of the claim "the floor causes the
     failures it was installed to prevent" is comparative, and there is no
     run without a floor. This sweep adds headroom 0 (safety layer OFF) and
     the weak-floor band 1.05-1.15. If unshielded PPO over-provisions and
     stays safe while h=1.37 collapses, the comparison IS the paper. If
     unshielded PPO also collapses, the finding is "PPO is seed-unstable in
     slicing" - still true, much smaller, and better known before drafting.

  2) THE MECHANISM IS AN ACTION-INVARIANT PLATEAU, NOT A SHIELD MISMATCH.
     Audit of gym_env._safety_filter: whenever BOTH latency-critical slices
     are clipped, the filter reserves their floors and hands the entire
     remaining capacity to the single discretionary slice - so

         granted = { urllc: d_u*h,  volte: d_v*h,  video: C - d_u*h - d_v*h }

     with the action appearing nowhere. Verified numerically: actions
     [-1,-1,1] and [-0.9,-0.5,1] give bit-identical allocations. The reward
     is a function of the granted allocation only, so on that region the
     reward is EXACTLY constant and the policy gradient is EXACTLY zero.

     That region grows with headroom. The steps it swallows are the
     high-demand steps - precisely the ones whose gradient would say
     "allocate more". The steps that survive are low-demand steps, where
     R7's over-allocation penalty says "allocate less". So the shield does
     not merely fail to teach; it CENSORS the upward gradient and leaves the
     downward one. Drift is therefore directional, not diffusive, and its
     probability rises with headroom - the 0/8 -> 6/8 curve seen in the
     preliminary (synthetic-trace) abdication sweep.

WHAT IS LOGGED THAT WAS NOT BEFORE

     alloc_ratio        mean granted_urllc / demand_urllc.  Directly
                        comparable to h* = 1.3765; works with the shield off,
                        which `ratio` (request vs floor) does not.
     frac_below_hstar   fraction of steps with granted_urllc/demand_urllc
                        < h*. Equals the URLLC violation fraction exactly (an
                        identity of the latency model, not a correlation).
                        With the floor on it accounts for every critical
                        violation; with it off, VoLTE can also violate and
                        `crit` exceeds frac_below_hstar * steps.
     frac_plateau       fraction of steps in the action-invariant region
                        (both critical slices clipped) = the fraction of the
                        rollout carrying zero gradient. The mechanism's own
                        dose variable.
     trajectory CSV     mean_req and frac_plateau at every checkpoint, so
                        abdication can be shown as drift over training rather
                        than inferred from an endpoint.

USAGE
     python sweeps/plateau_sweep.py                          # full sweep, 8 seeds
     python sweeps/plateau_sweep.py --headrooms 0 1.35       # control vs collapse
     python sweeps/plateau_sweep.py --seeds 4 --steps 150000 # quick look
     python sweeps/plateau_sweep.py --floor-mode reserve ... # reserve-then-allocate
     The defaults are the original exploratory settings (lr 3e-4, no KL cap,
     250k steps). The paper's configuration is given in README.md.

OUTPUT
     plateau_sweep.csv       one row per run
     plateau_traj.csv        one row per checkpoint per run
"""

import argparse, csv, os, sys, time

# Make the repository root importable when run as `python sweeps/plateau_sweep.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from slicing.gym_env import SlicingGymEnv
from slicing.clara_slices import clara_slices
from slicing.metrics import summarise


# ---------------------------------------------------------------------------
def self_sufficiency_threshold():
    """Headroom at which a binding floor alone meets the URLLC SLA.

    Derived, never measured: once the floor binds, alloc = demand * h, so the
    queue load is exactly 1/h and demand cancels. Solving the calibrated
    latency model L(h) = sla_target gives h*.
    """
    u = [s for s in clara_slices() if s.name == "urllc"][0]
    load_star = u.KNEE + ((u.sla_target - u.L_FLOOR) / u.A) ** (1.0 / u.K)
    return 1.0 / load_star, u


def make_env(headroom, reward, seed, floor_mode="clip"):
    """headroom == 0 means the safety layer is OFF - the control condition.

    floor_mode selects how the floor is enforced when it is on: "clip" is the
    projection the paper diagnoses, "reserve" the reserve-then-allocate
    parameterisation it proposes (see gym_env)."""
    return SlicingGymEnv(reward=reward, safe=(headroom > 0),
                         headroom=(headroom if headroom > 0 else 1.0),
                         episode_steps=144, seed=seed, floor_mode=floor_mode)


def open_csv(path, header):
    """Open `path` for appending, writing `header` if the file is new.

    Refuses to append to an existing file whose header differs: rows written
    under a stale header end up silently misaligned, which is how
    fix_sweep_real.csv came to have shifted columns (README, known issue 3).
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != header:
            raise SystemExit(f"{path} was written with a different header; "
                             "choose a new file with --out / --traj.")
        fh = open(path, "a", newline="")
        return fh, csv.writer(fh)
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    w.writerow(header)
    return fh, w


def policy_request(env, action):
    """The urllc units the policy itself asked for, before any filtering.

    Mirrors gym_env.step's softmax exactly. Kept separate from the granted
    allocation on purpose: `mean_req` measures intent, `alloc_ratio` measures
    what the network actually ran.
    """
    x = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    e = np.exp(2.0 * (x - x.max()))
    share = e / e.sum()
    return float(env.capacity * share[env.names.index("urllc")])


# ---------------------------------------------------------------------------
def evaluate(model, headroom, reward, h_star, eval_seed=100, episodes=3,
             floor_mode="clip"):
    """Roll out deterministically and collect the plateau diagnostics."""
    ev = make_env(headroom, reward, eval_seed, floor_mode)
    logs, req, floor, ratio, plateau = [], [], [], [], []
    # THROUGHPUT. The capacity pool is hard-capped, so "units granted" is a
    # near-constant under overload and says nothing. The quantity that varies,
    # and the one a reader means by throughput here, is CARRIED LOAD: how much
    # of the demand actually offered to the network was served. Tracked in
    # aggregate and per slice, because the whole question about a safety floor
    # is which slice pays for it.
    carried_num, carried_den = 0.0, 0.0
    served = {}

    for ep in range(episodes):
        obs, _ = ev.reset(seed=eval_seed + ep)
        crit = [s for s in ev.slices if s.name == "urllc"][0]
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            req.append(policy_request(ev, a))
            floor.append(min(crit.last_demand * max(headroom, 1.0),
                             ev.capacity * 0.5))
            obs, r, te, tr, info = ev.step(a)

            for n, ps in info["per_slice"].items():
                d, a_ = ps["demand"], ps["alloc"]
                carried_num += min(a_, d)
                carried_den += d
                served.setdefault(n, []).append(min(a_, d) / max(1e-9, d))

            p = info["per_slice"]["urllc"]
            ratio.append(p["alloc"] / max(1e-9, p["demand"]))
            # clipped == 2  <=>  both latency-critical slices floored
            #                <=>  granted allocation independent of the action
            plateau.append(1.0 if info.get("clipped", 0) >= 2 else 0.0)

            logs.append(info)
            done = te or tr

    k = summarise(logs)
    pv = k["per_slice_violations"]
    return {
        "carried":          float(carried_num / max(1e-9, carried_den)),
        "served_urllc":     float(np.mean(served.get("urllc", [0.0]))),
        "served_volte":     float(np.mean(served.get("volte", [0.0]))),
        "served_video":     float(np.mean(served.get("video", [0.0]))),
        "utilisation":      float(k["mean_utilisation"]),
        "alloc_ratio":      float(np.mean(ratio)),
        "frac_below_hstar": float(np.mean([x < h_star for x in ratio])),
        "frac_plateau":     float(np.mean(plateau)),
        "ratio":            float(np.mean(req) / max(1e-9, np.mean(floor))),
        "mean_req":         float(np.mean(req)),
        "mean_floor":       float(np.mean(floor)),
        "crit":             pv["urllc"] + pv["volte"],
        "urllc":            pv["urllc"],
        "viol_rate":        k["sla_violation_rate"],
    }


class Trajectory(BaseCallback):
    """Snapshot the drift every `every` steps - one short deterministic episode.

    This is what turns "the policy abdicated" into a learning curve. Costs one
    144-step rollout per checkpoint, which is noise next to training.
    """

    def __init__(self, headroom, reward, seed, h_star, writer, fh, every,
                 floor_mode="clip"):
        super().__init__()
        self.hr, self.reward, self.seed = headroom, reward, seed
        self.h_star, self.w, self.fh, self.every = h_star, writer, fh, every
        self.floor_mode = floor_mode
        self._next = every
        self.history = []            # (timestep, crit) - the window metric's input
        self.rows = []               # buffered rows; a parallel worker returns
                                     # these instead of holding a file handle

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        m = evaluate(self.model, self.hr, self.reward, self.h_star, episodes=1,
                     floor_mode=self.floor_mode)
        self.history.append((self.num_timesteps, m["crit"]))
        row = [self.hr, self.seed, self.num_timesteps,
               round(m["mean_req"], 3), round(m["alloc_ratio"], 4),
               round(m["frac_plateau"], 4),
               round(m["frac_below_hstar"], 4), m["crit"]]
        self.rows.append(row)
        if self.w is not None:
            self.w.writerow(row)
            self.fh.flush()
        return True


def window_metrics(history, window):
    """Time-averaged safety over the last `window` checkpoints.

    WHY THIS REPLACED THE ENDPOINT NUMBER
      The 1M-step run showed neither the shielded nor the unshielded policy
      converges: both oscillate between safe operation and total collapse,
      with excursions lasting 100-300k steps. A single evaluation at the end
      of training therefore samples the phase of an oscillation, not the
      quality of a policy - which is why earlier sweeps looked bimodal across
      seeds. Two seeds "behaving differently" were the same process caught at
      different moments.

      mean_crit_win  average violations per checkpoint     - the honest headline
      max_crit_win   worst checkpoint in the window        - the safety case
      frac_unsafe    share of checkpoints with any violation
    """
    tail = [c for _, c in history[-window:]] or [0]
    return {
        "mean_crit_win":  float(np.mean(tail)),
        "max_crit_win":   int(max(tail)),
        "frac_unsafe_win": float(np.mean([c > 0 for c in tail])),
        "n_win":          len(tail),
    }


def run_point(args_tuple):
    """One (headroom, seed) run. Takes a single tuple and returns a plain dict
    so it can be dispatched to a worker process by multiprocessing.Pool."""
    (headroom, reward, steps, seed, h_star, every, window, hp,
     floor_mode) = args_tuple
    try:                              # stop workers oversubscribing the cores
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    env = make_env(headroom, reward, seed, floor_mode)
    m = PPO("MlpPolicy", env, verbose=0, seed=seed,
            n_steps=hp["n_steps"], batch_size=hp["batch_size"],
            learning_rate=hp["lr"], ent_coef=hp["ent_coef"],
            target_kl=hp["target_kl"], gamma=0.95, device="cpu")
    cb = Trajectory(headroom, reward, seed, h_star, None, None, every,
                    floor_mode=floor_mode)
    t0 = time.time()
    m.learn(total_timesteps=steps, callback=cb)
    secs = time.time() - t0

    out = evaluate(m, headroom, reward, h_star, floor_mode=floor_mode)
    out.update(window_metrics(cb.history, window))
    out["abdicated"] = int(out["ratio"] < 1.0)   # legacy label, kept for continuity
    out["train_s"] = secs
    out["headroom"] = headroom
    out["seed"] = seed
    out["rows"] = cb.rows
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headrooms", type=float, nargs="+",
                    default=[0.0, 1.05, 1.10, 1.15, 1.20, 1.25,
                             1.30, 1.33, 1.35, 1.37])
    ap.add_argument("--reward", default="asymmetric_over0.5")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--ckpt-every", type=int, default=25_000)
    ap.add_argument("--window", type=int, default=8,
                    help="checkpoints averaged for the time-averaged metric")
    # PPO knobs, exposed so the oscillation can be attacked directly. The
    # question they answer: does the shield's effect survive a STABILISED
    # baseline, or was it an artefact of PPO collapsing and recovering?
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--target-kl", type=float, default=None)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel training runs; 1 keeps it sequential")
    ap.add_argument("--floor-mode", choices=["clip", "reserve"], default="clip",
                    help="how the floor is enforced when headroom > 0")
    ap.add_argument("--out", default="plateau_sweep.csv")
    ap.add_argument("--traj", default="plateau_traj.csv")
    a = ap.parse_args()
    hp = {"lr": a.lr, "ent_coef": a.ent_coef, "target_kl": a.target_kl,
          "n_steps": a.n_steps, "batch_size": a.batch_size}

    h_star, u = self_sufficiency_threshold()
    n = len(a.headrooms) * a.seeds
    print(f"h* = {h_star:.4f}  (derived: {u.L_FLOOR}+{u.A}*max(0,1/h-{u.KNEE})"
          f"^{u.K} = {u.sla_target} ms)")
    print("headroom 0 = safety layer OFF (the control)")
    print(f"floor mode: {a.floor_mode}")
    print(f"{n} runs: {len(a.headrooms)} headrooms x {a.seeds} seeds "
          f"@ {a.steps:,} steps, {a.jobs} in parallel\n")

    fs, ws = open_csv(a.out, [
        "headroom", "seed", "steps",
        "carried", "served_urllc", "served_volte", "served_video", "utilisation",
        "mean_crit_win", "max_crit_win", "frac_unsafe_win", "n_win",
        "alloc_ratio", "frac_below_hstar", "frac_plateau", "ratio",
        "abdicated", "crit", "urllc", "viol_rate", "mean_req",
        "mean_floor", "h_star", "lr", "ent_coef", "target_kl",
        "train_s", "floor_mode"])
    ft, wt = open_csv(a.traj, [
        "headroom", "seed", "timestep", "mean_req", "alloc_ratio",
        "frac_plateau", "frac_below_hstar", "crit"])

    jobs = [(hr, a.reward, a.steps, seed, h_star, a.ckpt_every, a.window, hp,
             a.floor_mode)
            for hr in a.headrooms for seed in range(a.seeds)]

    def emit(r):
        hr, seed = r["headroom"], r["seed"]
        for row in r["rows"]:
            wt.writerow(row)
        ws.writerow([hr, seed, a.steps,
                     round(r["carried"], 4), round(r["served_urllc"], 4),
                     round(r["served_volte"], 4), round(r["served_video"], 4),
                     round(r["utilisation"], 4),
                     round(r["mean_crit_win"], 2), r["max_crit_win"],
                         round(r["frac_unsafe_win"], 4), r["n_win"],
                         round(r["alloc_ratio"], 4),
                         round(r["frac_below_hstar"], 4),
                         round(r["frac_plateau"], 4), round(r["ratio"], 4),
                         r["abdicated"], r["crit"], r["urllc"],
                         round(r["viol_rate"], 4), round(r["mean_req"], 3),
                         round(r["mean_floor"], 3), round(h_star, 4),
                     a.lr, a.ent_coef, a.target_kl,
                     round(r["train_s"], 1), a.floor_mode])
        fs.flush(); ft.flush()
        label = "OFF  " if hr == 0 else f"{hr:<5}"
        print(f"hr={label} s{seed}  "
              f"win_mean={r['mean_crit_win']:6.1f}  "
              f"win_max={r['max_crit_win']:4d}  "
              f"unsafe={r['frac_unsafe_win']:5.1%}  "
              f"| endpoint crit={r['crit']:4d} (snapshot, do not report alone)")

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
