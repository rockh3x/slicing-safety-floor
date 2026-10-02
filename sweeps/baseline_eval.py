"""
baseline_eval.py - the fixed-rule allocators on the paper's evaluation protocol
==============================================================================
WHY THIS EXISTS

  The paper compares a shielded PPO agent against an unshielded PPO agent. A
  reviewer will ask the obvious next question: how does either compare to a
  static rule that needs no training at all? That question is sharper here than
  usual, because the paper's own design rule -- "grant the critical slices
  d * h with h > h*" -- IS a static rule. If it matches PPO on safety, the paper
  should say so itself rather than have it extracted in review.

  These allocators are deterministic, so nothing is trained. The protocol is
  identical to the one plateau_sweep.evaluate() uses for a policy: the same
  held-out evaluation seed, the same three 144-step episodes, the same metrics.

REPORTED
     crit        critical-slice SLA violations over 432 evaluated steps,
                 summed over urllc and volte (so at most 864)
     urllc       the urllc part of crit (at most 432)
     alloc/d     mean allocation ratio of the urllc slice
     tau         carried load, sum min(a,d) / sum d
     tau_<slice> per-slice served fraction

USAGE
     python sweeps/baseline_eval.py
     python sweeps/baseline_eval.py --headrooms 1.0 1.2 1.35 1.3765 1.5 2.0
"""

import argparse, csv, os, sys

# Make the repository root importable when run as `python sweeps/baseline_eval.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from slicing.clara_slices import clara_slices
from slicing.environment import SlicingEnv
from slicing.milan_traffic import MilanTraffic
from slicing.allocators import (EqualAllocator, DemandProportionalAllocator,
                                PrioritySafeAllocator, HeadroomAllocator)
from slicing.metrics import summarise
from plateau_sweep import self_sufficiency_threshold


def evaluate_rule(alloc, eval_seed=100, episodes=3, steps=144):
    """Deterministic rollout of a fixed rule, on the policy evaluation protocol."""
    logs, ratio = [], []
    carried_num = carried_den = 0.0
    served = {}

    for ep in range(episodes):
        slices = clara_slices()
        env = SlicingEnv(slices=slices,
                         traffic=MilanTraffic(slices, seed=eval_seed + ep, quiet=True),
                         capacity=100.0)
        env.reset()
        for _ in range(steps):
            demand = {s.name: s.last_demand for s in env.slices}
            units = alloc.decide(demand, env.capacity, env.slices)
            _, _, _, info = env.step(units)

            for n, ps in info["per_slice"].items():
                d, a = ps["demand"], ps["alloc"]
                carried_num += min(a, d)
                carried_den += d
                served.setdefault(n, []).append(min(a, d) / max(1e-9, d))

            p = info["per_slice"]["urllc"]
            ratio.append(p["alloc"] / max(1e-9, p["demand"]))
            logs.append(info)

    k = summarise(logs)
    pv = k["per_slice_violations"]
    return {
        "crit":         pv["urllc"] + pv["volte"],
        "urllc":        pv["urllc"],
        "alloc_ratio":  float(np.mean(ratio)),
        "carried":      float(carried_num / max(1e-9, carried_den)),
        "tau_urllc":    float(np.mean(served["urllc"])),
        "tau_volte":    float(np.mean(served["volte"])),
        "tau_video":    float(np.mean(served["video"])),
        "steps":        len(logs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headrooms", type=float, nargs="+",
                    default=[1.0, 1.2, 1.35, 1.5, 2.0])
    ap.add_argument("--out", default="baseline_eval.csv")
    a = ap.parse_args()

    h_star, _ = self_sufficiency_threshold()
    print(f"h* = {h_star:.4f}\n")

    rules = [("equal", EqualAllocator()),
             ("proportional", DemandProportionalAllocator()),
             ("priority-safe", PrioritySafeAllocator())]
    rules += [(f"headroom h={h:g}", HeadroomAllocator(h)) for h in a.headrooms]

    rows = []
    print(f"{'rule':<20}{'crit/864':>10}{'alloc/d':>10}{'tau':>8}"
          f"{'urllc':>8}{'volte':>8}{'video':>8}")
    print("-" * 72)
    for name, rule in rules:
        r = evaluate_rule(rule)
        r["rule"] = name
        rows.append(r)
        flag = ""
        if name.startswith("headroom"):
            h = float(name.split("=")[1])
            flag = "  <- below h*" if h < h_star else "  <- at/above h*"
        print(f"{name:<20}{r['crit']:>10}{r['alloc_ratio']:>10.2f}"
              f"{r['carried']:>8.3f}{r['tau_urllc']:>8.3f}"
              f"{r['tau_volte']:>8.3f}{r['tau_video']:>8.3f}{flag}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
