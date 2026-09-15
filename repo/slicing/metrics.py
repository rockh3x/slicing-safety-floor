"""
metrics.py
----------
Turns a run's per-step logs into the summary numbers you report.

These four are the KPIs you will defend and, later, the ones your RL reward is
built from. They are deliberately IMT-2030-flavoured:
    sla_violation_rate  -> reliability   (lower is better)
    mean_utilisation    -> efficiency    (higher is better, but not at any cost)
    mean_satisfaction   -> QoE           (higher is better)
    per_slice_violations -> where the pain lands (which SLA breaks, and how often)
"""

import numpy as np


def summarise(logs):
    """logs: list of 'info' dicts returned by env.step(). Returns a dict of KPIs."""
    steps = len(logs)
    if steps == 0:
        return {}

    violations = np.array([l["violations"] for l in logs])
    utils = np.array([l["utilisation"] for l in logs])
    sats = np.array([
        np.mean([ps["satisfaction"] for ps in l["per_slice"].values()])
        for l in logs
    ])

    # per-slice violation counts
    slice_names = list(logs[0]["per_slice"].keys())
    per_slice_viol = {n: 0 for n in slice_names}
    for l in logs:
        for n, ps in l["per_slice"].items():
            if not ps["sla_met"]:
                per_slice_viol[n] += 1

    return {
        "steps": steps,
        "sla_violation_rate": round(float((violations > 0).mean()), 3),  # frac of steps with >=1 break
        "mean_utilisation": round(float(utils.mean()), 3),
        "mean_satisfaction": round(float(sats.mean()), 3),
        "per_slice_violations": per_slice_viol,
    }


def print_report(name, kpis):
    print(f"\n=== {name} ===")
    print(f"  steps                : {kpis['steps']}")
    print(f"  SLA violation rate   : {kpis['sla_violation_rate']:.1%}  (frac of steps with a broken SLA)")
    print(f"  mean utilisation     : {kpis['mean_utilisation']:.1%}")
    print(f"  mean satisfaction    : {kpis['mean_satisfaction']:.3f}")
    pv = ", ".join(f"{n}={c}" for n, c in kpis["per_slice_violations"].items())
    print(f"  per-slice violations : {pv}")
