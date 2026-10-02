"""Section 6 (reserve-then-allocate): per-condition summary and paired tests.

Reads the committed result files and prints the numbers behind the
projection-versus-reservation table. No training; runs in seconds.

    python figures/reserve_analysis.py

Conditions (8 paired seeds each, paper configuration):
    no floor               results/plateau_*_real.csv, headroom 0
    clip  h=1.35           results/plateau_*_real.csv, headroom 1.35
    clip  h=1.2, 1.38      results/clip_*.csv
    reserve h=1.0 ... 1.38 results/reserve_*.csv
Carried load for no floor and clip 1.35 comes from the thr_plateau re-runs.
"""
import csv
import itertools
import statistics as st
from collections import defaultdict

D = "results/"
FULL = 140          # checkpoint counts as total failure at >= 140/144
EVERY = 50_000


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def runs_from_traj(path, key="headroom"):
    """{(headroom, seed): metrics} computed from per-checkpoint trajectories."""
    by = defaultdict(list)
    for r in rows(path):
        by[(float(r[key]), int(r["seed"]))].append(r)
    out = {}
    for k, rs in by.items():
        rs.sort(key=lambda r: int(r["timestep"]))
        crit = [int(r["crit"]) for r in rs]
        urllc = [round(float(r["frac_below_hstar"]) * 144) for r in rs]
        safe_at = -1
        for i in range(len(rs)):
            if all(c == 0 for c in crit[i:]):
                safe_at = int(rs[i]["timestep"])
                break
        out[k] = {
            "total": sum(crit),
            "urllc": sum(urllc),
            "volte": sum(crit) - sum(urllc),
            "stagnation": EVERY * sum(c >= FULL for c in crit),
            "safe_at": safe_at,
            "peak_plateau": max(float(r["frac_plateau"]) for r in rs),
            "n_ckpt": len(rs),
        }
    return out


def summaries(path):
    return {(float(r["headroom"]), int(r["seed"])): r for r in rows(path)}


def wilcoxon_exact(x, y):
    """Two-sided exact Wilcoxon signed-rank p (zero differences dropped)."""
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    if n == 0:
        return float("nan"), 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    w = min(w_plus, sum(ranks) - w_plus)
    count = total = 0
    for signs in itertools.product((0, 1), repeat=n):
        s = sum(r for r, g in zip(ranks, signs) if g)
        total += 1
        if min(s, sum(ranks) - s) <= w + 1e-9:
            count += 1
    return count / total, w


def main():
    old = runs_from_traj(D + "plateau_traj_real.csv")
    old_sum = summaries(D + "plateau_sweep_real.csv")
    thr = summaries(D + "thr_plateau.csv")
    clip = runs_from_traj(D + "clip_traj.csv")
    clip_sum = summaries(D + "clip_sweep.csv")
    res = runs_from_traj(D + "reserve_traj.csv")
    res_sum = summaries(D + "reserve_sweep.csv")

    conds = [("no floor", old, old_sum, thr, 0.0),
             ("clip 1.2", clip, clip_sum, clip_sum, 1.2),
             ("clip 1.35", old, old_sum, thr, 1.35),
             ("clip 1.38", clip, clip_sum, clip_sum, 1.38),
             ("reserve 1.0", res, res_sum, res_sum, 1.0),
             ("reserve 1.2", res, res_sum, res_sum, 1.2),
             ("reserve 1.35", res, res_sum, res_sum, 1.35),
             ("reserve 1.38", res, res_sum, res_sum, 1.38)]
    seeds = range(8)
    table = {}
    print(f"{'condition':<13}{'viol mean±sd':>15}{'urllc':>7}{'volte':>7}{'%train':>8}"
          f"{'stagn':>9}{'safe':>6}{'end/432':>9}{'alloc':>7}{'carried':>9}"
          f"{'video':>7}{'peak_pl':>8}")
    for name, traj, summ, thrs, h in conds:
        t = [traj[(h, s)] for s in seeds]
        sm = [summ[(h, s)] for s in seeds]
        th = [thrs[(h, s)] for s in seeds]
        tot = [x["total"] for x in t]
        table[name] = tot
        print(f"{name:<13}{st.mean(tot):>8.0f} ± {st.stdev(tot):<4.0f}"
              f"{st.mean(x['urllc'] for x in t):>7.0f}{st.mean(x['volte'] for x in t):>7.0f}"
              f"{100 * st.mean(tot) / 2880:>7.1f}%"
              f"{st.mean(x['stagnation'] for x in t):>9,.0f}"
              f"{sum(x['safe_at'] > 0 for x in t):>4d}/8"
              f"{st.mean(int(r['crit']) for r in sm):>9.1f}"
              f"{st.mean(float(r['alloc_ratio']) for r in sm):>7.2f}"
              f"{st.mean(float(r['carried']) for r in th):>9.3f}"
              f"{st.mean(float(r['served_video']) for r in th):>7.3f}"
              f"{max(x['peak_plateau'] for x in t):>8.3f}")
    print()
    for a, b in [("reserve 1.35", "clip 1.35"), ("reserve 1.35", "no floor"),
                 ("reserve 1.2", "clip 1.2"), ("reserve 1.2", "no floor"),
                 ("reserve 1.38", "clip 1.38"), ("reserve 1.0", "no floor"),
                 ("clip 1.2", "no floor"), ("clip 1.38", "no floor")]:
        x, y = table[a], table[b]
        p, w = wilcoxon_exact(x, y)
        lower = sum(u < v for u, v in zip(x, y))
        ties = sum(u == v for u, v in zip(x, y))
        sep = max(x) < min(y)
        print(f"{a:<13} vs {b:<11} lower on {lower}/8 (ties {ties})  "
              f"W={w:g}  p={p:.4f}  complete separation={sep}")


if __name__ == "__main__":
    main()
