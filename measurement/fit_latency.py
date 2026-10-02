"""
fit_latency.py - fit L(rho) = L0 + A * max(0, rho - rho_knee)^kappa to measurements
==================================================================================
Takes the CSV from run_sweep.sh and produces:

  * least-squares estimates of L0, A, rho_knee, kappa
  * bootstrap confidence intervals on each
  * the implied self-sufficiency threshold h*, WITH a confidence interval
  * a goodness-of-fit report and a residual plot

The h* interval is the output that matters. The paper's experimental design
sets the floor 1.9% below h*; if the interval is wide, or if it straddles the
floor you intend to use, the design has to be rebuilt around the measurement
rather than the other way round.

USAGE
    python fit_latency.py latency_measurements.csv --sla 5.0
    python fit_latency.py latency_measurements.csv --sla 5.0 --stat p99
"""

import argparse
import csv
import sys

import numpy as np

try:
    from scipy.optimize import curve_fit
except ImportError:
    sys.exit("scipy is required:  pip install scipy")


def model(rho, L0, A, knee, kappa):
    return L0 + A * np.maximum(0.0, rho - knee) ** kappa


def h_star(L0, A, knee, kappa, sla):
    """Floor multiplier at which a binding floor exactly meets the SLA.

    Returns nan when the SLA is unreachable (SLA below the floor latency L0)
    or when the knee alone already satisfies it, both of which are meaningful
    outcomes rather than errors.
    """
    if sla <= L0:
        return float("nan")
    inner = knee + ((sla - L0) / A) ** (1.0 / kappa)
    return float("nan") if inner <= 0 else 1.0 / inner


def fit(rho, lat, n_boot=2000, seed=0):
    # Multi-start: kappa sits in an exponent, so a single start finds local minima.
    best, best_sse = None, np.inf
    for k0 in (1.0, 1.5, 2.0, 3.0):
        for kn0 in (0.5, 0.6, 0.7, 0.8):
            try:
                p, _ = curve_fit(
                    model, rho, lat,
                    p0=[max(0.01, lat.min()), 100.0, kn0, k0],
                    bounds=([0.0, 1e-3, 0.0, 0.5], [np.inf, np.inf, 0.99, 6.0]),
                    maxfev=60000)
            except (RuntimeError, ValueError):
                continue
            sse = float(((model(rho, *p) - lat) ** 2).sum())
            if sse < best_sse:
                best, best_sse = p, sse
    if best is None:
        sys.exit("fit did not converge - check the data covers the knee region")

    rng = np.random.default_rng(seed)
    boots = []
    n = len(rho)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            p, _ = curve_fit(model, rho[idx], lat[idx], p0=best,
                             bounds=([0.0, 1e-3, 0.0, 0.5],
                                     [np.inf, np.inf, 0.99, 6.0]),
                             maxfev=20000)
            boots.append(p)
        except (RuntimeError, ValueError):
            continue
    return best, np.array(boots), best_sse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sla", type=float, required=True,
                    help="latency target in ms for the critical slice")
    ap.add_argument("--stat", default="p95",
                    choices=["mean", "p50", "p95", "p99", "max"])
    ap.add_argument("--plot", default="latency_fit.png")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    rho = np.array([float(r["rho"]) for r in rows])
    lat = np.array([float(r[f"{a.stat}_ms"]) for r in rows])
    loss = np.array([float(r.get("loss_pct", 0)) for r in rows])

    if (loss > 1.0).any():
        bad = rho[loss > 1.0]
        print(f"WARNING: loss above 1% at rho = {sorted(set(bad))}.\n"
              "         Beyond the point where the queue starts dropping, delay\n"
              "         stops rising and the curve flattens - those points bias\n"
              "         the fit downward. Consider a deeper pfifo or excluding "
              "them.\n")

    p, boots, sse = fit(rho, lat)
    names = ["L0 (ms)", "A", "rho_knee", "kappa"]

    print(f"fitted on {a.stat}, n = {len(rho)} points\n")
    print(f"{'parameter':<12}{'estimate':>12}{'95% CI':>26}")
    print("-" * 50)
    for i, nm in enumerate(names):
        if len(boots):
            lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
            print(f"{nm:<12}{p[i]:>12.4f}   [{lo:>9.4f}, {hi:>9.4f}]")
        else:
            print(f"{nm:<12}{p[i]:>12.4f}{'(bootstrap failed)':>26}")

    resid = model(rho, *p) - lat
    ss_tot = ((lat - lat.mean()) ** 2).sum()
    print(f"\nRMSE  {np.sqrt((resid ** 2).mean()):.4f} ms")
    print(f"R^2   {1 - sse / ss_tot:.4f}")

    hs = h_star(*p, a.sla)
    print(f"\nSLA = {a.sla} ms  ->  h* = {hs:.4f}")
    if len(boots):
        hb = np.array([h_star(*b, a.sla) for b in boots])
        hb = hb[np.isfinite(hb)]
        if len(hb):
            lo, hi = np.percentile(hb, [2.5, 97.5])
            print(f"            95% CI [{lo:.4f}, {hi:.4f}]")
            print(f"\nA floor is sub-threshold only if it is below {lo:.4f} "
                  "with confidence.")
            print("Pick the experimental floor from the LOWER end of this "
                  "interval, not the point estimate.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = np.linspace(rho.min(), rho.max(), 400)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), dpi=160,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.scatter(rho, lat, s=28, color="#2a78d6", zorder=3, label="measured")
        ax1.plot(g, model(g, *p), color="#eb6834", lw=2, zorder=4, label="fit")
        ax1.axhline(a.sla, ls="--", color="#0ca30c", lw=1.4, label=f"SLA {a.sla} ms")
        ax1.axvline(p[2], ls=":", color="#898781", lw=1.4, label="fitted knee")
        ax1.set_ylabel(f"RTT {a.stat} (ms)"); ax1.legend(frameon=False, fontsize=9)
        ax1.set_title("Measured delay against offered-load ratio", loc="left")
        ax2.axhline(0, color="#c3c2b7", lw=1)
        ax2.scatter(rho, resid, s=22, color="#52514e")
        ax2.set_xlabel(r"offered-load ratio $\rho$"); ax2.set_ylabel("residual (ms)")
        fig.tight_layout(); fig.savefig(a.plot, bbox_inches="tight")
        print(f"\nplot: {a.plot}  -- check the residuals for structure before "
              "trusting the fit")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
