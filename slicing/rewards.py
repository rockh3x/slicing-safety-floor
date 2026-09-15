"""
rewards.py - reward function designs for safety-critical slice allocation
=========================================================================
The open problem this module exists to attack.

DIAGNOSIS THAT MOTIVATED IT
    Under the original reward,  R = mean_sat + 0.2*util - 0.1*weighted_viol,
    PPO converges to a uniform split. That is not a training failure - it is
    the reward's own optimum being nearly flat:

        headroom x1.5   1.179    <- the better policy
        equal/uniform   1.080    <- where PPO lands
        proportional    0.566
        priority-safe  -1.362

    Only a 9% gap between the default policy and the good one. Worse, the
    components barely move:
        utilisation  = 1.000 for EVERY policy   -> zero gradient, dead weight
        mean_sat     = 0.93 .. 0.99             -> 6% dynamic range
        violations   = the only discriminating term, smallest coefficient
    PPO initialises at action mean ~0, which softmaxes to a uniform split, and
    there is almost no slope to climb away from it.

    SliceFed (2026) names this as the field's default failure: most DRL
    slicing formulations "treat slicing as an unconstrained MDP, relying on
    reward shaping and providing no formal guarantees on latency or
    reliability." This module makes the reward the object of study rather
    than an afterthought.

EACH REWARD TAKES
    per_slice : dict from env info - satisfaction, sla_met, latency_ms, alloc, demand
    slices    : the Slice objects (family, sla_target, priority)
    util      : capacity utilisation this step
"""

import numpy as np

PENALTY_WEIGHT = {"URLLC": 10.0, "eMBB": 1.0, "mMTC": 3.0}


def _weighted_violations(per_slice, slices):
    return sum(PENALTY_WEIGHT[s.family]
               for s in slices if not per_slice[s.name]["sla_met"])


# ---------------------------------------------------------------------------
def r_baseline(per_slice, slices, util):
    """R1 - the original. Flat landscape; PPO finds the uniform split."""
    sat = np.mean([p["satisfaction"] for p in per_slice.values()])
    return float(sat + 0.2 * util - 0.1 * _weighted_violations(per_slice, slices))


def r_no_util(per_slice, slices, util):
    """R2 - drop the utilisation term, which is 1.000 for every policy.

    Tests the hypothesis that a constant term is simply dead weight. It should
    not change WHICH policy is best, only remove an uninformative offset.
    """
    sat = np.mean([p["satisfaction"] for p in per_slice.values()])
    return float(sat - 0.1 * _weighted_violations(per_slice, slices))


def r_violation_heavy(per_slice, slices, util):
    """R3 - same shape, 5x the violation coefficient.

    Widens the gap between good and default policies. Crude but it directly
    tests whether the problem is purely gradient magnitude.
    """
    sat = np.mean([p["satisfaction"] for p in per_slice.values()])
    return float(sat - 0.5 * _weighted_violations(per_slice, slices))


def r_risk_sensitive(per_slice, slices, util):
    """R4 - penalise being NEAR a violation, not only crossing it.

    A binary SLA check gives the agent no warning that it is approaching the
    cliff: latency 4.9 ms and 0.1 ms score identically against a 5 ms target.
    Here a sigmoid in the ratio latency/target starts biting before the
    boundary, so there is a gradient pointing away from danger.

    This is the same idea as SafeSlice's (2025) sigmoid risk-sensitive reward.
    Arriving at it independently from the flat-landscape diagnosis is a good
    sign that it is the right shape.
    """
    total, n = 0.0, 0
    for s in slices:
        p = per_slice[s.name]
        if s.family == "URLLC":
            lat = p["latency_ms"] if p["latency_ms"] is not None else 0.0
            x = lat / max(1e-6, s.sla_target)          # 1.0 = exactly at the limit
            risk = 1.0 / (1.0 + np.exp(-6.0 * (x - 0.8)))   # bites from 80% of target
            total += (1.0 - risk) * PENALTY_WEIGHT[s.family]
        else:
            total += p["satisfaction"] * PENALTY_WEIGHT[s.family]
        n += PENALTY_WEIGHT[s.family]
    return float(total / n)


def r_headroom(per_slice, slices, util):
    """R5 - reward critical slices for holding headroom, not just for coping.

    Directly encodes the measured finding: allocating exactly demand pins
    queue utilisation at 1.0, where delay diverges. Reward peaks at ~1.5x
    demand and falls off on both sides - too little is unsafe, too much
    starves the elastic slice.
    """
    total, n = 0.0, 0
    for s in slices:
        p = per_slice[s.name]
        w = PENALTY_WEIGHT[s.family]
        if s.priority <= 2 and p["demand"] > 0:
            ratio = p["alloc"] / p["demand"]
            score = np.exp(-((ratio - 1.5) ** 2) / 0.5)    # peak at 1.5x
        else:
            score = p["satisfaction"]
        total += score * w
        n += w
    return float(total / n)


def r_hybrid(per_slice, slices, util):
    """R6 - risk-sensitivity for critical slices + satisfaction for elastic ones,
    with an explicit violation penalty retained.

    The combination the diagnosis points to: a smooth gradient away from the
    SLA cliff, an unchanged incentive to keep the elastic slice served, and a
    hard cost for actually crossing a limit.
    """
    risk_part = r_risk_sensitive(per_slice, slices, util)
    viol = _weighted_violations(per_slice, slices)
    return float(risk_part - 0.2 * viol)


def r_deviation(per_slice, slices, util):
    """R7 - penalise PER-STEP deviation below the headroom target.

    WHY THIS EXISTS - the R5 stability result that produced it:
      Five seeds trained on R5 to 2M steps all converged to nearly IDENTICAL
      mean allocations:
          seed 0 [0.090, 0.159, 0.751]   36.1% violations
          seed 1 [0.098, 0.164, 0.739]   27.8%
          seed 2 [0.097, 0.158, 0.745]   24.1%
          seed 3 [0.095, 0.167, 0.738]   37.7%
          seed 4 [0.096, 0.159, 0.746]   30.3%
      Splits within ~1% of each other, violation rates 14 points apart. The
      MEAN allocation was right; the TIMING was not.

      R5's Gaussian bump exp(-(ratio-1.5)^2/0.5) is symmetric and smooth, so
      a step at ratio 0.8 can be offset by a step at 2.2 and the average looks
      fine. Nothing in the reward constrains WHEN the headroom is applied,
      and demand spikes are exactly when it matters. Seed 0 even reached zero
      violations at 500k and then drifted back to 158 by 2M - it could not
      hold a good solution because the reward did not require holding it.

    THE FIX
      Score each step on its own, and make the penalty ASYMMETRIC:
      under-allocating a latency-critical slice is what breaks SLAs, so it is
      punished hard; over-allocating only wastes capacity, so it is punished
      gently. No amount of over-allocation can compensate for a starved step.

          deficit = max(0, target - alloc) / target      -> steep penalty
          excess  = max(0, alloc - target) / target      -> mild penalty
    """
    return _deviation_core(per_slice, slices, under_penalty=3.0, over_penalty=0.15)


def _deviation_core(per_slice, slices, under_penalty=3.0,
                    over_penalty=0.15, target_headroom=1.5):
    """Shared body so the coefficients can be swept without duplicating code.

    under_penalty    cost of UNDER-allocating a critical slice (breaks SLAs)
    over_penalty     cost of OVER-allocating it (only wastes capacity)
    target_headroom  allocation target, as a multiple of demand

    The asymmetry under_penalty >> over_penalty is the whole design: no
    amount of over-allocation can compensate for a single starved step.
    """
    total, n = 0.0, 0
    for s in slices:
        p = per_slice[s.name]
        w = PENALTY_WEIGHT[s.family]
        if s.priority <= 2 and p["demand"] > 0:
            target = p["demand"] * target_headroom
            shortfall = max(0.0, target - p["alloc"]) / target
            surplus = max(0.0, p["alloc"] - target) / target
            score = max(-1.0, 1.0 - under_penalty * shortfall
                              - over_penalty * surplus)
        else:
            score = p["satisfaction"]
        total += score * w
        n += w
    return float(total / n)


def make_deviation(over_penalty, under_penalty=3.0, target_headroom=1.5):
    """Factory for swept variants of R7.

    R7 is safe (0 critical violations on 5/5 seeds at every checkpoint) but
    WASTEFUL: with over_penalty=0.15 over-allocation is nearly free, so the agent
    buys safety with capacity the elastic slice wanted. Violation rate lands
    at 14-29% vs the headroom heuristic's 11.6%.

    Raising over_penalty should reclaim that capacity. The question this sweep
    answers: is there a value where violations stay at ZERO and efficiency
    beats the heuristic, or does efficiency only improve by trading safety
    back? Either answer is a result - the second maps a safety/efficiency
    frontier.
    """
    def fn(per_slice, slices, util):
        return _deviation_core(per_slice, slices, under_penalty,
                               over_penalty, target_headroom)
    fn.__name__ = f"asymmetric_over{over_penalty}"
    return fn


def r_deviation_hard(per_slice, slices, util):
    """R8 - R7 plus an explicit violation penalty.

    Tests whether the per-step deviation term alone is enough, or whether a
    hard cost for actually crossing an SLA is still needed on top of it.
    """
    return float(r_deviation(per_slice, slices, util)
                 - 0.15 * _weighted_violations(per_slice, slices))


REWARDS = {
    "R1_baseline":       r_baseline,
    "R2_no_util":        r_no_util,
    "R3_violation_heavy": r_violation_heavy,
    "R4_risk_sensitive": r_risk_sensitive,
    "R5_headroom":       r_headroom,
    "R6_hybrid":         r_hybrid,
    "R7_deviation":      r_deviation,
    "R8_deviation_hard": r_deviation_hard,
}

# Swept variants of R7, differing only in how harshly OVER-allocation is
# penalised. Readable names are canonical; the terse "R7e0.5" style keys are
# kept as aliases so existing scripts, CSVs and saved models still resolve.
for _over in [0.15, 0.3, 0.5, 1.0, 2.0]:
    _fn = make_deviation(_over)
    REWARDS[f"asymmetric_over{_over}"] = _fn      # readable
    REWARDS[f"R7e{_over}"] = _fn                  # legacy alias
