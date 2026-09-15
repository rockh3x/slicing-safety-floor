"""
allocators.py
-------------
The 'brain' that decides how many units each slice requests. Swapping the brain
is the whole experiment: the environment never changes, only this does.

Phase 1-2 ship three FIXED-RULE brains so you have honest baselines to beat.
Phase 3 (post-exam) adds an RL brain with the SAME .decide(obs, ...) signature,
which is exactly your Objective 2 - and because they share the interface, the
RL agent drops in with zero changes to the environment.

Every allocator implements:
    decide(demand: dict, capacity: float, slices: list) -> dict {name: units}
"""


class EqualAllocator:
    """Naive baseline: split capacity equally, ignore who needs what.

    This is the strawman. It wastes units on slices that don't need them and
    starves the ones that do. Great to show WHY intelligence is needed.
    """
    name = "equal"

    def decide(self, demand, capacity, slices):
        share = capacity / len(slices)
        return {s.name: share for s in slices}


class DemandProportionalAllocator:
    """Heuristic baseline: give each slice a share proportional to its demand.

    Much better than equal, and a genuinely respectable baseline. Its weakness:
    it is 'blind' to criticality - under heavy load it will shave the surgery
    slice just as readily as the video slice, because it only looks at size.
    """
    name = "proportional"

    def decide(self, demand, capacity, slices):
        total = sum(demand.values())
        if total <= 0:
            return {s.name: 0.0 for s in slices}
        return {s.name: capacity * demand[s.name] / total for s in slices}


class PrioritySafeAllocator:
    """Smarter heuristic: serve critical slices to their demand first, then
    share the remainder among the rest by demand.

    This is the 'hand-crafted safety' baseline. It usually keeps every SLA, and
    it is the bar your future RL agent must MATCH on safety while BEATING on
    efficiency. Being able to say that sentence in the viva is worth a lot.
    """
    name = "priority-safe"

    def decide(self, demand, capacity, slices):
        grant = {s.name: 0.0 for s in slices}
        remaining = capacity
        # critical slices (priority 1-2) served to full demand first, in order
        for s in sorted(slices, key=lambda x: x.priority):
            if s.priority <= 2:
                give = min(demand[s.name], remaining)
                grant[s.name] = give
                remaining -= give
        # share whatever is left among the non-critical slices by demand
        rest = [s for s in slices if s.priority > 2]
        rest_demand = sum(demand[s.name] for s in rest)
        for s in rest:
            if rest_demand > 0:
                grant[s.name] += remaining * demand[s.name] / rest_demand
        return grant


def all_allocators():
    return [EqualAllocator(), DemandProportionalAllocator(),
            PrioritySafeAllocator(), HeadroomAllocator(1.5)]


class HeadroomAllocator:
    """Latency-critical slices get demand x headroom; the rest share the remainder.

    WHY THIS EXISTS - the finding that produced it:
      Allocating a latency-critical slice EXACTLY its demand drives its queue
      utilisation to 1.0, and queueing delay grows without bound at full
      utilisation. The latency model used throughout this project is an
      ASSUMED convex form, L(rho) = L0 + A * max(0, rho - rho_knee)^kappa with
      L0 = 0.5 ms, A = 292.8, rho_knee = 0.70, kappa = 1.15. It is not fitted
      to measurements from any testbed, and no result here should be read as
      a hardware measurement.

      Under that model, on the Telecom Italia Milan day trace and the paper's
      evaluation protocol (3 x 144 steps, eval seed 100), PrioritySafeAllocator
      (which grants exactly demand) breaks the URLLC SLA in 432 of 432 steps.
      Headroom 1.5x takes that to 0, at the cost of carried load falling from
      0.952 to 0.897. See baseline_eval.py for the numbers and the sweep that
      brackets the self-sufficiency threshold h* = 1.3765.

    The headroom factor is a tunable safety/efficiency knob - and 'how much
    headroom, per slice, right now' is exactly the question a learning agent
    should answer (Objective 2), under a constraint that it never drops below
    the safe floor (Objective 3).
    """
    def __init__(self, headroom: float = 1.5):
        self.headroom = float(headroom)
        self.name = f"headroom x{headroom:g}"

    def decide(self, demand, capacity, slices):
        grant = {s.name: 0.0 for s in slices}
        remaining = capacity
        for s in sorted(slices, key=lambda x: x.priority):
            if s.priority <= 2:                       # latency-critical
                give = min(demand[s.name] * self.headroom, remaining)
                grant[s.name] = give
                remaining -= give
        rest = [s for s in slices if s.priority > 2]
        rest_demand = sum(demand[s.name] for s in rest)
        for s in rest:
            if rest_demand > 0:
                grant[s.name] += remaining * demand[s.name] / rest_demand
        return grant
