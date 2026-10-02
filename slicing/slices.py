"""
slices.py
---------
Defines a network slice and how well a given resource allocation satisfies
its Service Level Agreement (SLA).

The three slice families follow the 5G/IMT-2030 service classes:
    URLLC  - needs a latency guarantee, little bandwidth
    eMBB   - needs large bandwidth, tolerant of delay
    mMTC   - needs to keep a large number of devices connected

Design note:
    Each slice measures satisfaction in ITS OWN currency. That is the whole
    point of slicing - one number ('more bandwidth') cannot describe all three.
    So every slice converts (allocation, demand) into a satisfaction score in
    [0, 1] and a boolean 'sla_met' using a model appropriate to its family.
"""

from dataclasses import dataclass, field


@dataclass
class Slice:
    name: str            # human label, e.g. "surgery"
    family: str          # "URLLC" | "eMBB" | "mMTC"
    priority: int        # lower number = more critical (1 = most critical)
    # SLA parameters (meaning depends on family, see satisfaction() below):
    sla_target: float    # URLLC: max latency ms | eMBB: min quality ratio | mMTC: min connected fraction
    min_guarantee: float = 0.0   # units that must never be taken away (Objective 3 hook)

    # --- filled in each step, kept here so plots/logging can read them back ---
    last_demand: float = 0.0
    last_alloc: float = 0.0

    def satisfaction(self, alloc: float, demand: float, rho: float = 0.0,
                     isolated: bool = False) -> float:
        """Return a satisfaction score in [0, 1] for this allocation.

        1.0 means the slice is fully served; 0.0 means starved.
        The curve differs per family - this is where the 'own currency' idea lives.
        """
        if demand <= 0:
            return 1.0  # nothing requested -> trivially satisfied
        ratio = alloc / demand  # fraction of what the slice asked for

        if self.family == "URLLC":
            latency = self.latency_ms(alloc, demand, rho, isolated)
            # Satisfaction is 1 while under the target, then decays.
            return 1.0 if latency <= self.sla_target else max(0.0, self.sla_target / latency)

        if self.family == "eMBB":
            # Quality scales roughly with the share you get, capped at 1.
            return min(1.0, ratio)

        if self.family == "mMTC":
            # Fraction of devices we can keep attached scales with share.
            return min(1.0, ratio)

        raise ValueError(f"unknown slice family: {self.family}")

    # ---- LATENCY MODEL (ASSUMED, NOT FITTED) ---------------------------
    #
    #     L(load) = L_FLOOR + A * max(0, load - KNEE)^K      (capped at 400 ms)
    #
    # A convex proxy: flat while there is slack, rising steeply past a knee.
    # The four constants were chosen for shape. They are NOT fitted to
    # measurements from any testbed, and nothing here should be read as a
    # hardware measurement. The paper's results depend on the curve having a
    # knee below load = 1/h (its scope condition), not on these values; the
    # derived threshold h* = 1.3765 is specific to them.
    #
    # `load` is the slice's own demand/allocation when it has its own queue
    # (isolated=True, used throughout the paper), or the system load when it
    # shares one - see latency_ms().
    L_FLOOR = 0.5     # ms, delay floor
    A       = 292.8   # ms, scale
    KNEE    = 0.70    # load at which queueing delay begins to rise
    K       = 1.15    # exponent

    def latency_ms(self, alloc: float, demand: float, rho: float = 0.0,
                   isolated: bool = False) -> float:
        """Latency in ms for this slice.

        rho       = system load (total demand / capacity)
        isolated  = does this slice have its own queue?

        Which queue the slice sits in decides what its latency depends on:

          isolated=False (shared queue, 'soft'): the slice inherits the
             SYSTEM's congestion, whatever its own allocation. Allocation
             cannot protect a slice that shares a queue with a flooding
             neighbour.

          isolated=True (own queue, 'hard'): the slice only feels ITS OWN
             utilisation, demand/alloc. This is the mode used in the paper.

        This is the hard vs soft slicing distinction (Afolabi 2018,
        Sec. VII-B): isolation buys guarantees, sharing buys utilisation.
        """
        if demand <= 0:
            load = 0.0
        elif isolated:
            load = demand / alloc if alloc > 0 else 99.0   # own-queue utilisation
        else:
            load = rho                                     # shared-queue: system load
        lat = self.L_FLOOR + self.A * max(0.0, load - self.KNEE) ** self.K
        return min(400.0, lat)

    def sla_met(self, alloc: float, demand: float, rho: float = 0.0,
                isolated: bool = False) -> bool:
        """Boolean SLA check, evaluated in the slice's own currency."""
        if demand <= 0:
            return True
        ratio = alloc / demand
        if self.family == "URLLC":
            return self.latency_ms(alloc, demand, rho, isolated) <= self.sla_target
        # eMBB / mMTC: SLA is a minimum fraction of demand served
        return ratio >= self.sla_target


def default_slices():
    """Generic three-slice setup used by the early fixed-rule demo.

    The paper uses clara_slices.clara_slices() instead."""
    return [
        Slice(name="surgery", family="URLLC", priority=1, sla_target=5.0,  min_guarantee=8.0),
        Slice(name="video",   family="eMBB",  priority=3, sla_target=0.90, min_guarantee=0.0),
        Slice(name="sensors", family="mMTC",  priority=2, sla_target=0.90, min_guarantee=5.0),
    ]
