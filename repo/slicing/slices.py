"""
slices.py
---------
Defines a network slice and how well a given resource allocation satisfies
its Service Level Agreement (SLA).

The three slice archetypes match the 5G/IMT-2030 families you defend in the viva:
    URLLC  - surgery        : needs latency guarantee, tiny bandwidth
    eMBB   - video          : needs large bandwidth, tolerant of delay
    mMTC   - sensors         : needs to keep a huge number of devices connected

Design note (say this to the DC):
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

    # ---- CALIBRATED LATENCY MODEL -------------------------------------
    # Fitted to measurements on a Linux HTB testbed (10 Mbit link, 1000-packet
    # FIFO, three UDP classes, 448 probes per load level). See fig4_calibration.
    #
    # KEY FINDING that forced this rewrite: at system load rho = 0.93 every
    # flow received the throughput it requested (per-slice ratio = 1.0) and
    # latency was still 54 ms. The previous model, which depended only on
    # alloc/demand, predicted 4 ms - a 13.5x under-estimate. Queueing delay is
    # driven by SYSTEM utilisation, not only by whether a slice got its share.
    #
    #     L(ratio, rho) = [ L_FLOOR + A * max(0, rho - KNEE)^K ] / min(1, ratio)
    #
    # First bracket: queueing delay from system load (measurement-fitted).
    # Divisor: additional penalty when this slice is itself starved.
    #
    # Measured vs model:  rho 0.33 -> 0.55 ms (model 0.50)
    #                     rho 0.63 -> 0.73 ms (model 0.50)
    #                     rho 0.93 -> 53.99 ms (model 54.0)
    #                     rho 1.08 -> 95.03 ms (model 95.5)
    #
    # LIMITATION to state openly: fitted to four load points on one testbed
    # with one buffer configuration. It is a calibrated empirical model for
    # this regime, not a universal law.
    L_FLOOR = 0.5     # ms, unloaded round-trip
    A       = 292.8   # ms, fitted amplitude
    KNEE    = 0.70    # system load at which queueing begins to bite
    K       = 1.15    # fitted exponent

    def latency_ms(self, alloc: float, demand: float, rho: float = 0.0,
                   isolated: bool = False) -> float:
        """Latency in ms for this slice.

        rho       = system load (total demand / capacity)
        isolated  = does this slice have its own queue?

        WHICH QUEUE THE SLICE SITS IN DECIDES EVERYTHING - this is the second
        finding from the testbed, and the one the first patch got wrong.

          isolated=False (shared queue, 'soft'): the slice inherits the
             SYSTEM's congestion. Measured: at rho=1.08 all three classes
             showed ~96 ms regardless of their own demand. Allocation cannot
             protect you when you share a queue with a flooding neighbour.

          isolated=True (own queue, 'hard'): the slice only feels ITS OWN
             utilisation, demand/alloc. Measured: VoNR held 0.35 ms p95 while
             video in the same link sat at 114 ms.

        This is exactly the hard vs soft slicing distinction (Afolabi 2018,
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
    """The three-slice setup used across the demo, mirroring the website."""
    return [
        Slice(name="surgery", family="URLLC", priority=1, sla_target=5.0,  min_guarantee=8.0),
        Slice(name="video",   family="eMBB",  priority=3, sla_target=0.90, min_guarantee=0.0),
        Slice(name="sensors", family="mMTC",  priority=2, sla_target=0.90, min_guarantee=5.0),
    ]
