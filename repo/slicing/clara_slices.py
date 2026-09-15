"""
clara_slices.py
---------------
The three slices used in CLARA (Liu, Ding, Zhang & Liu, IEEE Big Data 2021):
video, VoLTE, and URLLC, sharing one bandwidth pool.

Mapping onto our framework (and the simplifications, stated honestly):

  CLARA slice   our family   SLA in our framework            why
  -----------   ----------   -----------------------------   ---------------------------
  video         eMBB         >= 90% of demanded bandwidth    quality scales with share
  VoLTE         URLLC        latency <= 20 ms                voice: small bw, delay-bound
  URLLC         URLLC        latency <= 5 ms                 strict delay guarantee

Honest simplification: CLARA formulates per-slice *delay* constraints computed
from queueing behaviour. Our latency proxy is the convex function in slices.py
(latency grows as allocation/demand ratio falls). Same direction of effect,
simpler mechanics - say this to the DC before they ask.

Priorities: URLLC (1) > VoLTE (2) > video (3), so the criticality-aware
baseline protects voice and URLLC and lets video absorb overload - which is
exactly the behaviour CLARA's constraints are meant to enforce.
"""

from .slices import Slice


def clara_slices():
    return [
        Slice(name="urllc", family="URLLC", priority=1, sla_target=5.0,  min_guarantee=6.0),
        Slice(name="volte", family="URLLC", priority=2, sla_target=20.0, min_guarantee=4.0),
        Slice(name="video", family="eMBB",  priority=3, sla_target=0.90, min_guarantee=0.0),
    ]
