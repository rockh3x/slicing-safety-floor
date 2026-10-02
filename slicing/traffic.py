"""
traffic.py
----------
Generates per-step demand for each slice from fixed scenarios.

Used by the early fixed-rule demo only; the paper's experiments are driven by
milan_traffic.MilanTraffic.

Scenarios:
    quiet   - 2 AM,  everything low
    office  - 11 AM, video climbs
    match   - 8 PM,  video demand explodes (the surge)
"""

import numpy as np

# Base demand per slice for each scenario, in resource units.
SCENARIOS = {
    "quiet":  {"surgery": 8,  "video": 30, "sensors": 18},
    "office": {"surgery": 10, "video": 55, "sensors": 25},
    "match":  {"surgery": 10, "video": 88, "sensors": 22},
}


class TrafficModel:
    """Produces noisy demand vectors so the environment is non-trivial."""

    def __init__(self, slices, scenario="office", noise=0.12, seed=0):
        self.slice_names = [s.name for s in slices]
        self.scenario = scenario
        self.noise = noise                     # relative Gaussian noise
        self.rng = np.random.default_rng(seed)  # reproducible -> defensible results

    def set_scenario(self, scenario):
        assert scenario in SCENARIOS, f"unknown scenario {scenario}"
        self.scenario = scenario

    def demand(self):
        """Return a dict {slice_name: demand_units} for the current step."""
        base = SCENARIOS[self.scenario]
        out = {}
        for name in self.slice_names:
            b = base[name]
            # multiplicative noise, clamped so demand stays non-negative
            factor = max(0.0, 1.0 + self.rng.normal(0, self.noise))
            out[name] = round(b * factor, 2)
        return out
