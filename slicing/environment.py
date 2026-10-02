"""
environment.py
--------------
The network slicing environment.

It deliberately uses the SAME interface shape as reinforcement-learning
environments (reset / step), so fixed-rule allocators and a learning agent
(through gym_env.SlicingGymEnv) drive the same code:

    obs            = env.reset()
    obs, reward, done, info = env.step(action)

    action = a dict {slice_name: units_requested}
    obs    = current demands + last allocations (what an agent would observe)
    reward = utilisation reward  -  penalty for SLA violations
    info   = full per-slice detail for logging, metrics and plots

Contention rule:
    Slices REQUEST units. If total request > capacity, the network cannot
    invent bandwidth, so requests are scaled down. We scale the *discretionary*
    part only: min_guarantee units are protected first, and the remainder is
    shared proportionally. (The demand-proportional safety floor studied in
    the paper is applied earlier, in gym_env.SlicingGymEnv._safety_filter.)
"""

import numpy as np


class SlicingEnv:
    def __init__(self, slices, traffic, capacity=100.0, sla_penalty=20.0,
                 isolation=True):
        """isolation=True  -> each slice gets its own queue (hard slicing).
           isolation=False -> one shared queue (soft / best-effort).
        The paper uses isolation=True throughout."""
        self.slices = slices
        self.traffic = traffic
        self.capacity = float(capacity)
        self.sla_penalty = float(sla_penalty)   # reward points lost per SLA violation
        self.isolation = bool(isolation)
        self.t = 0
        self._demand = None

    # ---- RL-style API -----------------------------------------------------
    def reset(self):
        self.t = 0
        self._demand = self.traffic.demand()
        for s in self.slices:
            s.last_demand = self._demand[s.name]
            s.last_alloc = 0.0
        return self._observation()

    def step(self, action: dict):
        """Apply an allocation, advance one step, return (obs, reward, done, info)."""
        alloc = self._resolve_contention(action)

        # System load ratio: total demand across all slices / capacity.
        # Only a slice in a SHARED queue (isolation=False) takes its latency
        # from this; an isolated slice uses its own demand/alloc - see
        # Slice.latency_ms in slices.py.
        rho = sum(self._demand.values()) / self.capacity if self.capacity > 0 else 0.0

        # score every slice against the demand it faced this step
        per_slice, violations, total_sat = {}, 0, 0.0
        for s in self.slices:
            d = self._demand[s.name]
            a = alloc[s.name]
            met = s.sla_met(a, d, rho, self.isolation)
            sat = s.satisfaction(a, d, rho, self.isolation)
            if not met:
                violations += 1
            total_sat += sat
            s.last_demand, s.last_alloc = d, a
            per_slice[s.name] = {
                "demand": d, "alloc": a, "sla_met": met,
                "satisfaction": round(sat, 3),
                "latency_ms": round(s.latency_ms(a, d, rho, self.isolation), 1) if s.family == "URLLC" else None,
            }

        used = sum(alloc.values())
        utilisation = used / self.capacity
        # Reward: reward useful work (mean satisfaction) and utilisation,
        # then subtract a stiff penalty for each SLA broken.
        reward = (total_sat / len(self.slices)) + 0.2 * utilisation \
            - self.sla_penalty * violations

        info = {
            "t": self.t, "scenario": self.traffic.scenario, "rho": round(rho, 3),
            "per_slice": per_slice, "violations": violations,
            "utilisation": round(utilisation, 3), "used": round(used, 2),
            "reward": round(reward, 3),
        }

        # advance to next step's demand
        self.t += 1
        self._demand = self.traffic.demand()
        for s in self.slices:
            s.last_demand = self._demand[s.name]
        return self._observation(), reward, False, info

    # ---- helpers ----------------------------------------------------------
    def _resolve_contention(self, action):
        """Turn requested units into granted units under a hard capacity cap."""
        req = {s.name: max(0.0, float(action.get(s.name, 0.0))) for s in self.slices}

        # 1) protect guarantees first
        guaranteed = {s.name: min(req[s.name], s.min_guarantee) for s in self.slices}
        g_total = sum(guaranteed.values())

        # 2) share whatever capacity remains across the discretionary requests
        remaining_cap = max(0.0, self.capacity - g_total)
        disc = {s.name: req[s.name] - guaranteed[s.name] for s in self.slices}
        disc_total = sum(disc.values())

        grant = {}
        if disc_total <= remaining_cap or disc_total == 0:
            # everything fits (or nothing discretionary requested)
            for s in self.slices:
                grant[s.name] = guaranteed[s.name] + disc[s.name]
        else:
            scale = remaining_cap / disc_total  # proportional fair share
            for s in self.slices:
                grant[s.name] = guaranteed[s.name] + disc[s.name] * scale
        return {k: round(v, 3) for k, v in grant.items()}

    def _observation(self):
        """What an agent sees: current demand + last allocation per slice."""
        obs = []
        for s in self.slices:
            obs.append(self._demand[s.name])
            obs.append(s.last_alloc)
        return np.array(obs, dtype=float)
