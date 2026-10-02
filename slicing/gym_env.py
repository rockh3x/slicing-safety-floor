"""
gym_env.py - Gymnasium wrapper around the slicing environment
=============================================================
Turns SlicingEnv into a standard RL environment so Stable-Baselines3 (or any
Gymnasium-compatible library) can train on it without custom glue.

WHY A WRAPPER RATHER THAN A REWRITE
    SlicingEnv already has reset/step and the latency model. This only
    translates between dict-of-slices and the flat float arrays that RL
    libraries expect - and adds the safety layer as a toggle, so constrained
    and unconstrained agents run on an otherwise identical environment. That
    is what makes the comparison fair.

OBSERVATION  (3 values per slice, normalised to roughly [0, 2])
    demand / capacity        what this slice is asking for
    last_alloc / capacity    what it got last step
    rho                      system load, total demand / capacity (repeated
                             per slice)

ACTION  (1 value per slice, in [-1, 1])  ->  softmax over the vector
    A BUG WORTH RECORDING: the first version used weights in [0,1] normalised
    by their sum. That mapping is scale-invariant - [1,1,1] and [0.3,0.3,0.3]
    produce the SAME allocation - so "increase everything" costs nothing and
    the policy drifts to the boundary. Both agents saturated: unconstrained
    settled on [1,1,1] (a uniform split, identical to the 'equal' baseline)
    and the constrained one on [0,0.1,1], dumping everything into video and
    letting the safety filter rescue the critical slices.

    Softmax over [-1,1] removes the redundancy: only RELATIVE values matter,
    the mapping is one-to-one, and the bounds are not attractors.

REWARD
    Selected by name from rewards.REWARDS (reward=...). The paper uses
    "asymmetric_over0.5": per-step asymmetric deviation from 1.5x demand for
    the latency-critical slices, satisfaction for the elastic one, weighted by
    PENALTY_WEIGHT. The default, "R1_baseline", is the original design kept
    for the reward study.

SAFETY LAYER  (safe=True)
    Raises each latency-critical slice to its floor, min(demand * headroom,
    capacity / 2), and reclaims the excess only from what each slice asked
    for above its own floor - see _safety_filter. Every clip is counted in
    info["clipped"] and the capacity the filter moved in info["override"], so
    "how often did the policy try to violate safety?" is measurable.
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:      # see the note in fix_sweep.py: no silent
    raise ImportError(          # fallback to the unmaintained `gym` package
        "gymnasium is required (pip install gymnasium). Legacy `gym` is not "
        "a substitute -- it is unmaintained and incompatible with NumPy 2.") from exc

from .rewards import REWARDS
from .clara_slices import clara_slices
from .milan_traffic import MilanTraffic
from .environment import SlicingEnv


# violation cost by slice family - the knob that decides what the agent learns
PENALTY_WEIGHT = {"URLLC": 10.0, "eMBB": 1.0, "mMTC": 3.0}


class SlicingGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, capacity=100.0, episode_steps=144, seed=0,
                 safe=False, headroom=1.5, isolation=True, verbose=False,
                 reward="R1_baseline"):
        super().__init__()
        self.reward_name = reward
        self.reward_fn = REWARDS[reward]
        self.capacity = float(capacity)
        self.episode_steps = int(episode_steps)
        self.safe = bool(safe)
        self.headroom = float(headroom)
        self.isolation = bool(isolation)
        self._seed = seed
        self.verbose = verbose

        self.slices = clara_slices()
        self.n = len(self.slices)
        self.names = [s.name for s in self.slices]

        self.observation_space = spaces.Box(low=0.0, high=5.0,
                                            shape=(3 * self.n,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n,), dtype=np.float32)
        self._build(seed)

    # ---------------------------------------------------------------
    def _build(self, seed):
        self.slices = clara_slices()
        if self.verbose:
            self.traffic = MilanTraffic(self.slices, seed=seed)
        else:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                self.traffic = MilanTraffic(self.slices, seed=seed)
        self.env = SlicingEnv(self.slices, self.traffic,
                              capacity=self.capacity, isolation=self.isolation)

    def _obs(self):
        rho = sum(s.last_demand for s in self.slices) / self.capacity
        v = []
        for s in self.slices:
            v += [s.last_demand / self.capacity,
                  s.last_alloc / self.capacity,
                  rho]
        return np.asarray(v, dtype=np.float32)

    # ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)          # required by the Gymnasium API
        if seed is not None:
            self._seed = seed
        self._build(self._seed)
        self.env.reset()
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        e = np.exp(2.0 * (a - a.max()))          # temperature 2 -> usable spread
        share = e / e.sum()
        units = {n: float(self.capacity * share[i])
                 for i, n in enumerate(self.names)}

        policy_request = dict(units)     # what the agent itself asked for
        clipped = 0
        if self.safe:
            units, clipped = self._safety_filter(units)

        # How far did the filter move the agent's action? If this is large on
        # every step, the constraint is not shaping a learned policy - it is
        # REPLACING it, and the agent is decorative. Measured, not assumed.
        override = sum(abs(units[n] - policy_request[n])
                       for n in units) / self.capacity

        _, _, _, info = self.env.step(units)

        reward = float(self.reward_fn(info["per_slice"], self.slices,
                                      info["utilisation"]))
        weighted_viol = sum(
            PENALTY_WEIGHT[s.family]
            for s in self.slices
            if not info["per_slice"][s.name]["sla_met"]
        )

        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_steps
        info["clipped"] = clipped
        info["override"] = round(override, 4)     # fraction of capacity moved
        info["raw_share"] = {n: round(policy_request[n] / self.capacity, 4)
                             for n in policy_request}
        info["weighted_violations"] = weighted_viol
        return self._obs(), reward, terminated, truncated, info

    # ---------------------------------------------------------------
    def _safety_filter(self, requested):
        """Objective 3: no critical slice below demand * headroom.

        requested -> what the policy asked for
        returns   -> (granted, n_clipped) after enforcing the floors

        Rationale: allocating a latency-critical slice exactly its demand pins
        its queue load at 1.0, well past the latency model's knee, so the floor
        is set at a multiple of demand. Whether that multiple is high enough
        is the paper's question: below h* = 1.3765 a binding floor cannot meet
        the 5 ms URLLC target.
        """
        # minimum each latency-critical slice must receive this step
        minimum_required, n_clipped = {}, 0
        for s in self.slices:
            if s.priority <= 2:                       # latency-critical
                minimum_required[s.name] = min(s.last_demand * self.headroom,
                                               self.capacity * 0.5)
            else:
                minimum_required[s.name] = 0.0

        granted = dict(requested)
        for name, minimum in minimum_required.items():
            if granted[name] < minimum:
                n_clipped += 1
                granted[name] = minimum

        # if raising the floors pushed us over capacity, shrink only the
        # DISCRETIONARY part (what sits above each floor), never the floors
        if sum(granted.values()) > self.capacity:
            reserved = sum(minimum_required.values())
            spare = max(0.0, self.capacity - reserved)
            discretionary = {n: max(0.0, granted[n] - minimum_required[n])
                             for n in granted}
            total_discretionary = sum(discretionary.values())
            for n in granted:
                granted[n] = minimum_required[n] + (
                    discretionary[n] * spare / total_discretionary
                    if total_discretionary > 0 else 0.0)
        return granted, n_clipped


def make_env(**kw):
    """Convenience factory for SB3's make_vec_env."""
    return SlicingGymEnv(**kw)
