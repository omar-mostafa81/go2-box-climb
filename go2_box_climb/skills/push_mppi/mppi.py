"""MPPI planner (Model Predictive Path Integral control) for the push.

Each replan:
    1. Take the current plan U (T steps of (vx, vy, wz)), warm-started from the last replan.
    2. Sample K noisy copies of it (Gaussian noise, clipped to the allowed commands).
       Sample 0 is the plan itself without noise, so a good plan is never lost.
    3. Roll every copy out on the planar model and add up its cost S_k.
    4. New plan = weighted average of the samples with weights exp(-(S_k - min S) / lambda):
       cheap rollouts count much more than expensive ones.
    5. Send the plan's first command; shift the plan one step left for next time.
"""

from dataclasses import dataclass, field

import numpy as np

from go2_box_climb.skills.locomotion import COMMAND_LIMITS

from .cost import CostWeights, running_cost, terminal_cost
from .model import PlanarParams, rollout


def _limits(lo_hi: tuple[float, float], key: str) -> tuple[float, float]:
    """Intersect a push limit with the walking policy's trained range."""
    lo, hi = COMMAND_LIMITS[key]
    return max(lo, lo_hi[0]), min(hi, lo_hi[1])


@dataclass
class MPPIConfig:
    dt: float = 0.1  # model step = replan period [s]
    horizon: int = 30  # steps (4 s)
    num_samples: int = 512  # K
    noise_std: tuple[float, float, float] = (0.15, 0.1, 0.3)  # (vx, vy, wz)
    temperature: float = 1.0  # lambda
    # Push commands stay slow (the walking policy then tracks them closely).
    vx_limits: tuple[float, float] = (-0.3, 0.4)  # [m/s]
    vy_limits: tuple[float, float] = (-0.3, 0.3)  # [m/s]
    wz_limits: tuple[float, float] = (-0.6, 0.6)  # [rad/s]
    seed: int = 0
    weights: CostWeights = field(default_factory=CostWeights)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lims = [_limits(self.vx_limits, "vx"), _limits(self.vy_limits, "vy"), _limits(self.wz_limits, "wz")]
        return np.array([lo for lo, _ in lims]), np.array([hi for _, hi in lims])


def evaluate(x0, U, u_prev, p: PlanarParams, w: CostWeights) -> tuple[np.ndarray, dict, np.ndarray]:
    """Total cost (K,), per-term sums over the horizon {name: (K,)}, and the states (K, T+1, 6)."""
    X, info = rollout(x0, U, p)
    K, T, _ = U.shape
    total = np.zeros(K)
    term_sums: dict[str, np.ndarray] = {}
    for t in range(T):
        prev = np.broadcast_to(u_prev, (K, 3)) if t == 0 else U[:, t - 1]
        step_info = {k: v[:, t] for k, v in info.items()}
        c, terms = running_cost(X[:, t + 1], X[:, t], U[:, t], prev, step_info, p, w)
        total += c
        for k, v in terms.items():
            term_sums[k] = term_sums.get(k, 0.0) + v
    term_sums["terminal"] = terminal_cost(X[:, -1], w)
    total += term_sums["terminal"]
    return total, term_sums, X


class MPPIPlanner:
    def __init__(self, head_x: float, config: MPPIConfig | None = None):
        self.cfg = config or MPPIConfig()
        self.params = PlanarParams(dt=self.cfg.dt, head_x=head_x)
        self.lo, self.hi = self.cfg.bounds()
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)
        self.U = np.zeros((self.cfg.horizon, 3))  # current plan
        self.u_prev = np.zeros(3)  # last command sent
        self.last_terms: dict[str, float] = {}  # cost breakdown of the chosen plan (for logging)
        self.last_states: np.ndarray | None = None  # predicted states of the chosen plan (T+1, 6)

    def plan(self, x0: np.ndarray) -> np.ndarray:
        """One MPPI update from state x0 (6,); returns the command (vx, vy, wz) to send now."""
        cfg = self.cfg
        noise = self.rng.normal(size=(cfg.num_samples, cfg.horizon, 3)) * np.array(cfg.noise_std)
        noise[0] = 0.0  # keep the current plan as one of the samples
        samples = np.clip(self.U + noise, self.lo, self.hi)

        costs, _, _ = evaluate(x0, samples, self.u_prev, self.params, cfg.weights)
        weights = np.exp(-(costs - costs.min()) / cfg.temperature)
        weights /= weights.sum()
        self.U = np.einsum("k,ktc->tc", weights, samples)

        # breakdown of the plan we actually follow (one extra rollout, cheap)
        _, terms, X = evaluate(x0, self.U[None], self.u_prev, self.params, cfg.weights)
        self.last_terms = {k: float(v[0]) for k, v in terms.items()}
        self.last_states = X[0]

        u0 = self.U[0].copy()
        self.u_prev = u0
        self.U = np.concatenate([self.U[1:], self.U[-1:]])  # warm start for the next replan
        return u0
