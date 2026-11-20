"""Run the trained walking policy outside the training code (Isaac-free).

    command (vx, vy, wz, h) + TaskState  ->  12 joint actions

The policy was trained by ``go2_box_climb/envs/locomotion``. This file rebuilds
EXACTLY the observation it saw during training, from a plain-numpy ``TaskState``,
and runs the exported network (TorchScript file ``policy.pt``). The push skill
(MPPI) and keyboard driving use it to make the robot walk.

Observation (49 numbers for stage B policies, 48 for stage A; this order, no scaling):
    [0:3]   base linear velocity, body frame [m/s]
    [3:6]   base angular velocity, body frame [rad/s]
    [6:9]   gravity direction in the body frame (unit vector; (0, 0, -1) when level)
    [9:12]  velocity command (vx, vy, wz) [m/s, m/s, rad/s]
    [12:24] joint positions minus DEFAULT_JOINT_POS [rad], JOINT_NAMES order
    [24:36] joint velocities [rad/s], JOINT_NAMES order
    [36:48] the previous action (raw network output)
    [48]    body-height command, as (h - 0.32 m) * 20  (stage B policies only)

Action (12 numbers): joint targets are  DEFAULT_JOINT_POS + ACTION_SCALE * action.
"""

from pathlib import Path

import numpy as np
import torch

from go2_box_climb.common import constants as C
from go2_box_climb.common.state import TaskState

# Command ranges the policy was trained on (walk_env_cfg.py imports these, so they
# always match). Commands outside them are clipped: the policy never saw them.
COMMAND_LIMITS = {
    "vx": (-1.0, 1.0),  # forward [m/s]
    "vy": (-0.6, 0.6),  # left [m/s]
    "wz": (-1.0, 1.0),  # yaw rate, counter-clockwise [rad/s]
    "h": (0.25, 0.34),  # body (base) height above the ground [m]  (stage B policies only)
}
DEFAULT_BODY_HEIGHT = 0.32  # normal walking height [m]
PUSH_BODY_HEIGHT = 0.27  # height to command while pushing the box [m]: head center ~0.22 m, below the 0.25 m box top
OBS_DIM = 49  # stage B: 48 + body-height command (stage A policies have 48 and no height input)
# The height command is fed to the network as (h - DEFAULT_BODY_HEIGHT) * HEIGHT_OBS_SCALE, so it
# varies by about +-1 like the other inputs (raw h only varies by a few cm and was ignored).
HEIGHT_OBS_SCALE = 20.0


def height_observation(h):
    """The network input for body height h (works for floats, numpy arrays and torch tensors)."""
    return (h - DEFAULT_BODY_HEIGHT) * HEIGHT_OBS_SCALE

# The released policy (committed to git). Pass another path to test a checkpoint
# export, e.g. logs/rsl_rl/go2_walk_push/<run>/exported/policy.pt
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[3] / "policies" / "walking" / "policy.pt"


def export_checkpoint(checkpoint: str | Path) -> Path:
    """Turn an rsl_rl training checkpoint (model_<N>.pt) into a TorchScript policy file.

    The checkpoint holds the actor's weights ("actor_state_dict": an MLP whose Linear
    layers are mlp.0, mlp.2, ... with ELU between them). We rebuild that network,
    keep only its mean output (no exploration noise) and save it next to the run:
        <run>/exported/<model_N>_policy.pt
    Returns that path (re-uses it if it already exists).
    """
    checkpoint = Path(checkpoint)
    out = checkpoint.parent / "exported" / f"{checkpoint.stem}_policy.pt"
    if out.exists() and out.stat().st_mtime >= checkpoint.stat().st_mtime:
        return out
    actor = torch.load(checkpoint, weights_only=False, map_location="cpu").get("actor_state_dict")
    if actor is None or not any(k.startswith("mlp.") for k in actor):
        raise ValueError(f"{checkpoint} is neither a TorchScript policy nor an rsl_rl training checkpoint.")
    if any("normalizer" in k for k in actor):
        raise ValueError(f"{checkpoint} uses observation normalization, which export_checkpoint does not support.")
    ids = sorted({int(k.split(".")[1]) for k in actor if k.startswith("mlp.") and k.endswith(".weight")})
    layers: list[torch.nn.Module] = []
    for n, i in enumerate(ids):
        w = actor[f"mlp.{i}.weight"]
        layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
        if n < len(ids) - 1:
            layers.append(torch.nn.ELU())
    net = torch.nn.Sequential(*layers)
    net.load_state_dict({f"{n * 2}.{p}": actor[f"mlp.{i}.{p}"] for n, i in enumerate(ids) for p in ("weight", "bias")})
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(net.eval()).save(str(out))
    return out


def resolve_policy_file(path: str | Path) -> Path:
    """Accept an exported TorchScript policy.pt OR a training checkpoint model_<N>.pt."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Walking policy not found: {path}. Pass an exported policy.pt or a training checkpoint model_<N>.pt, "
            "or use the released one in policies/walking/."
        )
    try:
        torch.jit.load(str(path), map_location="cpu")
        return path
    except RuntimeError:  # not TorchScript: treat it as a training checkpoint
        return export_checkpoint(path)


def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Express a world-frame vector ``v`` in the body frame of orientation ``q`` (x, y, z, w)."""
    x, y, z, w = q_xyzw
    u = np.array([x, y, z])
    # R^T v  for a unit quaternion
    return v * (2 * w * w - 1) - np.cross(u, v) * 2 * w + u * (2 * np.dot(u, v))


class WalkingPolicy:
    """The trained walking network plus the bookkeeping it needs (the last action)."""

    def __init__(self, policy_path: str | Path = DEFAULT_POLICY_PATH, device: str = "cpu"):
        """``policy_path``: an exported TorchScript policy.pt or a training checkpoint model_<N>.pt."""
        self.policy_file = resolve_policy_file(policy_path)  # exports a model_<N>.pt checkpoint if needed
        self.device = device
        self.net = torch.jit.load(str(self.policy_file), map_location=device).eval()
        # input size of the network: 49 = has the body-height input (stage B), 48 = stage A
        self.obs_dim = next(p.shape[1] for p in self.net.parameters() if p.dim() == 2)
        self.has_height_input = self.obs_dim == OBS_DIM
        self.command = np.zeros(3)
        self.height = DEFAULT_BODY_HEIGHT
        self.last_action = np.zeros(C.NUM_JOINTS)

    def reset(self) -> None:
        """Call when the robot is reset or when another skill hands over control."""
        self.last_action[:] = 0.0
        if hasattr(self.net, "reset"):  # recurrent policies; ours is a plain MLP
            self.net.reset()

    def set_command(self, vx: float, vy: float, wz: float, h: float = DEFAULT_BODY_HEIGHT) -> np.ndarray:
        """Set the velocity command (body frame) and body height.

        Returns (vx, vy, wz, h) after clipping to the trained ranges. ``h`` is ignored by
        stage A policies, which have no height input.
        """
        self.command = np.array(
            [np.clip(v, *COMMAND_LIMITS[k]) for k, v in zip(("vx", "vy", "wz"), (vx, vy, wz))], dtype=float
        )
        self.height = float(np.clip(h, *COMMAND_LIMITS["h"]))
        return np.append(self.command, self.height)

    def observation(self, state: TaskState) -> np.ndarray:
        """The observation, built exactly as during training (49 or 48 numbers)."""
        q = state.base_quat
        parts = [

                quat_rotate_inverse(q, state.base_lin_vel),
                quat_rotate_inverse(q, state.base_ang_vel),
                quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0])),
                self.command,
                state.joint_pos - C.DEFAULT_JOINT_POS,
                state.joint_vel,
                self.last_action,
        ]
        if self.has_height_input:
            parts.append(np.array([height_observation(self.height)]))
        return np.concatenate(parts).astype(np.float32)

    @torch.inference_mode()
    def act(self, state: TaskState) -> np.ndarray:
        """12 joint actions for the current state and command (call once per POLICY_DT)."""
        obs = torch.from_numpy(self.observation(state)).to(self.device).unsqueeze(0)
        action = self.net(obs)[0].cpu().numpy().astype(float)
        self.last_action = action
        return action
