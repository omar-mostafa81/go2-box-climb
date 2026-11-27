"""Run a trained jump-tracking policy outside the training code (Isaac-free).

    TaskState + time since the jump started  ->  12 joint actions

Builds the observation with the same function as training (observation.py), so
it is exactly what the network saw, and runs the exported network (TorchScript).
At ``reset`` the robot's current position and heading become the start the
reference is anchored to, as at the start of every training episode.

Action (12 numbers): joint targets = DEFAULT_JOINT_POS + ACTION_SCALE * action
(the shared robot contract).
"""

from pathlib import Path

import numpy as np
import torch

from go2_box_climb.common import constants as C
from go2_box_climb.common.state import TaskState
from go2_box_climb.skills.locomotion.walking_policy import resolve_policy_file
from go2_box_climb.utils.runs import REPO_ROOT

from .jump_math import yaw_of
from .observation import OBS_DIM, jump_observation
from .reference import JUMPS, check_jump, load_reference

#: Released policies (committed to git once trained), one per jump.
POLICY_DIR = REPO_ROOT / "policies" / "jump_from_tracking"
DEFAULT_POLICY_PATHS = {jump: POLICY_DIR / f"{jump}.pt" for jump in JUMPS}

#: Skill names used in logs (TRIAL_LOG_COLUMNS "skill").
SKILL_NAMES = {"ground_to_box": "jump_ground_box", "box_to_platform": "jump_box_platform"}


def state_tensors(state: TaskState, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """The TaskState fields as (1, ...) float32 tensors."""

    def f(x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).unsqueeze(0)

    return {
        "base_pos": f(state.base_pos),
        "base_quat": f(state.base_quat),
        "base_lin_vel": f(state.base_lin_vel),
        "base_ang_vel": f(state.base_ang_vel),
        "joint_pos": f(state.joint_pos),
        "joint_vel": f(state.joint_vel),
        "box_pos": f(state.box_pos),
        "box_quat": f(state.box_quat),
    }


class JumpTrackingPolicy:
    """The trained jump network plus what it needs: the reference, the start pose and the last action."""

    def __init__(
        self,
        jump: str,
        policy_path: str | Path | None = None,
        reference_path: str | Path | None = None,
        device: str = "cpu",
    ):
        """
        Args:
            jump: "ground_to_box" or "box_to_platform".
            policy_path: exported TorchScript policy.pt or a training checkpoint model_<N>.pt
                (default: the released policy in policies/jump_from_tracking/).
            reference_path: the reference the policy was trained with (default: references/).
        """
        self.jump = check_jump(jump)
        self.device = device
        self.track = load_reference(jump, reference_path, device=device)
        path = Path(policy_path) if policy_path else DEFAULT_POLICY_PATHS[jump]
        if not path.exists():
            raise FileNotFoundError(
                f"Jump policy not found: {path}. Pass an exported policy.pt or a training checkpoint "
                "logs/rsl_rl/go2_jump_track_<jump>/<run>/model_<N>.pt."
            )
        self.policy_file = resolve_policy_file(path)  # exports a model_<N>.pt checkpoint if needed
        self.net = torch.jit.load(str(self.policy_file), map_location=device).eval()
        in_dim = next(p.shape[1] for p in self.net.parameters() if p.dim() == 2)
        if in_dim != OBS_DIM:
            raise ValueError(f"{self.policy_file} expects {in_dim} inputs; the jump observation has {OBS_DIM}.")
        self.start_xy = torch.zeros(1, 2, device=device)
        self.start_yaw = torch.zeros(1, device=device)
        self.start_time = 0.0
        self.last_action = torch.zeros(1, C.NUM_JOINTS, device=device)

    def reset(self, state: TaskState) -> None:
        """Call when the jump starts: the current pose becomes the reference's start."""
        s = state_tensors(state, self.device)
        self.start_xy = s["base_pos"][:, :2].clone()
        self.start_yaw = yaw_of(s["base_quat"])
        self.start_time = float(state.time)
        self.last_action = torch.zeros(1, C.NUM_JOINTS, device=self.device)
        if hasattr(self.net, "reset"):  # recurrent policies; ours is a plain MLP
            self.net.reset()

    def elapsed(self, state: TaskState) -> float:
        """Time since the jump started [s]."""
        return float(state.time) - self.start_time

    def observation(self, state: TaskState) -> torch.Tensor:
        """(1, OBS_DIM) observation, built exactly as during training."""
        s = state_tensors(state, self.device)
        return jump_observation(
            track=self.track,
            time=torch.tensor([self.elapsed(state)], dtype=torch.float32, device=self.device),
            start_xy=self.start_xy,
            start_yaw=self.start_yaw,
            last_action=self.last_action,
            **s,
        )

    @torch.no_grad()  # not inference_mode: its tensors (the stored last action) could not be changed later
    def act(self, state: TaskState) -> np.ndarray:
        """12 joint actions for the current state (call once per POLICY_DT)."""
        action = self.net(self.observation(state))
        self.last_action = action.reshape(1, C.NUM_JOINTS).clone()
        return action[0].cpu().numpy().astype(float)
