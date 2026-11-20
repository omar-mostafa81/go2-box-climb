"""Start a training stage from a checkpoint whose network has FEWER inputs (Isaac-free).

Stage B adds one observation (the body-height command) at the END of stage A's 48.
To continue from stage A, the first layer of the actor and of the critic gets one
extra input column filled with ZEROS. The widened network therefore produces
exactly the same outputs as stage A at the start of stage B, and learns to use the
new input from there. The optimizer state is not carried over (its shapes changed).
"""

from pathlib import Path

import torch


def first_layer_inputs(checkpoint: dict) -> int:
    return checkpoint["actor_state_dict"]["mlp.0.weight"].shape[1]


def widen_checkpoint_inputs(src: str | Path, obs_dim: int, dst: str | Path) -> bool:
    """Write a copy of ``src`` with zero-padded first layers if it has fewer than ``obs_dim`` inputs.

    Returns True if a widened copy was written to ``dst``, False if ``src`` already fits.
    """
    ckpt = torch.load(src, weights_only=False, map_location="cpu")
    n_old = first_layer_inputs(ckpt)
    if n_old == obs_dim:
        return False
    if n_old > obs_dim:
        raise ValueError(f"{src} has {n_old} inputs, more than the environment's {obs_dim}.")
    for key in ("actor_state_dict", "critic_state_dict"):
        w = ckpt[key]["mlp.0.weight"]
        ckpt[key]["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], obs_dim - n_old, dtype=w.dtype)], dim=1)
    ckpt.pop("optimizer_state_dict", None)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    return True
