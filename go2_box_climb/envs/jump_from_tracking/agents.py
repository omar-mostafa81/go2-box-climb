"""PPO settings (rsl_rl) for the jump-tracking policy: the walking policy's settings as a start."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from go2_box_climb.skills.jump.jump_from_tracking.reference import check_jump

STEPS_PER_ITERATION = 24  # policy steps each robot collects per PPO update (a jump attempt is at most 150)


@configclass
class JumpTrackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = STEPS_PER_ITERATION
    max_iterations = 1500
    save_interval = 50
    experiment_name = "go2_jump_track_ground_to_box"  # logs/rsl_rl/<experiment_name>/<date_time>; see for_jump
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,  # the skill's exporter (walking_policy.export_checkpoint) needs plain MLPs
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",  # learning rate adapts to keep the policy change (KL) near desired_kl
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def for_jump(self, jump: str) -> "JumpTrackPPORunnerCfg":
        """One run folder family per jump: logs/rsl_rl/go2_jump_track_<jump>/."""
        self.experiment_name = f"go2_jump_track_{check_jump(jump)}"
        return self
