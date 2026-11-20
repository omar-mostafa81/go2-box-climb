"""PPO settings (rsl_rl) for the walking policy, stages A and B."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

STEPS_PER_ITERATION = 24  # policy steps each robot collects per PPO update


@configclass
class WalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage A: train from scratch."""

    num_steps_per_env = STEPS_PER_ITERATION
    max_iterations = 1000
    save_interval = 50
    experiment_name = "go2_walk"  # logs/rsl_rl/go2_walk/<date_time>
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
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


@configclass
class WalkPushPPORunnerCfg(WalkPPORunnerCfg):
    """Stage B: continue from stage A with the steady push."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500  # extra iterations on top of stage A (boxes + height command need more than 800)
        self.experiment_name = "go2_walk_push"  # logs/rsl_rl/go2_walk_push/<date_time>
