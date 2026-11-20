"""Walking task (PPO velocity tracker), stages A and B. Part: walking policy (team leader).

Isaac-dependent: import only after launching the app.
    walk_env_cfg.py  environment configs (WalkEnvCfg = stage A, WalkPushEnvCfg = stage B)
    agents.py        PPO settings (WalkPPORunnerCfg, WalkPushPPORunnerCfg)
    mdp.py           the steady push and its ramp (stage B)
"""
