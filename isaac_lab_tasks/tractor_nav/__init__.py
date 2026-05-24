"""Tractor navigation task — RGB + proprio exploration with skrl PPO.

Registers `Isaac-Tractor-Nav-v0` with gymnasium. All entry points are
strings so this module can be imported BEFORE Isaac Sim's AppLauncher
boots — concrete env/cfg classes pull in USD (`pxr`), which isn't
available until the sim app is running.
"""

import os

import gymnasium as gym

_AGENTS_DIR = os.path.join(os.path.dirname(__file__), "agents")

gym.register(
    id="Isaac-Tractor-Nav-v0",
    entry_point=f"{__name__}.tractor_env:TractorNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tractor_env_cfg:TractorNavEnvCfg",
        "skrl_cfg_entry_point": os.path.join(_AGENTS_DIR, "skrl_ppo_cfg.yaml"),
    },
)
