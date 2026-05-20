"""Tractor navigation task — pure exploration with RGB camera + proprio.

Registers `Isaac-Tractor-Nav-v0` with gymnasium so Isaac Lab's
train.py can pick it up via --task.
"""

import gymnasium as gym

from . import agents
from .tractor_env import TractorNavEnv
from .tractor_env_cfg import TractorNavEnvCfg

gym.register(
    id="Isaac-Tractor-Nav-v0",
    entry_point=f"{__name__}.tractor_env:TractorNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TractorNavEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:TractorNavPPORunnerCfg",
    },
)
