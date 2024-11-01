"""
@FileName：__init__.py
@Description：
@Author：Ferry
@Time：2024/10/31 下午8:56
@Copyright：©2024-2024 ShanghaiTech University-RIMLAB
"""
""" Unitree-H1 balancing environment. """

import gymnasium as gym

from . import agents
from .Humanoid import HumanoidEnv, HumanoidCfg
from .LocomotionEnv import LocomotionEnv
##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humanoid-Mine-v0",
    entry_point="Create_a_Direct_Workflow_RL_Env.Humanoid:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HumanoidCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)