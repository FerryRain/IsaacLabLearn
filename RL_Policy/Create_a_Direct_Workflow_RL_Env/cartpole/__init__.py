"""
@FileName：__init__.py
@Description：
@Author：Ferry
@Time：2024/10/31 下午4:51
@Copyright：©2024-2024 ShanghaiTech University-RIMLAB
"""
""" Cartpole balancing environment. """

import gymnasium as gym

from . import agent as agents
from .direct_rl_scene_cartpole_env import CartpoleEnv, CartpoleEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Cartpole-Direct-Mine-v0",
    entry_point="Create_a_Direct_Workflow_RL_Env.cartpole:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
