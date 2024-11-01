"""
@FileName：__init__.py.py
@Description：
@Author：Ferry
@Time：2024/10/31 下午4:53
@Copyright：©2024-2024 ShanghaiTech University-RIMLAB
"""
import gymnasium as gym

from . import agents
from .shadow_hand_env_cfg import ShadowHandEnvCfg
from .shadow_hand_env import HandsEnv
##
# Register Gym environments.
##

gym.register(
    id="Isaac-ShadowHand-Cube-v0",
    entry_point="Create_a_Direct_Workflow_RL_Env.shadowhands:HandsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)