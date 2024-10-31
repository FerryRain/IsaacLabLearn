"""
@FileName：run_cartpole_rl_env.py
@Description：Run the environment for CartPole with file cartpole_env_cfg.py
@Author：Ferry
@Time：2024/10/30 下午8:51
@Copyright：©2024-2024 ShanghaiTech University-RIMLAB
"""

import argparse
from Basic.Basic_utils import create_applauncher
argparse = argparse.ArgumentParser(description="Tutorial on running the cartpole RL env")
argparse.add_argument("--num_envs", type=int, default=32, help="Number of environments")
argparse.add_argument("--debug_vis", type=bool, default=True, help="Debug visualization")
arg_clis, sim_launcher, sim_app = create_applauncher(argparse)

""" Reset everything follows. """
import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from RL_Policy.Create_a_Manager_based_RL_Env.cartpole_env_cfg import CartpoleEnvCfg

if __name__ == '__main__':
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = arg_clis.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg,debug_vis=False)

    count = 0
    while sim_app.is_running():
        with torch.inference_mode():
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO] Resetting...")

            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the env
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint : ", obs["policy"][0][1].item())
            count += 1
    env.close()
    sim_app.close()