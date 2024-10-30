"""
@FileName：8_Create_a_Manager_Based_Based_env.py
@Description：How to used manager based env like env.ManagerBasedEnv and env.BasedBasedRLEnv.
              BasedBasedRLEnv is an env design for RL train which contain rewards and terminations additionally.
              This scripts only look at ManagerBasedEnv and its corresponding configuration class env.ManagerBasedEnvCfg.
              It combines the concepts of scene, action, observation and event managers to create an environment.
@Used-Envs: Cart-pole
@Author：Ferry
@Time：2024/10/29 下午9:04
@Copyright：©2024-2024 ShanghaiTech University-RIMLAB
"""

import argparse
from Basic.Basic_utils import create_applauncher, create_simulator

parser = argparse.ArgumentParser(description="Tutorial on creating a manager based env")
parser.add_argument("--num_envs", type=int, default=16, help="Number of envs")

arg_cli, applauncher, appsim = create_applauncher(parser)

""" Reset everything follows """
import math
import torch
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as obsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg

@configclass
class ActionsCfg:
    """ Action sprcifications for the environment. """
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)

@configclass
class ObservationCfg:
    """ Observation for policy group. """

    @configclass
    class PolicyCfg(obsGroup):
        """ Observation for policy group. """
        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """ Configuration for events. """

    # on startup
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        }
    )

    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, .1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-.125 * math.pi, .125 * math.pi),
            "velocity_range": (-.01 * math.pi, .01 * math.pi),
        },
    )

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """ Configuration for cartpole env. """

    #  Scene setting
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic setting
    observations = ObservationCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """ Post initialization. """
        # viewer settings
        self.viewer.eye = [4.5, .0, 6.0]
        self.viewer.lookat = [.0, .0, 2.0]
        #  step settings
        self.decimation = 4 # env step every 4 sim steps: 200HZ / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005

if __name__ == '__main__':
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = arg_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while appsim.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting the environment...")
            # Sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][:])
            count += 1

    env.close()
    appsim.close()