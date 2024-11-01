"""
@FileName：Unitree_H1.py
@Description：The Unitree-H1 env cfg
@Author：Ferry
@Time：2024/10/31 下午8:57
@Copyright：©2024-2024 ShanghaiTech University-RIMLAB
"""
from __future__ import annotations

from omni.isaac.lab_assets import HUMANOID_CFG, H1_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from Create_a_Direct_Workflow_RL_Env.Humanoid.LocomotionEnv import LocomotionEnv

@configclass
class UnitreeH1Cfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0

    # Unitree-H1
    action_space = 19
    observation_space = 69
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    # Unitree-H1 robot
    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        50.0,  # left_hip_yaw
        50.0,  # right_hip_yaw
        50.0,  # torso
        50.0,  # left_hip_roll
        50.0,  # right_hip_roll
        50.0,  # left_shoulder_pitch
        50.0,  # right_shoulder_pitch
        50.0,  # left_hip_pitch
        50.0,  # right_hip_pitch
        50.0,  # left_shoulder_roll
        50.0,  # right_shoulder_roll
        50.0,  # left_knee
        50.0,  # right_knee
        50.0,  # left_shoulder_yaw
        50.0,  # right_shoulder_yaw
        50.0,  # left_ankle
        50.0,  # right_ankle
        50.0,  # left_elbow
        50.0,  # right_elbow
    ]

    heading_weight: float = .5
    up_weight: float = .1

    energy_cost_scale: float = .05
    actions_cost_scale: float = .01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = .1

    death_cost: float = -1.0
    termination_height: float = .8
    angular_velocity_scale: float = .25
    contact_force_scale: float = .01

class UnitreeH1Env(LocomotionEnv):
    cfg: UnitreeH1Cfg
    def __init__(self, cfg: UnitreeH1Cfg, render_mode:str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)