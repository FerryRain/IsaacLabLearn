"""
@FileName：cartpole_env_cfg.py
@Description：Create a manager-based cartpole scene for RL.
@Author：Ferry
@Time：2024/10/30 下午6:35
@Copyright：©2024-2024 ShanghaiTech University-RIMLAB
"""
import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedRLEnv

import omni.isaac.lab_tasks.manager_based.classic.cartpole.mdp as mdp

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG
import torch

def constant_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ The generated command from the command generator. """
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)

""" Scene definition. """

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """ Configuration for a cart-pole scene. """
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(color=(.9, .9, .9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/distant_light",
        spawn=sim_utils.DistantLightCfg(color=(.9, .9, .9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(.738, .477, .477, .0)),
    )

""" MDP sttings """

@configclass
class ActionsCfg:
    """ Action specifications for the MDP. """

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)

@configclass
class ObservationsCfg:
    """ Observation specifications for the MDP. """
    @configclass
    class PolicyCfg(ObsGroup):
        """ Observations for policy group. """

        # obvervation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        velocity_commands = ObsTerm(func=constant_commands)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """ Configuration for events. """
    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-.25 * math.pi, .25 * math.pi),
            "velocity_range": (-.25 * math.pi, .25 * math.pi),
        },
    )

@configclass
class RewardsCfg:
    """ Reward terms for the MDP. """

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0}
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )

    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )

@configclass
class TerminationCfg:
    """ Termination terms for the MDP. """

    # (1) Time Out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )

@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """ COnfiguration for the cartpole environment. """

    # Scene settings
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # MDP setting
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationCfg = TerminationCfg()

    # post initialzation
    def __post_init__(self) -> None:
        """ Post initialization. """
        # general setting
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1/120
        self.sim.render_interval = self.decimation
