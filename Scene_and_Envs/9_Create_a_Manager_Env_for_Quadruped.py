"""
@FileName：9_Create_a_Manager_Env_for_Quadruped.py
@Description：Like tutorial 8, this tutorial creates a manager environment for Quadruped
@Author：Ferry
@Time：2024/10/30 上午11:32
@Copyright：©2024-2024 ShanghaiTech University-RIMLAB
"""
import argparse
from Basic.Basic_utils import create_applauncher, create_simulator

parser = argparse.ArgumentParser(description="Create a manager environment for Quadruped.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")

args_cli, simapplauncher, sim_app = create_applauncher(parser)

""" Reset everying follows. """
import torch
import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

""" Pre define configs. """
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG

""" Create observation terms. """
def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """ The generated command from the command generator. """
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)

""" Scene definition. """
@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    """ Example scene configeration. """

    # add terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # add robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(.0, .0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(.75, .75, .75), intensity=3000.0),
    )

@configclass
class ActionCfg:
    """ Action spectifications for the MDPS. """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=.5, use_default_offset=True)

@configclass
class ObservationsCfg:
    """ Observation specifications for the MDPs. """
    # observation terms (order preserved)
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-.1, n_max=.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-.2, n_max=.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-.05, n_max=.05),
        )
        velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-.01, n_max=.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-.1, n_max=.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    #  observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """ Configuration for events. """

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """ COnfiguration for the locomotion velocity-tracking environment. """

    # Scene setting
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic setting
    observations: ObservationsCfg = ObservationsCfg()
    actions : ActionCfg = ActionCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """ Post initialization. """
        # general settings
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt # 50Hz


if __name__ == '__main__':
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    policy_path = ISAACLAB_NUCLEUS_DIR + "/Policies/ANYmal-C/HeightScan/policy.pt"
    # check
    if not check_file_path(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    file_bytes = read_file(policy_path)
    # jit load the policy
    policy = torch.jit.load(file_bytes).to(env.device).eval()

    # simulate physics
    count = 0
    obs, _ = env.reset()
    while sim_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 100)
                print("[INFO] Resetting...")

            action = policy(obs["policy"])
            # step env
            obs, _ = env.step(action)
            count += 1
    env.close()
    sim_app.close()