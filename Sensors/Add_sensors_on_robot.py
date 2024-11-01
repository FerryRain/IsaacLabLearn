"""
@FileName：Add_sensors_on_robot.py
@Description：How to add sensors on robot
@Author：Ferry
@Time：2024/10/31 下午10:52
@Copyright：©2024-2024 ShanghaiTech University-RIMLAB
"""
import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser =argparse.ArgumentParser(description=" Tutorial on adding sensors on robot. ")
parser.add_argument("--num_envs", type=int, default=2, help="How many envs to add")
from Basic.Basic_utils import create_applauncher,create_simulator
args_cli, app_laucher, sim_app = create_applauncher(parser)

import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass

""" Pre-defined configs """

from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG

@configclass
class SensorsCfg(InteractiveSceneCfg):
    """ Design the scene with sensors on the robot. """
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/plane",spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/dome",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(.75, .75, .75)),
    )

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_Camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(.510, .0, .015), rot=(.5, .5, -.5, -.5), convention="ros"),
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/plane"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """ Run the simulation. """
    sim_dt = sim.get_physics_dt()
    sim_time = .0
    count = 0

    while sim_app.is_running():
        # Reset
        if count % 500 == 0:
            count = 0
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_state_to_sim(root_state)

            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print("[INFO]: Resetting robot state...")

        # Apply default actions to the robot
        # --generate actions/commands
        target = scene["robot"].data.default_joint_pos
        scene["robot"].set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        print(scene["height_scanner"])
        print("Received max height value: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        print("-------------------------------")
        print(scene["contact_forces"])
        print("Received max contact force of: ", torch.max(scene["contact_forces"].data.net_forces_w).item())

if __name__ == '__main__':
    sim = create_simulator(args_cli)
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)
    sim_app.close()