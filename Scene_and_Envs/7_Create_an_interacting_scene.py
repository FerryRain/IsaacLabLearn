from Basic.Basic_utils import create_applauncher, create_simulator
import argparse

parser = argparse.ArgumentParser(description="Tutorial on creating an interacting scene.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments.")
args_cli, app_launcher, Si_app = create_applauncher(parser)

""" Reset everything follows. """
import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass

Si = create_simulator(args_cli)
""" Pre-defined configs """

from omni.isaac.lab_assets import CARTPOLE_CFG

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """ Configuration for a cart-pole scene """
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(.75, .75, .75))
    )

    # articulation
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
# robot = Articulation(CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NX}/Robot"))

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """ Run the simulation loop """
    # Extract scene entities
    robot = scene["cartpole"]
    # Define simulation stepping
    sim_dt = Si.get_physics_dt()
    count = 0
    while Si_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)

            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffer
            scene.reset()
        # Apply random action
        # --generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to robots
        robot.set_joint_effort_target(efforts)
        # write data into sim
        scene.write_data_to_sim()
        sim.step()
        count +=1
        scene.update(sim_dt)

if __name__ == '__main__':
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    Si.reset()
    print("[INFO]: Setup complete...")
    run_simulator(Si, scene)

    Si_app.close()