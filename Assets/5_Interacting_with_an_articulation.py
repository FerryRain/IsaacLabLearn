from Basic.Basic_utils import create_applauncher, create_simulator
import argparse

parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")

args_cli, app_launcher, simulation_app = create_applauncher(parser)

"""Reset everything folows."""
import torch
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation

sim = create_simulator(args_cli)

"""Pre-defined configs"""
from omni.isaac.lab_assets import CARTPOLE_CFG # isort:skip

def design_scene() -> tuple[dict, list[list[float]]]:
    """ Design scene """
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/GroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(.75, .75, .75))
    cfg.func("/World/DomeLightCfg", cfg)

    # Create separate groups called "Origin1 Origin2"
    # Each group will have a robot in it.
    origins = [[.0, .0, .0],
               [-1.0, .0, .0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Ariculation
    cartple_cfg = CARTPOLE_CFG.copy()
    cartple_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartple_cfg)

    # return the scene information
    scene_entities = {"cartpole": cartpole}
    return scene_entities, origins

def run_simulation(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor) -> None:
    """ Run simulation loop """
    # Extract scene entities
    robot = entities["cartpole"]
    # Defube simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            #  reset counter
            count = 0
            # reset the scene entities
            # root state
            root_state =robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)

            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clean internal buffers
            robot.reset()
            print("[INFO]: Reseting robot state...")
        # Apply random action
        # --generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        count += 1
        robot.update(sim_dt)

if __name__ == '__main__':
    scene_entities, origins = design_scene()
    scene_origins = torch.tensor(origins, device=args_cli.device)

    # Play
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulation(sim, scene_entities, scene_origins)

    simulation_app.close()