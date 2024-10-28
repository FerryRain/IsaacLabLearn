from Basic.Basic_utils import create_applauncher, create_simulator
import argparse

parser = argparse.ArgumentParser(description="Tutorial on spawning and interection with rigid objects")
args_cli, app_launcher, simulation_app = create_applauncher(parser)

""" Reset everything follows """
import torch

import omni.isaac.core.utils.prims as prims_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg

def design_scene():
    """ Design the scene """
    # ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Light
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(.8, .8, .8))
    cfg.func("/World/defaultLightCfg", cfg)

    # Create separate groups called "Origin1 Origin2 Origin3"
    # each group will have robot in it
    origins= [[.25, .25, .0],
              [-.25, .25, .0],
              [.25, -.25, .0],
              [-.25, -.25, .0]]

    for i, origin in enumerate(origins):
        prims_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Ragid Objects
    cone_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.0, 1.0, .0))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cone_object = RigidObject(cfg=cone_cfg)

    # return the scene information
    scene_entilies = {"cone": cone_object}
    return scene_entilies, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """ Run the simulation """
    # Extract scene entilies
    cone_object = entities["cone"]
    sim_dt = sim.get_physics_dt()
    sim_time = .0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counters
            sim_time = .0
            count = 0
            # reset robot state
            root_state = cone_object.data.default_root_state.clone()
            # sample a random position on a cylinder around the origins
            root_state[:, :3] += origins
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=.1, h_range=(.25, .5), size=cone_object.num_instances, device=cone_object.device
            )
            # write root state to simulation
            cone_object.write_root_state_to_sim(root_state)
            # reset buffers
            cone_object.reset()
            print("------------------------------------")
            print("[INFO]: Resetting object state...")
        # apply sim data
        cone_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time = sim_time + sim_dt
        count += 1
        # update buffer
        cone_object.update(sim_dt)
        # print robot position
        if count % 50 == 0:
            print(f"Root position (in world): {cone_object.data.root_state_w[:, :3]}")

if __name__ == '__main__':
    sim = create_simulator(args_cli)

    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play
    sim.reset()

    run_simulator(sim, scene_entities, scene_origins)

    simulation_app.close()
