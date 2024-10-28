from Basic.Basic_utils import create_applauncher, create_simulator
import argparse

parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object.")

args_cli, app_launcher, simulation_app = create_applauncher(parser)

""" Reset everything follows. """
import torch
import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import DeformableObject, DeformableObjectCfg

sim = create_simulator(args_cli)

def design_scene():
    """ Design the scene. """
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/GroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000, color=(.8, .8, .8))
    cfg.func("/World/DomeLightCfg", cfg)

    # Create separate groups called "Origin1 Origin2...."
    origins = [[.25, .25, .0],
               [-.25, .25, .0],
               [.25, -.25, .0],
               [-.25, -.25, .0],]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Deformable Object
    cfg = DeformableObjectCfg(
        prim_path = "/World/Origin.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(.25, .25, .25),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.8, .8, .8)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(.0, .0, 1.0)),
        debug_vis=True,
    )
    cube_object = DeformableObject(cfg=cfg)

    # return the scene information
    scene_entities = {"cube_object": cube_object}
    return scene_entities, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject], origins: torch.Tensor):
    """ Run the simulation with the given entities. """
    # Extract scene entities
    cube_object = entities["cube_object"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Nodal kinematic targets of the deformable bodies
    nodal_kinematic_target = cube_object.data.nodal_kinematic_target.clone()

    # Simulation physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            sim_time = .0
            count = 0

            # reset the nodal state of the object
            nodal_state = cube_object.data.default_nodal_state_w.clone()
            # apply random pose to the object
            pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins
            quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

            # write kinematic target to nodal state and free all vertices
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            # reset buffers
            cube_object.reset()

            print("--------------------------------------")
            print("[INFO]: Resetting object state...")

        # Update the kinematic target for cubes at index 0 and index 3
        # Slightly move the cube in the z-direction by picking the vertex at index 0
        nodal_kinematic_target[[0, 3], 0, 2] += 0.001
        # Set vertex at index 0 to be kinematically constrained
        # 0: constrained, 1: free
        nodal_kinematic_target[[0, 3], 0, 3] = 0.0
        # write kinematic target to simulation
        cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

        # write internal data to simulation
        cube_object.write_data_to_sim()

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1

        # update buffers
        cube_object.update(sim_dt)
        #print the root position
        if count % 50 == 0:
            print(f"Root position (in world):{cube_object.data.root_pos_w[:, :3]}")


if __name__ == '__main__':
    scene_entities, origins = design_scene()
    scene_origins = torch.tensor(origins, device=sim.device)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)

    simulation_app.close()