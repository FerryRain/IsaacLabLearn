import argparse

from omni.isaac.lab.app import AppLauncher

# Create argparser
parser = argparse.ArgumentParser(description="Launch an AppLauncher")
parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
# SimulationApp arguments https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html?highlight=simulationapp#omni.isaac.kit.SimulationApp
parser.add_argument(
    "--width", type=int, default=1280, help="Width of viewport and generated image. Default is 1280."
)
parser.add_argument("--height", type=int, default=720, help="Height of viewport and generated image. Default is 720")

# append Applauncher cli args
AppLauncher.add_app_launcher_args(parser)
#parse arg
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

""" Reset everything follows"""
import omni.isaac.lab.sim as sim_utils

def design_scene():
    """ Design scene (ground plane, light, objects and mesh from urd files)"""
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Spawn distant light
    cfg_light = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color = (.75, .75, .75),
    )
    cfg_light.func("/World/light", cfg_light, translation=(1, 0, 10))

    # spawn a cuboid
    cfg_cuboid = sim_utils.CuboidCfg(
        size=[args_cli.size] * 3,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.75, .75, .75)),
    )
    cfg_cuboid.func("/World/cuboid", cfg_cuboid, translation=(0, 0, args_cli.size /2))

if __name__ == '__main__':
    """ Main function"""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [.0, .0, .0])

    # Design scene by adding assets to it
    design_scene()

    # Play!
    sim.reset()

    while simulation_app.is_running():
        sim.step()

    simulation_app.close()