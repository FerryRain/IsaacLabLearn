import argparse


def create_applauncher(parser: argparse.ArgumentParser):
    """
    Create an Isaac Lab applauncher
    :param parser: The argument parser of the applauncher and objects
    :return: args_cli: args of applauncher
             applauncher: Isaac Lab applauncher
             simulator: Isaac Lab simulator
    """
    from omni.isaac.lab.app import AppLauncher

    # append Applauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse arg
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    return args_cli, app_launcher, simulation_app

def create_simulator(args_cli, dt=0.01):
    """
    Create an Isaac Lab simulator
    :param args_cli:
    :param dt: time in Simulatior
    :return:
    """
    import omni.isaac.lab.sim as sim_utils


    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-.5, .0, .5])
    return sim

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
        size= (.5, .5, .5),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.75, .75, .75)),
    )
    cfg_cuboid.func("/World/cuboid", cfg_cuboid, translation=(0, 0, 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch an AppLauncher")
    parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
    # SimulationApp arguments https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html?highlight=simulationapp#omni.isaac.kit.SimulationApp
    parser.add_argument(
        "--width", type=int, default=1280, help="Width of viewport and generated image. Default is 1280."
    )
    parser.add_argument("--height", type=int, default=720,
                        help="Height of viewport and generated image. Default is 720")

    from omni.isaac.lab.app import AppLauncher

    args_cli, app_launcher, simulation_app = create_applauncher(parser)

    import omni.isaac.lab.sim as sim_utils
    sim = create_simulator(args_cli)

    design_scene()
    # Play!
    sim.reset()

    while simulation_app.is_running():
        sim.step()

    simulation_app.close()

