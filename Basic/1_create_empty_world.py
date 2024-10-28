import argparse
from omni.isaac.lab.app import  AppLauncher


def start_Simulation(parser: argparse.ArgumentParser):
    from omni.isaac.lab.app import AppLauncher

    AppLauncher().add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    return simulation_app

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.lab.sim import SimulationCfg, SimulationContext

sim_cfg = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([0.0, 0.0, 0.0],[0.0, 0.0, 0.0])



if __name__ == '__main__':
    sim.reset()

    while simulation_app.is_running():
        sim.step()

    simulation_app.close()
