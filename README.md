# IsaacLabLearn

Learn to Use IsaacLab and add some common utils for convenient using.

And then use command `pip install -e .` in the directory `./RL_Policy` to install the packet to register the environment.
`import Create_a_Direct_Workflow_RL_Env`
### Basic
 - 1_create_empty_world.py: How to create an Isaac sim launcher and simulation. 
 - 2_spawn_prims.py: How to generate different objects in the simulation.
 - 3_Learn_About_Applauncher.py: How to use argparse to setup simulation and applauncher
 - Basic_utils.py: The utils about quickly build Isaac Lab applauncher and simulation.
### Assets
 - 4_Interacting_with_rigid_objects.py: How to interact with rigid objects.
 - 5_Interacting_with_an_articulation.py: How to interact with articulations(cartpole env as the example).
 - 6_Interacting_with_deformable_object.py: How to interact with deformable objects. (Simulation of deformable objects can only with gpu)
### Scene_and_Envs
 - 7_Create_an_interacting_scene.py: How to Create a scene contain ground, light and objects.
 - 8_Create_a_Manager_Based_env.py: How to create a scene used manager scene class for cartpole env.
 - 9_Create_a_Manager_Env_for_Quadruped.py: Like tutorial 8, but change the env to quadruped.
### RL_Policy (python packet)
 - **config**
   - extension.toml
 - **Create_a_Direct_Workflow_RL_Env**
   - **cartpole**
     - **agent**
       - \_\_init\_\_.py: Setup the RL algorithm.
       - rl_games_ppo_cfg.yaml
     - \_\_init\_\_.py: Setup the cartpole balancing environment.
     - direct_rl_scene_cartpole_env.py: A more usefully env for RL scene create by direct RL env scene
   - **shadowhands**
      - **agent**
        - \_\_init\_\_.py: Setup the RL algorithm.
        - rl_games_ppo_cfg.yaml
      - \_\_init\_\_.py: Setup the Shadowhands environment.
      - direct_rl_scene_Shadowhands_env.py: A more usefully env for RL scene create by direct RL env scene
   - **utils**: tools about rl environment.
   - \_\_init\_\_.py: Create a python packet.
   - train_by_rl_games.py: train the RL agent for direct_cartpole_env
 - **Create_a_Manager_based_RL_Env**
   - 10.1 cartpole_env_cfg.py: Create a Manager RL Scene cfg class for cartpole env to train RL policy
   - 10.2 run_cartpole_rl_env: Run the sample action to the created cfg env.
 - pyproject.toml
 - setup.py
 - README.md: How to register a gymnasium environment.


---
# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)


**Isaac Lab** is a unified and modular framework for robot learning that aims to simplify common workflows
in robotics research (such as RL, learning from demonstrations, and motion planning). It is built upon
[NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) to leverage the latest
simulation capabilities for photo-realistic scenes and fast and accurate simulation.

Please refer to our [documentation page](https://isaac-sim.github.io/IsaacLab) to learn more about the
installation steps, features, tutorials, and how to set up your project with Isaac Lab.

## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
