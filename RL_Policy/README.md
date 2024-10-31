# How to register a gymnasium environment.

Build the files directory as the example below.

- Packet_root_directory
  - **config**
    - extension.toml
  - **packet_name**
    - **sub_packet_name**
      - **agent**
        - rl frameworks yaml files
        - \_\_init\_\_.py
      - env.py
      - \_\_init\_\_.py
    - **utils**
      - The RL frameworks utils
    - \_\_init\_\_.py
  - pyproject.toml
  - setup.py

And then use command `pip install -e .` to install the packet.  
For use the gymnasium env, you need to import the root packet to register all of the env automatically like 
`import Create_a_Direct_Workflow_RL_Env`

### For setup.py file
Build like below template.


```python
import itertools
import os
import platform
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch==2.4.0",
    "torchvision>=0.14.1",  # ensure compatibility with torch 1.13.1
    # 5.26.0 introduced a breaking change, so we restricted it for now.
    # See issue https://github.com/tensorflow/tensorboard/issues/6808 for details.
    "protobuf>=3.20.2, < 5.0.0",
    # configuration management
    "hydra-core",
    # data collection
    "h5py",
    # basic logger
    "tensorboard",
    # video recording
    "moviepy",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Extra dependencies for RL agents
EXTRAS_REQUIRE = {
    "sb3": ["stable-baselines3>=2.1"],
    "rl-games": ["rl-games==1.6.1", "gym"],  # rl-games still needs gym :(
    "robomimic": [],
} #REQUIRED PACKETS

# Add the names with hyphens as aliases for convenience
EXTRAS_REQUIRE["rl_games"] = EXTRAS_REQUIRE["rl-games"]
# EXTRAS_REQUIRE["rsl_rl"] = EXTRAS_REQUIRE["rsl-rl"]

# Check if the platform is Linux and add the dependency
if platform.system() == "Linux":
    EXTRAS_REQUIRE["robomimic"].append("robomimic@git+https://github.com/ARISE-Initiative/robomimic.git")

# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))


# Installation operation
setup(
    name="RL-Policy",
    author="Ferry",
    maintainer="ShanghaiTech University RIMLAB",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    extras_require=EXTRAS_REQUIRE,
    packages=["Create_a_Direct_Workflow_RL_Env"], # Packets root directory
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.2.0",
        "Isaac Sim :: 4.1.0",
    ],
    zip_safe=False,
)
```

The __init__.py file in root _Create_a_Direct_Workflow_RL_Env_ directory
```python
import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

from .utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
```

The __init__.py file in subdirectory
```python
""" Cartpole balancing environment. """

import gymnasium as gym

from . import agent as agents
from .direct_rl_scene_cartpole_env import CartpoleEnv, CartpoleEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Cartpole-Direct-Mine-v0",
    entry_point="Create_a_Direct_Workflow_RL_Env.cartpole:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

```