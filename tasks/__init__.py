"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages
import gymnasium as gym

# from .dextrah_kuka_allegro import agents as dextrah_agents
# from .dextrah_kuka_allegro.dextrah_kuka_allegro_env import DextrahKukaAllegroEnv
# from .dextrah_kuka_allegro.dextrah_kuka_allegro_env_cfg import DextrahKukaAllegroEnvCfg

from .openarm import agents as openarm_agents
from .openarm.openarm_env import OpenarmEnv
from .openarm.openarm_env_cfg import OpenarmEnvCfg

##
# Register Gym environments.
##

# gym.register(
#     id="Dextrah-Kuka-Allegro",
#     entry_point="dextrah_lab.tasks.dextrah_kuka_allegro.dextrah_kuka_allegro_env:DextrahKukaAllegroEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": DextrahKukaAllegroEnvCfg,
#         #"rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rl_games_cfg_entry_point": f"{dextrah_agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
#         # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_scratch_cnn_aux.yaml",
#         #"rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ShadowHandPPORunnerCfg,
#     },
# )

gym.register(
    id="Openarm",
    entry_point="dextrah_lab.tasks.openarm.openarm_env:OpenarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenarmEnvCfg,
        #"rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{openarm_agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_scratch_cnn_aux.yaml",
        #"rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ShadowHandPPORunnerCfg,
    },
)

gym.register(
    id="Openarm_ik",
    entry_point="dextrah_lab.tasks.openarm.openarm_ik_env:OpenarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenarmEnvCfg,
        #"rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{openarm_agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_scratch_cnn_aux.yaml",
        #"rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ShadowHandPPORunnerCfg,
    },
)

gym.register(
    id="Openarm-Depth",
    entry_point="dextrah_lab.tasks.openarm.openarm_env:OpenarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenarmEnvCfg,
        "rl_games_cfg_entry_point": f"{openarm_agents.__name__}:rl_games_ppo_depth_lstm_cfg.yaml",
    },
)

gym.register(
    id="Openarm-Depth-skrl",
    entry_point="dextrah_lab.tasks.openarm.openarm_env:OpenarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenarmEnvCfg,
        "skrl_cfg_entry_point": f"{openarm_agents.__name__}:skrl_ppo_cfg.yaml",
    },
)