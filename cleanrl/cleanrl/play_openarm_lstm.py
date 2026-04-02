"""Inference script for ppo_openarm_lstm trained agent."""

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run trained OpenArm LSTM policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=1000, help="Steps to run per env.")
parser.add_argument("--task", type=str, default="Openarm", help="Task name.")
parser.add_argument("--deterministic", action="store_true", default=True, help="Use mean action (no sampling).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1000)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dextrah_lab.tasks

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg, DirectMARLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from ppo_openarm_lstm import Agent, _split_obs


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = str(device)
    env_cfg.use_depth_teacher = True

    run_name = f"play__{args_cli.task}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        envs = gym.wrappers.RecordVideo(
            envs,
            video_folder=f"runs/{run_name}/videos",
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    agent = Agent(envs.unwrapped).to(device)
    agent.load_state_dict(torch.load(args_cli.checkpoint, map_location=device))
    agent.eval()
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    lstm_state = (
        torch.zeros(agent.lstm.num_layers, args_cli.num_envs, agent.lstm.hidden_size, device=device),
        torch.zeros(agent.lstm.num_layers, args_cli.num_envs, agent.lstm.hidden_size, device=device),
    )

    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    done = torch.zeros(args_cli.num_envs, device=device)

    episode_rewards = torch.zeros(args_cli.num_envs, device=device)
    episode_count = 0
    total_reward = 0.0

    print(f"[INFO] Running {args_cli.num_steps} steps ...")
    for step in range(args_cli.num_steps):
        with torch.no_grad():
            if args_cli.deterministic:
                _, action, _, lstm_state = agent.get_action_mu_sigma(obs, lstm_state, done)
            else:
                action, _, _, _, lstm_state = agent.get_action_and_value(obs, lstm_state, done)

        obs, reward, terminations, truncations, infos = envs.step(action)
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        done = torch.tensor((terminations | truncations), dtype=torch.float32, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)

        episode_rewards += reward

        finished = done.bool()
        if finished.any():
            mean_ep_r = episode_rewards[finished].mean().item()
            total_reward += mean_ep_r
            episode_count += 1
            writer.add_scalar("play/episode_reward", mean_ep_r, step)
            print(f"[step {step:6d}] episode_reward={mean_ep_r:.3f}")
            episode_rewards[finished] = 0.0

    if episode_count > 0:
        print(f"\n[DONE] Episodes: {episode_count}  |  Mean reward: {total_reward / episode_count:.3f}")
    else:
        print(f"\n[DONE] No full episodes completed in {args_cli.num_steps} steps.")

    writer.close()
    envs.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
