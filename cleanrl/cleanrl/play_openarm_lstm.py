"""Inference script for ppo_openarm_lstm trained agent."""

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run trained OpenArm LSTM policy.")
parser.add_argument("--checkpoint", type=str, default="/home/neubility-sim/isaac_ws/DEXTRAH_CAM/dextrah_lab/cleanrl/runs/ppo_openarm_lstm_step150732800.pth", help="Path to .pth checkpoint file.")
#parser.add_argument("--checkpoint", type=str, default="/home/neubility-sim/isaac_ws/DEXTRAH_CAM/dextrah_lab/cleanrl/runs/Openarm__ppo_openarm_lstm__1__1776147867/ppo_openarm_lstm_step488243200.pth", help="Path to .pth checkpoint file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=10000000, help="Steps to run per env.")
parser.add_argument("--task", type=str, default="test_Openarm", help="Task name.")
parser.add_argument("--deterministic", action="store_true", default=False, help="Use mean action (no sampling).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--cuda", type=bool, default=True)

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



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def _split_obs(states, img_h, img_w, proprio_dim):
    """Split flat states into proprio and depth.

    states: (B, 48 + 160*120) -> proprio (B, 48), depth (B, 1, 160, 120)
    """
    proprio = states[:, :proprio_dim]
    head_depth = states[:, proprio_dim:proprio_dim + img_h * img_w].reshape(-1, 1, img_h, img_w)
    wrist_L_depth = states[:, proprio_dim + img_h * img_w:proprio_dim + 2 * img_h * img_w].reshape(-1, 1, img_h, img_w)
    return proprio, head_depth, wrist_L_depth

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.img_w, self.img_h = int(envs.cfg.head_img_width), int(envs.cfg.head_img_height)
        self.proprio_dim = envs.cfg.num_observations

        # Separate CNNs for head and wrist
        self.head_cnn, self.head_lns, self.head_pool = self._make_cnn()
        self.wrist_cnn, self.wrist_lns, self.wrist_pool = self._make_cnn()

        self.head_fc = nn.Linear(256, 32)
        self.wrist_fc = nn.Linear(256, 32)

        self.lstm = nn.LSTM(32 + 32 + self.proprio_dim, 1024, num_layers=2)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ELU(),
            nn.Linear(512,  512), nn.ELU(),
            nn.Linear(512,  256), nn.ELU(),
        )

        self.actor_mean = layer_init(nn.Linear(256, envs.cfg.num_actions), std=0.1)
        self.actor_logstd = nn.Parameter(torch.zeros(1, envs.cfg.num_actions))
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)

    def _make_cnn(self):
        convs = nn.ModuleList([
            nn.Conv2d(1,  16, kernel_size=8, stride=4),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
        ])
        lns = nn.ModuleList()
        x = torch.zeros(1, 1, self.img_h, self.img_w)
        for conv in convs:
            x = conv(x)
            lns.append(nn.LayerNorm(x.shape[1:]))
        pool = nn.AdaptiveAvgPool2d((1, 1))
        return convs, lns, pool

    def _cnn_forward(self, x, convs, lns, pool, fc):
        for conv, ln in zip(convs, lns):
            x = F.relu(ln(conv(x)))
        
        x = x.flatten(1)
        x = fc(x)

        return x

    def get_states(self, x, lstm_state, done):
        
        proprio, head_depth, wrist_L_depth = _split_obs(x, self.img_h, self.img_w, self.proprio_dim)

        # import cv2
        # img = wrist_L_depth[0, 0].detach().cpu().numpy()
        # # normalize to 0~255
        # img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        # img_uint8 = (img_norm * 255).astype("uint8")
        # cv2.imwrite("depth_debug.png", img_uint8)

        head_cnn_out = self._cnn_forward(head_depth/2., self.head_cnn, self.head_lns, self.head_pool, self.head_fc)
        wrist_cnn_out = self._cnn_forward(wrist_L_depth/2., self.wrist_cnn, self.wrist_lns, self.wrist_pool, self.wrist_fc)
        hidden = torch.cat([head_cnn_out, wrist_cnn_out, proprio], dim=-1)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.float().reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        hidden = self.mlp(hidden)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        hidden = self.mlp(hidden)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden), lstm_state

    def get_action_mu_sigma(self, x, lstm_state, done):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        hidden = self.mlp(hidden)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()

        return action, action_mean, action_std, lstm_state


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    device_str = "cuda" if torch.cuda.is_available() and args_cli.cuda else "cpu"
    device = torch.device(device_str)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device_str
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
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "agent" in ckpt:
        agent.load_state_dict(ckpt["agent"])
    else:
        agent.load_state_dict(ckpt)
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
    for _ in range(args_cli.num_steps):
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

        # finished = done.bool()
        # if finished.any():
        #     mean_ep_r = episode_rewards[finished].mean().item()
        #     total_reward += mean_ep_r
        #     episode_count += 1
        #     writer.add_scalar("play/episode_reward", mean_ep_r, step)
        #     print(f"[step {step:6d}] episode_reward={mean_ep_r:.3f}")
        #     episode_rewards[finished] = 0.0

    # if episode_count > 0:
    #     print(f"\n[DONE] Episodes: {episode_count}  |  Mean reward: {total_reward / episode_count:.3f}")
    # else:
    #     print(f"\n[DONE] No full episodes completed in {args_cli.num_steps} steps.")

    writer.close()
    envs.close()


if __name__ == "__main__":
    main()
    simulation_app.close()