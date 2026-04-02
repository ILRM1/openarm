# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass

import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import torch
import numpy as np
from isaaclab.app import AppLauncher

from cleanrl_utils.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    headless: bool = False
    """Whether to run the simulation in headless mode."""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Openarm"
    """the id of the environment"""
    total_timesteps: int = 100000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.002
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    save_model: bool = True
    """whether to save the model at the end of training"""
    save_interval: int = 200
    """save model every N updates (0 = only at the end)"""
    video: bool = True
    """whether to record videos during training"""
    video_length: int = 1000
    """length of the recorded video (in steps)"""
    video_interval: int = 20000
    """interval between video recordings (in steps)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 1)
        return env

    return thunk


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

        self.img_w, self.img_h = int(envs.cfg.head_img_width/2), int(envs.cfg.head_img_height/2)
        self.proprio_dim = envs.cfg.num_observations

        # Separate CNNs for head and wrist
        self.head_cnn, self.head_lns, self.head_pool = self._make_cnn()
        self.wrist_cnn, self.wrist_lns, self.wrist_pool = self._make_cnn()

        self.lstm = nn.LSTM(128 + 128 + self.proprio_dim, 1024, num_layers=2)
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

        self.actor_mean = layer_init(nn.Linear(256, 7), std=0.1)
        self.actor_logstd = nn.Parameter(torch.zeros(1, 7))
        self.critic = layer_init(nn.Linear(256, 1), std=1.0)

    def _make_cnn(self):
        convs = nn.ModuleList([
            nn.Conv2d(1,  16, kernel_size=8, stride=4),
            nn.Conv2d(16, 32, kernel_size=6, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
        ])
        lns = nn.ModuleList()
        x = torch.zeros(1, 1, self.img_h, self.img_w)
        for conv in convs:
            x = conv(x)
            lns.append(nn.LayerNorm(x.shape[1:]))
        pool = nn.AdaptiveAvgPool2d((1, 1))
        return convs, lns, pool

    def _cnn_forward(self, x, convs, lns, pool):
        for conv, ln in zip(convs, lns):
            x = F.relu(ln(conv(x)))
        x = pool(x)
        x = x.flatten(1)
        return x

    def get_states(self, x, lstm_state, done):
        
        proprio, head_depth, wrist_L_depth = _split_obs(x, self.img_h, self.img_w, self.proprio_dim)

        # import cv2
        # img = wrist_L_depth[0, 0].detach().cpu().numpy()
        # # normalize to 0~255
        # img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        # img_uint8 = (img_norm * 255).astype("uint8")
        # cv2.imwrite("depth_debug.png", img_uint8)

        head_cnn_out = self._cnn_forward(head_depth/2., self.head_cnn, self.head_lns, self.head_pool)
        wrist_cnn_out = self._cnn_forward(wrist_L_depth/2., self.wrist_cnn, self.wrist_lns, self.wrist_pool)
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
    

args = tyro.cli(Args)

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=args.num_envs, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=args.num_steps, help="Number of steps to run each environment.")
parser.add_argument("--task", type=str, default=args.env_id, help="Name of the task.")
parser.add_argument("--seed", type=int, default=args.seed, help="Seed used for the environment")

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
args_cli.enable_cameras = True
args_cli.headless =args.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import dextrah_lab.tasks

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    
    # append AppLauncher cli args
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    # )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.use_depth_teacher = True

    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        video_kwargs = {
            "video_folder": f"runs/{run_name}/videos",
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        envs = gym.wrappers.RecordVideo(envs, **video_kwargs)

    agent = Agent(envs.unwrapped).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    observation_space = (envs.unwrapped.cfg.num_observations
        + int(envs.unwrapped.cfg.head_img_width/2) * int(envs.unwrapped.cfg.head_img_height/2)
        + int(envs.unwrapped.cfg.wrist_img_width/2) * int(envs.unwrapped.cfg.wrist_img_height/2))

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (observation_space,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + (env_cfg.num_actions,)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    record_reward = 0.
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
           
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = (terminations | truncations).float()
            rewards[step] = reward 

            record_reward += reward[0]                
            if next_done[0]:
                writer.add_scalar("rewards", record_reward, global_step)
                record_reward = 0


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (observation_space,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (env_cfg.num_actions,))
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.save_model and args.save_interval > 0 and iteration % args.save_interval == 0:
            ckpt_path = f"runs/{run_name}/{args.exp_name}_step{global_step}.pth"
            torch.save(agent.state_dict(), ckpt_path)
            print(f"[INFO] Checkpoint saved: {ckpt_path}")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pth"
        torch.save(agent.state_dict(), model_path)
        print(f"[INFO] Final model saved: {model_path}")

    envs.close()
    writer.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
