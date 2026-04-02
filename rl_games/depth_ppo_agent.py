"""
Custom PPO agent for depth-based RL training with RL-Games.

Extends the standard A2CAgent to:
1. Capture depth images from the env camera after each step
2. Store downscaled depth in a parallel buffer
3. Re-encode depth with gradients during PPO mini-batch training
   so the CNN encoder receives gradient signal
"""

import time

import torch
import torch.nn.functional as F

from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch import torch_ext
from rl_games.common.a2c_common import swap_and_flatten01


class DepthA2CAgent(A2CAgent):
    """A2C agent with depth image processing for end-to-end visuomotor RL."""

    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        # Get depth resolution from network config
        net_params = params['network']
        self.depth_height = net_params.get('depth_height', 120)
        self.depth_width = net_params.get('depth_width', 160)

        # Reference to the underlying Isaac Lab env for camera access
        self._ov_env = None

    @property
    def ov_env(self):
        """Lazily resolve the underlying OpenArm env."""
        if self._ov_env is None:
            env = self.vec_env.env
            # Navigate through wrappers to get the actual env
            while hasattr(env, 'env'):
                env = env.env
            if hasattr(env, 'unwrapped'):
                env = env.unwrapped
            self._ov_env = env
        return self._ov_env

    def init_tensors(self):
        super().init_tensors()
        # Allocate depth rollout buffer: (horizon, num_envs, 1, H, W)
        self.depth_buffer = torch.zeros(
            self.horizon_length, self.num_actors, 1,
            self.depth_height, self.depth_width,
            device=self.ppo_device,
        )

    def _get_depth_from_env(self):
        """Read preprocessed depth from env (already processed in _get_observations).

        Returns:
            (num_envs, 1, depth_height, depth_width) tensor
        """
        depth = self.ov_env._latest_depth  # (N, 1, H, W), already preprocessed

        # Downscale to target resolution if needed
        if depth.shape[-2:] != (self.depth_height, self.depth_width):
            depth = F.interpolate(
                depth, size=(self.depth_height, self.depth_width),
                mode='bilinear', align_corners=False,
            )
        return depth

    def get_action_values(self, obs):
        """Override to pass depth through input_dict to network forward."""
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()

        # Capture depth - will be passed through forward() which calls encode_depth internally
        self._current_depth = self._get_depth_from_env()

        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'depth': self._current_depth,
            'rnn_states': self.rnn_states,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {'is_train': False, 'states': states}
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        return res_dict

    def get_values(self, obs):
        """Override to include depth features for value estimation."""
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {'is_train': False, 'states': states}
                value = self.get_central_value(input_dict)
                return value
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])

                input_dict = {
                    'is_train': False,
                    'prev_actions': None,
                    'obs': processed_obs,
                    'depth': self._get_depth_from_env(),
                    'rnn_states': self.rnn_states,
                }
                result = self.model(input_dict)
                value = result['values']
                return value

    def play_steps_rnn(self):
        """Override to capture and store depth at each step."""
        update_list = self.update_list
        mb_rnn_states = self.mb_rnn_states
        step_time = 0.0

        for n in range(self.horizon_length):
            if n % self.seq_length == 0:
                for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                    mb_s[n // self.seq_length, :, :, :] = s

            if self.has_central_value:
                self.central_value_net.pre_step_rnn(n)

            # get_action_values also captures self._current_depth
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones.byte())

            # Store depth for this step (captured during get_action_values)
            self.depth_buffer[n] = self._current_depth

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.perf_counter()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.perf_counter()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            if len(all_done_indices) > 0:
                if self.zero_rnn_on_done:
                    for s in self.rnn_states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_central_value:
                    self.central_value_net.post_step_rnn(all_done_indices)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()

        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)

        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        # Flatten depth buffer: (horizon, num_envs, 1, H, W) -> (horizon*num_envs, 1, H, W)
        batch_dict['depth'] = swap_and_flatten01(self.depth_buffer)

        states = []
        for mb_s in mb_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))

        batch_dict['rnn_states'] = states
        batch_dict['step_time'] = step_time

        return batch_dict

    def prepare_dataset(self, batch_dict):
        """Override to include depth in the training dataset."""
        super().prepare_dataset(batch_dict)

        # Add depth to the dataset so it's available in mini-batches
        if 'depth' in batch_dict:
            self.dataset.values_dict['depth'] = batch_dict['depth']

    def calc_gradients(self, input_dict):
        """Override to re-encode depth WITH gradients for CNN training."""
        from rl_games.common import common_losses

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        # Pass depth through forward() which calls encode_depth WITH gradients
        if 'depth' in input_dict:
            batch_dict['depth'] = input_dict['depth']

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(
                old_action_log_probs_batch, action_log_probs,
                advantage, self.ppo, curr_e_clip,
            )

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(
                    self.model, value_preds_batch, values,
                    curr_e_clip, return_batch, self.clip_value,
                )
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)],
                rnn_masks,
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = (
                a_loss
                + 0.5 * c_loss * self.critic_coef
                - entropy * self.entropy_coef
                + b_loss * self.bounds_loss_coef
            )

            aux_loss = None
            self.aux_loss_dict = {}
            if aux_loss is not None:
                for k, v in aux_loss.items():
                    loss += v
                    if k in self.aux_loss_dict:
                        self.aux_loss_dict[k] = v.detach()
                    else:
                        self.aux_loss_dict[k] = [v.detach()]

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(),
                old_mu_batch, old_sigma_batch, reduce_kl,
            )
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(self, {
            'values': value_preds_batch,
            'returns': return_batch,
            'new_neglogp': action_log_probs,
            'old_neglogp': old_action_log_probs_batch,
            'masks': rnn_masks,
        }, curr_e_clip, 0)

        self.train_result = (
            a_loss, c_loss, entropy,
            kl_dist, self.last_lr, lr_mul,
            mu.detach(), sigma.detach(), b_loss,
        )