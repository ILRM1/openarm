"""
CNN + MLP + LSTM network for depth-based RL training with RL-Games.

Uses a lightweight CNN encoder to extract features from depth images,
concatenated with proprioceptive observations, then processed by MLP + LSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.d2rl import D2RLNet
from rl_games.common.layers.recurrent import GRUWithDones, LSTMWithDones
from rl_games.common.layers.value import DefaultValue
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import object_factory


CNN_OUT_FEATURES = 32


def _create_initializer(func, **kwargs):
    return lambda v: func(v, **kwargs)


def conv_output_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(pad, int):
        pad = (pad, pad)
    h = (h_w[0] + 2 * pad[0] - dilation * (kernel_size[0] - 1) - 1) // stride[0] + 1
    w = (h_w[1] + 2 * pad[1] - dilation * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return h, w


class DepthCNNEncoder(nn.Module):
    """Lightweight 4-layer CNN encoder for depth images.

    Takes (N, 1, H, W) depth input and outputs (N, CNN_OUT_FEATURES) features.
    Uses AdaptiveAvgPool so it works with any input resolution.
    """

    def __init__(self, input_height, input_width):
        super().__init__()
        h, w = input_height, input_width

        h, w = conv_output_size((h, w), kernel_size=6, stride=2)
        layer1_norm = [16, h, w]
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer2_norm = [32, h, w]
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer3_norm = [64, h, w]
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer4_norm = [128, h, w]

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.LayerNorm(layer1_norm),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LayerNorm(layer2_norm),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LayerNorm(layer3_norm),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.LayerNorm(layer4_norm),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear = nn.Linear(128, CNN_OUT_FEATURES)

    def forward(self, depth_img):
        """Encode depth image to feature vector.

        Args:
            depth_img: (N, 1, H, W) depth tensor
        Returns:
            (N, CNN_OUT_FEATURES) feature tensor
        """
        x = self.cnn(depth_img)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class DepthA2CBuilder:
    """RL-Games compatible network builder for depth-based A2C."""

    def __init__(self, **kwargs):
        pass

    def load(self, params):
        self.params = params

    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            super().__init__()
            self.actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = kwargs.pop('num_seqs', 1)

            self._load_params(params)

            # CNN encoder for depth
            self.depth_height = params.get('depth_height', 120)
            self.depth_width = params.get('depth_width', 160)
            self.depth_encoder = DepthCNNEncoder(self.depth_height, self.depth_width)
            self.depth_running_mean_std = RunningMeanStd((1, self.depth_height, self.depth_width))

            # MLP input = flat obs + CNN features
            mlp_input_size = input_shape[0] + CNN_OUT_FEATURES

            # Build MLP
            self.actor_mlp = self._build_mlp(mlp_input_size)
            out_size = self.units[-1] if self.units else mlp_input_size
            in_mlp_shape = mlp_input_size

            # Build RNN
            if self.has_rnn:
                if self.is_rnn_before_mlp:
                    rnn_in_size = in_mlp_shape
                    in_mlp_shape = self.rnn_units
                    self.actor_mlp = self._build_mlp(in_mlp_shape)
                    out_size = self.units[-1] if self.units else in_mlp_shape
                else:
                    rnn_in_size = out_size
                    if self.rnn_concat_input:
                        rnn_in_size += in_mlp_shape
                    out_size = self.rnn_units

                self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                if self.rnn_ln:
                    self.layer_norm = nn.LayerNorm(self.rnn_units)

            # Value head
            self.value = nn.Linear(out_size, self.value_size)

            # Continuous action heads
            self.mu = nn.Linear(out_size, self.actions_num)
            self.mu_act = self._get_activation(self.space_config['mu_activation'])
            if self.fixed_sigma:
                self.sigma = nn.Parameter(
                    torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.sigma = nn.Linear(out_size, self.actions_num)
            self.sigma_act = self._get_activation(self.space_config['sigma_activation'])

            # Init weights
            self._init_weights(params)

        def _get_activation(self, name):
            activations = {
                'None': nn.Identity(),
                'relu': nn.ReLU(),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
                'elu': nn.ELU(),
                'selu': nn.SELU(),
                'swish': nn.SiLU(),
                'softplus': nn.Softplus(),
            }
            return activations.get(name, nn.Identity())

        def _build_mlp(self, input_size):
            layers = []
            in_size = input_size
            for unit in self.units:
                layers.append(nn.Linear(in_size, unit))
                layers.append(self._get_activation(self.activation))
                in_size = unit
            return nn.Sequential(*layers)

        def _build_rnn(self, name, input_size, hidden_size, num_layers):
            if name == 'lstm':
                return LSTMWithDones(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
            elif name == 'gru':
                return GRUWithDones(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
            else:
                return torch_ext.IdentityRNN(input_size, hidden_size)

        def _init_weights(self, params):
            mlp_init_name = params['mlp']['initializer']['name']
            if mlp_init_name == 'default':
                mlp_init = nn.Identity()
            elif mlp_init_name == 'orthogonal_initializer':
                mlp_init = lambda v: nn.init.orthogonal_(v)
            else:
                mlp_init = lambda v: nn.init.xavier_uniform_(v)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if mlp_init_name != 'default':
                        mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        nn.init.zeros_(m.bias)

            # Sigma init
            sigma_init_cfg = self.space_config['sigma_init']
            if sigma_init_cfg['name'] == 'const_initializer':
                val = sigma_init_cfg.get('val', 0)
                if self.fixed_sigma:
                    nn.init.constant_(self.sigma, val)
                else:
                    nn.init.constant_(self.sigma.weight, val)

        def _load_params(self, params):
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.has_rnn = 'rnn' in params
            self.space_config = params['space']['continuous']
            self.fixed_sigma = self.space_config['fixed_sigma']

            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.rnn_ln = params['rnn'].get('layer_norm', False)
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_concat_input = params['rnn'].get('concat_input', False)

        def encode_depth(self, depth_img):
            """Run CNN encoder on depth image.

            Args:
                depth_img: (N, 1, H, W) depth tensor (any resolution, will be resized)
            Returns:
                (N, CNN_OUT_FEATURES) feature tensor
            """
            # Resize to encoder's expected resolution
            if depth_img.shape[-2:] != (self.depth_height, self.depth_width):
                depth_img = F.interpolate(
                    depth_img, size=(self.depth_height, self.depth_width),
                    mode='bilinear', align_corners=False,
                )
            # Normalize
            depth_img = self.depth_running_mean_std(depth_img)
            return self.depth_encoder(depth_img)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            # Concatenate depth features if available
            if 'depth_features' in obs_dict:
                obs = torch.cat([obs, obs_dict['depth_features']], dim=-1)
            elif 'depth' in obs_dict:
                features = self.encode_depth(obs_dict['depth'])
                obs = torch.cat([obs, features], dim=-1)

            out = obs

            if self.has_rnn:
                seq_length = obs_dict.get('seq_length', 1)
                out_in = out

                if not self.is_rnn_before_mlp:
                    out = self.actor_mlp(out)
                    if self.rnn_concat_input:
                        out = torch.cat([out, out_in], dim=1)

                batch_size = out.size(0)
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1).transpose(0, 1)

                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)

                if len(states) == 1:
                    states = states[0]

                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1).contiguous().reshape(batch_size, -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)

                if type(states) is not tuple:
                    states = (states,)

                if self.is_rnn_before_mlp:
                    out = self.actor_mlp(out)
            else:
                out = self.actor_mlp(out)

            value = self.value(out)
            mu = self.mu_act(self.mu(out))

            if self.fixed_sigma:
                sigma = self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(out))

            return mu, mu * 0 + sigma, value, states

        def is_rnn(self):
            return self.has_rnn

        def is_separate_critic(self):
            return False

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            num_layers = self.rnn_layers
            rnn_units = self.rnn_units
            if self.rnn_name == 'lstm':
                return (
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, rnn_units)),
                )
            else:
                return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)

    def build(self, name, **kwargs):
        net = DepthA2CBuilder.Network(self.params, **kwargs)
        return net
