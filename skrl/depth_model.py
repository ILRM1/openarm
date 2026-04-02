import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
import torch.nn.functional as F

IMG_H, IMG_W = 160, 120
PROPRIO_DIM = 47
CNN_OUT = 128


def _split_obs(states, img_h, img_w, proprio_dim):
    """Split flat states into proprio and depth.

    states: (B, 48 + 160*120) -> proprio (B, 48), depth (B, 1, 160, 120)
    """
    proprio = states[:, :proprio_dim]
    head_depth = states[:, proprio_dim:proprio_dim + img_h * img_w].reshape(-1, 1, img_h, img_w)
    wrist_L_depth = states[:, proprio_dim + img_h * img_w:proprio_dim + 2 * img_h * img_w].reshape(-1, 1, img_h, img_w)
    return proprio, head_depth, wrist_L_depth


class Policy(GaussianMixin, Model):
    def __init__(self, proprio_dim, action_space, head_img_height, head_img_width, wrist_img_height,wrist_img_width, device):
        observation_space = proprio_dim + head_img_height * head_img_width + wrist_img_height * wrist_img_width
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False)

        # 4-layer CNN + LayerNorm
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1,  16, kernel_size=6, stride=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
        ])
        self.layer_norms = nn.ModuleList()
        x = torch.zeros(1, 1, IMG_H, IMG_W)
        for conv in self.conv_layers:
            x = conv(x)
            self.layer_norms.append(nn.LayerNorm(x.shape[1:]))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM x 2 (concat with proprio)
        lstm_input = PROPRIO_DIM
        self.lstm1 = nn.LSTMCell(lstm_input, 1024)
        self.lstm2 = nn.LSTMCell(1024, 1024)

        # MLP [512, 512, 256]
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ELU(),
            nn.Linear(512,  512), nn.ELU(),
            nn.Linear(512,  256), nn.ELU(),
        )

        self.mean_layer = nn.Linear(256, 7)
        self.log_std = nn.Parameter(torch.zeros(7))

        # LSTM hidden state
        self.h1 = self.h2 = self.c1 = self.c2 = None

    def _cnn_forward(self, x):
        for conv, ln in zip(self.conv_layers, self.layer_norms):
            x = F.relu(ln(conv(x)))
        x = self.pool(x)
        x = x.squeeze()
        return x

    def compute(self, inputs, role=""):
        proprio, depth = _split_obs(inputs["states"])

        # CNN -> embedding
        cnn_out = self._cnn_forward(depth)  # (B, CNN_OUT)

        # concat depth embedding + proprio
        #x = torch.cat([cnn_out, proprio], dim=-1)
        x=proprio

        # LSTM x 2 (detach to prevent backprop through previous timesteps)
        if self.h1 is None or self.h1.shape[0] != x.shape[0]:
            self.h1 = torch.zeros(x.shape[0], 1024, device=x.device)
            self.c1 = torch.zeros(x.shape[0], 1024, device=x.device)
            self.h2 = torch.zeros(x.shape[0], 1024, device=x.device)
            self.c2 = torch.zeros(x.shape[0], 1024, device=x.device)

        self.h1, self.c1 = self.lstm1(x,  (self.h1.detach(), self.c1.detach()))
        self.h2, self.c2 = self.lstm2(self.h1, (self.h2.detach(), self.c2.detach()))

        # MLP
        out = self.mlp(self.h2)

        return self.mean_layer(out), self.log_std.expand(out.shape[0], -1), {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # 4-layer CNN + LayerNorm
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1,  16, kernel_size=6, stride=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
        ])
        self.layer_norms = nn.ModuleList()
        x = torch.zeros(1, 1, IMG_H, IMG_W)
        for conv in self.conv_layers:
            x = conv(x)
            self.layer_norms.append(nn.LayerNorm(x.shape[1:]))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM x 2 (concat with proprio)
        lstm_input =  PROPRIO_DIM
        self.lstm1 = nn.LSTMCell(lstm_input, 1024)
        self.lstm2 = nn.LSTMCell(1024, 1024)

        # MLP [512, 512, 256]
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.ELU(),
            nn.Linear(512,  512), nn.ELU(),
            nn.Linear(512,  256), nn.ELU(),
        )

        self.value_layer = nn.Linear(256, 1)

        # LSTM hidden state
        self.h1 = self.h2 = self.c1 = self.c2 = None

    def _cnn_forward(self, x):
        for conv, ln in zip(self.conv_layers, self.layer_norms):
            x = F.relu(ln(conv(x)))
        x = self.pool(x)
        x = x.squeeze()
        return x

    def compute(self, inputs, role=""):
        proprio, depth = _split_obs(inputs["states"])

        # CNN -> embedding
        cnn_out = self._cnn_forward(depth)  # (B, CNN_OUT)

        # concat depth embedding + proprio
        #x = torch.cat([cnn_out, proprio], dim=-1)
        x=proprio

        # LSTM x 2
        if self.h1 is None or self.h1.shape[0] != x.shape[0]:
            self.h1 = torch.zeros(x.shape[0], 1024, device=x.device)
            self.c1 = torch.zeros(x.shape[0], 1024, device=x.device)
            self.h2 = torch.zeros(x.shape[0], 1024, device=x.device)
            self.c2 = torch.zeros(x.shape[0], 1024, device=x.device)

        self.h1, self.c1 = self.lstm1(x,  (self.h1.detach(), self.c1.detach()))
        self.h2, self.c2 = self.lstm2(self.h1, (self.h2.detach(), self.c2.detach()))

        # MLP
        out = self.mlp(self.h2)

        return self.value_layer(out), {}


# import torch
# import torch.nn as nn
# from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
# import torch.nn.functional as F

# IMG_H, IMG_W = 160, 120
# PROPRIO_DIM = 34
# CNN_OUT = 128


# def _split_obs(states):
#     """Split flat states into proprio and depth.

#     states: (B, 34 + 160*120) -> proprio (B, 34), depth (B, 1, 160, 120)
#     """
#     proprio = states[:, :PROPRIO_DIM]
#     depth = states[:, PROPRIO_DIM:].reshape(-1, 1, IMG_H, IMG_W)

#     depth = torch.clamp(depth, 0.0, 5.0) / 5.0
#     return proprio, depth


# class Policy(GaussianMixin, Model):
#     def __init__(self, observation_space, action_space, device):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions=False)

#         # 4-layer CNN + LayerNorm
#         self.conv_layers = nn.ModuleList([
#             nn.Conv2d(1,  16, kernel_size=6, stride=2),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2),
#         ])
#         self.layer_norms = nn.ModuleList()
#         x = torch.zeros(1, 1, IMG_H, IMG_W)
#         for conv in self.conv_layers:
#             x = conv(x)
#             self.layer_norms.append(nn.LayerNorm(x.shape[1:]))

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))

#         # MLP [512, 512, 256]
#         self.mlp = nn.Sequential(
#             nn.Linear( CNN_OUT + PROPRIO_DIM, 512), nn.ELU(),
#             nn.Linear(512,  512), nn.ELU(),
#             nn.Linear(512,  256), nn.ELU(),
#         )

#         self.mean_layer = nn.Linear(256, self.num_actions)
#         self.log_std = nn.Parameter(torch.zeros(self.num_actions))


#     def _cnn_forward(self, x):
#         for conv, ln in zip(self.conv_layers, self.layer_norms):
#             x = F.relu(ln(conv(x)))
#         x = self.pool(x)
#         x = x.squeeze()
#         return x

#     def compute(self, inputs, role):
#         proprio, depth = _split_obs(inputs["states"])

#         # CNN -> embedding
#         cnn_out = self._cnn_forward(depth)  # (B, CNN_OUT)

#         # concat depth embedding + proprio
#         x = torch.cat([cnn_out, proprio], dim=-1)

#         # MLP
#         out = self.mlp(x)

#         return self.mean_layer(out), self.log_std.expand(out.shape[0], -1), {}


# class Value(DeterministicMixin, Model):
#     def __init__(self, observation_space, action_space, device):
#         Model.__init__(self, observation_space, action_space, device)
#         DeterministicMixin.__init__(self, clip_actions=False)

#         # 4-layer CNN + LayerNorm
#         self.conv_layers = nn.ModuleList([
#             nn.Conv2d(1,  16, kernel_size=6, stride=2),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2),
#         ])
#         self.layer_norms = nn.ModuleList()
#         x = torch.zeros(1, 1, IMG_H, IMG_W)
#         for conv in self.conv_layers:
#             x = conv(x)
#             self.layer_norms.append(nn.LayerNorm(x.shape[1:]))

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))

#         self.mlp = nn.Sequential(
#             nn.Linear(CNN_OUT + PROPRIO_DIM, 512), nn.ELU(),
#             nn.Linear(512,  512), nn.ELU(),
#             nn.Linear(512,  256), nn.ELU(),
#         )

#         self.value_layer = nn.Linear(256, 1)

#     def _cnn_forward(self, x):
#         for conv, ln in zip(self.conv_layers, self.layer_norms):
#             x = F.relu(ln(conv(x)))
#         x = self.pool(x)
#         x = x.squeeze()
#         return x

#     def compute(self, inputs, role):
#         proprio, depth = _split_obs(inputs["states"])

#         # CNN -> embedding
#         cnn_out = self._cnn_forward(depth)  # (B, CNN_OUT)

#         # concat depth embedding + proprio
#         x = torch.cat([cnn_out, proprio], dim=-1)

#         # MLP
#         out = self.mlp(x)

#         return self.value_layer(out), {}

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

# IMG_H, IMG_W = 160, 120
# PROPRIO_DIM = 47
# DEPTH_DIM = IMG_H * IMG_W


# def split_obs(states: torch.Tensor):
#     """
#     states: (B, PROPRIO_DIM + DEPTH_DIM)
#     returns:
#         proprio: (B, PROPRIO_DIM)
#         depth:   (B, 1, IMG_H, IMG_W)
#     """
#     proprio = states[:, :PROPRIO_DIM]
#     depth = states[:, PROPRIO_DIM:].reshape(-1, 1, IMG_H, IMG_W)
#     return proprio, depth


# class DepthEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv_layers = nn.ModuleList([
#             nn.Conv2d(1, 16, kernel_size=6, stride=2),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2),
#         ])

#         self.layer_norms = nn.ModuleList()

#         with torch.no_grad():
#             x = torch.zeros(1, 1, IMG_H, IMG_W)
#             for conv in self.conv_layers:
#                 x = conv(x)
#                 self.layer_norms.append(nn.LayerNorm(x.shape[1:]))

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.output_dim = 128

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for conv, ln in zip(self.conv_layers, self.layer_norms):
#             x = conv(x)
#             x = ln(x)
#             x = F.relu(x)

#         x = self.pool(x)          # (B, 128, 1, 1)
#         x = torch.flatten(x, 1)   # (B, 128)
#         return x


# class Policy(GaussianMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True,
#                  min_log_std=-20, max_log_std=2):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(
#             self,
#             clip_actions=clip_actions,
#             clip_log_std=clip_log_std,
#             min_log_std=min_log_std,
#             max_log_std=max_log_std,
#         )

#         self.encoder = DepthEncoder()

#         self.mlp = nn.Sequential(
#             nn.Linear(self.encoder.output_dim + PROPRIO_DIM, 512),
#             nn.ELU(),
#             nn.Linear(512, 512),
#             nn.ELU(),
#             nn.Linear(512, 256),
#             nn.ELU(),
#         )

#         self.mean_layer = nn.Linear(256, self.num_actions)
#         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

#     def compute(self, inputs, role):
#         states = inputs["states"]

#         proprio, depth = split_obs(states)

#         # 필요하면 여기서 depth normalize
#         # depth = torch.clamp(depth, 0.0, 5.0) / 5.0

#         depth_feature = self.encoder(depth)
#         x = torch.cat([depth_feature, proprio], dim=-1)

#         hidden = self.mlp(x)
#         mean = self.mean_layer(hidden)
#         log_std = self.log_std_parameter.unsqueeze(0).expand(mean.shape[0], -1)

#         return mean, log_std, {}


# class Value(DeterministicMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=False):
#         Model.__init__(self, observation_space, action_space, device)
#         DeterministicMixin.__init__(self, clip_actions=clip_actions)

#         self.encoder = DepthEncoder()

#         self.mlp = nn.Sequential(
#             nn.Linear(self.encoder.output_dim + PROPRIO_DIM, 512),
#             nn.ELU(),
#             nn.Linear(512, 512),
#             nn.ELU(),
#             nn.Linear(512, 256),
#             nn.ELU(),
#         )

#         self.value_layer = nn.Linear(256, 1)

#     def compute(self, inputs, role):
#         states = inputs["states"]

#         proprio, depth = split_obs(states)

#         # 필요하면 여기서 depth normalize
#         # depth = torch.clamp(depth, 0.0, 5.0) / 5.0

#         depth_feature = self.encoder(depth)
#         x = torch.cat([depth_feature, proprio], dim=-1)

#         hidden = self.mlp(x)
#         value = self.value_layer(hidden)

#         return value, {}